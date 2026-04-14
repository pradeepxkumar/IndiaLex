"""
IndiaLexABSA — MCA Portal Scraper
==================================
Scrapes the Ministry of Corporate Affairs consultation portal
to collect stakeholder submission PDF links for the Digital
Competition Bill public comment process.

Usage:
    python data/scraper/mca_scraper.py --output data/raw/pdf_urls.json
"""
from __future__ import annotations

import argparse
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from loguru import logger
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import requests
from bs4 import BeautifulSoup


# ─── Target URLs ─────────────────────────────────────────────────────────────
MCA_BASE = "https://www.mca.gov.in"
CONSULTATION_URLS = [
    "https://www.mca.gov.in/content/mca/global/en/public-consultation.html",
    # Known direct consultation pages
    "https://www.mca.gov.in/content/mca/global/en/data/web-content/digital-competition-bill-2024.html",
]

# Additional known repositories where MCA submissions are published
SUPPLEMENTARY_URLS = [
    "https://prsindia.org/billtrack/digital-competition-bill-2024",
    "https://www.meity.gov.in/content/public-consultation",
]


@dataclass
class PDFRecord:
    url: str
    filename: str
    submitter: Optional[str]
    date: Optional[str]
    source_page: str
    category: str  # law_firm / startup / individual / industry_body / government


class MCAScraper:
    """Selenium-based scraper for MCA consultation PDFs."""

    def __init__(self, headless: bool = True, delay_range: tuple[float, float] = (1.5, 3.5)):
        self.headless = headless
        self.delay_range = delay_range
        self.driver: Optional[webdriver.Chrome] = None
        self.records: list[PDFRecord] = []

    # ─── Driver lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        opts = Options()
        if self.headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=opts)
        self.driver.implicitly_wait(10)
        logger.info("Chrome driver started")

    def stop(self) -> None:
        if self.driver:
            self.driver.quit()
            logger.info("Chrome driver stopped")

    # ─── Scraping helpers ─────────────────────────────────────────────────────

    def _random_delay(self) -> None:
        time.sleep(random.uniform(*self.delay_range))

    def _wait_for_element(self, by: By, value: str, timeout: int = 15) -> None:
        WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )

    def _extract_pdf_links(self, page_url: str) -> list[PDFRecord]:
        """Extract all PDF links from a single page."""
        try:
            self.driver.get(page_url)
            self._random_delay()
            soup = BeautifulSoup(self.driver.page_source, "lxml")
        except Exception as exc:
            logger.error(f"Failed to load {page_url}: {exc}")
            return []

        records = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if not href.lower().endswith(".pdf"):
                continue

            # Resolve relative URLs
            if href.startswith("http"):
                full_url = href
            elif href.startswith("/"):
                full_url = MCA_BASE + href
            else:
                continue

            filename = full_url.split("/")[-1]
            submitter = self._infer_submitter(a_tag.get_text(strip=True), filename)
            category = self._classify_submitter(submitter)

            records.append(
                PDFRecord(
                    url=full_url,
                    filename=filename,
                    submitter=submitter,
                    date=None,
                    source_page=page_url,
                    category=category,
                )
            )
            logger.debug(f"Found PDF: {filename}")

        logger.info(f"Found {len(records)} PDFs on {page_url}")
        return records

    def _paginate(self, base_url: str) -> list[str]:
        """Return all paginated URLs from a listing page."""
        pages = [base_url]
        try:
            self.driver.get(base_url)
            self._random_delay()
            soup = BeautifulSoup(self.driver.page_source, "lxml")
            # Look for pagination links
            for a in soup.find_all("a", href=True):
                text = a.get_text(strip=True).lower()
                if any(kw in text for kw in ["next", "page", "2", "3", "4", "5"]):
                    href = a["href"]
                    if href.startswith("/"):
                        href = MCA_BASE + href
                    if href not in pages and MCA_BASE in href:
                        pages.append(href)
        except Exception as exc:
            logger.warning(f"Pagination failed for {base_url}: {exc}")
        return pages

    @staticmethod
    def _infer_submitter(link_text: str, filename: str) -> str:
        """Extract submitter name from link text or filename."""
        # Prefer link text if informative
        if len(link_text) > 5 and link_text.lower() not in ("pdf", "download", "view"):
            return link_text.strip()
        # Fall back to filename sans extension
        return filename.replace("_", " ").replace("-", " ").rsplit(".", 1)[0].strip()

    @staticmethod
    def _classify_submitter(name: str) -> str:
        """Classify submitter into a category based on name heuristics."""
        name_lower = name.lower()
        law_keywords = ["law", "llp", "legal", "advocate", "counsel", "barrister", "attorney", "solicitor"]
        startup_keywords = ["startup", "tech", "digital", "platform", "pvt ltd", "private limited"]
        industry_keywords = ["nasscom", "ficci", "cii", "assocham", "federation", "association", "council", "chamber"]
        govt_keywords = ["ministry", "government", "department", "authority", "commission", "tribunal"]

        if any(kw in name_lower for kw in govt_keywords):
            return "government"
        if any(kw in name_lower for kw in law_keywords):
            return "law_firm"
        if any(kw in name_lower for kw in industry_keywords):
            return "industry_body"
        if any(kw in name_lower for kw in startup_keywords):
            return "startup"
        return "individual"

    # ─── Public API ───────────────────────────────────────────────────────────

    def scrape(self) -> list[PDFRecord]:
        """Run the full scrape across all configured URLs."""
        if not self.driver:
            self.start()

        seen_urls: set[str] = set()

        for base_url in CONSULTATION_URLS:
            logger.info(f"Scraping: {base_url}")
            pages = self._paginate(base_url)
            for page_url in pages:
                records = self._extract_pdf_links(page_url)
                for rec in records:
                    if rec.url not in seen_urls:
                        seen_urls.add(rec.url)
                        self.records.append(rec)
                self._random_delay()

        # Also scrape supplementary sources with requests (lighter weight)
        for url in SUPPLEMENTARY_URLS:
            self._scrape_with_requests(url, seen_urls)

        logger.info(f"Total unique PDFs found: {len(self.records)}")
        return self.records

    def _scrape_with_requests(self, url: str, seen_urls: set[str]) -> None:
        """Lightweight scrape using requests (no JS needed)."""
        try:
            resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".pdf") and href not in seen_urls:
                    full_url = href if href.startswith("http") else "https:" + href if href.startswith("//") else url + href
                    seen_urls.add(full_url)
                    self.records.append(PDFRecord(
                        url=full_url,
                        filename=full_url.split("/")[-1],
                        submitter=self._infer_submitter(a.get_text(strip=True), full_url.split("/")[-1]),
                        date=None,
                        source_page=url,
                        category="individual",
                    ))
        except Exception as exc:
            logger.warning(f"Requests scrape failed for {url}: {exc}")

    def save(self, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(r) for r in self.records]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} records → {output_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MCA consultation PDF scraper")
    parser.add_argument("--output", default="data/raw/pdf_urls.json")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    scraper = MCAScraper(headless=args.headless)
    try:
        scraper.scrape()
        scraper.save(args.output)
    finally:
        scraper.stop()


if __name__ == "__main__":
    main()
