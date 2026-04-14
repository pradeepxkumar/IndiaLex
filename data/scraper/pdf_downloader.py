"""
IndiaLexABSA — PDF Downloader
==============================
Downloads PDFs from the URL list produced by mca_scraper.py.
Features:
  - SHA-256 deduplication (skips already-downloaded files)
  - Resumable downloads (range requests)
  - Concurrent downloads via asyncio + aiohttp
  - Updates the data registry after each download
  - Respects rate limits with configurable delay

Usage:
    python data/scraper/pdf_downloader.py \
        --urls data/raw/pdf_urls.json \
        --output data/raw/ \
        --workers 4
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import aiohttp
from loguru import logger
from tqdm.asyncio import tqdm_asyncio


# ─── Helpers ─────────────────────────────────────────────────────────────────

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize_filename(name: str) -> str:
    """Remove characters unsafe for filenames."""
    keepchars = (" ", ".", "_", "-")
    return "".join(c for c in name if c.isalnum() or c in keepchars).rstrip()


# ─── Async downloader ─────────────────────────────────────────────────────────

class PDFDownloader:
    def __init__(
        self,
        output_dir: str,
        max_workers: int = 4,
        delay_seconds: float = 1.0,
        max_file_mb: int = 50,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.delay = delay_seconds
        self.max_bytes = max_file_mb * 1024 * 1024

        # SHA-256 → filename map for dedup
        self._seen_hashes: dict[str, str] = {}
        self._load_existing()

        # Registry updates
        self._registry_updates: list[dict] = []

    def _load_existing(self) -> None:
        """Pre-compute hashes of already-downloaded files."""
        for pdf_path in self.output_dir.glob("*.pdf"):
            try:
                h = sha256_of_file(pdf_path)
                self._seen_hashes[h] = pdf_path.name
            except Exception:
                pass
        logger.info(f"Pre-loaded {len(self._seen_hashes)} existing PDFs for dedup")

    async def _download_one(
        self, session: aiohttp.ClientSession, record: dict, semaphore: asyncio.Semaphore
    ) -> Optional[dict]:
        """Download a single PDF. Returns registry entry or None."""
        url = record["url"]
        raw_name = record.get("filename") or url.split("/")[-1]
        safe_name = sanitize_filename(raw_name)
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"

        dest = self.output_dir / safe_name

        async with semaphore:
            try:
                headers = {"User-Agent": "Mozilla/5.0 (compatible; IndiaLexABSA/1.0)"}
                # Resume if partial file exists
                if dest.exists():
                    headers["Range"] = f"bytes={dest.stat().st_size}-"

                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status == 416:  # Range not satisfiable → file complete
                        pass
                    elif resp.status not in (200, 206):
                        logger.warning(f"HTTP {resp.status} for {url}")
                        return None

                    content = await resp.read()
                    if len(content) > self.max_bytes:
                        logger.warning(f"Skipping {safe_name} — too large ({len(content)//1024//1024} MB)")
                        return None

                    mode = "ab" if resp.status == 206 else "wb"
                    with open(dest, mode) as f:
                        f.write(content)

            except asyncio.TimeoutError:
                logger.error(f"Timeout downloading {url}")
                return None
            except Exception as exc:
                logger.error(f"Error downloading {url}: {exc}")
                return None

        # Dedup check
        file_hash = sha256_of_file(dest)
        if file_hash in self._seen_hashes and self._seen_hashes[file_hash] != safe_name:
            logger.info(f"Duplicate detected: {safe_name} == {self._seen_hashes[file_hash]}, removing")
            dest.unlink()
            return None

        self._seen_hashes[file_hash] = safe_name
        await asyncio.sleep(self.delay)

        entry = {
            "filename": safe_name,
            "url": url,
            "sha256": file_hash,
            "size_bytes": dest.stat().st_size,
            "submitter": record.get("submitter"),
            "category": record.get("category"),
            "extracted": False,
        }
        logger.debug(f"Downloaded: {safe_name}")
        return entry

    async def download_all_async(self, records: list[dict]) -> list[dict]:
        """Download all PDFs concurrently."""
        semaphore = asyncio.Semaphore(self.max_workers)
        connector = aiohttp.TCPConnector(limit=self.max_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self._download_one(session, rec, semaphore) for rec in records]
            results = await tqdm_asyncio.gather(*tasks, desc="Downloading PDFs")

        successful = [r for r in results if r is not None]
        logger.info(f"Downloaded {len(successful)}/{len(records)} PDFs successfully")
        return successful

    def download_all(self, records: list[dict]) -> list[dict]:
        return asyncio.run(self.download_all_async(records))

    def save_registry(self, entries: list[dict], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Merge with existing registry
        existing = []
        if Path(path).exists():
            with open(path) as f:
                existing = json.load(f)
        existing_names = {e["filename"] for e in existing}
        new_entries = [e for e in entries if e["filename"] not in existing_names]
        all_entries = existing + new_entries
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_entries, f, indent=2, ensure_ascii=False)
        logger.info(f"Registry updated: {len(all_entries)} total entries → {path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk PDF downloader")
    parser.add_argument("--urls", default="data/raw/pdf_urls.json")
    parser.add_argument("--output", default="data/raw/")
    parser.add_argument("--registry", default="data/raw/registry.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.urls, encoding="utf-8") as f:
        records = json.load(f)
    logger.info(f"Loaded {len(records)} PDF URLs")

    downloader = PDFDownloader(args.output, args.workers, args.delay)
    entries = downloader.download_all(records)
    downloader.save_registry(entries, args.registry)


if __name__ == "__main__":
    main()
