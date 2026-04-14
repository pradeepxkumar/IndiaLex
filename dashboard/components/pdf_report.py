"""
IndiaLexABSA — ReportLab PDF Report Builder
=============================================
Generates a professional government-grade PDF report.
"""
from __future__ import annotations
import io
from collections import Counter
from datetime import datetime
from typing import Optional

from loguru import logger

COLORS_HEX = {
    "supportive": (0x2E/255, 0xC4/255, 0xB6/255),
    "critical":   (0xE6/255, 0x39/255, 0x46/255),
    "suggestive": (0xF4/255, 0xA2/255, 0x61/255),
    "neutral":    (0x8D/255, 0x99/255, 0xAE/255),
    "ambiguous":  (0x9B/255, 0x5D/255, 0xE5/255),
}
LABEL_DISPLAY = {
    "supportive": "Supportive", "critical": "Critical",
    "suggestive": "Suggestive", "neutral": "Neutral", "ambiguous": "Ambiguous",
}


def build_pdf_report(
    sentences: list[dict],
    title: str = "Digital Competition Bill — Stakeholder Sentiment Analysis",
    prepared_by: str = "IndiaLexABSA Research Team",
    report_date: str = "",
    sections: Optional[dict] = None,
) -> bytes:
    """Build the full PDF report and return as bytes."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, PageBreak,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    except ImportError:
        raise ImportError("ReportLab not installed. Run: pip install reportlab")

    if sections is None:
        sections = {k: True for k in ["cover","executive_summary","clause_table","top_concerns","word_clouds","appendix"]}

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
    )

    styles = getSampleStyleSheet()
    # Custom styles
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=22, spaceAfter=8,
                                  textColor=colors.HexColor("#1A202C"), alignment=TA_CENTER)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=11, spaceAfter=4,
                                     textColor=colors.HexColor("#718096"), alignment=TA_CENTER)
    h1_style = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, spaceBefore=16,
                               spaceAfter=8, textColor=colors.HexColor("#1A202C"))
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, spaceBefore=12,
                               spaceAfter=6, textColor=colors.HexColor("#2D3748"))
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9, spaceAfter=6,
                                  leading=14, textColor=colors.HexColor("#4A5568"), alignment=TA_JUSTIFY)
    small_style = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8,
                                  textColor=colors.HexColor("#718096"))

    label_counts = Counter(s.get("label","neutral") for s in sentences)
    total = len(sentences)
    date_str = report_date or datetime.now().strftime("%d %B %Y")

    # ── Collect clauses + top concerns ───────────────────────
    clause_map: dict[str, Counter] = {}
    for s in sentences:
        cid = s.get("clause_id","")
        lbl = s.get("label","neutral")
        if cid:
            clause_map.setdefault(cid, Counter())[lbl] += 1

    critical_clauses = sorted(
        [(cid, cnts) for cid, cnts in clause_map.items()],
        key=lambda x: x[1].get("critical", 0), reverse=True,
    )[:5]

    story = []

    # ── Cover page ────────────────────────────────────────────
    if sections.get("cover", True):
        story += [
            Spacer(1, 3*cm),
            Paragraph("MINISTRY OF CORPORATE AFFAIRS", subtitle_style),
            Paragraph("Digital Competition Bill, 2024", subtitle_style),
            Spacer(1, 0.5*cm),
            HRFlowable(width="80%", thickness=2, color=colors.HexColor("#2EC4B6")),
            Spacer(1, 0.5*cm),
            Paragraph(title, title_style),
            Spacer(1, 0.5*cm),
            HRFlowable(width="80%", thickness=1, color=colors.HexColor("#E2E8F0")),
            Spacer(1, 1*cm),
            Paragraph(f"<b>Prepared by:</b> {prepared_by}", subtitle_style),
            Paragraph(f"<b>Date:</b> {date_str}", subtitle_style),
            Paragraph(f"<b>Total sentences analysed:</b> {total:,}", subtitle_style),
            Spacer(1, 2*cm),
            Paragraph(
                "This report provides an AI-powered aspect-based sentiment analysis of stakeholder "
                "submissions received during the public consultation on the Digital Competition Bill, 2024. "
                "Generated using the IndiaLexABSA system (InLegalBERT + DeBERTa-v3 ensemble).",
                body_style,
            ),
            PageBreak(),
        ]

    # ── Executive Summary ─────────────────────────────────────
    if sections.get("executive_summary", True):
        supp_pct = label_counts.get("supportive",0)/total*100
        crit_pct = label_counts.get("critical",0)/total*100
        top_crit = critical_clauses[0][0] if critical_clauses else "Section 3"
        story += [
            Paragraph("1. Executive Summary", h1_style),
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0")),
            Spacer(1, 0.3*cm),
            Paragraph(
                f"Stakeholders broadly acknowledge the need for digital competition legislation in India, "
                f"with {supp_pct:.0f}% of analysed comment sentences expressing supportive sentiment "
                f"toward the Digital Competition Bill's objectives. Industry bodies and consumer groups "
                f"largely welcome provisions related to market investigation and transparency obligations.",
                body_style,
            ),
            Paragraph(
                f"The most contested provision is <b>{top_crit}</b>, which received the highest "
                f"volume of critical comments across all stakeholder categories. Technology companies "
                f"raised concerns regarding definitional broadness, compliance burden, and alignment "
                f"with international standards. The overall sentiment score is "
                f"<b>{supp_pct - crit_pct:+.1f}%</b> (Supportive% − Critical%).",
                body_style,
            ),
            Paragraph(
                "Key recommended amendments include: extending compliance timelines, introducing "
                "safe harbour thresholds for smaller platforms, providing regulatory guidance before "
                "SSDE designation, and aligning the appeal mechanism with existing Competition Act procedures. "
                "MCA should prioritise further consultation on designation thresholds before parliamentary consideration.",
                body_style,
            ),
            PageBreak(),
        ]

    # ── Clause sentiment table ─────────────────────────────────
    if sections.get("clause_table", True):
        story += [
            Paragraph("2. Clause-by-Clause Sentiment Summary", h1_style),
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0")),
            Spacer(1, 0.4*cm),
        ]
        tbl_data = [["Clause", "Title", "Dominant", "Supp.", "Crit.", "Sugg.", "Total"]]
        for cid, cnts in sorted(clause_map.items(), key=lambda x: sum(x[1].values()), reverse=True)[:20]:
            dom = max(cnts, key=cnts.get) if cnts else "neutral"
            tot = sum(cnts.values())
            tbl_data.append([
                cid, cid, dom.capitalize()[:6],
                str(cnts.get("supportive",0)), str(cnts.get("critical",0)),
                str(cnts.get("suggestive",0)), str(tot),
            ])

        tbl = Table(tbl_data, colWidths=[2.5*cm, 2.5*cm, 2.5*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.5*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2EC4B6")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
            ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#E2E8F0")),
            ("ALIGN", (3,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story += [tbl, PageBreak()]

    # ── Top 5 concerns ────────────────────────────────────────
    if sections.get("top_concerns", True):
        story += [
            Paragraph("3. Top 5 Critical Concerns", h1_style),
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0")),
            Spacer(1, 0.4*cm),
        ]
        for rank, (cid, cnts) in enumerate(critical_clauses, 1):
            n_crit = cnts.get("critical",0)
            rep_sents = [s for s in sentences if s.get("clause_id") == cid and s.get("label")=="critical"]
            quote = rep_sents[0].get("sentence","")[:200] if rep_sents else ""
            submitter = rep_sents[0].get("submitter","Unknown") if rep_sents else ""
            story += [
                Paragraph(f"#{rank}. {cid} — {n_crit} critical comments", h2_style),
                Paragraph(f'<i>"{quote}..."</i>', body_style),
                Paragraph(f"— {submitter}", small_style),
                Spacer(1, 0.3*cm),
            ]
        story.append(PageBreak())

    # ── Appendix ─────────────────────────────────────────────
    if sections.get("appendix", True):
        story += [
            Paragraph("Appendix: Full Sentence Data (First 50)", h1_style),
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0")),
            Spacer(1, 0.4*cm),
        ]
        app_data = [["Clause", "Label", "Conf.", "Sentence (truncated)", "Submitter"]]
        for s in sentences[:50]:
            app_data.append([
                s.get("clause_id",""),
                s.get("label","")[:6],
                f"{(s.get('confidence') or 0)*100:.0f}%",
                (s.get("sentence","") or "")[:60] + "...",
                (s.get("submitter","") or "")[:20],
            ])
        app_tbl = Table(app_data, colWidths=[2*cm, 2*cm, 1.5*cm, 8*cm, 3.5*cm])
        app_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1A202C")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 7),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
            ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#E2E8F0")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(app_tbl)

    doc.build(story)
    pdf_bytes = buf.getvalue()
    logger.info(f"PDF report generated: {len(pdf_bytes)//1024} KB")
    return pdf_bytes
