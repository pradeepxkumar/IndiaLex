"""
IndiaLexABSA — Demo Data Provider
===================================
Provides realistic mock data for demo mode.
Covers 3 submitters, 47 clauses, 287 sentences with
realistic sentiment distributions.
"""
from __future__ import annotations

import random
from typing import Optional

random.seed(2024)

LABELS = ["supportive", "critical", "suggestive", "neutral", "ambiguous"]
LABEL_WEIGHTS = [0.20, 0.32, 0.22, 0.18, 0.08]  # realistic DCB distribution

SUBMITTERS = [
    {"submitter": "Google India Pvt Ltd", "category": "startup"},
    {"submitter": "NASSCOM", "category": "industry_body"},
    {"submitter": "Cyril Amarchand Mangaldas", "category": "law_firm"},
]

SENTENCE_POOL = {
    "critical": [
        "We strongly oppose the designation thresholds in Section 3 as they are unworkably low and will capture companies that pose no systemic risk.",
        "The mandatory data sharing requirements under Section 12 would fundamentally compromise trade secrets and must be rejected.",
        "Section 5's per se prohibition approach is disproportionate and departs from the well-established rule-of-reason analysis.",
        "The definition of 'digital market' in Section 2(1)(g) is so broad as to create significant regulatory uncertainty.",
        "The monetary thresholds for SSDE designation are inconsistent with international standards such as the EU Digital Markets Act.",
        "Section 4's interoperability mandate imposes unrealistic technical requirements without adequate implementation guidance.",
        "We are deeply concerned that Section 26's penalty regime fails to account for good-faith compliance attempts.",
        "The timeline of 90 days for compliance under Section 10 is wholly inadequate for companies of any size.",
    ],
    "supportive": [
        "We welcome the Digital Competition Bill as a necessary step toward ensuring fair competition in digital markets.",
        "The self-assessment framework under Section 9 is a sensible, proportionate compliance approach.",
        "NASSCOM broadly supports the objectives of preventing anti-competitive conduct outlined in Section 7.",
        "The CCI's proposed investigative powers under Section 16 are reasonable and proportionate.",
        "We commend the government's effort to align India's digital competition law with global best practices.",
        "Section 35's market study provisions will provide valuable insights for evidence-based regulation.",
        "The commitment mechanism under Section 21 is a constructive addition that reduces litigation burden.",
        "We support the inclusion of consumer protection provisions in Section 36.",
    ],
    "suggestive": [
        "We recommend including a safe harbour provision in Section 7 for startups with annual revenue below ₹500 crore.",
        "The definition of SSDE in Section 2 should be revised to exclude B2B platforms from its scope.",
        "We suggest that the CCI be required to publish detailed guidance on the SSDE designation process.",
        "Section 40 should explicitly provide for interim stay of penalties pending appeal.",
        "We propose that the compliance timeline under Section 10 be extended to 180 days for initial implementation.",
        "It would be beneficial to include a consultation mechanism between CCI and sectoral regulators under Section 38.",
        "We recommend that Section 12's data sharing obligations be subject to a data localisation requirement.",
        "The appeal mechanism in Section 40 should align with established Competition Act 2002 procedures.",
    ],
    "neutral": [
        "Section 3 provides for the designation of Systemically Significant Digital Enterprises based on user base and revenue thresholds.",
        "The Act establishes a compliance officer requirement under Section 11.",
        "Section 47 contains a three-year sunset review clause.",
        "The Digital Competition Bill was released for public consultation in March 2024.",
        "Section 2(1)(e) defines 'Systemically Significant Digital Enterprise' for purposes of this Act.",
        "The Bill proposes amendments to the Competition Act 2002 through Section 46.",
        "Section 15 provides the Commission with powers to conduct investigations.",
        "The Bill covers 47 sections addressing digital market conduct.",
    ],
    "ambiguous": [
        "While we appreciate the intent of Section 4, its operational implications require further regulatory guidance.",
        "Our members hold mixed views on Section 12 — some see merit in data portability while others cite privacy concerns.",
        "The provisions of Section 3 may be appropriate in certain contexts but create uncertainty for growing platforms.",
        "Section 7's anti-steering provisions could be beneficial or harmful depending on how the CCI interprets them.",
    ],
}

CLAUSE_POOL = [
    {"clause_id": f"S{i}", "clause_title": title, "section_num": i}
    for i, title in [
        (2, "Definitions"),
        (3, "Designation of SSDEs"),
        (4, "Interoperability Obligations"),
        (5, "Per se Prohibitions"),
        (7, "Anti-steering Obligations"),
        (9, "Self-assessment Framework"),
        (10, "Compliance Requirements"),
        (11, "Compliance Officer"),
        (12, "Data Access and Sharing"),
        (14, "Powers of the Commission"),
        (15, "Investigation Powers"),
        (16, "Inquiry Procedures"),
        (21, "Commitment Mechanism"),
        (26, "Penalty Provisions"),
        (35, "Market Study"),
        (36, "Consumer Protection"),
        (38, "Sectoral Regulator Interface"),
        (40, "Appellate Tribunal"),
        (46, "Amendments to Competition Act"),
        (47, "Sunset and Review"),
    ]
]


def get_demo_sentences(n: int = 287) -> list[dict]:
    """Generate realistic demo sentence data."""
    sentences = []
    sent_id = 0

    for submitter_info in SUBMITTERS:
        submitter = submitter_info["submitter"]
        category = submitter_info["category"]
        doc_id = f"demo_{submitter[:4].lower().replace(' ', '')}"

        for _ in range(n // len(SUBMITTERS)):
            label = random.choices(LABELS, weights=LABEL_WEIGHTS)[0]
            pool = SENTENCE_POOL.get(label, SENTENCE_POOL["neutral"])
            text = random.choice(pool)
            clause = random.choice(CLAUSE_POOL)
            conf = random.uniform(0.55, 0.95)

            sentences.append({
                "sent_id": f"demo_{sent_id:04d}",
                "doc_id": doc_id,
                "submitter": submitter,
                "category": category,
                "sentence": text,
                "text": text,
                "clause_id": clause["clause_id"],
                "clause_title": clause["clause_title"],
                "section_num": clause["section_num"],
                "label": label,
                "confidence": round(conf, 3),
                "similarity_score": round(random.uniform(0.4, 0.92), 3),
                "language": "en",
                "was_translated": False,
                "label_source": random.choice(["inlegalbert", "ensemble", "deberta"]),
            })
            sent_id += 1

    random.shuffle(sentences)
    return sentences[:n]


def get_demo_docs() -> list[dict]:
    """Return demo document metadata."""
    return [
        {
            "doc_id": f"demo_{s['submitter'][:4].lower().replace(' ', '')}",
            "filename": f"{s['submitter'].lower().replace(' ', '_')}_submission.pdf",
            "submitter": s["submitter"],
            "category": s["category"],
            "word_count": random.randint(2000, 8000),
            "num_pages": random.randint(4, 20),
            "extraction_method": "demo",
        }
        for s in SUBMITTERS
    ]


def get_demo_heatmap_data() -> dict:
    """Return clause×sentiment matrix for heatmap."""
    clauses = [c["clause_id"] for c in CLAUSE_POOL]
    labels = LABELS
    import numpy as np
    np.random.seed(2024)

    # Realistic distribution: critical skewed toward regulatory clauses
    matrix = {}
    for clause in CLAUSE_POOL:
        cid = clause["clause_id"]
        sec = clause["section_num"]
        # More critical comments on data/penalty/interop sections
        weights = [0.15, 0.38, 0.25, 0.15, 0.07] if sec in [3, 4, 5, 12, 26] else [0.25, 0.20, 0.25, 0.22, 0.08]
        total = np.random.randint(4, 45)
        counts = np.random.multinomial(total, weights).tolist()
        matrix[cid] = {l: c for l, c in zip(labels, counts)}

    return {"clauses": CLAUSE_POOL, "matrix": matrix}
