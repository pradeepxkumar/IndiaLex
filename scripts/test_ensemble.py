"""Quick test of the ensemble model."""
import sys
sys.path.insert(0, ".")
from models.ensemble import EnsemblePredictor

pred = EnsemblePredictor(demo_mode=False)

test_sentences = [
    "We strongly oppose the mandatory data sharing provisions under Section 12.",
    "We welcome the Digital Competition Bill as a necessary step forward.",
    "The definition of SSDE should be revised to include clearer thresholds.",
    "Section 3 provides for the designation of SSDEs based on turnover.",
]

print("=" * 60)
print("ENSEMBLE MODEL TEST")
print("=" * 60)
for sent in test_sentences:
    r = pred.predict_one(sent, "")
    lbl = r["label"]
    conf = r["confidence"]
    src = r["source"]
    print(f"\nInput: {sent[:65]}...")
    print(f"  -> Label: {lbl:12s} | Confidence: {conf:.3f} | Source: {src}")

print(f"\nStats: {pred.get_stats()}")
print("\nDONE!")
