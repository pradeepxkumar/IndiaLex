"""
Train DistilBERT as challenger model — fast on CPU.
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score

VALID_LABELS = sorted(["supportive", "critical", "suggestive", "neutral", "ambiguous"])
label2id = {l: i for i, l in enumerate(VALID_LABELS)}
id2label = {i: l for l, i in label2id.items()}

train_data = [json.loads(l) for l in open(ROOT / "data/processed/train.jsonl", encoding="utf-8")]
val_data = [json.loads(l) for l in open(ROOT / "data/processed/val.jsonl", encoding="utf-8")]
logger.info(f"Train={len(train_data)}, Val={len(val_data)}")

MODEL_ID = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=5, id2label=id2label, label2id=label2id,
    ignore_mismatched_sizes=True,
)
logger.info(f"Loaded {MODEL_ID} ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")


class LazyDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        enc = self.tok(self.data[idx]["text"], truncation=True, max_length=self.max_len)
        enc["labels"] = label2id.get(self.data[idx].get("label", "neutral"), 3)
        return enc


ckpt_dir = ROOT / "models" / "checkpoints" / "deberta_absa"
ckpt_dir.mkdir(parents=True, exist_ok=True)

args = TrainingArguments(
    output_dir=str(ckpt_dir),
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=5,
    report_to="none",
    dataloader_pin_memory=False,
)

def compute_metrics(ep):
    logits, labels = ep
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=LazyDataset(train_data, tokenizer),
    eval_dataset=LazyDataset(val_data, tokenizer),
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

logger.info("Starting DistilBERT training...")
start = time.time()
trainer.train()

best_dir = ckpt_dir / "best"
trainer.save_model(str(best_dir))
tokenizer.save_pretrained(str(best_dir))
results = trainer.evaluate()
elapsed = time.time() - start

logger.info(f"DONE in {elapsed/60:.1f} min | {results}")
logger.info(f"Saved to {best_dir}")
