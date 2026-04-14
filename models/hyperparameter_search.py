"""
IndiaLexABSA — Optuna Hyperparameter Search
=============================================
30-trial TPE search over learning rate, batch size, warmup, dropout.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger


def run_hpo(model_type: str = "inlegalbert", n_trials: int = 30) -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("Optuna not installed: pip install optuna")
        return {}

    dataset_dir = Path("dataset/IndiaLexABSA_v1")

    def _load(split):
        items = []
        with open(dataset_dir / f"{split}.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    train_data = _load("train")
    val_data = _load("validation")

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        batch = trial.suggest_categorical("batch_size", [8, 16, 32])
        warmup = trial.suggest_float("warmup_ratio", 0.05, 0.2)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)

        try:
            if model_type == "inlegalbert":
                from models.inlegalbert_absa import InLegalBERTABSA
                model = InLegalBERTABSA(
                    checkpoint_dir=f"models/checkpoints/hpo_trial_{trial.number}",
                )
                model.train(train_data, val_data, num_epochs=2,
                            learning_rate=lr, batch_size=batch, warmup_ratio=warmup)
                preds = model.predict(
                    [d.get("sentence","") for d in val_data[:100]],
                    [d.get("clause_title","") for d in val_data[:100]],
                )
            else:
                from models.deberta_absa import DeBERTaABSA
                model = DeBERTaABSA(
                    checkpoint_dir=f"models/checkpoints/hpo_trial_{trial.number}",
                )
                model.train(train_data, val_data, num_epochs=2,
                            learning_rate=lr, batch_size=batch, warmup_ratio=warmup)
                preds = model.predict(
                    [d.get("sentence","") for d in val_data[:100]],
                )

            from sklearn.metrics import f1_score
            y_true = [d.get("label","neutral") for d in val_data[:100]]
            y_pred = [p["label"] for p in preds]
            return f1_score(y_true, y_pred, average="macro", zero_division=0)

        except Exception as exc:
            logger.warning(f"Trial {trial.number} failed: {exc}")
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    logger.info(f"Best hyperparameters: {best}")
    logger.info(f"Best macro-F1: {study.best_value:.4f}")
    return {"best_params": best, "best_f1": study.best_value}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="inlegalbert")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    run_hpo(args.model, args.trials)
