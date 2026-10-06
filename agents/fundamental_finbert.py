"""
Fine-tuned FinBERT inference helper for the Fundamental Agent.

This module is optional at runtime. If the model folder or ML dependencies are
missing, callers can keep using the deterministic fundamental fallback.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import inspect
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "fundamental_agent_finbert"
DEFAULT_TRAIN_JSONL = PROJECT_ROOT / "data" / "gold" / "fundamental_finetune_data.jsonl"
LABELS = ["LOW", "MODERATE", "HIGH"]
LABEL2ID = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
ID2LABEL = {0: "LOW", 1: "MODERATE", 2: "HIGH"}

_RUNTIME: dict[str, Any] | None = None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "nan", "NaN"):
            return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def create_fundamental_text(row: dict[str, Any]) -> str:
    ticker = str(row.get("ticker") or "UNKNOWN").upper()
    company = str(row.get("company") or ticker).replace("\xa0", " ").strip()
    end_date = row.get("end_date") or row.get("filed") or row.get("as_of_date") or "unknown"
    form = row.get("form") or "SEC filing"
    cash = row.get("cash_usd_m", "unknown")
    debt = row.get("long_term_debt", row.get("long_term_debt_usd_m", "unknown"))
    net_income = row.get("net_income_usd_m", "unknown")
    operating_income = row.get("operating_income_usd_m", "unknown")
    revenue = row.get("revenue", row.get("revenue_usd_m", row.get("revenue_alt", "unknown")))
    equity = row.get("stockholders_equity", row.get("stockholders_equity_usd_m", "unknown"))
    return (
        f"Company {company} with ticker {ticker} filed {form} for period ending {end_date}. "
        f"Cash is {cash} million USD. Long-term debt is {debt} million USD. "
        f"Net income is {net_income} million USD. Operating income is {operating_income} million USD. "
        f"Revenue is {revenue} million USD. Stockholders equity is {equity} million USD. "
        "Classify the company's fundamental financial risk as LOW, MODERATE, or HIGH."
    )


def score_from_label(label: str, confidence: float) -> float:
    if label == "LOW":
        return round(max(1.0, 3.5 - confidence * 1.5), 2)
    if label == "MODERATE":
        return round(4.25 + confidence * 1.75, 2)
    return round(min(10.0, 6.25 + confidence * 2.5), 2)


def financial_signals(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    positive, negative = [], []
    cash = safe_float(row.get("cash_usd_m"))
    debt = safe_float(row.get("long_term_debt", row.get("long_term_debt_usd_m")))
    if abs(debt) > 1_000_000:
        debt = debt / 1_000_000.0
    net_income = safe_float(row.get("net_income_usd_m"))
    operating_income = safe_float(row.get("operating_income_usd_m"))
    revenue = safe_float(row.get("revenue", row.get("revenue_usd_m", row.get("revenue_alt"))))
    equity = safe_float(row.get("stockholders_equity", row.get("stockholders_equity_usd_m")))

    debt_to_cash = debt / (cash + 1.0)
    debt_to_equity = debt / (equity + 1.0) if equity > 0 else 999.0
    profit_margin = net_income / (revenue + 1.0) if revenue > 0 else 0.0
    operating_margin = operating_income / (revenue + 1.0) if revenue > 0 else 0.0

    if debt_to_cash <= 2:
        positive.append("Cash reserves are reasonable relative to long-term debt.")
    else:
        negative.append("Long-term debt is elevated relative to cash reserves.")

    if debt_to_equity <= 1:
        positive.append("Debt-to-equity does not indicate severe leverage pressure.")
    elif debt_to_equity <= 2:
        negative.append("Debt-to-equity indicates moderate leverage exposure.")
    else:
        negative.append("Debt-to-equity indicates high leverage pressure.")

    if profit_margin >= 0.08:
        positive.append("Profit margin indicates positive earnings strength.")
    elif profit_margin >= 0:
        negative.append("Profit margin is low, limiting financial flexibility.")
    else:
        negative.append("Negative profit margin indicates profitability stress.")

    if operating_margin >= 0.10:
        positive.append("Operating margin shows stable core business performance.")
    elif operating_margin >= 0:
        negative.append("Operating margin is thin, suggesting limited operational cushion.")
    else:
        negative.append("Negative operating margin indicates weak core operations.")

    return positive, negative


def _load_runtime(model_dir: str | Path | None = None) -> dict[str, Any] | None:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    path = Path(model_dir or os.getenv("FUNDAMENTAL_FINBERT_MODEL_DIR", DEFAULT_MODEL_DIR)).expanduser()
    if not (path / "config.json").exists():
        return None

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        _RUNTIME = {"tokenizer": tokenizer, "model": model, "device": device, "model_dir": path}
        return _RUNTIME
    except Exception:
        return None


def predict_fundamental_finbert(row: dict[str, Any], model_dir: str | Path | None = None) -> dict[str, Any] | None:
    runtime = _load_runtime(model_dir)
    if runtime is None:
        return None

    try:
        import torch

        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        device = runtime["device"]
        text = create_fundamental_text(row)
        encoded = tokenizer(text, truncation=True, max_length=180, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**encoded)
            probs = torch.softmax(output.logits, dim=1).cpu().numpy()[0]

        pred_id = int(probs.argmax())
        label = model.config.id2label.get(pred_id, LABELS[pred_id] if pred_id < len(LABELS) else "MODERATE")
        label = str(label).upper()
        confidence = round(float(probs[pred_id]), 4)
        risk_score = score_from_label(label, confidence)
        positive, negative = financial_signals(row)
        ticker = str(row.get("ticker") or "UNKNOWN").upper()
        company = str(row.get("company") or ticker).replace("\xa0", " ").strip()
        return {
            "agent": "Fundamental Agent",
            "ticker": ticker,
            "company": company,
            "risk_score": risk_score,
            "risk_label": label,
            "confidence": confidence,
            "claim_type": "FINBERT_MODEL_INFERENCE",
            "model_name": "ProsusAI/finbert fine-tuned fundamental classifier",
            "model_dir": str(runtime["model_dir"].relative_to(PROJECT_ROOT) if runtime["model_dir"].is_relative_to(PROJECT_ROOT) else runtime["model_dir"]),
            "evidence": [
                f"Fine-tuned FinBERT classified fundamental risk as {label} with confidence {confidence:.2f}.",
                "Input text was built from SEC-derived cash, debt, income, revenue, and equity fields.",
            ],
            "positive_signals": positive,
            "negative_signals": negative,
            "overall_assessment": (
                f"{company} ({ticker}) is classified as {label} fundamental risk by the fine-tuned FinBERT model."
            ),
            "raw_probabilities": {LABELS[i]: round(float(prob), 4) for i, prob in enumerate(probs[: len(LABELS)])},
        }
    except Exception:
        return None


def _parse_json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def _row_from_chat(example: dict[str, Any]) -> dict[str, Any] | None:
    messages = example.get("messages") or []
    user_message = next((m for m in messages if m.get("role") == "user"), None)
    assistant_message = next((m for m in messages if m.get("role") == "assistant"), None)
    if not user_message or not assistant_message:
        return None

    user_payload = _parse_json_object(user_message.get("content", "{}"))
    assistant_payload = _parse_json_object(assistant_message.get("content", "{}"))
    label = str(assistant_payload.get("fundamental_risk_label") or assistant_payload.get("risk_label") or "").upper()
    if label not in LABEL2ID:
        return None

    features = user_payload.get("features") or {}
    if not isinstance(features, dict):
        features = {}
    row = {
        **features,
        "ticker": user_payload.get("ticker"),
        "company": user_payload.get("company") or user_payload.get("ticker"),
        "fundamental_risk_label": label,
        "fundamental_risk_score": assistant_payload.get("fundamental_risk_score"),
    }
    return {
        "text": create_fundamental_text(row),
        "label": LABEL2ID[label],
        "risk_label": label,
        "ticker": row.get("ticker"),
    }


def load_training_examples(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                row = _row_from_chat(parsed)
                if row is not None:
                    rows.append(row)
            except Exception as exc:
                print(f"Skipping line {line_number}: {exc}")
    if not rows:
        raise ValueError(f"No trainable examples found in {path}")
    return rows


def stratified_split(rows: list[dict[str, Any]], val_size: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["label"]), []).append(row)

    train_rows, val_rows = [], []
    for group in grouped.values():
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_size)) if len(group) > 1 else 0
        val_rows.extend(group[:n_val])
        train_rows.extend(group[n_val:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def train_fundamental_finbert(args: Any) -> None:
    try:
        import torch
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, f1_score
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install with:\n"
            "  pip install torch transformers datasets scikit-learn accelerate safetensors\n"
            f"Original error: {exc}"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_jsonl = Path(args.train_jsonl)
    output_dir = Path(args.output_dir)
    rows = load_training_examples(train_jsonl)
    train_rows, val_rows = stratified_split(rows, args.val_size, args.seed)

    label_counts = {label: sum(1 for row in rows if row["risk_label"] == label) for label in LABELS}
    print(f"Loaded {len(rows)} examples from {train_jsonl}")
    print(f"Label counts: {label_counts}")
    print(f"Train rows: {len(train_rows)} | Validation rows: {len(val_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    train_dataset = Dataset.from_list([{k: row[k] for k in ("text", "label")} for row in train_rows])
    val_dataset = Dataset.from_list([{k: row[k] for k in ("text", "label")} for row in val_rows])
    train_dataset = train_dataset.map(tokenize, batched=True).remove_columns(["text"])
    val_dataset = val_dataset.map(tokenize, batched=True).remove_columns(["text"])
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved FinBERT fundamental model to {output_dir}")
    print(json.dumps(metrics, indent=2))


def predict_from_cli(args: Any) -> None:
    import csv

    path = Path(args.silver_dir) / f"silver_edgar_{args.ticker.upper()}.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8-sig")))
    row = rows[-1]
    row["ticker"] = args.ticker.upper()
    result = predict_fundamental_finbert(row, args.model_dir)
    if result is None:
        raise SystemExit("FinBERT model unavailable. Train it first or check dependencies.")
    print(json.dumps(result, indent=2))


def main() -> None:
    import argparse

    # Backward-compatible shorthand:
    #   python3 agents/fundamental_finbert.py AAPL
    # becomes:
    #   python3 agents/fundamental_finbert.py predict AAPL
    if len(sys.argv) > 1 and sys.argv[1] not in {"train", "predict", "-h", "--help"}:
        sys.argv.insert(1, "predict")

    parser = argparse.ArgumentParser(description="Train or run the Fundamental FinBERT classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the FinBERT classifier.")
    train_parser.add_argument("--train-jsonl", default=str(DEFAULT_TRAIN_JSONL))
    train_parser.add_argument("--output-dir", default=str(DEFAULT_MODEL_DIR))
    train_parser.add_argument("--base-model", default="ProsusAI/finbert")
    train_parser.add_argument("--epochs", type=float, default=4)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--max-len", type=int, default=180)
    train_parser.add_argument("--val-size", type=float, default=0.20)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(func=train_fundamental_finbert)

    predict_parser = subparsers.add_parser("predict", help="Run the trained FinBERT classifier.")
    predict_parser.add_argument("ticker")
    predict_parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    predict_parser.add_argument("--silver-dir", default=str(PROJECT_ROOT / "data" / "silver"))
    predict_parser.set_defaults(func=predict_from_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
