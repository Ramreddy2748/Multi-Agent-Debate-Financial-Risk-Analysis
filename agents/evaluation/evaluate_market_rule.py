"""Evaluate the deterministic market rule on the same JSONL holdout as LoRA models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = ["LOW", "MODERATE", "HIGH"]


def clip(value: float) -> float:
    return max(1.0, min(10.0, value))


def label(score: float) -> str:
    return "LOW" if score <= 3.5 else "MODERATE" if score <= 6.0 else "HIGH"


def score(features: dict) -> float:
    vol = float(features.get("annualized_volatility") or 0.0)
    beta = float(features.get("beta") or 1.0)
    drawdown = abs(float(features.get("max_drawdown") or 0.0))
    high_gap = abs(float(features.get("price_vs_52w_high_pct") or 0.0))
    sma_gap = float(features.get("sma_50_gap") or 0.0)
    vol_score = clip((vol / 0.50) * 10)
    beta_score = clip(beta * 5)
    drawdown_score = clip(drawdown / 0.50 * 10)
    trend_score = clip(5 - sma_gap * 25)
    high_gap_score = clip(high_gap / 50 * 10)
    return round(clip(
        vol_score * 0.35 + beta_score * 0.20 + drawdown_score * 0.25
        + trend_score * 0.10 + high_gap_score * 0.10
    ), 4)


def macro_f1(true: list[str], pred: list[str]) -> float:
    values = []
    for current in LABELS:
        tp = sum(t == current and p == current for t, p in zip(true, pred))
        fp = sum(t != current and p == current for t, p in zip(true, pred))
        fn = sum(t == current and p != current for t, p in zip(true, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(sum(values) / len(values), 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    true, predicted, score_errors = [], [], []
    for row in rows:
        messages = row["messages"]
        user = next(item["content"] for item in messages if item["role"] == "user")
        assistant = next(item["content"] for item in messages if item["role"] == "assistant")
        features = json.loads(user).get("features", {})
        target = json.loads(assistant)
        rule_score = score(features)
        true.append(str(target["market_risk_label"]).upper())
        predicted.append(label(rule_score))
        score_errors.append(abs(rule_score - float(target["market_risk_score"])))
    metrics = {
        "model": "deterministic_market_rule",
        "examples": len(rows),
        "accuracy": round(sum(t == p for t, p in zip(true, predicted)) / len(true), 4),
        "macro_f1": macro_f1(true, predicted),
        "score_mae": round(sum(score_errors) / len(score_errors), 4),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
