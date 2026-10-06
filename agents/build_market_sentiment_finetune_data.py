"""
Build market/volatility/news-sentiment LoRA training data.

Input artifacts:
  - data/gold/gold_risk_scores_ALL.csv
  - data/silver/silver_prices_<TICKER>.csv
  - data/silver/silver_news_sentiment.csv

Output:
  - data/gold/market_sentiment_finetune_data.jsonl

Each JSONL row follows the chat-message format expected by
agents/finetune_lora.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from agents.market_sentiment_agent import MarketSentimentAgent


DEFAULT_GOLD_PATH = PROJECT_ROOT / config.LOCAL_GOLD / "gold_risk_scores_ALL.csv"
DEFAULT_OUT_PATH = PROJECT_ROOT / config.LOCAL_GOLD / "market_sentiment_finetune_data.jsonl"


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, "", "nan", "NaN"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _risk_label(score: float) -> str:
    if score <= config.RISK_SCORE_LOW:
        return "LOW"
    if score <= config.RISK_SCORE_HIGH:
        return "MODERATE"
    return "HIGH"


def _read_gold(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _rounded_features(features: dict[str, Any]) -> dict[str, Any]:
    rounded = {}
    for key, value in features.items():
        number = _safe_float(value)
        if number is None:
            rounded[key] = value
        else:
            rounded[key] = round(number, 6)
    return rounded


def _target_from_gold(row: dict[str, str]) -> tuple[float, str]:
    volatility = _safe_float(row.get("volatility_risk"))
    sentiment = _safe_float(row.get("sentiment_risk"))
    composite = _safe_float(row.get("composite_risk"), 5.0) or 5.0

    if volatility is not None and sentiment is not None:
        score = volatility * 0.70 + sentiment * 0.30
    elif volatility is not None:
        score = volatility
    elif sentiment is not None:
        score = sentiment
    else:
        score = composite

    score = round(max(1.0, min(10.0, score)), 2)
    return score, _risk_label(score)


def build_example(row: dict[str, str], agent: MarketSentimentAgent) -> dict[str, Any] | None:
    ticker = row["ticker"].upper()
    try:
        analysis = asdict(agent.analyze(ticker))
    except Exception as exc:
        print(f"Skipping {ticker}: {exc}")
        return None

    target_score, target_label = _target_from_gold(row)
    features = _rounded_features(analysis["features"])
    features.update({
        "sector": row.get("sector") or "Unknown sector",
        "gold_volatility_risk": round(_safe_float(row.get("volatility_risk"), 5.0) or 5.0, 2),
        "gold_sentiment_risk": round(_safe_float(row.get("sentiment_risk"), 5.0) or 5.0, 2),
        "gold_composite_risk": round(_safe_float(row.get("composite_risk"), 5.0) or 5.0, 2),
        "gold_risk_label": row.get("risk_label") or "MODERATE",
    })

    prompt = {
        "ticker": ticker,
        "task": "Assess market volatility and financial news sentiment risk.",
        "features": features,
        "evidence": analysis["evidence"],
        "output_rules": [
            "Return strict JSON only.",
            "Use Yahoo-derived market features and NewsAPI sentiment features.",
            "Classify risk as LOW, MODERATE, or HIGH.",
        ],
    }
    response = {
        "market_sentiment_risk_score": target_score,
        "market_sentiment_risk_label": target_label,
        "explanation": (
            "Use volatility, beta, drawdown, trend, YTD return, and news sentiment "
            "evidence to justify the market/news risk score."
        ),
    }

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a market volatility and financial news sentiment risk analyst. Return strict JSON.",
            },
            {"role": "user", "content": json.dumps(prompt, separators=(",", ":"))},
            {"role": "assistant", "content": json.dumps(response, separators=(",", ":"))},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build market/news fine-tuning JSONL.")
    parser.add_argument("--gold-path", default=str(DEFAULT_GOLD_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    args = parser.parse_args()

    gold_path = Path(args.gold_path)
    out_path = Path(args.out)
    if not gold_path.exists():
        raise SystemExit(f"Gold table not found: {gold_path}")

    agent = MarketSentimentAgent(
        silver_dir=str(PROJECT_ROOT / config.LOCAL_SILVER),
        news_path=str(PROJECT_ROOT / config.LOCAL_SILVER / "silver_news_sentiment.csv"),
    )
    rows = _read_gold(gold_path)
    examples = []
    label_counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0}

    for row in rows:
        example = build_example(row, agent)
        if example is None:
            continue
        examples.append(example)
        assistant = json.loads(example["messages"][-1]["content"])
        label_counts[assistant["market_sentiment_risk_label"]] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=True) + "\n")

    print(f"Wrote {len(examples)} examples to {out_path}")
    print(f"Label counts: {label_counts}")


if __name__ == "__main__":
    main()
