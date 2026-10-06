"""Sentiment Agent using NewsAPI headlines and ProsusAI/finbert."""

from __future__ import annotations

import csv
import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze"
SILVER_DIR = PROJECT_ROOT / "data" / "silver"

_SENTIMENT_FINBERT_RUNTIME: dict[str, Any] | None = None
_SENTIMENT_FINBERT_LAST_ERROR: str | None = None


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _label_from_score(score: float) -> str:
    if score <= 3.5:
        return "LOW"
    if score <= 6.0:
        return "MODERATE"
    return "HIGH"


def _clamp_score(score: float) -> float:
    return round(max(1.0, min(10.0, float(score))), 2)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _load_raw_news(ticker: str, limit: int = 40) -> list[dict[str, str]]:
    ticker = ticker.upper()
    rows = [
        row for row in _read_csv(BRONZE_DIR / "newsapi_headlines_raw.csv")
        if str(row.get("ticker", "")).upper() == ticker and row.get("title")
    ]
    return sorted(rows, key=lambda row: str(row.get("published_at", "")), reverse=True)[:limit]


def _load_sentiment_finbert_runtime() -> dict[str, Any] | None:
    global _SENTIMENT_FINBERT_RUNTIME, _SENTIMENT_FINBERT_LAST_ERROR
    _SENTIMENT_FINBERT_LAST_ERROR = None
    if _SENTIMENT_FINBERT_RUNTIME is not None:
        return _SENTIMENT_FINBERT_RUNTIME

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        _SENTIMENT_FINBERT_LAST_ERROR = f"Missing FinBERT sentiment dependency: {exc}"
        return None

    try:
        model_name = os.getenv("SENTIMENT_FINBERT_MODEL", "ProsusAI/finbert")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
            else "cpu"
        )
        model = model.to(device)
        model.eval()
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        _SENTIMENT_FINBERT_RUNTIME = {
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
            "id2label": id2label,
            "model_name": model_name,
        }
        return _SENTIMENT_FINBERT_RUNTIME
    except Exception as exc:
        _SENTIMENT_FINBERT_LAST_ERROR = f"Failed to load FinBERT sentiment runtime: {type(exc).__name__}: {exc}"
        return None


def _score_headlines_with_finbert(news_rows: list[dict[str, str]]) -> dict[str, Any] | None:
    runtime = _load_sentiment_finbert_runtime()
    if runtime is None:
        return None

    try:
        import torch

        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        device = runtime["device"]
        id2label = runtime["id2label"]
        scored = []
        for row in news_rows:
            text = ". ".join(part for part in [row.get("title", ""), row.get("description", "")] if part).strip()
            if not text:
                continue
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=192).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits[0]
                probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
            label_idx = max(range(len(probs)), key=lambda idx: probs[idx])
            label = id2label.get(label_idx, str(label_idx)).lower()
            sentiment_value = 1.0 if label == "positive" else -1.0 if label == "negative" else 0.0
            scored.append({
                "title": row.get("title", ""),
                "source": row.get("source", ""),
                "published_at": row.get("published_at", ""),
                "url": row.get("url", ""),
                "label": label.upper(),
                "confidence": round(float(probs[label_idx]), 4),
                "sentiment_value": sentiment_value,
            })
        return {"items": scored, "model_name": runtime["model_name"]}
    except Exception as exc:
        global _SENTIMENT_FINBERT_LAST_ERROR
        _SENTIMENT_FINBERT_LAST_ERROR = f"FinBERT sentiment inference failed: {type(exc).__name__}: {exc}"
        return None


def _headline_summary(scored: list[dict[str, Any]], target_label: str, limit: int = 3) -> list[str]:
    matches = [item for item in scored if item.get("label") == target_label]
    matches.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    return [
        f"{item.get('label')} headline ({item.get('source') or 'unknown'}): {item.get('title')}"
        for item in matches[:limit]
    ]


def run_sentiment_agent(ticker: str, company_name: Optional[str] = None) -> dict[str, Any]:
    """News sentiment specialist agent backed by ProsusAI/finbert."""
    ticker = ticker.upper()
    news_rows = _load_raw_news(ticker)
    silver_rows = [
        row for row in _read_csv(SILVER_DIR / "silver_news_sentiment.csv")
        if str(row.get("ticker", "")).upper() == ticker
    ]

    if not news_rows and not silver_rows:
        return {
            "agent": "Sentiment Agent",
            "ticker": ticker,
            "company": company_name or ticker,
            "risk_score": 5.0,
            "risk_label": "MODERATE",
            "confidence": 0.10,
            "claim_type": "NO_NEWS_FALLBACK",
            "model_name": "ProsusAI/finbert",
            "metrics": {
                "article_count": 0,
                "finbert_positive": 0,
                "finbert_neutral": 0,
                "finbert_negative": 0,
                "avg_finbert_sentiment": 0.0,
            },
            "evidence": [f"No NewsAPI headlines found for {ticker}; sentiment risk cannot be model-scored."],
            "positive_signals": [],
            "negative_signals": ["News sentiment coverage is unavailable for this ticker."],
            "overall_assessment": "Sentiment risk is unavailable because no ticker-specific headlines were found.",
        }

    finbert = _score_headlines_with_finbert(news_rows) if news_rows else None
    scored = finbert.get("items", []) if finbert else []

    if scored:
        positive_count = sum(1 for item in scored if item.get("label") == "POSITIVE")
        negative_count = sum(1 for item in scored if item.get("label") == "NEGATIVE")
        neutral_count = sum(1 for item in scored if item.get("label") == "NEUTRAL")
        avg_sentiment = statistics.mean(float(item.get("sentiment_value") or 0.0) for item in scored)
        confidence = statistics.mean(float(item.get("confidence") or 0.0) for item in scored)
        score = _clamp_score(5.0 - avg_sentiment * 3.0 + min(negative_count / max(len(scored), 1), 1.0) * 2.0)
        positive_headlines = _headline_summary(scored, "POSITIVE")
        negative_headlines = _headline_summary(scored, "NEGATIVE")
        evidence = [
            f"FinBERT scored {len(scored)} recent NewsAPI headlines for {ticker}.",
            f"Headline sentiment mix: {positive_count} positive, {neutral_count} neutral, {negative_count} negative.",
            f"Average FinBERT sentiment value is {avg_sentiment:.3f}.",
        ]
        evidence.extend((negative_headlines + positive_headlines)[:4])
        return {
            "agent": "Sentiment Agent",
            "ticker": ticker,
            "company": company_name or ticker,
            "risk_score": score,
            "risk_label": _label_from_score(score),
            "confidence": round(max(0.0, min(1.0, confidence)), 2),
            "claim_type": "FINBERT_SENTIMENT_INFERENCE",
            "model_name": finbert.get("model_name", "ProsusAI/finbert"),
            "metrics": {
                "article_count": len(scored),
                "finbert_positive": positive_count,
                "finbert_neutral": neutral_count,
                "finbert_negative": negative_count,
                "avg_finbert_sentiment": round(avg_sentiment, 4),
            },
            "evidence": evidence,
            "positive_signals": positive_headlines[:3],
            "negative_signals": negative_headlines[:3],
            "top_positive_headlines": positive_headlines,
            "top_negative_headlines": negative_headlines,
            "overall_assessment": f"News sentiment risk is {_label_from_score(score)} based on FinBERT headline classification.",
        }

    avg_vader = statistics.mean(
        _safe_float(row.get("sentiment_mean"), 0.0) or 0.0 for row in silver_rows
    ) if silver_rows else 0.0
    article_count = sum(int(_safe_float(row.get("article_count"), 0) or 0) for row in silver_rows)
    score = _clamp_score(5.0 - avg_vader * 4.0)
    evidence = [
        f"FinBERT sentiment unavailable; VADER silver baseline used for {ticker}.",
        f"VADER average sentiment is {avg_vader:.3f} across {article_count} articles.",
    ]
    if _SENTIMENT_FINBERT_LAST_ERROR:
        evidence.append(_SENTIMENT_FINBERT_LAST_ERROR)
    return {
        "agent": "Sentiment Agent",
        "ticker": ticker,
        "company": company_name or ticker,
        "risk_score": score,
        "risk_label": _label_from_score(score),
        "confidence": 0.45 if article_count else 0.15,
        "claim_type": "VADER_BASELINE_FALLBACK",
        "model_name": "VADER baseline; ProsusAI/finbert unavailable",
        "metrics": {
            "article_count": article_count,
            "vader_sentiment_mean": round(avg_vader, 4),
            "finbert_positive": 0,
            "finbert_neutral": 0,
            "finbert_negative": 0,
            "avg_finbert_sentiment": None,
        },
        "evidence": evidence,
        "positive_signals": ["VADER sentiment baseline is non-negative."] if avg_vader >= 0 else [],
        "negative_signals": ["VADER sentiment baseline is negative."] if avg_vader < 0 else [],
        "overall_assessment": f"Sentiment risk is {_label_from_score(score)} using VADER fallback sentiment.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dedicated FinBERT news sentiment agent.")
    parser.add_argument("ticker", help="Ticker symbol to score, for example AAPL.")
    parser.add_argument("--company", default=None, help="Optional company name for report output.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    result = run_sentiment_agent(args.ticker, company_name=args.company)
    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent, sort_keys=bool(indent)))


if __name__ == "__main__":
    main()
