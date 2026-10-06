"""
Agent 2 — Market / Volatility / Sentiment Agent
Uses:
  - data/silver/silver_prices_<TICKER>.csv
  - data/silver/silver_news_sentiment.csv

Proposed fine-tuned model:
  Market/Volatility + Sentiment risk model using numeric price features and
  financial news sentiment signals.

Current version is deterministic and explainable. Later, this file can keep the
same JSON interface while the scoring section is replaced by fine-tuned model
inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def _risk_label(score: float) -> str:
    if score <= config.RISK_SCORE_LOW:
        return "LOW"
    if score <= config.RISK_SCORE_HIGH:
        return "MODERATE"
    return "HIGH"


def _clip_score(value: float) -> float:
    return float(max(1.0, min(10.0, value)))


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, "", "nan", "NaN"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError):
        return datetime.min


def _mean(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


@dataclass
class MarketSentimentAgentResult:
    agent: str
    ticker: str
    model_name: str
    risk_score: float
    risk_label: str
    confidence: str
    features: dict[str, Any]
    evidence: list[str]


class MarketSentimentAgent:
    def __init__(
        self,
        silver_dir: str = config.LOCAL_SILVER,
        news_path: str | None = None,
    ):
        self.silver_dir = silver_dir
        self.news_path = news_path or os.path.join(config.LOCAL_SILVER, "silver_news_sentiment.csv")
        self.model_name = "Fine-tuned market volatility + news sentiment agent candidate"

    def _price_path(self, ticker: str) -> str:
        return os.path.join(self.silver_dir, f"silver_prices_{ticker.upper()}.csv")

    def load_prices(self, ticker: str) -> list[dict[str, str]]:
        path = self._price_path(ticker)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing price silver file: {path}")
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return sorted(rows, key=lambda row: _parse_date(row.get("date", "")))

    def load_news(self) -> list[dict[str, str]]:
        if not os.path.exists(self.news_path):
            return []
        with open(self.news_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _market_score(self, rows: list[dict[str, str]]) -> tuple[float, dict[str, Any], list[str]]:
        latest = rows[-1]
        close_values = [_safe_float(row.get("close")) for row in rows]
        close_values = [value for value in close_values if value is not None and value > 0]
        returns = [_safe_float(row.get("daily_return")) for row in rows]
        returns = [value for value in returns if value is not None]

        latest_close = _safe_float(latest.get("close"))
        latest_beta = _safe_float(latest.get("beta"), 1.0) or 1.0
        latest_vol_30d = _safe_float(latest.get("rolling_vol_30d"))
        latest_vol_60d = _safe_float(latest.get("rolling_vol_60d"))
        latest_drawdown_52w = _safe_float(latest.get("price_vs_52w_high_pct"))
        latest_sma_50 = _safe_float(latest.get("sma_50"))

        if returns:
            mean_return = sum(returns) / len(returns)
            variance = sum((value - mean_return) ** 2 for value in returns) / max(len(returns) - 1, 1)
            annual_volatility = math.sqrt(variance) * math.sqrt(252)
        else:
            annual_volatility = latest_vol_60d or latest_vol_30d or 0.0

        max_drawdown = 0.0
        if close_values:
            peak = close_values[0]
            drawdowns = []
            for value in close_values:
                peak = max(peak, value)
                drawdowns.append((value - peak) / peak)
            max_drawdown = min(drawdowns)

        sma_gap = None
        if latest_close and latest_sma_50:
            sma_gap = (latest_close - latest_sma_50) / latest_sma_50

        vol_score = _clip_score((annual_volatility / 0.50) * 10)
        beta_score = _clip_score(latest_beta * 5)
        drawdown_score = _clip_score(abs(max_drawdown) / 0.50 * 10)
        trend_score = 5.0 if sma_gap is None else _clip_score(5 - sma_gap * 25)
        high_gap_score = 5.0
        if latest_drawdown_52w is not None:
            high_gap_score = _clip_score(abs(latest_drawdown_52w) / 50 * 10)

        score = _clip_score(
            vol_score * 0.35 +
            beta_score * 0.20 +
            drawdown_score * 0.25 +
            trend_score * 0.10 +
            high_gap_score * 0.10
        )

        features = {
            "latest_close": latest_close,
            "annualized_volatility": annual_volatility,
            "rolling_vol_30d": latest_vol_30d,
            "rolling_vol_60d": latest_vol_60d,
            "beta": latest_beta,
            "max_drawdown": max_drawdown,
            "price_vs_52w_high_pct": latest_drawdown_52w,
            "sma_50_gap": sma_gap,
            "price_rows_used": len(rows),
        }
        evidence = [
            f"Annualized volatility is {annual_volatility:.1%}.",
            f"Beta is {latest_beta:.2f}.",
            f"Max drawdown over available window is {max_drawdown:.1%}.",
        ]
        if latest_drawdown_52w is not None:
            evidence.append(f"Price is {latest_drawdown_52w:.1f}% from 52-week high.")
        if sma_gap is not None:
            evidence.append(f"Latest close vs SMA-50 gap is {sma_gap:.1%}.")

        return score, features, evidence

    def _sentiment_score(self, ticker: str) -> tuple[float, dict[str, Any], list[str], str]:
        rows_all = self.load_news()
        rows = [row for row in rows_all if str(row.get("ticker", "")).upper() == ticker.upper()]

        if not rows:
            return (
                5.0,
                {
                    "sentiment_mean": None,
                    "sentiment_std": None,
                    "article_count": 0,
                    "news_days_covered": 0,
                },
                ["No company-specific news sentiment rows found; sentiment defaulted to neutral."],
                "LOW",
            )

        rows = sorted(rows, key=lambda row: _parse_date(row.get("date", "")))
        article_counts = [_safe_float(row.get("article_count"), 0.0) or 0.0 for row in rows]
        total_articles = int(sum(article_counts))

        weighted_total = 0.0
        weight_sum = 0.0
        for row, count in zip(rows, article_counts):
            sentiment = _safe_float(row.get("sentiment_mean"), 0.0) or 0.0
            weight = max(count, 1.0)
            weighted_total += sentiment * weight
            weight_sum += weight
        weighted_sentiment = weighted_total / weight_sum if weight_sum else 0.0

        sentiment_std = _mean([
            value
            for value in (_safe_float(row.get("sentiment_std")) for row in rows)
            if value is not None
        ])
        if sentiment_std is None:
            sentiment_std = 0.0

        score = _clip_score(5.5 - weighted_sentiment * 5)
        if sentiment_std > 0.50:
            score = _clip_score(score + 0.75)

        if total_articles >= 10:
            confidence = "HIGH"
        elif total_articles >= 3:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        dates = [_parse_date(row.get("date", "")) for row in rows]
        days_covered = len(set(row.get("date") for row in rows))
        features = {
            "sentiment_mean": weighted_sentiment,
            "sentiment_std": sentiment_std,
            "article_count": total_articles,
            "news_days_covered": days_covered,
            "first_news_date": min(dates).date().isoformat(),
            "last_news_date": max(dates).date().isoformat(),
        }
        evidence = [
            f"Weighted news sentiment_mean is {weighted_sentiment:.3f}.",
            f"Article count is {total_articles} across {days_covered} day(s).",
            f"Average sentiment_std is {sentiment_std:.3f}.",
        ]
        if total_articles < 3:
            evidence.append("Low article volume makes the sentiment component less reliable.")

        return score, features, evidence, confidence

    def analyze(self, ticker: str) -> MarketSentimentAgentResult:
        ticker = ticker.upper()
        price_rows = self.load_prices(ticker)
        if not price_rows:
            raise ValueError(f"No price rows found for {ticker}")

        market_score, market_features, market_evidence = self._market_score(price_rows)
        sentiment_score, sentiment_features, sentiment_evidence, sentiment_confidence = self._sentiment_score(ticker)

        sentiment_weight = 0.25
        market_weight = 0.75
        if sentiment_confidence == "LOW":
            sentiment_weight = 0.10
            market_weight = 0.90

        score = round(_clip_score(market_score * market_weight + sentiment_score * sentiment_weight), 2)
        confidence = "HIGH" if len(price_rows) >= 120 and sentiment_confidence != "LOW" else "MEDIUM"

        evidence = [
            f"Market component score is {market_score:.2f}.",
            f"Sentiment component score is {sentiment_score:.2f}.",
            f"Component weights are market={market_weight:.2f}, sentiment={sentiment_weight:.2f}.",
        ] + market_evidence + sentiment_evidence

        return MarketSentimentAgentResult(
            agent="market_sentiment_agent",
            ticker=ticker,
            model_name=self.model_name,
            risk_score=score,
            risk_label=_risk_label(score),
            confidence=confidence,
            features={
                **market_features,
                **sentiment_features,
                "market_component_score": market_score,
                "sentiment_component_score": sentiment_score,
                "market_weight": market_weight,
                "sentiment_weight": sentiment_weight,
                "sentiment_confidence": sentiment_confidence,
            },
            evidence=evidence,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Market / Volatility / Sentiment Agent.")
    parser.add_argument("ticker", help="Ticker symbol, for example KO or AAPL")
    parser.add_argument("--out", help="Optional JSON output path")
    args = parser.parse_args()

    result = asdict(MarketSentimentAgent().analyze(args.ticker))
    payload = json.dumps(result, indent=2)
    print(payload)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload + "\n")


if __name__ == "__main__":
    main()
