"""
Agent 1 — Fundamental Analysis Agent
Uses: data/silver/silver_edgar_<TICKER>.csv
Proposed fine-tuned model: Mistral Small 4

Current version is deterministic and explainable. Later, the scoring section can
be replaced with fine-tuned model inference while keeping the same JSON output.
"""

from __future__ import annotations

import argparse
import csv
import json
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


@dataclass
class FundamentalAgentResult:
    agent: str
    ticker: str
    model_name: str
    risk_score: float
    risk_label: str
    confidence: str
    features: dict[str, Any]
    evidence: list[str]


class FundamentalAnalysisAgent:
    def __init__(self, silver_dir: str = config.LOCAL_SILVER):
        self.silver_dir = silver_dir
        self.model_name = "Mistral Small 4 - fine-tuned fundamental agent candidate"

    def _path(self, ticker: str) -> str:
        return os.path.join(self.silver_dir, f"silver_edgar_{ticker.upper()}.csv")

    def load(self, ticker: str) -> list[dict[str, str]]:
        path = self._path(ticker)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing SEC silver file: {path}")
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return sorted(rows, key=lambda row: _parse_date(row.get("end_date", "")))

    def analyze(self, ticker: str) -> FundamentalAgentResult:
        rows = self.load(ticker)
        if not rows:
            raise ValueError(f"No SEC rows found for {ticker.upper()}")

        latest = rows[-1]
        assets = _safe_float(latest.get("total_assets_usd_m"))
        liabilities = _safe_float(latest.get("total_liabilities_usd_m"))
        equity = _safe_float(latest.get("stockholders_equity_usd_m"))
        cash = _safe_float(latest.get("cash_usd_m"))
        net_income = _safe_float(latest.get("net_income_usd_m"))
        operating_income = _safe_float(latest.get("operating_income_usd_m"))
        long_term_debt = _safe_float(latest.get("long_term_debt"))

        # Most EDGAR silver values are USD millions; some long_term_debt values
        # are still raw dollars, so normalize when the value is obviously raw.
        if long_term_debt is not None and long_term_debt > 1_000_000:
            long_term_debt = long_term_debt / 1_000_000

        net_income_series = [
            value
            for value in (_safe_float(row.get("net_income_usd_m")) for row in rows)
            if value is not None
        ]
        income_trend = 0.0
        if len(net_income_series) >= 2:
            first = net_income_series[0]
            last = net_income_series[-1]
            if abs(first) > 1e-9:
                income_trend = float((last - first) / abs(first))

        debt_to_assets = long_term_debt / assets if assets and long_term_debt is not None else None
        liabilities_to_assets = liabilities / assets if assets and liabilities is not None else None
        equity_to_assets = equity / assets if assets and equity is not None else None
        cash_to_debt = cash / long_term_debt if long_term_debt and cash is not None else None
        net_income_to_assets = net_income / assets if assets and net_income is not None else None

        risk_parts = []
        evidence = []

        if debt_to_assets is not None:
            debt_score = _clip_score(1 + debt_to_assets * 12)
            risk_parts.append((debt_score, 0.30))
            evidence.append(f"Long-term debt/assets is {debt_to_assets:.2f}.")

        if liabilities_to_assets is not None:
            liability_score = _clip_score(1 + liabilities_to_assets * 9)
            risk_parts.append((liability_score, 0.25))
            evidence.append(f"Total liabilities/assets is {liabilities_to_assets:.2f}.")

        if equity_to_assets is not None:
            equity_score = _clip_score(10 - equity_to_assets * 10)
            risk_parts.append((equity_score, 0.20))
            evidence.append(f"Equity/assets is {equity_to_assets:.2f}.")

        if net_income_to_assets is not None:
            income_score = _clip_score(6 - net_income_to_assets * 40)
            risk_parts.append((income_score, 0.15))
            evidence.append(f"Net income/assets is {net_income_to_assets:.2f}.")

        if cash_to_debt is not None:
            cash_score = _clip_score(8 - cash_to_debt * 4)
            risk_parts.append((cash_score, 0.10))
            evidence.append(f"Cash/debt is {cash_to_debt:.2f}.")

        if income_trend < -0.20:
            evidence.append(f"Net income trend is weakening ({income_trend:.1%}).")
        elif income_trend > 0.20:
            evidence.append(f"Net income trend is improving ({income_trend:.1%}).")

        if not risk_parts:
            score = 5.0
            confidence = "LOW"
            evidence.append("Not enough SEC numeric fields were available; defaulted to neutral risk.")
        else:
            total_weight = sum(weight for _, weight in risk_parts)
            score = sum(part_score * weight for part_score, weight in risk_parts) / total_weight
            score = round(_clip_score(score), 2)
            confidence = "HIGH" if len(risk_parts) >= 4 else "MEDIUM"

        return FundamentalAgentResult(
            agent="fundamental_analysis_agent",
            ticker=ticker.upper(),
            model_name=self.model_name,
            risk_score=score,
            risk_label=_risk_label(score),
            confidence=confidence,
            features={
                "assets_usd_m": assets,
                "liabilities_usd_m": liabilities,
                "equity_usd_m": equity,
                "cash_usd_m": cash,
                "long_term_debt_usd_m": long_term_debt,
                "net_income_usd_m": net_income,
                "operating_income_usd_m": operating_income,
                "debt_to_assets": debt_to_assets,
                "liabilities_to_assets": liabilities_to_assets,
                "equity_to_assets": equity_to_assets,
                "cash_to_debt": cash_to_debt,
                "net_income_to_assets": net_income_to_assets,
                "net_income_trend": income_trend,
            },
            evidence=evidence,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Fundamental Analysis Agent.")
    parser.add_argument("ticker", help="Ticker symbol, for example KO or AAPL")
    parser.add_argument("--out", help="Optional JSON output path")
    args = parser.parse_args()

    result = asdict(FundamentalAnalysisAgent().analyze(args.ticker))
    payload = json.dumps(result, indent=2)
    print(payload)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload + "\n")


if __name__ == "__main__":
    main()
