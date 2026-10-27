"""Macro-Economic Agent wrapper.

This keeps the orchestrator-facing `run_macro_agent()` interface stable while
Macro v2 moves the heavy work into a point-in-time data pipeline.
"""

from __future__ import annotations

import math
import os
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MACRO_V2_ROOT = PROJECT_ROOT / "data"
MACRO_V2_MODEL_DIR = Path(__file__).resolve().parent / "macro_v2"

SECTOR_TO_ETF = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _clamp_score(score: float) -> float:
    return round(max(1.0, min(10.0, float(score))), 2)


def _label_from_score(score: float) -> str:
    if score <= 3.5:
        return "LOW"
    if score <= 6.0:
        return "MODERATE"
    return "HIGH"


def _macro_v2_root() -> Path:
    return Path(os.getenv("MACRO_V2_DATA_ROOT", str(DEFAULT_MACRO_V2_ROOT))).expanduser()


def _latest_snapshot_path() -> Path:
    return _macro_v2_root() / "gold" / "macro_v2" / "latest_snapshot.parquet"


def _latest_forecast_path() -> Path:
    return Path(os.getenv(
        "MACRO_V2_FORECAST_PATH",
        str(MACRO_V2_MODEL_DIR / "latest_forecast_per_sector.json"),
    )).expanduser()


def _load_latest_snapshot() -> pd.DataFrame:
    path = _latest_snapshot_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Macro v2 latest snapshot not found at {path}. "
            "Run: python3 agents/macro/run_macro_v2.py --out data"
        )
    return pd.read_parquet(path)


def _load_latest_forecasts() -> list[dict[str, Any]]:
    path = _latest_forecast_path()
    if not path.exists():
        raise FileNotFoundError(f"Macro v2 forecast artifact not found at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Macro v2 forecast artifact is empty or invalid: {path}")
    return [row for row in data if isinstance(row, dict)]


def _select_forecast_row(forecasts: list[dict[str, Any]], sector: str) -> dict[str, Any]:
    etf = SECTOR_TO_ETF.get(sector, "")
    if etf:
        for row in forecasts:
            if str(row.get("etf", "")).upper() == etf:
                return row
    for row in forecasts:
        if str(row.get("sector", "")).lower() == sector.lower():
            return row
    return forecasts[-1]


def _select_sector_row(snapshot: pd.DataFrame, sector: str) -> pd.Series:
    etf = SECTOR_TO_ETF.get(sector, "")
    if etf and "etf" in snapshot.columns:
        match = snapshot[snapshot["etf"].astype(str).str.upper() == etf]
        if not match.empty:
            return match.iloc[-1]
    if "sector" in snapshot.columns:
        match = snapshot[snapshot["sector"].astype(str).str.lower() == sector.lower()]
        if not match.empty:
            return match.iloc[-1]
    return snapshot.iloc[-1]


def _macro_v2_forecast_result(
    ticker: str,
    sector: str,
    company_name: str | None,
    row: dict[str, Any],
) -> dict[str, Any]:
    score = _clamp_score(_safe_float(row.get("pred_macro_risk_score"), 5.0) or 5.0)
    label = str(row.get("pred_risk_label") or _label_from_score(score)).upper()
    if label not in {"LOW", "MODERATE", "HIGH"}:
        label = _label_from_score(score)
    expected_vol = _safe_float(row.get("pred_expected_vol_pct"))
    as_of = row.get("as_of_date")

    evidence = [
        f"Macro v2 Phi-4-mini LoRA forecast maps {sector} to sector ETF {row.get('etf', 'N/A')}.",
        f"Forecast as-of date is {as_of or 'N/A'}.",
    ]
    if expected_vol is not None:
        evidence.append(f"Model forecasts next-60-trading-day annualized volatility at {expected_vol:.2f}%.")
    evidence.append(f"Model predicted macro risk label {label} with score {score:.2f}/10.")

    positive = []
    negative = []
    if score <= 3.5:
        positive.append("Macro model forecast is in a low turbulence regime.")
    elif score <= 6.0:
        positive.append("Macro model forecast is not in the highest turbulence bucket.")
        negative.append("Macro model forecast still indicates moderate systematic pressure.")
    else:
        negative.append("Macro model forecast indicates elevated systematic risk.")

    return {
        "agent": "Macro-Economic Agent",
        "ticker": ticker,
        "company": company_name or ticker,
        "sector": sector,
        "as_of_date": as_of,
        "risk_score": score,
        "macro_risk_score": score,
        "risk_label": label,
        "confidence": 0.68,
        "claim_type": "MACRO_V2_LORA_FORECAST",
        "metrics": {
            "etf": row.get("etf"),
            "expected_vol_pct": expected_vol,
            "forecast_split": row.get("split"),
            "valid_json": row.get("valid_json"),
            "has_actual_label": row.get("has_label"),
        },
        "evidence": evidence,
        "positive_signals": positive,
        "negative_signals": negative,
        "sector_sensitivity": f"Macro v2 maps {sector} to sector ETF {row.get('etf', 'N/A')}.",
        "overall_assessment": (
            f"Macro v2 Phi-4-mini LoRA forecast is {label} for {sector}, "
            f"with score {score:.2f}/10."
        ),
    }


def _metric(row: pd.Series, *names: str, default: float | None = None) -> float | None:
    for name in names:
        if name in row.index:
            value = _safe_float(row.get(name))
            if value is not None:
                return value
    return default


def _score_from_macro_v2(row: pd.Series, sector: str) -> tuple[float, list[str], list[str]]:
    """Transparent score from Macro v2 features and sector sensitivities."""
    fed = _metric(row, "DFF", "FEDFUNDS", default=0.0) or 0.0
    ten_year = _metric(row, "DGS10", "GS10", default=0.0) or 0.0
    yield_curve = _metric(row, "T10Y2Y")
    vix = _metric(row, "VIXCLS")
    credit = _metric(row, "BAMLH0A0HYM2", "BAA10Y")
    unemployment = _metric(row, "UNRATE")
    cpi_yoy = _metric(row, "CPIAUCSL_yoy")
    oil_yoy = _metric(row, "DCOILWTICO_yoy")
    beta_rate = abs(_metric(row, "beta_d_dgs10", default=0.0) or 0.0)
    beta_vix = abs(_metric(row, "beta_d_vix", default=0.0) or 0.0)
    beta_credit = abs(_metric(row, "beta_d_credit", default=0.0) or 0.0)

    score = 2.7
    score += min(max(fed - 2.5, 0.0) / 3.5, 1.0) * 1.3
    score += min(max(ten_year - 4.0, 0.0) / 2.0, 1.0) * 0.9
    score += 0.9 if yield_curve is not None and yield_curve < 0 else 0.0
    score += min(max((vix or 0.0) - 18.0, 0.0) / 20.0, 1.0) * 1.2 if vix is not None else 0.0
    score += min(max((credit or 0.0) - 4.0, 0.0) / 6.0, 1.0) * 1.1 if credit is not None else 0.0
    score += min(max((unemployment or 0.0) - 4.2, 0.0) / 2.5, 1.0) * 0.8 if unemployment is not None else 0.0
    score += min(max((cpi_yoy or 0.0) - 2.5, 0.0) / 3.0, 1.0) * 0.8 if cpi_yoy is not None else 0.0
    score += min(beta_rate + beta_vix + beta_credit, 1.5) * 0.7

    if sector in {"Real Estate", "Utilities"}:
        score += 0.5
    elif sector in {"Consumer Staples", "Health Care", "Healthcare"}:
        score -= 0.3
    elif sector in {"Information Technology", "Technology", "Consumer Discretionary"}:
        score += 0.25

    evidence = [
        f"Macro v2 point-in-time snapshot uses sector ETF {row.get('etf', 'N/A')} for {sector}.",
        f"Fed funds is {fed:.2f}%; 10Y Treasury is {ten_year:.2f}%.",
    ]
    if yield_curve is not None:
        evidence.append(f"10Y-2Y yield spread is {yield_curve:.2f}%.")
    if vix is not None:
        evidence.append(f"VIX is {vix:.2f}.")
    if credit is not None:
        evidence.append(f"Credit spread proxy is {credit:.2f}.")
    if cpi_yoy is not None:
        evidence.append(f"CPI YoY is {cpi_yoy:.2f}%.")
    if oil_yoy is not None:
        evidence.append(f"WTI oil YoY is {oil_yoy:.2f}%.")

    positive = []
    negative = []
    if yield_curve is not None and yield_curve >= 0:
        positive.append("Yield curve is not inverted.")
    if vix is not None and vix <= 20:
        positive.append("VIX is not elevated versus common stress thresholds.")
    if beta_rate + beta_vix + beta_credit > 0.8:
        negative.append("Sector ETF sensitivities show elevated macro factor exposure.")
    if credit is not None and credit > 4:
        negative.append("Credit spreads indicate tighter financing conditions.")

    return _clamp_score(score), evidence, positive + negative


def _silver_macro_unavailable_fallback(
    ticker: str,
    sector: str,
    company_name: str | None,
    error: str,
) -> dict[str, Any]:
    return {
        "agent": "Macro-Economic Agent",
        "ticker": ticker,
        "company": company_name or ticker,
        "sector": sector,
        "risk_score": 5.0,
        "macro_risk_score": 5.0,
        "risk_label": "MODERATE",
        "confidence": 0.20,
        "claim_type": "MACRO_V2_UNAVAILABLE",
        "evidence": [
            "Macro v2 agent is installed, but its latest snapshot artifact is not available yet.",
            error,
        ],
        "positive_signals": [],
        "negative_signals": ["Run the Macro v2 pipeline before expecting point-in-time macro agent inference."],
        "overall_assessment": "Macro v2 data artifact is missing; no model-grade macro verdict was produced.",
    }


def run_macro_agent(
    ticker: str,
    sector: str,
    company_name: str | None = None,
    macro_path: str | None = None,
    max_retries: int = 1,
) -> dict[str, Any]:
    """Run Macro v2 sector-aware point-in-time inference.

    `macro_path` and `max_retries` are accepted for compatibility with the old
    DeepSeek macro agent signature.
    """
    del macro_path, max_retries
    ticker = ticker.upper()
    sector = sector if isinstance(sector, str) and sector else "General"

    forecast_error: Exception | None = None
    if os.getenv("MACRO_V2_USE_LORA_FORECAST", "1") not in {"0", "false", "False"}:
        try:
            forecasts = _load_latest_forecasts()
            forecast_row = _select_forecast_row(forecasts, sector)
            return _macro_v2_forecast_result(ticker, sector, company_name, forecast_row)
        except Exception as exc:
            forecast_error = exc

    try:
        snapshot = _load_latest_snapshot()
        row = _select_sector_row(snapshot, sector)
        score, evidence, signals = _score_from_macro_v2(row, sector)
    except Exception as exc:
        if forecast_error:
            exc = RuntimeError(f"{forecast_error}; snapshot fallback also failed: {exc}")
        return _silver_macro_unavailable_fallback(ticker, sector, company_name, error=str(exc))

    as_of = row.get("as_of_date")
    if pd.isna(as_of):
        as_of = row.get("date")
    as_of_text = str(pd.Timestamp(as_of).date()) if as_of is not None and not pd.isna(as_of) else None

    return {
        "agent": "Macro-Economic Agent",
        "ticker": ticker,
        "company": company_name or ticker,
        "sector": sector,
        "as_of_date": as_of_text,
        "risk_score": score,
        "macro_risk_score": score,
        "risk_label": _label_from_score(score),
        "confidence": 0.72,
        "claim_type": "MACRO_V2_POINT_IN_TIME",
        "metrics": {
            "etf": row.get("etf"),
            "fed_funds_rate": _metric(row, "DFF", "FEDFUNDS"),
            "treasury_10y": _metric(row, "DGS10", "GS10"),
            "yield_curve_10y2y": _metric(row, "T10Y2Y"),
            "vix": _metric(row, "VIXCLS"),
            "credit_spread": _metric(row, "BAMLH0A0HYM2", "BAA10Y"),
            "unemployment": _metric(row, "UNRATE"),
            "cpi_yoy": _metric(row, "CPIAUCSL_yoy"),
            "beta_rate": _metric(row, "beta_d_dgs10"),
            "beta_vix": _metric(row, "beta_d_vix"),
            "beta_credit": _metric(row, "beta_d_credit"),
        },
        "evidence": evidence,
        "positive_signals": [s for s in signals if s.startswith(("Yield", "VIX is not"))],
        "negative_signals": [s for s in signals if not s.startswith(("Yield", "VIX is not"))],
        "sector_sensitivity": f"Macro v2 maps {sector} to sector ETF {row.get('etf', 'N/A')}.",
        "overall_assessment": f"Macro v2 risk is {_label_from_score(score)} for {sector}.",
    }
