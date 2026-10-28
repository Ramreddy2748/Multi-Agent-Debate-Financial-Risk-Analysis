"""Build an auditable feature-priority and data-freshness layer.

The layer sits between Gold/Silver data and the agents. It does not change the
agent scores. It ranks the evidence that deserves attention first and records
whether each value is complete and fresh enough to trust.

Outputs:
  data/gold/feature_priority_scores.csv
  data/gold/feature_priority_summary.csv
  data/gold/feature_freshness.csv
  data/gold/feature_priority_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLD = PROJECT_ROOT / "data" / "gold" / "gold_risk_scores_ALL.csv"
DEFAULT_SILVER = PROJECT_ROOT / "data" / "silver"
DEFAULT_OUT = PROJECT_ROOT / "data" / "gold"


AGENT_RULES = {
    "Fundamental Agent": {
        "fresh_days": 120,
        "features": [
            ("cash_usd_m", "cash_usd_m", "cash", "higher_is_better", 1.00, 100_000),
            ("long_term_debt_usd_m", "long_term_debt", "long_term_debt", "higher_is_worse", 0.95, 100_000),
            ("debt_to_assets", "derived", "debt_to_assets", "higher_is_worse", 1.00, 1.0),
            ("cash_to_debt", "derived", "cash_to_debt", "higher_is_better", 0.90, 2.0),
            ("profit_margin", "derived", "profit_margin", "higher_is_better", 0.85, 0.25),
            ("operating_margin", "derived", "operating_margin", "higher_is_better", 0.80, 0.25),
        ],
    },
    "Market + Volatility Agent": {
        "fresh_days": 7,
        "features": [
            ("annualized_volatility_pct", "ann_volatility_pct", "annualized_volatility", "higher_is_worse", 1.00, 50.0),
            ("max_drawdown_pct", "max_drawdown_pct", "max_drawdown", "higher_is_worse", 1.00, 50.0),
            ("beta", "beta", "beta", "higher_is_worse", 0.85, 2.0),
            ("ytd_return_pct", "ytd_return_pct", "ytd_return", "higher_is_better", 0.75, 35.0),
            ("sharpe_ratio", "sharpe_ratio", "sharpe_ratio", "higher_is_better", 0.70, 2.0),
        ],
    },
    "Sentiment Agent": {
        "fresh_days": 3,
        "features": [
            ("sentiment_mean", "sentiment_mean", "sentiment_mean", "higher_is_better", 1.00, 1.0),
            ("article_count", "article_count", "article_count", "higher_is_better", 0.95, 20.0),
            ("sentiment_std", "sentiment_std", "sentiment_dispersion", "higher_is_worse", 0.70, 0.5),
        ],
    },
    "Macro-Economic Agent": {
        "fresh_days": 45,
        "features": [
            ("vix", "vix", "vix", "higher_is_worse", 1.00, 30.0),
            ("yield_curve_10y2y", "yield_curve_10y2y", "yield_curve", "higher_is_better", 0.90, 2.0),
            ("fed_funds_rate", "fed_funds_rate", "policy_rate", "higher_is_worse", 0.80, 6.0),
            ("unemployment", "unemployment", "unemployment", "higher_is_worse", 0.75, 10.0),
            ("wti_oil", "wti_oil", "oil_price", "higher_is_worse", 0.55, 120.0),
        ],
    },
}


def _float(value: Any) -> float | None:
    try:
        if value in (None, "", "nan", "NaN"):
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _date(value: Any) -> date | None:
    text = str(value or "")[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _latest(rows: list[dict[str, str]], columns: tuple[str, ...]) -> dict[str, str]:
    dated = [(row, _date(next((row.get(column) for column in columns), ""))) for row in rows]
    dated = [(row, stamp) for row, stamp in dated if stamp]
    return max(dated, key=lambda item: item[1])[0] if dated else (rows[-1] if rows else {})


def _freshness(source_date: date | None, allowed_days: int) -> tuple[int | None, str, float]:
    if source_date is None:
        return None, "MISSING_DATE", 0.35
    age = max((date.today() - source_date).days, 0)
    if age <= allowed_days:
        status = "FRESH" if age <= max(1, allowed_days // 2) else "RECENT"
        freshness_factor = 1.0 if status == "FRESH" else 0.85
    elif age <= allowed_days * 2:
        status, freshness_factor = "STALE", 0.60
    else:
        status, freshness_factor = "EXPIRED", 0.30
    return age, status, freshness_factor


def _quality(value: float | None) -> float:
    return 1.0 if value is not None else 0.0


def _impact(value: float | None, scale: float, direction: str) -> float:
    if value is None:
        return 0.0
    normalized = min(abs(value) / max(scale, 1e-9), 1.0)
    # The absolute magnitude identifies an important feature; direction is
    # retained as metadata for the UI and explanation layer.
    return round(normalized, 4)


def _fundamental_values(row: dict[str, str]) -> dict[str, float | None]:
    cash = _float(row.get("cash_usd_m"))
    debt = _float(row.get("long_term_debt"))
    assets = _float(row.get("total_assets_usd_m"))
    net_income = _float(row.get("net_income_usd_m"))
    operating_income = _float(row.get("operating_income_usd_m"))
    revenue = _float(row.get("revenue")) or _float(row.get("revenue_alt"))
    if debt is not None and abs(debt) > 1_000_000:
        debt /= 1_000_000
    return {
        "cash_usd_m": cash,
        "long_term_debt_usd_m": debt,
        "debt_to_assets": debt / assets if debt is not None and assets else None,
        "cash_to_debt": cash / debt if cash is not None and debt else None,
        "profit_margin": net_income / revenue if net_income is not None and revenue else None,
        "operating_margin": operating_income / revenue if operating_income is not None and revenue else None,
    }


def _add_row(output: list[dict[str, Any]], agent: str, ticker: str, feature: str,
             value: float | None, source: str, source_date: date | None,
             direction: str, base_priority: float, scale: float, allowed_days: int) -> None:
    age_days, freshness_status, freshness_factor = _freshness(source_date, allowed_days)
    quality = _quality(value)
    impact = _impact(value, scale, direction)
    priority_score = round(base_priority * max(impact, 0.10) * quality * freshness_factor, 4)
    output.append({
        "agent": agent,
        "ticker": ticker,
        "feature": feature,
        "value": "" if value is None else round(value, 6),
        "risk_direction": direction,
        "base_priority": base_priority,
        "risk_impact": impact,
        "data_quality": quality,
        "source": source,
        "source_date": source_date.isoformat() if source_date else "",
        "age_days": "" if age_days is None else age_days,
        "freshness_status": freshness_status,
        "freshness_factor": freshness_factor,
        "priority_score": priority_score,
        "is_high_priority": "Y" if priority_score >= 0.45 else "N",
    })


def build_priority_rows(gold_path: Path, silver_dir: Path) -> list[dict[str, Any]]:
    gold_rows = _read_csv(gold_path)
    gold_by_ticker = {str(row.get("ticker", "")).upper(): row for row in gold_rows}
    output: list[dict[str, Any]] = []

    macro_rows = _read_csv(silver_dir / "silver_macro.csv")
    macro = _latest(macro_rows, ("date",))
    macro_date = _date(macro.get("date"))

    news_rows = _read_csv(silver_dir / "silver_news_sentiment.csv")
    news_by_ticker: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in news_rows:
        news_by_ticker[str(row.get("ticker", "")).upper()].append(row)

    for ticker, gold in sorted(gold_by_ticker.items()):
        fundamental_rows = _read_csv(silver_dir / f"silver_edgar_{ticker}.csv")
        fundamental = _latest(fundamental_rows, ("filed", "end_date"))
        fundamental_date = _date(fundamental.get("filed") or fundamental.get("end_date"))
        for feature, field, _, direction, base, scale in AGENT_RULES["Fundamental Agent"]["features"]:
            value = _fundamental_values(fundamental).get(feature)
            _add_row(output, "Fundamental Agent", ticker, feature, value,
                     f"data/silver/silver_edgar_{ticker}.csv", fundamental_date,
                     direction, base, scale, AGENT_RULES["Fundamental Agent"]["fresh_days"])

        for feature, field, _, direction, base, scale in AGENT_RULES["Market + Volatility Agent"]["features"]:
            value = _float(gold.get(field))
            _add_row(output, "Market + Volatility Agent", ticker, feature, value,
                     "data/gold/gold_risk_scores_ALL.csv", _date(gold.get("as_of_date")),
                     direction, base, scale, AGENT_RULES["Market + Volatility Agent"]["fresh_days"])

        ticker_news = sorted(news_by_ticker.get(ticker, []), key=lambda row: _date(row.get("date")) or date.min)
        latest_news = ticker_news[-1] if ticker_news else {}
        news_date = _date(latest_news.get("date"))
        for feature, field, _, direction, base, scale in AGENT_RULES["Sentiment Agent"]["features"]:
            values = [_float(row.get(field)) for row in ticker_news if _float(row.get(field)) is not None]
            if feature == "article_count":
                value = sum(int(_float(row.get(field)) or 0) for row in ticker_news)
            else:
                value = sum(values) / len(values) if values else None
            _add_row(output, "Sentiment Agent", ticker, feature, value,
                     "data/silver/silver_news_sentiment.csv", news_date,
                     direction, base, scale, AGENT_RULES["Sentiment Agent"]["fresh_days"])

        for feature, field, _, direction, base, scale in AGENT_RULES["Macro-Economic Agent"]["features"]:
            _add_row(output, "Macro-Economic Agent", ticker, feature, _float(macro.get(field)),
                     "data/silver/silver_macro.csv", macro_date, direction, base, scale,
                     AGENT_RULES["Macro-Economic Agent"]["fresh_days"])

    return output


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    scores_path = output_dir / "feature_priority_scores.csv"
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["ticker"], row["agent"])].append(row)
    summary_rows = []
    for (ticker, agent), items in sorted(grouped.items()):
        ranked = sorted(items, key=lambda row: float(row["priority_score"]), reverse=True)
        top = ranked[:3]
        summary_rows.append({
            "ticker": ticker,
            "agent": agent,
            "top_feature_1": top[0]["feature"] if top else "",
            "top_feature_1_score": top[0]["priority_score"] if top else "",
            "top_feature_2": top[1]["feature"] if len(top) > 1 else "",
            "top_feature_2_score": top[1]["priority_score"] if len(top) > 1 else "",
            "top_feature_3": top[2]["feature"] if len(top) > 2 else "",
            "top_feature_3_score": top[2]["priority_score"] if len(top) > 2 else "",
            "high_priority_count": sum(row["is_high_priority"] == "Y" for row in items),
            "stale_or_expired_count": sum(row["freshness_status"] in {"STALE", "EXPIRED"} for row in items),
        })
    summary_path = output_dir / "feature_priority_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    freshness = []
    for (ticker, agent), items in sorted(grouped.items()):
        statuses = [row["freshness_status"] for row in items]
        freshness.append({
            "ticker": ticker,
            "agent": agent,
            "latest_source_date": max((row["source_date"] for row in items), default=""),
            "freshness_status": "EXPIRED" if "EXPIRED" in statuses else "STALE" if "STALE" in statuses else "RECENT" if "RECENT" in statuses else "FRESH",
            "stale_or_expired_count": sum(status in {"STALE", "EXPIRED"} for status in statuses),
            "missing_value_count": sum(row["data_quality"] == 0.0 for row in items),
        })
    freshness_path = output_dir / "feature_freshness.csv"
    with freshness_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(freshness[0]))
        writer.writeheader()
        writer.writerows(freshness)

    high = sorted(rows, key=lambda row: float(row["priority_score"]), reverse=True)[:20]
    report = [
        "# Feature Priority and Freshness Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Priority score combines risk impact, configured feature importance, data completeness, and freshness.",
        "RAG definitions and thresholds explain the features; this layer computes the numeric ranking.",
        "",
        "## Highest-priority evidence",
        "",
        "| Agent | Ticker | Feature | Value | Priority | Freshness | Source date |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in high:
        report.append(f"| {row['agent']} | {row['ticker']} | {row['feature']} | {row['value']} | {row['priority_score']} | {row['freshness_status']} | {row['source_date']} |")
    (output_dir / "feature_priority_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature priority and freshness artifacts.")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--silver-dir", default=str(DEFAULT_SILVER))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    rows = build_priority_rows(Path(args.gold), Path(args.silver_dir))
    if not rows:
        raise SystemExit("No Gold rows found; run the Gold pipeline first.")
    write_outputs(rows, Path(args.out_dir))
    print(f"Wrote {len(rows)} feature-priority rows for {len({row['ticker'] for row in rows})} tickers.")
    print(f"Outputs: {Path(args.out_dir) / 'feature_priority_scores.csv'}")


if __name__ == "__main__":
    main()
