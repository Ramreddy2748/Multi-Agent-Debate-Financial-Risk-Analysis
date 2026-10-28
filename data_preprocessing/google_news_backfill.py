"""Backfill ticker news from Google News RSS into the existing news files.

This is useful when NewsAPI plan limits block historical pulls. Google News RSS
does not provide a full archive, but it can return recent/historical search
results using query windows such as ``when:365d``.

Example:
  python3 data_preprocessing/google_news_backfill.py --ticker AAPL --company "Apple Inc." --when 365d --clean
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from data_preprocessing import clean


BRONZE_NEWS_PATH = PROJECT_ROOT / config.LOCAL_BRONZE / "newsapi_headlines_raw.csv"
GOLD_PATH = PROJECT_ROOT / config.LOCAL_GOLD / "gold_risk_scores_ALL.csv"
TICKER_ALIASES = {
    "BRK.B": "BRK-B",
    "BRK-B": "BRK-B",
    "BF.B": "BF-B",
    "BF-B": "BF-B",
}


def _read_existing() -> pd.DataFrame:
    if not BRONZE_NEWS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(BRONZE_NEWS_PATH)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _company_name(ticker: str) -> str:
    price_path = PROJECT_ROOT / config.LOCAL_SILVER / f"silver_prices_{ticker}.csv"
    rows = _read_csv(price_path)
    if rows:
        latest = rows[-1]
        return latest.get("long_name") or latest.get("company") or ticker
    return ticker


def _top_market_cap_tickers(limit: int) -> list[tuple[str, str]]:
    rows = _read_csv(GOLD_PATH)
    ranked = []
    for row in rows:
        ticker = str(row.get("ticker", "")).upper().strip()
        try:
            market_cap = float(row.get("market_cap_usd_m") or 0)
        except ValueError:
            market_cap = 0.0
        if ticker:
            ranked.append((ticker, market_cap))
    ranked.sort(key=lambda item: item[1], reverse=True)
    targets: list[tuple[str, str]] = []
    seen_companies: set[str] = set()
    for ticker, _ in ranked:
        company = _company_name(ticker)
        company_key = company.lower().replace(" class a", "").replace(" class c", "").strip()
        if company_key in seen_companies:
            continue
        seen_companies.add(company_key)
        targets.append((ticker, company))
        if len(targets) >= limit:
            break
    return targets


def _ticker_query_terms(ticker: str) -> str:
    canonical = TICKER_ALIASES.get(ticker.upper(), ticker.upper())
    terms = {canonical, canonical.replace("-", ".")}
    return " OR ".join(f'"{term}"' for term in sorted(terms))


def _clean_google_title(title: str) -> str:
    # Google News RSS titles commonly append " - Source". Keep the headline body.
    return re.sub(r"\s+-\s+[^-]{2,80}$", "", title).strip() or title.strip()


def _rss_url(ticker: str, company: str, when: str) -> str:
    query = f'({_ticker_query_terms(ticker)} OR "{company}") stock OR earnings OR shares when:{when}'
    return (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + "&hl=en-US&gl=US&ceid=US:en"
    )


def _published_at(pub_date: str | None) -> str | None:
    if not pub_date:
        return None
    try:
        return parsedate_to_datetime(pub_date).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def fetch_google_news_rows(ticker: str, company: str, when: str) -> list[dict[str, Any]]:
    url = _rss_url(ticker, company, when)
    response = requests.get(url, headers={"User-Agent": "DATA298B-financial-risk-research/1.0"}, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    rows = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        source_el = item.find("source")
        source = source_el.text.strip() if source_el is not None and source_el.text else "Google News"
        rows.append({
            "ticker": ticker,
            "company": company,
            "published_at": _published_at(item.findtext("pubDate")),
            "title": _clean_google_title(title),
            "description": "",
            "source": f"Google News:{source}",
            "author": "",
            "url": link,
        })
    return rows


def append_and_clean(rows: list[dict[str, Any]], clean_output: bool) -> None:
    existing = _read_existing()
    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    if not combined.empty and "url" in combined.columns:
        combined["url"] = combined["url"].fillna("").astype(str).str.strip()
        with_url = combined[combined["url"] != ""].drop_duplicates(subset=["url"], keep="last")
        without_url = combined[combined["url"] == ""].drop_duplicates(
            subset=["ticker", "published_at", "title"],
            keep="last",
        )
        combined = pd.concat([with_url, without_url], ignore_index=True)
    if not combined.empty and "published_at" in combined.columns:
        combined["published_at"] = pd.to_datetime(
            combined["published_at"], format="mixed", utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        combined = combined.dropna(subset=["published_at"])

    BRONZE_NEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(BRONZE_NEWS_PATH, index=False)
    print(f"Saved raw news rows: {len(combined)}")
    print(f"New rows before URL dedupe: {len(rows)}")

    if clean_output:
        cleaned = clean.clean_news(combined)
        print(f"Cleaned sentiment rows: {len(cleaned)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical ticker news from Google News RSS.")
    parser.add_argument("--ticker")
    parser.add_argument("--company")
    parser.add_argument("--tickers", nargs="*", help="Optional ticker batch. Company names are read from Silver prices.")
    parser.add_argument("--top-n", type=int, help="Backfill top N tickers by Gold market_cap_usd_m.")
    parser.add_argument("--when", default="365d", help="Google News time window, e.g. 30d, 365d.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between Google News RSS requests.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    targets: list[tuple[str, str]] = []
    if args.top_n:
        targets = _top_market_cap_tickers(args.top_n)
    elif args.tickers:
        targets = [(ticker.upper().strip(), _company_name(ticker.upper().strip())) for ticker in args.tickers]
    elif args.ticker:
        company = args.company or _company_name(args.ticker.upper().strip())
        targets = [(args.ticker.upper().strip(), company)]
    else:
        raise SystemExit("Provide --ticker, --tickers, or --top-n.")

    all_rows: list[dict[str, Any]] = []
    print("Google News backfill targets:")
    for ticker, company in targets:
        print(f"  {ticker} - {company}")

    for idx, (ticker, company) in enumerate(targets, start=1):
        rows = fetch_google_news_rows(ticker=ticker, company=company, when=args.when)
        print(f"[{idx}/{len(targets)}] {ticker}: fetched {len(rows)} rows")
        all_rows.extend(rows)
        time.sleep(args.sleep)

    print(f"Fetched Google News rows: {len(all_rows)}")
    append_and_clean(all_rows, clean_output=args.clean)


if __name__ == "__main__":
    main()
