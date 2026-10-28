"""Backfill historical company news into the existing Bronze/Silver news files.

NewsAPI free/developer plans often only allow recent articles. This script uses
the public GDELT DOC API for historical headline backfills, then normalizes rows
to the same schema used by ``newsapi_only.py``.

Example:
  python3 data_preprocessing/historical_news_backfill.py --ticker AAPL --company "Apple Inc." --days 365 --clean
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from data_preprocessing import clean


BRONZE_NEWS_PATH = PROJECT_ROOT / config.LOCAL_BRONZE / "newsapi_headlines_raw.csv"


def _read_existing() -> pd.DataFrame:
    if not BRONZE_NEWS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(BRONZE_NEWS_PATH)


def _gdelt_datetime(value: datetime) -> str:
    return value.strftime("%Y%m%d%H%M%S")


def _chunk_ranges(start: datetime, end: datetime, chunk_days: int) -> list[tuple[datetime, datetime]]:
    ranges = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        ranges.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(seconds=1)
    return ranges


def _query(ticker: str, company: str) -> str:
    return (
        f'("{ticker}" OR "{company}") '
        "(stock OR shares OR earnings OR revenue OR profit OR guidance OR analyst OR downgrade OR upgrade)"
    )


def fetch_gdelt_rows(
    ticker: str,
    company: str,
    start: datetime,
    end: datetime,
    chunk_days: int,
    max_records: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "DATA298B-financial-risk-research/1.0"})
    rows: list[dict[str, Any]] = []

    for idx, (chunk_start, chunk_end) in enumerate(_chunk_ranges(start, end, chunk_days), start=1):
        params = {
            "query": _query(ticker, company),
            "mode": "ArtList",
            "format": "json",
            "maxrecords": max_records,
            "sort": "HybridRel",
            "startdatetime": _gdelt_datetime(chunk_start),
            "enddatetime": _gdelt_datetime(chunk_end),
        }
        print(f"[{idx}] {chunk_start.date()} -> {chunk_end.date()}")
        response = session.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=30)
        if response.status_code == 429:
            print(f"  rate limited; sleeping {sleep_seconds * 3:.1f}s then retrying once")
            time.sleep(sleep_seconds * 3)
            response = session.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=30)
        if response.status_code == 429:
            print("  still rate limited; stopping and saving rows collected so far")
            break
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError:
            print(f"  non-JSON response; stopping and saving rows collected so far: {response.text[:120]}")
            break
        articles = payload.get("articles", [])
        print(f"  articles: {len(articles)}")
        for article in articles:
            seen = article.get("seendate") or article.get("publishedAt")
            rows.append({
                "ticker": ticker,
                "company": company,
                "published_at": seen,
                "title": article.get("title", ""),
                "description": article.get("snippet", ""),
                "source": f"GDELT:{article.get('domain', '')}".rstrip(":"),
                "author": "",
                "url": article.get("url", ""),
            })
        time.sleep(sleep_seconds)

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
    parser = argparse.ArgumentParser(description="Backfill historical ticker news with GDELT.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL.")
    parser.add_argument("--company", required=True, help="Company name, e.g. Apple Inc.")
    parser.add_argument("--days", type=int, default=365, help="Days of history to fetch.")
    parser.add_argument("--chunk-days", type=int, default=30, help="Days per GDELT request.")
    parser.add_argument("--max-records", type=int, default=100, help="Max GDELT records per chunk.")
    parser.add_argument("--sleep", type=float, default=5.0, help="Seconds to sleep between GDELT requests.")
    parser.add_argument("--clean", action="store_true", help="Regenerate Silver sentiment after appending.")
    args = parser.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    ticker = args.ticker.upper().strip()

    rows = fetch_gdelt_rows(
        ticker=ticker,
        company=args.company,
        start=start,
        end=end,
        chunk_days=args.chunk_days,
        max_records=args.max_records,
        sleep_seconds=args.sleep,
    )
    append_and_clean(rows, clean_output=args.clean)


if __name__ == "__main__":
    main()
