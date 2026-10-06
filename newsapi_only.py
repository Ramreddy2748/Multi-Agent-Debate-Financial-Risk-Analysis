"""
NewsAPI-only ingestion for the market/news agent.

This avoids rerunning Yahoo, SEC, and FRED when you only need better news
coverage. It appends to a cumulative raw file, deduplicates by URL, resumes by
skipping tickers already present, and can regenerate the Silver sentiment file.

Examples:
  python3 newsapi_only.py --days 1 --page-size 20 --sleep 2 --clean
  python3 newsapi_only.py --limit 50 --days 10 --page-size 20 --clean
  python3 newsapi_only.py --tickers AAPL MSFT NVDA --days 7 --clean --no-resume
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

import clean
import config
from ingest import _load_api_key


PROJECT_ROOT = Path(__file__).resolve().parent
GOLD_PATH = PROJECT_ROOT / config.LOCAL_GOLD / "gold_risk_scores_ALL.csv"
BRONZE_NEWS_PATH = PROJECT_ROOT / config.LOCAL_BRONZE / "newsapi_headlines_raw.csv"
PROGRESS_PATH = PROJECT_ROOT / config.LOCAL_BRONZE / "newsapi_progress.json"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _ticker_rows() -> list[dict[str, str]]:
    rows = _read_csv(GOLD_PATH)
    if not rows:
        raise SystemExit(f"Gold table not found or empty: {GOLD_PATH}")
    seen = set()
    unique = []
    for row in rows:
        ticker = str(row.get("ticker", "")).upper().strip()
        if ticker and ticker not in seen:
            seen.add(ticker)
            unique.append({"ticker": ticker, "sector": row.get("sector", "")})
    return unique


def _company_name(ticker: str) -> str:
    price_path = PROJECT_ROOT / config.LOCAL_SILVER / f"silver_prices_{ticker}.csv"
    rows = _read_csv(price_path)
    if rows:
        latest = rows[-1]
        return latest.get("long_name") or ticker
    return ticker


def _existing_raw() -> pd.DataFrame:
    if not BRONZE_NEWS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(BRONZE_NEWS_PATH)


def _completed_tickers(raw_df: pd.DataFrame) -> set[str]:
    if raw_df.empty or "ticker" not in raw_df.columns:
        return set()
    return {str(value).upper() for value in raw_df["ticker"].dropna().unique()}


def _query(ticker: str, company: str) -> str:
    return (
        f'("{ticker}" OR "{company}") AND '
        "(stock OR shares OR earnings OR revenue OR profit OR guidance OR analyst OR downgrade OR upgrade)"
    )


def _request_error_message(exc: Exception) -> str:
    message = str(exc)
    if "apiKey=" in message:
        before, _, after = message.partition("apiKey=")
        for separator in ("&", " ", "'"):
            if separator in after:
                _, _, tail = after.partition(separator)
                return before + "apiKey=<redacted>" + (separator + tail if tail else "")
        return before + "apiKey=<redacted>"
    return message


def _save_raw(rows: list[dict[str, Any]], existing: pd.DataFrame) -> pd.DataFrame:
    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    if not combined.empty and "url" in combined.columns:
        combined = combined.drop_duplicates(subset=["url"], keep="last")
    BRONZE_NEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(BRONZE_NEWS_PATH, index=False)
    return combined


def _save_progress(done: list[str], stopped_reason: str | None = None) -> None:
    payload = {
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "done_count": len(set(done)),
        "done_tickers": sorted(set(done)),
        "stopped_reason": stopped_reason,
    }
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def fetch_news_only(
    tickers: list[str],
    days: int,
    page_size: int,
    sleep_seconds: float,
    resume: bool,
    max_requests: int | None,
) -> pd.DataFrame:
    api_key = _load_api_key("NEWS_API_KEY")
    if not api_key:
        raise SystemExit("NEWS_API_KEY not found. Add it to .env first.")

    existing = _existing_raw()
    done = _completed_tickers(existing) if resume else set()
    pending = [ticker for ticker in tickers if ticker not in done]
    from_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    session = requests.Session()

    print(f"NewsAPI only run")
    print(f"  total tickers: {len(tickers)}")
    print(f"  already in raw file: {len(done)}")
    print(f"  pending: {len(pending)}")
    print(f"  days: {days}, page_size: {page_size}, sleep: {sleep_seconds}s")
    print(f"  raw output: {BRONZE_NEWS_PATH}")

    new_rows: list[dict[str, Any]] = []
    completed_this_run: list[str] = []
    requests_used = 0

    for idx, ticker in enumerate(pending, start=1):
        if max_requests is not None and requests_used >= max_requests:
            print(f"Reached --max-requests={max_requests}; saving partial results.")
            break

        company = _company_name(ticker)
        params = {
            "q": _query(ticker, company),
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": page_size,
            "apiKey": api_key,
        }
        print(f"[{idx}/{len(pending)}] {ticker} - {company}")
        try:
            response = session.get("https://newsapi.org/v2/everything", params=params, timeout=20)
            requests_used += 1
        except requests.RequestException as exc:
            print(f"  request failed without saving key: {_request_error_message(exc)}")
            combined = _save_raw(new_rows, existing)
            _save_progress(list(done) + completed_this_run, "request_error")
            print(f"Saved raw news rows: {len(combined)}")
            return combined

        if response.status_code == 401:
            _save_raw(new_rows, existing)
            _save_progress(list(done) + completed_this_run, "invalid_api_key")
            raise SystemExit("NewsAPI returned 401. Regenerate/check NEWS_API_KEY.")
        if response.status_code == 426:
            _save_raw(new_rows, existing)
            _save_progress(list(done) + completed_this_run, "plan_limit")
            raise SystemExit("NewsAPI returned 426. Free plan may only allow recent news; retry with --days 1.")
        if response.status_code == 429:
            print("NewsAPI returned 429 rate/quota limit. Saving partial results and stopping cleanly.")
            break

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            print(f"  failed: {exc}")
            time.sleep(sleep_seconds)
            continue

        payload = response.json()
        articles = payload.get("articles", [])
        print(f"  articles: {len(articles)}")
        for article in articles:
            new_rows.append({
                "ticker": ticker,
                "company": company,
                "published_at": article.get("publishedAt"),
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "source": article.get("source", {}).get("name", ""),
                "author": article.get("author", ""),
                "url": article.get("url", ""),
            })
        completed_this_run.append(ticker)
        time.sleep(sleep_seconds)

    combined = _save_raw(new_rows, existing)
    _save_progress(list(done) + completed_this_run)
    print(f"Saved raw news rows: {len(combined)}")
    print(f"New rows before URL dedupe: {len(new_rows)}")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch only NewsAPI data and optionally clean sentiment.")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional explicit tickers. Defaults to Gold table.")
    parser.add_argument("--limit", type=int, default=None, help="Limit pending tickers for a controlled run.")
    parser.add_argument("--days", type=int, default=1, help="Days of NewsAPI history to request.")
    parser.add_argument("--page-size", type=int, default=20, help="Articles per ticker request.")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to sleep between ticker requests.")
    parser.add_argument("--max-requests", type=int, default=None, help="Stop after N NewsAPI requests.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip tickers already in raw news file.")
    parser.add_argument("--clean", action="store_true", help="Regenerate data/silver/silver_news_sentiment.csv after fetch.")
    args = parser.parse_args()

    if args.tickers:
        tickers = [ticker.upper().strip() for ticker in args.tickers if ticker.strip()]
    else:
        tickers = [row["ticker"] for row in _ticker_rows()]

    if args.limit is not None:
        existing = _existing_raw()
        done = _completed_tickers(existing) if not args.no_resume else set()
        pending = [ticker for ticker in tickers if ticker not in done]
        tickers = sorted(done) + pending[:args.limit] if not args.no_resume else tickers[:args.limit]

    combined = fetch_news_only(
        tickers=tickers,
        days=args.days,
        page_size=args.page_size,
        sleep_seconds=args.sleep,
        resume=not args.no_resume,
        max_requests=args.max_requests,
    )

    if args.clean:
        if combined.empty:
            print("No raw news rows to clean.")
            return
        combined["published_at"] = pd.to_datetime(combined["published_at"], errors="coerce")
        cleaned = clean.clean_news(combined)
        print(f"Cleaned sentiment rows: {len(cleaned)}")
        print(f"Silver output: {PROJECT_ROOT / config.LOCAL_SILVER / 'silver_news_sentiment.csv'}")


if __name__ == "__main__":
    main()
