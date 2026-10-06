
import os
import time
import logging
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
log = logging.getLogger("ingest")


# ─── Retry-enabled HTTP session ───────────────────────────────────────────────
def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session

SESSION = _make_session()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dirs():
    Path(config.LOCAL_BRONZE).mkdir(parents=True, exist_ok=True)

def _save_bronze(df: pd.DataFrame, filename: str) -> str:
    _ensure_dirs()
    path = os.path.join(config.LOCAL_BRONZE, filename)
    df.to_csv(path, index=True)
    log.info(f"  ✔ Bronze saved: {path}  ({len(df):,} rows x {df.shape[1]} cols)")
    return path

def _upload_to_gcs(local_path: str, gcs_path: str):
    try:
        from google.cloud import storage
        client = storage.Client(project=config.GCP_PROJECT_ID)
        bucket = client.bucket(config.GCS_BUCKET)
        bucket.blob(gcs_path).upload_from_filename(local_path)
        log.info(f"  ☁  GCS: gs://{config.GCS_BUCKET}/{gcs_path}")
    except ImportError:
        log.warning("google-cloud-storage not installed — skipping GCS upload.")
    except Exception as e:
        log.error(f"  GCS upload failed: {e}")

def _load_api_key(env_var: str, env_file: str = ".env") -> Optional[str]:
    """Read API key from environment variable, then .env file as fallback."""
    key = os.getenv(env_var)
    if key and "YOUR_" not in key:
        return key
    env_path = Path(env_file)
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith(env_var + "="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    log.info(f"  Loaded {env_var} from .env file")
                    return val
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1 — YAHOO FINANCE
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yahoo_finance(
    tickers: List[str],
    start_date: str = config.START_DATE,
    end_date:   str = config.END_DATE,
    upload_gcs: bool = False,
) -> dict:
    log.info("=" * 60)
    log.info(f"[Yahoo Finance] Fetching {len(tickers)} tickers: {tickers}")
    results = {}

    for ticker in tickers:
        try:
            log.info(f"  -> {ticker}...")
            stock = yf.Ticker(ticker)
            hist  = stock.history(start=start_date, end=end_date)
            if hist.empty:
                log.warning(f"    No data for {ticker}")
                continue

            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            hist.index.name = "date"
            hist["ticker"] = ticker

            info = stock.info
            for col, key in [
                ("pe_ratio", "trailingPE"), ("eps", "trailingEps"),
                ("market_cap", "marketCap"), ("revenue_ttm", "totalRevenue"),
                ("debt_to_equity", "debtToEquity"), ("beta", "beta"),
                ("52w_high", "fiftyTwoWeekHigh"), ("52w_low", "fiftyTwoWeekLow"),
                ("sector", "sector"), ("industry", "industry"),
                ("long_name", "longName"),
            ]:
                hist[col] = info.get(key)

            log.info(f"    OK: {len(hist)} trading days")
            filename = f"yahoo_{ticker}_{start_date}_to_{end_date}.csv"
            path = _save_bronze(hist, filename)
            if upload_gcs:
                _upload_to_gcs(path, f"bronze/yahoo/{filename}")

            results[ticker] = hist
            time.sleep(0.5)

        except Exception as e:
            log.error(f"    FAILED {ticker}: {e}")

    log.info(f"[Yahoo Finance] Done — {len(results)}/{len(tickers)} fetched.\n")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2 — SEC EDGAR  (v2 — single call per ticker, full facts JSON)
# ─────────────────────────────────────────────────────────────────────────────

EDGAR_HEADERS = {
    "User-Agent": "FinancialRiskPipeline venkatasiddarth.gullipalli@sjsu.edu",
    "Accept": "application/json",
}

EDGAR_CONCEPTS = {
    "Revenues":                                            "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue_alt",
    "NetIncomeLoss":                                       "net_income",
    "Assets":                                              "total_assets",
    "Liabilities":                                         "total_liabilities",
    "StockholdersEquity":                                  "stockholders_equity",
    "EarningsPerShareBasic":                               "eps_basic",
    "EarningsPerShareDiluted":                             "eps_diluted",
    "OperatingIncomeLoss":                                 "operating_income",
    "CashAndCashEquivalentsAtCarryingValue":               "cash",
    "LongTermDebt":                                        "long_term_debt",
    "CommonStockSharesOutstanding":                        "shares_outstanding",
}

def _get_cik_map() -> dict:
    try:
        log.info("  Downloading SEC CIK map...")
        resp = SESSION.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=EDGAR_HEADERS, timeout=20
        )
        resp.raise_for_status()
        data = resp.json()
        cik_map = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}
        log.info(f"  CIK map loaded: {len(cik_map):,} companies")
        return cik_map
    except Exception as e:
        log.error(f"  CIK map download failed: {e}")
        return {}

def fetch_sec_edgar(
    tickers: List[str],
    upload_gcs: bool = False,
) -> dict:
    log.info("=" * 60)
    log.info(f"[SEC EDGAR] Fetching annual financials for: {tickers}")

    cik_map = _get_cik_map()
    if not cik_map:
        log.error("  EDGAR skipped — could not load CIK map")
        return {}

    results = {}

    for ticker in tickers:
        cik = cik_map.get(ticker.upper())
        if not cik:
            log.warning(f"  -> {ticker}: CIK not found — skipping")
            continue

        log.info(f"  -> {ticker} (CIK: {cik})")
        try:
            url  = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            resp = SESSION.get(url, headers=EDGAR_HEADERS, timeout=30)

            if resp.status_code == 404:
                log.warning(f"    No XBRL data for {ticker}")
                continue
            resp.raise_for_status()

            facts       = resp.json()
            entity_name = facts.get("entityName", ticker)
            us_gaap     = facts.get("facts", {}).get("us-gaap", {})
            log.info(f"    Entity: {entity_name}")

            rows = []
            for xbrl_tag, col_name in EDGAR_CONCEPTS.items():
                concept = us_gaap.get(xbrl_tag, {})
                units   = concept.get("units", {})
                points  = units.get("USD", units.get("shares", []))
                annual  = [d for d in points
                           if d.get("form") in ("10-K", "20-F")
                           and d.get("val") is not None]
                for d in annual:
                    rows.append({
                        "ticker":   ticker,
                        "company":  entity_name,
                        "concept":  col_name,
                        "value":    d.get("val"),
                        "end_date": d.get("end"),
                        "filed":    d.get("filed"),
                        "form":     d.get("form"),
                    })

            if not rows:
                log.warning(f"    No annual 10-K data found for {ticker}")
                continue

            df_long = pd.DataFrame(rows)
            log.info(f"    Found {len(df_long)} data points across {df_long['concept'].nunique()} concepts")

            # Save raw long-format
            path_long = _save_bronze(df_long, f"edgar_{ticker}_raw_long.csv")
            if upload_gcs:
                _upload_to_gcs(path_long, f"bronze/edgar/edgar_{ticker}_raw_long.csv")

            # Pivot to wide (one row per annual filing)
            df_wide = (
                df_long
                .pivot_table(
                    index=["ticker", "company", "end_date", "form", "filed"],
                    columns="concept", values="value", aggfunc="last"
                )
                .reset_index()
            )
            df_wide.columns.name = None
            df_wide = df_wide.sort_values("end_date", ascending=False)

            path_wide = _save_bronze(df_wide, f"edgar_{ticker}_annual_financials.csv")
            if upload_gcs:
                _upload_to_gcs(path_wide, f"bronze/edgar/edgar_{ticker}_annual_financials.csv")

            log.info(f"    Saved {len(df_wide)} annual filings")
            results[ticker] = df_wide
            time.sleep(0.6)  # SEC rate limit

        except requests.exceptions.Timeout:
            log.error(f"    TIMEOUT for {ticker} — SEC may be slow, try again later")
        except Exception as e:
            log.error(f"    FAILED {ticker}: {e}")

    log.info(f"[SEC EDGAR] Done — {len(results)}/{len(tickers)} fetched.\n")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3 — FRED API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_fred(
    start_date: str = config.START_DATE,
    end_date:   str = config.END_DATE,
    upload_gcs: bool = False,
) -> pd.DataFrame:
    log.info("=" * 60)
    log.info(f"[FRED] Fetching {len(config.FRED_SERIES)} macro series...")

    api_key = _load_api_key("FRED_API_KEY")
    if not api_key:
        log.warning("  FRED_API_KEY not set — skipping. Add to .env file or set as env var.")
        return pd.DataFrame()

    series_dfs = []
    for col_name, series_id in config.FRED_SERIES.items():
        try:
            resp = SESSION.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={"series_id": series_id, "observation_start": start_date,
                        "observation_end": end_date, "api_key": api_key, "file_type": "json"},
                timeout=15
            )
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
            df  = pd.DataFrame(obs)[["date", "value"]].copy()
            df["date"]  = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.rename(columns={"value": col_name}).set_index("date")
            series_dfs.append(df)
            log.info(f"  OK: {series_id} ({col_name}) — {len(df)} observations")
            time.sleep(0.3)
        except Exception as e:
            log.error(f"  FAILED {series_id}: {e}")

    if not series_dfs:
        return pd.DataFrame()

    macro_df = pd.concat(series_dfs, axis=1).sort_index()
    filename = f"fred_macro_{start_date}_to_{end_date}.csv"
    path = _save_bronze(macro_df, filename)
    if upload_gcs:
        _upload_to_gcs(path, f"bronze/fred/{filename}")

    log.info(f"[FRED] Done — {macro_df.shape[0]} rows x {macro_df.shape[1]} series.\n")
    return macro_df


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 4 — NEWSAPI  (v2 — .env fallback + better error messages)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_news(
    tickers:   List[str],
    days_back: int = 10,
    upload_gcs: bool = False,
) -> pd.DataFrame:
    log.info("=" * 60)
    log.info(f"[NewsAPI] Fetching headlines for: {tickers}")

    api_key = _load_api_key("NEWS_API_KEY")
    if not api_key:
        log.warning("  NEWS_API_KEY not set — skipping.")
        log.warning("  Fix: add NEWS_API_KEY=your_key to a .env file in your project folder")
        log.warning("  Get free key at: https://newsapi.org/register")
        return pd.DataFrame()

    log.info("  NewsAPI key loaded OK")
    from_date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    all_rows  = []

    for ticker in tickers:
        try:
            stock     = yf.Ticker(ticker)
            long_name = stock.info.get("longName", ticker)
            query     = f'"{ticker}" stock OR "{long_name}" earnings OR revenue'

            log.info(f"  -> {ticker} ({long_name})")
            resp = SESSION.get(
                "https://newsapi.org/v2/everything",
                params={"q": query, "from": from_date, "sortBy": "publishedAt",
                        "language": "en", "pageSize": 100, "apiKey": api_key},
                timeout=15
            )

            if resp.status_code == 401:
                log.error("  INVALID NewsAPI key — check your key at newsapi.org/account")
                return pd.DataFrame()
            if resp.status_code == 426:
                log.error("  NewsAPI 426 error — free tier only allows today's news.")
                log.error("  Fix: upgrade at newsapi.org OR the pipeline will skip news.")
                log.warning("  Continuing without news data...")
                return pd.DataFrame()

            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            log.info(f"    {len(articles)} articles fetched")

            for art in articles:
                all_rows.append({
                    "ticker":       ticker,
                    "company":      long_name,
                    "published_at": art.get("publishedAt"),
                    "title":        art.get("title", ""),
                    "description":  art.get("description", ""),
                    "source":       art.get("source", {}).get("name", ""),
                    "author":       art.get("author", ""),
                    "url":          art.get("url", ""),
                })
            time.sleep(0.5)

        except Exception as e:
            log.error(f"  FAILED {ticker}: {e}")

    if not all_rows:
        log.warning("  No articles fetched")
        return pd.DataFrame()

    news_df = pd.DataFrame(all_rows)
    news_df["published_at"] = pd.to_datetime(news_df["published_at"])
    news_df = news_df.drop_duplicates(subset=["url"])
    log.info(f"  Total: {len(news_df):,} unique articles")

    path = _save_bronze(news_df, "newsapi_headlines_raw.csv")
    if upload_gcs:
        _upload_to_gcs(path, "bronze/news/newsapi_headlines_raw.csv")

    log.info("[NewsAPI] Done.\n")
    return news_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion(
    tickers:    List[str] = config.DEFAULT_TICKERS,
    upload_gcs: bool = False,
) -> dict:
    log.info("=" * 60)
    log.info("FINANCIAL RISK PIPELINE — DATA INGESTION (Bronze Layer)")
    log.info(f"Tickers : {tickers}")
    log.info(f"Period  : {config.START_DATE} -> {config.END_DATE}")
    log.info("=" * 60 + "\n")

    return {
        "prices": fetch_yahoo_finance(tickers, upload_gcs=upload_gcs),
        "edgar":  fetch_sec_edgar(tickers,    upload_gcs=upload_gcs),
        "macro":  fetch_fred(                 upload_gcs=upload_gcs),
        "news":   fetch_news(tickers,         upload_gcs=upload_gcs),
    }


if __name__ == "__main__":
    import sys
    tickers = [t.upper() for t in sys.argv[1:]] if len(sys.argv) > 1 else config.DEFAULT_TICKERS
    run_ingestion(tickers=tickers)
