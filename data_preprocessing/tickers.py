"""
tickers.py
─────────────────────────────────────────────────────────────────────────────
TICKER REGISTRY — S&P 500 + NASDAQ 100 (~600 companies)
Fetches live ticker lists from Wikipedia (always up to date).
Groups companies by GICS sector for batched processing.

Usage:
    from tickers import get_all_tickers, get_tickers_by_sector, SECTORS
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import time
from pathlib import Path
import pandas as pd
import requests
from typing import Optional

log = logging.getLogger("tickers")

# ─── GICS Sectors ─────────────────────────────────────────────────────────────
SECTORS = [
    "Information Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
]

# ─── Wikipedia URLs ───────────────────────────────────────────────────────────
SP500_URL   = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ_URL  = "https://en.wikipedia.org/wiki/Nasdaq-100"
CACHED_UNIVERSE_PATH = Path("data/gold/ticker_universe.csv")


def _read_html_with_headers(url: str) -> list:
    """
    Fetch Wikipedia page with browser-like headers to avoid 403 blocks,
    then parse tables with pd.read_html.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    from io import StringIO
    return pd.read_html(StringIO(resp.text))


def fetch_sp500() -> pd.DataFrame:
    """
    Fetch current S&P 500 constituents from Wikipedia.
    Returns DataFrame with columns: ticker, company, sector, sub_industry, index
    """
    log.info("  Fetching S&P 500 from Wikipedia...")
    try:
        tables = _read_html_with_headers(SP500_URL)
        df = tables[0].copy()

        df = df.rename(columns={
            "Symbol":                "ticker",
            "Security":              "company",
            "GICS Sector":           "sector",
            "GICS Sub-Industry":     "sub_industry",
            "Headquarters Location": "headquarters",
            "Date added":            "date_added",
            "CIK":                   "cik",
            "Founded":               "founded",
        })

        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False).str.strip()
        df["index"]  = "SP500"
        log.info(f"  ✔ S&P 500: {len(df)} companies across {df['sector'].nunique()} sectors")
        return df[["ticker", "company", "sector", "sub_industry", "index"]]

    except Exception as e:
        log.error(f"  Failed to fetch S&P 500: {e}")
        return pd.DataFrame()


def fetch_nasdaq100() -> pd.DataFrame:
    """
    Fetch current NASDAQ-100 constituents from Wikipedia.
    Returns DataFrame with columns: ticker, company, sector, sub_industry, index
    """
    log.info("  Fetching NASDAQ-100 from Wikipedia...")
    try:
        tables = _read_html_with_headers(NASDAQ_URL)

        df = None
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("ticker" in c or "symbol" in c for c in cols):
                df = t.copy()
                break

        if df is None:
            log.warning("  Could not find NASDAQ-100 table — using fallback list")
            return _nasdaq100_fallback()

        df.columns = [str(c).lower().strip() for c in df.columns]
        ticker_col  = next((c for c in df.columns if "ticker" in c or "symbol" in c), None)
        company_col = next((c for c in df.columns if "company" in c or "security" in c), None)
        sector_col  = next((c for c in df.columns if "sector" in c or "industry" in c), None)

        if not ticker_col:
            return _nasdaq100_fallback()

        result = pd.DataFrame()
        result["ticker"]       = df[ticker_col].str.strip()
        result["company"]      = df[company_col].str.strip() if company_col else ""
        result["sector"]       = df[sector_col].str.strip()  if sector_col  else "Unknown"
        result["sub_industry"] = ""
        result["index"]        = "NASDAQ100"

        log.info(f"  ✔ NASDAQ-100: {len(result)} companies")
        return result

    except Exception as e:
        log.warning(f"  NASDAQ-100 fetch failed ({e}) — using fallback list")
        return _nasdaq100_fallback()


def _nasdaq100_fallback() -> pd.DataFrame:
    """Hardcoded NASDAQ-100 fallback in case Wikipedia scraping fails."""
    nasdaq_tickers = [
        ("AAPL",  "Apple Inc.",              "Information Technology"),
        ("MSFT",  "Microsoft Corporation",   "Information Technology"),
        ("NVDA",  "NVIDIA Corporation",      "Information Technology"),
        ("AMZN",  "Amazon.com Inc.",         "Consumer Discretionary"),
        ("META",  "Meta Platforms Inc.",     "Communication Services"),
        ("GOOGL", "Alphabet Inc. Class A",   "Communication Services"),
        ("GOOG",  "Alphabet Inc. Class C",   "Communication Services"),
        ("TSLA",  "Tesla Inc.",              "Consumer Discretionary"),
        ("AVGO",  "Broadcom Inc.",           "Information Technology"),
        ("COST",  "Costco Wholesale",        "Consumer Staples"),
        ("ASML",  "ASML Holding",            "Information Technology"),
        ("NFLX",  "Netflix Inc.",            "Communication Services"),
        ("AZN",   "AstraZeneca",             "Health Care"),
        ("AMD",   "Advanced Micro Devices",  "Information Technology"),
        ("CSCO",  "Cisco Systems",           "Information Technology"),
        ("ADBE",  "Adobe Inc.",              "Information Technology"),
        ("INTC",  "Intel Corporation",       "Information Technology"),
        ("QCOM",  "Qualcomm Inc.",           "Information Technology"),
        ("INTU",  "Intuit Inc.",             "Information Technology"),
        ("TXN",   "Texas Instruments",       "Information Technology"),
        ("AMGN",  "Amgen Inc.",              "Health Care"),
        ("HON",   "Honeywell International", "Industrials"),
        ("AMAT",  "Applied Materials",       "Information Technology"),
        ("SBUX",  "Starbucks Corporation",   "Consumer Discretionary"),
        ("GILD",  "Gilead Sciences",         "Health Care"),
        ("ADI",   "Analog Devices",          "Information Technology"),
        ("MDLZ",  "Mondelez International",  "Consumer Staples"),
        ("REGN",  "Regeneron Pharmaceuticals","Health Care"),
        ("VRTX",  "Vertex Pharmaceuticals",  "Health Care"),
        ("PANW",  "Palo Alto Networks",      "Information Technology"),
        ("LRCX",  "Lam Research",            "Information Technology"),
        ("KLAC",  "KLA Corporation",         "Information Technology"),
        ("SNPS",  "Synopsys Inc.",           "Information Technology"),
        ("CDNS",  "Cadence Design Systems",  "Information Technology"),
        ("MRVL",  "Marvell Technology",      "Information Technology"),
        ("CRWD",  "CrowdStrike Holdings",    "Information Technology"),
        ("ABNB",  "Airbnb Inc.",             "Consumer Discretionary"),
        ("ORLY",  "O'Reilly Automotive",     "Consumer Discretionary"),
        ("CTAS",  "Cintas Corporation",      "Industrials"),
        ("FTNT",  "Fortinet Inc.",           "Information Technology"),
    ]
    df = pd.DataFrame(nasdaq_tickers, columns=["ticker", "company", "sector"])
    df["sub_industry"] = ""
    df["index"] = "NASDAQ100"
    log.info(f"  NASDAQ-100 fallback: {len(df)} companies")
    return df


def _cached_universe_fallback() -> pd.DataFrame:
    """Use the last saved ticker universe when live Wikipedia fetches fail."""
    if not CACHED_UNIVERSE_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(CACHED_UNIVERSE_PATH)
        required = {"ticker", "company", "sector"}
        if not required.issubset(df.columns):
            log.warning(f"  Cached ticker universe missing required columns: {CACHED_UNIVERSE_PATH}")
            return pd.DataFrame()
        if "sub_industry" not in df.columns:
            df["sub_industry"] = ""
        if "index" not in df.columns:
            df["index"] = "CACHED"
        df["ticker"] = df["ticker"].astype(str).str.strip()
        df = df.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"], keep="first")
        log.info(f"  Cached ticker universe fallback: {len(df)} companies from {CACHED_UNIVERSE_PATH}")
        return df[["ticker", "company", "sector", "sub_industry", "index"]].reset_index(drop=True)
    except Exception as exc:
        log.warning(f"  Cached ticker universe fallback failed: {exc}")
        return pd.DataFrame()


def get_all_tickers(deduplicate: bool = True) -> pd.DataFrame:
    """
    Fetch and combine S&P 500 + NASDAQ-100 tickers.

    Returns:
        DataFrame with columns: ticker, company, sector, sub_industry, index
        'index' column shows 'SP500', 'NASDAQ100', or 'BOTH'
    """
    log.info("[Tickers] Building S&P 500 + NASDAQ-100 universe...")

    sp500   = fetch_sp500()
    nasdaq  = fetch_nasdaq100()

    if sp500.empty and nasdaq.empty:
        cached = _cached_universe_fallback()
        if not cached.empty:
            return cached
        log.error("Could not fetch any tickers!")
        return pd.DataFrame()

    if sp500.empty and len(nasdaq) <= 40:
        cached = _cached_universe_fallback()
        if not cached.empty and len(cached) > len(nasdaq):
            log.warning("  Live S&P 500 fetch failed; using larger cached ticker universe instead of small NASDAQ fallback.")
            return cached

    combined = pd.concat([sp500, nasdaq], ignore_index=True)

    if deduplicate:
        # Mark tickers appearing in both indices
        dupes = combined[combined.duplicated(subset=["ticker"], keep=False)]
        both  = dupes["ticker"].unique()

        combined = combined.drop_duplicates(subset=["ticker"], keep="first")
        combined.loc[combined["ticker"].isin(both), "index"] = "BOTH"

        log.info(f"{len(both)} tickers appear in both indices (marked as BOTH)")

    log.info(f"[Tickers] Total universe: {len(combined)} unique companies\n")
    return combined.reset_index(drop=True)


def get_tickers_by_sector(df: Optional[pd.DataFrame] = None) -> dict:
    """
    Group tickers by GICS sector.

    Returns:
        dict of {sector_name: [ticker1, ticker2, ...]}
    """
    if df is None:
        df = get_all_tickers()

    sector_map = {}
    for sector in df["sector"].dropna().unique():
        tickers = df[df["sector"] == sector]["ticker"].tolist()
        sector_map[sector] = tickers

    # Log summary
    log.info("[Tickers] Sector breakdown:")
    for sector, tickers in sorted(sector_map.items(), key=lambda x: -len(x[1])):
        log.info(f"{sector:<35} {len(tickers):>4} companies")

    return sector_map


def get_sector_batches(
    df: Optional[pd.DataFrame] = None,
    batch_size: int = 50,
) -> list:
    """
    Split each sector into batches of `batch_size` tickers.
    This prevents rate limiting and makes failures recoverable.

    Returns:
        List of dicts: [{sector, batch_num, tickers}, ...]
    """
    if df is None:
        df = get_all_tickers()

    batches = []
    sector_map = get_tickers_by_sector(df)

    for sector, tickers in sector_map.items():
        # Split sector into chunks
        for i in range(0, len(tickers), batch_size):
            chunk = tickers[i:i + batch_size]
            batches.append({
                "sector":    sector,
                "batch_num": i // batch_size + 1,
                "total_batches": (len(tickers) + batch_size - 1) // batch_size,
                "tickers":   chunk,
                "count":     len(chunk),
            })

    log.info(f"[Tickers] Created {len(batches)} batches of up to {batch_size} tickers each")
    return batches


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    df = get_all_tickers()
    print(f"\nTotal companies: {len(df)}")
    print(f"Sectors: {df['sector'].nunique()}")
    print(f"\nSample:\n{df.head(10).to_string()}")

    batches = get_sector_batches(df, batch_size=50)
    print(f"\nTotal batches: {len(batches)}")
