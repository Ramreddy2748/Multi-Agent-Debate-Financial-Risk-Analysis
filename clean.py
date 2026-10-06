"""
clean.py
─────────────────────────────────────────────────────────────────────────────
DATA CLEANING MODULE  (Bronze → Silver Layer)
Steps:
  1. Handle missing values  (forward-fill, row dropping)
  2. Detect & cap outliers  (Z-score on returns, IQR on fundamentals)
  3. Normalize data         (Min-Max for prices, standardize risk features)
  4. Format standardization (dates → ISO UTC, currency → USD millions,
                             snake_case columns, sentiment scoring)
  5. Feature engineering    (daily returns, annualized volatility, VADER
                             sentiment score for news headlines)

Output: Cleaned CSVs saved to Silver layer (local or GCS)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from scipy import stats

import config

log = logging.getLogger("clean")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dirs():
    Path(config.LOCAL_SILVER).mkdir(parents=True, exist_ok=True)
    Path(config.LOCAL_SILVER_PARQUET).mkdir(parents=True, exist_ok=True)

def _save_silver(df: pd.DataFrame, filename: str) -> str:
    _ensure_dirs()
    path = os.path.join(config.LOCAL_SILVER, filename)
    df.to_csv(path, index=True)
    log.info(f"  ✔ Saved Silver: {path}  ({len(df):,} rows × {df.shape[1]} cols)")
    return path

def _save_silver_parquet(df: pd.DataFrame, filename: str) -> Optional[str]:
    _ensure_dirs()
    path = os.path.join(config.LOCAL_SILVER_PARQUET, filename)
    try:
        df.to_parquet(path, index=True)
        log.info(f"  ✔ Saved Silver Parquet: {path}")
        return path
    except ImportError:
        log.warning("  pyarrow or fastparquet not installed — skipping Silver Parquet export.")
    except Exception as e:
        log.warning(f"  Silver Parquet export skipped for {filename}: {e}")
    return None

def _upload_to_gcs(local_path: str, gcs_path: str):
    try:
        from google.cloud import storage
        client = storage.Client(project=config.GCP_PROJECT_ID)
        bucket = client.bucket(config.GCS_BUCKET)
        bucket.blob(gcs_path).upload_from_filename(local_path)
        log.info(f"  ☁  Uploaded: gs://{config.GCS_BUCKET}/{gcs_path}")
    except ImportError:
        log.warning("google-cloud-storage not installed — skipping GCS upload.")
    except Exception as e:
        log.error(f"GCS upload failed: {e}")

def _log_cleaning_report(label: str, before: int, after: int, extra: str = ""):
    dropped = before - after
    pct     = (dropped / before * 100) if before > 0 else 0
    log.info(f"  [{label}] Rows: {before:,} → {after:,}  (dropped {dropped:,} = {pct:.1f}%){' | ' + extra if extra else ''}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    - Forward-fill up to FFILL_LIMIT consecutive NaNs (covers weekends/holidays)
    - Drop rows where >MAX_NAN_PCT_ROW of numeric columns are NaN
    """
    original_len = len(df)

    # Forward-fill time-series gaps
    df[numeric_cols] = df[numeric_cols].ffill(limit=config.FFILL_LIMIT)

    # Drop rows with excessive missing values
    nan_pct = df[numeric_cols].isna().mean(axis=1)
    df = df[nan_pct <= config.MAX_NAN_PCT_ROW].copy()

    nan_remaining = df[numeric_cols].isna().sum().sum()
    _log_cleaning_report("Missing Values", original_len, len(df),
                         f"{nan_remaining} NaNs remaining after ffill")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — OUTLIER DETECTION & CAPPING
# ─────────────────────────────────────────────────────────────────────────────

def remove_price_outliers(df: pd.DataFrame, return_col: str = "daily_return") -> pd.DataFrame:
    """
    Z-score outlier detection on daily returns.
    Values beyond ±ZSCORE_THRESHOLD std are capped (winsorized), not dropped.
    """
    if return_col not in df.columns:
        return df

    z = np.abs(stats.zscore(df[return_col].dropna()))
    outlier_mask = z > config.ZSCORE_THRESHOLD
    n_outliers   = outlier_mask.sum()

    # Cap at ±3σ instead of dropping (preserves time series continuity)
    mean, std = df[return_col].mean(), df[return_col].std()
    df[return_col] = df[return_col].clip(
        lower=mean - config.ZSCORE_THRESHOLD * std,
        upper=mean + config.ZSCORE_THRESHOLD * std,
    )
    log.info(f"  [Outliers] {n_outliers} outlier returns capped at ±{config.ZSCORE_THRESHOLD}σ")
    return df

def cap_fundamental_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    IQR-based capping for fundamental ratio columns.
    """
    for col in cols:
        if col not in df.columns or df[col].dropna().empty:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr    = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_before = df[col].notna().sum()
        df[col]  = df[col].clip(lower=lower, upper=upper)
        log.info(f"  [IQR Cap] {col}: clipped to [{lower:.2f}, {upper:.2f}]")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_minmax(df: pd.DataFrame, cols: list, suffix: str = "_norm") -> pd.DataFrame:
    """Min-Max normalization → [0, 1] for each column."""
    for col in cols:
        if col not in df.columns:
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        denom   = col_max - col_min
        if denom == 0:
            df[f"{col}{suffix}"] = 0.0
        else:
            df[f"{col}{suffix}"] = (df[col] - col_min) / denom
    return df

def map_to_risk_scale(value: float, min_val: float, max_val: float) -> float:
    """Map a raw metric to a 1–10 risk score."""
    if max_val == min_val:
        return 5.0
    return 1.0 + 9.0 * (value - min_val) / (max_val - min_val)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FORMAT STANDARDIZATION
# ─────────────────────────────────────────────────────────────────────────────

def standardize_formats(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Rename columns to snake_case
    - Parse dates to datetime (UTC)
    - Convert market_cap/revenue from raw USD to USD millions
    """
    # snake_case columns
    df.columns = [
        c.lower()
         .replace(" ", "_")
         .replace("-", "_")
         .replace("/", "_")
        for c in df.columns
    ]

    # Ensure index is datetime
    if df.index.dtype != "datetime64[ns]":
        try:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        except Exception:
            pass

    # Scale market_cap and revenue to USD millions
    for col in ["market_cap", "revenue_ttm", "total_assets", "total_liabilities",
                "operating_income", "net_income", "stockholders_equity", "cash"]:
        if col in df.columns:
            df[col] = df[col] / 1e6   # → USD millions
            df = df.rename(columns={col: f"{col}_usd_m"})

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns to price DataFrame:
      - daily_return          : pct change in Close
      - log_return            : log of price ratio (better for stats)
      - rolling_vol_30d       : 30-day rolling annualized volatility
      - rolling_vol_60d       : 60-day rolling annualized volatility
      - sma_20, sma_50        : simple moving averages
      - price_vs_52w_high_pct : drawdown from 52-week high
    """
    df = df.sort_index()
    close = df["close"] if "close" in df.columns else df["Close"]

    df["daily_return"]    = close.pct_change()
    df["log_return"]      = np.log(close / close.shift(1))
    df["rolling_vol_30d"] = df["daily_return"].rolling(30).std()  * np.sqrt(252)
    df["rolling_vol_60d"] = df["daily_return"].rolling(60).std()  * np.sqrt(252)
    df["sma_20"]          = close.rolling(20).mean()
    df["sma_50"]          = close.rolling(50).mean()

    if "52w_high" in df.columns:
        df["price_vs_52w_high_pct"] = (close - df["52w_high"]) / df["52w_high"] * 100

    return df

def score_news_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply VADER sentiment scoring to news headlines + descriptions.
    Adds columns: vader_compound, vader_label (positive/neutral/negative)

    Requires: pip install vaderSentiment
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()

        def score(row):
            text = f"{row.get('title', '')} {row.get('description', '')}"
            return analyzer.polarity_scores(text)["compound"]

        news_df = news_df.copy()
        news_df["vader_compound"] = news_df.apply(score, axis=1)
        news_df["vader_label"]    = news_df["vader_compound"].apply(
            lambda x: "positive" if x >= 0.05 else ("negative" if x <= -0.05 else "neutral")
        )
        log.info(f"  [Sentiment] Scored {len(news_df):,} headlines with VADER")

    except ImportError:
        log.warning("  vaderSentiment not installed — run: pip install vaderSentiment")
        news_df["vader_compound"] = 0.0
        news_df["vader_label"]    = "neutral"

    return news_df


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING PIPELINE PER DATA SOURCE
# ─────────────────────────────────────────────────────────────────────────────

def clean_prices(
    price_data: dict[str, pd.DataFrame],
    upload_gcs: bool = False,
) -> dict[str, pd.DataFrame]:
    """Clean Yahoo Finance price DataFrames."""
    log.info("[Clean] Stock Prices ─────────────────────────")
    cleaned = {}
    fundamental_cols = ["pe_ratio", "eps", "beta", "debt_to_equity"]

    for ticker, df in price_data.items():
        log.info(f"  → {ticker}")
        df = df.copy()
        df = standardize_formats(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = handle_missing(df, numeric_cols)
        df = engineer_price_features(df)
        df = remove_price_outliers(df, return_col="daily_return")
        df = cap_fundamental_outliers(df, fundamental_cols)
        df = normalize_minmax(df, ["close", "volume"])

        df = df.drop_duplicates()
        filename = f"silver_prices_{ticker}.csv"
        path = _save_silver(df, filename)
        _save_silver_parquet(df, f"silver_prices_{ticker}.parquet")
        if upload_gcs:
            _upload_to_gcs(path, f"silver/prices/{filename}")

        cleaned[ticker] = df

    log.info("")
    return cleaned

def clean_macro(
    macro_df: pd.DataFrame,
    upload_gcs: bool = False,
) -> pd.DataFrame:
    """Clean FRED macroeconomic DataFrame."""
    if macro_df.empty:
        log.info("[Clean] Macro — skipped (no data)\n")
        return macro_df

    log.info("[Clean] Macro Indicators ─────────────────────")
    df = macro_df.copy()
    df = standardize_formats(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = handle_missing(df, numeric_cols)

    # Forward-fill monthly series to daily
    df = df.resample("D").ffill()

    filename = "silver_macro.csv"
    path = _save_silver(df, filename)
    _save_silver_parquet(df, "silver_macro.parquet")
    if upload_gcs:
        _upload_to_gcs(path, "silver/macro/silver_macro.csv")

    log.info("")
    return df

def clean_news(
    news_df: pd.DataFrame,
    upload_gcs: bool = False,
) -> pd.DataFrame:
    """Clean NewsAPI headlines and apply sentiment scoring."""
    if news_df.empty:
        log.info("[Clean] News — skipped (no data)\n")
        return news_df

    log.info("[Clean] News Sentiment ───────────────────────")
    df = news_df.copy()

    # Drop rows with missing headline
    before = len(df)
    df = df.dropna(subset=["title"])
    df = df[df["title"].str.strip() != ""]
    _log_cleaning_report("Headlines", before, len(df), "removed blank titles")

    # Deduplicate by URL
    df = df.drop_duplicates(subset=["url"])

    # Standardize dates
    df["published_at"] = pd.to_datetime(df["published_at"]).dt.tz_localize(None)

    # VADER sentiment scoring
    df = score_news_sentiment(df)

    # Daily average sentiment per ticker
    df["date"] = df["published_at"].dt.date
    daily_sentiment = (
        df.groupby(["ticker", "date"])["vader_compound"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "sentiment_mean", "std": "sentiment_std", "count": "article_count"})
    )

    filename = "silver_news_sentiment.csv"
    path = _save_silver(daily_sentiment, filename)
    _save_silver_parquet(daily_sentiment, "silver_news_sentiment.parquet")
    if upload_gcs:
        _upload_to_gcs(path, "silver/news/silver_news_sentiment.csv")

    log.info("")
    return daily_sentiment

def clean_edgar(
    edgar_data: dict[str, pd.DataFrame],
    upload_gcs: bool = False,
) -> dict[str, pd.DataFrame]:
    """Clean SEC EDGAR financial statement DataFrames."""
    log.info("[Clean] SEC EDGAR Financials ─────────────────")
    cleaned = {}

    for ticker, df in edgar_data.items():
        log.info(f"  → {ticker}")
        df = df.copy()
        df = standardize_formats(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = handle_missing(df, numeric_cols)
        df = df.sort_values("end_date", ascending=True)
        df = df.drop_duplicates(subset=["end_date"], keep="last")

        filename = f"silver_edgar_{ticker}.csv"
        path = _save_silver(df, filename)
        _save_silver_parquet(df, f"silver_edgar_{ticker}.parquet")
        if upload_gcs:
            _upload_to_gcs(path, f"silver/edgar/{filename}")

        cleaned[ticker] = df

    log.info("")
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_cleaning(raw_data: dict, upload_gcs: bool = False) -> dict:
    """
    Run full cleaning pipeline on raw ingested data.

    Args:
        raw_data:   Output dict from ingest.run_ingestion()
        upload_gcs: If True, upload Silver CSVs to GCS

    Returns:
        dict with keys: prices, edgar, macro, news
    """
    log.info("=" * 70)
    log.info("FINANCIAL RISK PIPELINE — DATA CLEANING (Silver Layer)")
    log.info("=" * 70 + "\n")

    return {
        "prices": clean_prices(raw_data.get("prices", {}), upload_gcs=upload_gcs),
        "edgar":  clean_edgar(raw_data.get("edgar",  {}), upload_gcs=upload_gcs),
        "macro":  clean_macro(raw_data.get("macro",  pd.DataFrame()), upload_gcs=upload_gcs),
        "news":   clean_news(raw_data.get("news",    pd.DataFrame()), upload_gcs=upload_gcs),
    }


if __name__ == "__main__":
    # Quick test: load Bronze CSVs and clean them
    import glob
    price_dfs = {}
    for f in glob.glob(f"{config.LOCAL_BRONZE}/yahoo_*.csv"):
        ticker = f.split("yahoo_")[1].split("_")[0]
        price_dfs[ticker] = pd.read_csv(f, index_col=0, parse_dates=True)
    run_cleaning({"prices": price_dfs, "edgar": {}, "macro": pd.DataFrame(), "news": pd.DataFrame()})



