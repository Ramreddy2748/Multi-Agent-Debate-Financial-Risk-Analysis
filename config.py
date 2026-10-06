"""
config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration for the Financial Risk Data Pipeline.
All API keys, GCP settings, and pipeline parameters live here.
─────────────────────────────────────────────────────────────────────────────
"""

import os
from datetime import datetime, timedelta

# ─── TICKERS (dynamic — pass any list at runtime) ─────────────────────────────
DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "JPM"]

# ─── DATE RANGE ───────────────────────────────────────────────────────────────
END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

# ─── API KEYS (set as environment variables — never hardcode) ─────────────────
FRED_API_KEY    = os.getenv("FRED_API_KEY", "YOUR_FRED_API_KEY_HERE")
NEWS_API_KEY    = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY_HERE")

# ─── GCP SETTINGS ─────────────────────────────────────────────────────────────
GCP_PROJECT_ID  = os.getenv("GCP_PROJECT_ID", "sacred-catfish-488122-u7")
GCS_BUCKET      = os.getenv("GCS_BUCKET",     "financial-risk-pipeline")

# Medallion Architecture paths in GCS
GCS_BRONZE      = f"gs://{GCS_BUCKET}/bronze"   # Raw data as-is
GCS_SILVER      = f"gs://{GCS_BUCKET}/silver"   # Cleaned & standardized
GCS_GOLD        = f"gs://{GCS_BUCKET}/gold"     # Analytics-ready

# BigQuery dataset
BQ_DATASET      = "financial_risk"
BQ_TABLE_PRICES = f"{GCP_PROJECT_ID}.{BQ_DATASET}.stock_prices"
BQ_TABLE_MACRO  = f"{GCP_PROJECT_ID}.{BQ_DATASET}.macro_indicators"
BQ_TABLE_NEWS   = f"{GCP_PROJECT_ID}.{BQ_DATASET}.news_sentiment"
BQ_TABLE_RISK   = f"{GCP_PROJECT_ID}.{BQ_DATASET}.risk_scores"

# ─── LOCAL PATHS (for dev / before GCP is set up) ─────────────────────────────
LOCAL_DATA_DIR  = "data"
LOCAL_BRONZE    = f"{LOCAL_DATA_DIR}/bronze"
LOCAL_SILVER    = f"{LOCAL_DATA_DIR}/silver"
LOCAL_SILVER_PARQUET = f"{LOCAL_DATA_DIR}/silver_parquet"
LOCAL_GOLD      = f"{LOCAL_DATA_DIR}/gold"
LOCAL_REPORTS   = f"{LOCAL_DATA_DIR}/reports"
LOCAL_VERDICTS   = f"{LOCAL_DATA_DIR}/verdicts"

# ─── CLEANING THRESHOLDS ──────────────────────────────────────────────────────
ZSCORE_THRESHOLD        = 3.0    # Outlier detection on daily returns
MAX_NAN_PCT_ROW         = 0.20   # Drop rows with >20% missing values
FFILL_LIMIT             = 2      # Forward-fill up to N consecutive NaNs

# ─── EDA SETTINGS ─────────────────────────────────────────────────────────────
RISK_SCORE_LOW          = 3.5    # <= LOW risk
RISK_SCORE_HIGH         = 6.0    # >  HIGH risk (between = MODERATE)
PLOT_STYLE = "seaborn-whitegrid"
FIGURE_DPI              = 150

# ─── FRED MACRO SERIES ────────────────────────────────────────────────────────
FRED_SERIES = {
    "fed_funds_rate":   "FEDFUNDS",    # Federal Funds Rate
    "cpi":              "CPIAUCSL",    # Consumer Price Index
    "treasury_10y":     "GS10",        # 10-Year Treasury Yield
    "unemployment":     "UNRATE",      # Unemployment Rate
    "gdp_growth":       "A191RL1Q225SBEA",  # Real GDP growth rate
    "yield_curve_10y2y":"T10Y2Y",      # 10-Year minus 2-Year Treasury spread
    "vix":              "VIXCLS",      # CBOE Volatility Index
    "wti_oil":          "DCOILWTICO",  # WTI crude oil price
}
