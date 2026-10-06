# Financial Risk Data Pipeline — Complete Technical Deep Dive

**Project:** Debate-Based Multi-Agent Reasoning for Financial Risk Modeling  
**Course:** DATA 298A — MSDA Project I  
**Team 3:** Harshitha Boinepally · Kalyan Mamidi · Venkata Ramireddy Seelam · Venkata Siddarth Gullipalli

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Architecture Overview](#2-architecture-overview)
3. [config.py — Central Configuration](#3-configpy--central-configuration)
4. [tickers.py — Ticker Registry](#4-tickerspy--ticker-registry)
5. [ingest.py — Bronze Layer (Data Extraction)](#5-ingestpy--bronze-layer-data-extraction)
6. [clean.py — Silver Layer (Data Preprocessing)](#6-cleanpy--silver-layer-data-preprocessing)
7. [eda.py — Gold Layer (Analytics + Risk Scoring)](#7-edapy--gold-layer-analytics--risk-scoring)
8. [pipeline_large_scale.py — Orchestrator](#8-pipeline_large_scalepy--orchestrator)
9. [split.py — Train/Val/Test Split](#9-splitpy--trainvaltest-split)
10. [stats_summary.py — Reporting](#10-stats_summarypy--reporting)
11. [GCP Integration — Storage and BigQuery](#11-gcp-integration--storage-and-bigquery)
12. [Data Flow: End to End](#12-data-flow-end-to-end)
13. [Risk Scoring Model](#13-risk-scoring-model)
14. [File Output Reference](#14-file-output-reference)
15. [Configuration Reference](#15-configuration-reference)

---

## 1. Project Goal

This pipeline builds the **data foundation** for a multi-agent AI system that performs structured investment risk assessment. The goal of this data layer is to:

- Collect financial data for S&P 500 + NASDAQ-100 companies (~600 tickers) from 4 independent sources
- Clean, standardize, and engineer features from that raw data
- Compute a multi-dimensional risk score (1–10) for each company across 4 risk dimensions
- Produce analytics-ready datasets (Gold layer) that feed into downstream AI agent models
- Optionally persist all outputs to Google Cloud Storage and BigQuery

The pipeline follows the **Medallion Architecture** pattern: Bronze (raw) → Silver (clean) → Gold (analytics-ready).

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                   │
│                                                                         │
│   Yahoo Finance     SEC EDGAR       FRED API       NewsAPI              │
│   (prices/OHLCV)    (10-K filings)  (macro data)   (headlines)          │
└──────────┬───────────────┬──────────────┬──────────────┬────────────────┘
           │               │              │              │
           └───────────────┴──────────────┴──────────────┘
                                    │
                              ┌─────▼──────┐
                              │  ingest.py │   ← Bronze Layer
                              └─────┬──────┘
                                    │  data/bronze/*.csv
                                    │
                              ┌─────▼──────┐
                              │  clean.py  │   ← Silver Layer
                              └─────┬──────┘
                                    │  data/silver/*.csv
                                    │
                              ┌─────▼──────┐
                              │   eda.py   │   ← Gold Layer
                              └─────┬──────┘
                                    │  data/gold/*.csv + data/reports/*.png
                                    │
                              ┌─────▼──────┐
                              │  split.py  │   ← Train / Val / Test
                              └─────┬──────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             split_train.csv   split_val.csv   split_test.csv
                  (70%)            (15%)           (15%)
                                    │
                              ┌─────▼──────┐
                              │    GCP     │   ← Cloud Storage + BigQuery
                              └────────────┘
```

The large-scale orchestrator `pipeline_large_scale.py` wires all four scripts together and runs them sector by sector across all 600 companies.

---

## 3. config.py — Central Configuration

**File:** `config.py`  
**Purpose:** Single source of truth for all constants, thresholds, API keys, and file paths used across every module.

### What it defines

#### Tickers
```python
DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "JPM"]
```
Used when running `ingest.py` directly without arguments. The large-scale pipeline overrides this with live ticker lists from Wikipedia.

#### Date Range
```python
END_DATE   = datetime.today().strftime("%Y-%m-%d")      # today
START_DATE = (datetime.today() - timedelta(days=365))   # 1 year ago
```
Every run automatically targets the trailing 365 days. This means you always get the most recent year of price history without hardcoding dates.

#### API Keys
```python
FRED_API_KEY = os.getenv("FRED_API_KEY", "YOUR_FRED_API_KEY_HERE")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY_HERE")
```
Keys are read from environment variables. If not set, the pipeline falls back to reading a `.env` file (handled inside `ingest._load_api_key()`). Yahoo Finance and SEC EDGAR require no keys.

#### GCP Settings
```python
GCP_PROJECT_ID = "sacred-catfish-488122-u7"
GCS_BUCKET     = "financial-risk-pipeline"

GCS_BRONZE = "gs://financial-risk-pipeline/bronze"
GCS_SILVER = "gs://financial-risk-pipeline/silver"
GCS_GOLD   = "gs://financial-risk-pipeline/gold"

BQ_TABLE_RISK = "sacred-catfish-488122-u7.financial_risk.risk_scores"
```

#### Local Paths
```python
LOCAL_BRONZE  = "data/bronze"
LOCAL_SILVER  = "data/silver"
LOCAL_GOLD    = "data/gold"
LOCAL_REPORTS = "data/reports"
```
All scripts write to these local directories first, then optionally upload to GCS.

#### Cleaning Thresholds
```python
ZSCORE_THRESHOLD = 3.0   # cap daily returns beyond ±3 standard deviations
MAX_NAN_PCT_ROW  = 0.20  # drop rows with more than 20% missing numeric values
FFILL_LIMIT      = 2     # forward-fill at most 2 consecutive NaN values
```

#### Risk Score Thresholds
```python
RISK_SCORE_LOW  = 3.5   # composite score ≤ 3.5 → LOW
RISK_SCORE_HIGH = 6.0   # composite score > 6.0 → HIGH (between = MODERATE)
```

#### FRED Macro Series
```python
FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi":            "CPIAUCSL",
    "treasury_10y":   "GS10",
    "unemployment":   "UNRATE",
    "gdp_growth":     "A191RL1Q225SBEA",
}
```
Each key becomes a column name in the macro DataFrame; each value is the FRED series ID passed to the API.

---

## 4. tickers.py — Ticker Registry

**File:** `tickers.py`  
**Purpose:** Build and manage the universe of ~600 companies from S&P 500 and NASDAQ-100. Groups them by GICS sector and creates batches for rate-limit-safe processing.

### How it fetches tickers

#### S&P 500 — `fetch_sp500()`
```
Wikipedia URL: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
       ↓
pd.read_html() parses the HTML table
       ↓
Columns renamed: Symbol→ticker, Security→company, GICS Sector→sector
       ↓
Dots in tickers replaced with dashes (e.g., BRK.B → BRK-B)
       ↓
Returns DataFrame: [ticker, company, sector, sub_industry, index="SP500"]
```

#### NASDAQ-100 — `fetch_nasdaq100()`
```
Wikipedia URL: https://en.wikipedia.org/wiki/Nasdaq-100
       ↓
Searches all HTML tables for a column with "ticker" or "symbol"
       ↓
If found: parse dynamically (column names vary by Wikipedia edit)
If not found: fall back to hardcoded list of 40 major NASDAQ companies
       ↓
Returns DataFrame: [ticker, company, sector, sub_industry, index="NASDAQ100"]
```

Both functions use browser-like headers to avoid 403 blocks from Wikipedia:
```python
"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36..."
```

#### Combining — `get_all_tickers()`
```
S&P 500 DataFrame + NASDAQ-100 DataFrame
       ↓
pd.concat() → ~600 rows
       ↓
Find duplicates (tickers in both indices)
       ↓
Drop duplicates, keep first occurrence
Mark tickers appearing in both as index="BOTH"
       ↓
Returns ~600 unique companies
```

### Sector Grouping — `get_tickers_by_sector()`

Groups the combined DataFrame into a dict:
```python
{
    "Information Technology": ["AAPL", "MSFT", "NVDA", ...],
    "Health Care":            ["JNJ", "UNH", "ABT", ...],
    "Financials":             ["JPM", "BAC", "WFC", ...],
    ...  # 11 GICS sectors total
}
```

### Batch Creation — `get_sector_batches()`

Splits each sector's tickers into chunks of 50:
```python
[
    {"sector": "Information Technology", "batch_num": 1, "total_batches": 3, "tickers": [...50...]},
    {"sector": "Information Technology", "batch_num": 2, "total_batches": 3, "tickers": [...50...]},
    ...
]
```
This prevents rate limiting by never hitting any API with more than 50 requests in a single batch.

---

## 5. ingest.py — Bronze Layer (Data Extraction)

**File:** `ingest.py`  
**Purpose:** Pull raw data from all 4 external sources and save it locally as CSV files (Bronze layer). Optionally upload to GCS.

### Retry-Enabled HTTP Session

Before any data is fetched, a shared `requests.Session` is created:
```python
retry = Retry(total=3, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
```
- Retries up to 3 times on rate-limit (429) or server errors (5xx)
- Waits 1.5s, then 3s, then 4.5s between retries (exponential backoff)
- All 4 source functions share this session

---

### Source 1 — Yahoo Finance (`fetch_yahoo_finance`)

**What it fetches:**
- Daily OHLCV (Open, High, Low, Close, Volume) price history
- 11 fundamental metrics attached to every row:
  `pe_ratio, eps, market_cap, revenue_ttm, debt_to_equity, beta, 52w_high, 52w_low, sector, industry, long_name`

**Step-by-step:**
```
For each ticker:
    1. yf.Ticker(ticker).history(start=START_DATE, end=END_DATE)
       → Returns DataFrame indexed by trading date
    
    2. Strip timezone from DatetimeIndex (tz_localize(None))
       → Prevents merge issues with timezone-naive columns later
    
    3. Add ticker column to DataFrame
    
    4. Attach fundamentals from stock.info dict
       → These are scalar values (same for all rows of a given ticker)
    
    5. Save as: data/bronze/yahoo_{TICKER}_{start}_to_{end}.csv
    
    6. Sleep 0.5s before next ticker (Yahoo rate limit protection)
```

**Output file example:** `data/bronze/yahoo_AAPL_2024-05-06_to_2025-05-06.csv`  
**Rows:** ~252 trading days per ticker  
**Columns:** Open, High, Low, Close, Volume, Dividends, Stock Splits, ticker, pe_ratio, eps, market_cap, revenue_ttm, debt_to_equity, beta, 52w_high, 52w_low, sector, industry, long_name

---

### Source 2 — SEC EDGAR (`fetch_sec_edgar`)

**What it fetches:**
- Annual financial statement data (10-K filings) directly from the SEC XBRL API
- 12 accounting concepts per company

**Step-by-step:**
```
Step 1: Download CIK map
    GET https://www.sec.gov/files/company_tickers.json
    → JSON of all ~10,000 SEC-registered companies
    → Build dict: {"AAPL": "0000320193", "MSFT": "0000789019", ...}

Step 2: For each ticker, get CIK then fetch all facts
    GET https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
    → One large JSON with every XBRL fact ever reported by this company

Step 3: Extract 12 specific XBRL concepts (10-K/20-F only):
    Revenues                        → revenue
    NetIncomeLoss                   → net_income
    Assets                          → total_assets
    Liabilities                     → total_liabilities
    StockholdersEquity              → stockholders_equity
    EarningsPerShareBasic           → eps_basic
    EarningsPerShareDiluted         → eps_diluted
    OperatingIncomeLoss             → operating_income
    CashAndCashEquivalentsAtCarryingValue → cash
    LongTermDebt                    → long_term_debt
    CommonStockSharesOutstanding    → shares_outstanding
    RevenueFromContractWithCustomer → revenue_alt

Step 4: Build long-format DataFrame
    One row per (ticker, concept, year, filing_date)

Step 5: Pivot to wide format
    One row per (ticker, year) with each concept as a column

Step 6: Save both formats:
    data/bronze/edgar_{TICKER}_raw_long.csv       (long format)
    data/bronze/edgar_{TICKER}_annual_financials.csv  (wide format)

Step 7: Sleep 0.6s between tickers (SEC rate limit: 10 req/sec)
```

**Why two formats?**  
The long format is useful for auditing and debugging (you can see exactly what value came from which filing). The wide format is what the cleaning module actually uses (one row = one year of financials).

---

### Source 3 — FRED API (`fetch_fred`)

**What it fetches:**  
5 macroeconomic time series published by the Federal Reserve Bank of St. Louis.

**Step-by-step:**
```
For each series (fed_funds_rate, cpi, treasury_10y, unemployment, gdp_growth):
    GET https://api.stlouisfed.org/fred/series/observations
    Params: series_id, observation_start, observation_end, api_key
    
    → JSON list of {date, value} pairs
    → Convert to DataFrame, set date as index
    → Rename value column to series name (e.g., "fed_funds_rate")
    → Sleep 0.3s between series

All 5 series DataFrames joined on date index (outer join):
    pd.concat([df1, df2, df3, df4, df5], axis=1)

Saved as: data/bronze/fred_macro_{start}_to_{end}.csv
```

**Output:** A wide DataFrame where each column is one macro indicator and each row is a date. FRED series are monthly (FEDFUNDS, CPI) or quarterly (GDP), so the DataFrame will have many NaN values that get forward-filled in the Silver layer.

---

### Source 4 — NewsAPI (`fetch_news`)

**What it fetches:**  
Financial news headlines and descriptions mentioning each ticker.

**Step-by-step:**
```
For each ticker:
    1. Look up company long name via yf.Ticker(ticker).info
    
    2. Build search query:
       '"AAPL" stock OR "Apple Inc." earnings OR revenue'
    
    3. GET https://newsapi.org/v2/everything
       Params: q=query, pageSize=100, language=en, sortBy=publishedAt
    
    4. Extract fields per article:
       ticker, company, published_at, title, description, source, author, url
    
    5. Sleep 0.5s between tickers

Combine all articles → drop duplicate URLs → save as:
    data/bronze/newsapi_headlines_raw.csv
```

**API limitations handled:**
- `401` → Invalid key, log error and return empty DataFrame
- `426` → Free tier restriction (only today's news), log warning and skip gracefully

---

### Helper Functions in ingest.py

| Function | What it does |
|----------|-------------|
| `_make_session()` | Creates retry-enabled requests Session |
| `_ensure_dirs()` | Creates `data/bronze/` if it doesn't exist |
| `_save_bronze(df, filename)` | Saves DataFrame as CSV to Bronze path, logs row count |
| `_upload_to_gcs(local_path, gcs_path)` | Uploads file to GCS bucket (skips gracefully if not installed) |
| `_load_api_key(env_var)` | Reads API key from env variable, falls back to `.env` file |

---

### Main Entry Point — `run_ingestion()`

```python
return {
    "prices": fetch_yahoo_finance(tickers),   # dict of {ticker: DataFrame}
    "edgar":  fetch_sec_edgar(tickers),        # dict of {ticker: DataFrame}
    "macro":  fetch_fred(),                    # single DataFrame
    "news":   fetch_news(tickers),             # single DataFrame
}
```

This dict is passed directly into `clean.run_cleaning()` in the next step.

---

## 6. clean.py — Silver Layer (Data Preprocessing)

**File:** `clean.py`  
**Purpose:** Transform raw Bronze data into analysis-ready Silver data. Applies 5 sequential cleaning steps to each data source.

### Step 1 — Missing Value Handling (`handle_missing`)

```
Input: DataFrame with numeric columns

1. Forward-fill (ffill) up to FFILL_LIMIT=2 consecutive NaNs
   → Fills weekend/holiday gaps in daily price data
   → Does not fill large gaps (e.g., a stock that stopped trading)

2. Calculate NaN percentage per row:
   nan_pct = (number of NaN numeric columns) / (total numeric columns)

3. Drop rows where nan_pct > MAX_NAN_PCT_ROW (0.20)
   → Removes rows that are mostly missing (unreliable data)

Logs: rows before → rows after, NaNs remaining
```

**Why forward-fill instead of interpolation?**  
Financial prices on non-trading days genuinely take the last known value. Forward-filling is the correct financial assumption — a stock's "price" on Saturday is Friday's closing price.

---

### Step 2 — Outlier Detection and Capping

#### Price Outliers — `remove_price_outliers()`
```
Applied to: daily_return column (computed in Step 5)

1. Compute Z-score for all non-NaN daily returns
   z = |x - mean| / std

2. Count returns where z > ZSCORE_THRESHOLD (3.0)

3. Cap (winsorize) — do NOT drop:
   lower_bound = mean - 3σ
   upper_bound = mean + 3σ
   df["daily_return"] = df["daily_return"].clip(lower, upper)
```

**Why cap instead of drop?**  
Dropping extreme return days would create gaps in the time series. A −20% crash day is real information. Capping preserves the direction (negative) and the extreme nature while preventing it from dominating statistical calculations.

#### Fundamental Outliers — `cap_fundamental_outliers()`
```
Applied to: pe_ratio, eps, beta, debt_to_equity

For each column:
    Q1 = 25th percentile
    Q3 = 75th percentile
    IQR = Q3 - Q1
    lower = Q1 - 1.5 × IQR
    upper = Q3 + 1.5 × IQR
    df[col] = df[col].clip(lower, upper)
```

**Why IQR for fundamentals instead of Z-score?**  
Fundamental ratios are often highly skewed (P/E of 1000× for loss-making companies). Z-score assumes normality. IQR is distribution-free and handles skewed data better.

---

### Step 3 — Normalization (`normalize_minmax`)

```
For price and volume columns:
    normalized = (value - min) / (max - min)
    → Result is always in [0, 1]
    → New column named: {original_col}_norm

For risk score mapping (map_to_risk_scale):
    score = 1.0 + 9.0 × (value - min) / (max - min)
    → Result is always in [1, 10]
```

Min-Max normalization is used rather than standardization (Z-score) because:
- Price data has a clear natural minimum of 0
- [0, 1] scale is intuitive for plotting and comparison
- The risk 1–10 scale is explicitly mapped this way for interpretability

---

### Step 4 — Format Standardization (`standardize_formats`)

```
1. Column names → snake_case
   "Close Price" → "close_price"
   "Market-Cap"  → "market_cap"
   "P/E"         → "p_e"

2. Index → datetime (UTC, timezone stripped)
   pd.to_datetime(df.index).tz_localize(None)

3. Monetary columns scaled to USD millions (÷ 1,000,000):
   market_cap, revenue_ttm, total_assets, total_liabilities,
   operating_income, net_income, stockholders_equity, cash
   → Renamed to: market_cap_usd_m, revenue_ttm_usd_m, etc.
```

**Why USD millions?**  
Apple's market cap is ~$3 trillion. Storing that as 3,000,000,000,000 causes floating point precision issues and makes log displays unreadable. USD millions gives clean 4–7 digit numbers for most large-cap companies.

---

### Step 5 — Feature Engineering

#### Price Features — `engineer_price_features()`

```python
daily_return    = close.pct_change()
                # % change from previous close
                # Used for: volatility, Sharpe, risk scoring

log_return      = log(close / close.shift(1))
                # Continuously compounded return
                # More statistically stable than simple returns

rolling_vol_30d = daily_return.rolling(30).std() * sqrt(252)
                # Annualized 30-day volatility
                # sqrt(252) converts daily std to annual

rolling_vol_60d = daily_return.rolling(60).std() * sqrt(252)
                # Longer window — smoother, less reactive

sma_20          = close.rolling(20).mean()
                # 20-day simple moving average (short-term trend)

sma_50          = close.rolling(50).mean()
                # 50-day simple moving average (medium-term trend)

price_vs_52w_high_pct = (close - 52w_high) / 52w_high * 100
                # Drawdown from 52-week high (always ≤ 0)
                # -15 means "currently 15% below yearly peak"
```

#### News Sentiment — `score_news_sentiment()`

```
Uses VADER (Valence Aware Dictionary and sEntiment Reasoner)
Designed specifically for social media and financial text

For each article:
    text = title + " " + description
    scores = VaderSentimentAnalyzer.polarity_scores(text)
    compound = scores["compound"]   # range: -1.0 to +1.0

vader_label:
    compound >= +0.05  → "positive"
    compound <= -0.05  → "negative"
    else               → "neutral"

Then aggregate per (ticker, date):
    sentiment_mean  = average compound score
    sentiment_std   = standard deviation (consistency)
    article_count   = number of articles that day
```

---

### Cleaning Pipeline Per Source

| Source | Function | Key Steps |
|--------|----------|-----------|
| Yahoo prices | `clean_prices()` | standardize → handle_missing → engineer_price_features → remove_price_outliers → cap_fundamentals → normalize close/volume |
| FRED macro | `clean_macro()` | standardize → handle_missing → resample("D").ffill() to convert monthly → daily |
| EDGAR financials | `clean_edgar()` | standardize → handle_missing → sort by end_date → deduplicate (keep most recent 10-K per year) |
| News | `clean_news()` | drop blank titles → deduplicate by URL → parse dates → VADER scoring → aggregate daily per ticker |

---

### Main Entry Point — `run_cleaning()`

```python
return {
    "prices": clean_prices(raw_data["prices"]),
    "edgar":  clean_edgar(raw_data["edgar"]),
    "macro":  clean_macro(raw_data["macro"]),
    "news":   clean_news(raw_data["news"]),
}
```

Returns the same dict structure as ingestion, but with Silver-layer DataFrames.

---

## 7. eda.py — Gold Layer (Analytics + Risk Scoring)

**File:** `eda.py`  
**Purpose:** Compute descriptive statistics, multi-dimensional risk scores, generate 7 visualization charts, produce auto-generated insights, and export Gold-layer CSVs ready for BigQuery or model training.

---

### Part 1 — Descriptive Statistics (`compute_statistics`)

For each ticker, computes:

| Metric | Formula | Meaning |
|--------|---------|---------|
| `current_price` | `close.iloc[-1]` | Latest closing price |
| `ytd_return_pct` | `(current - start) / start × 100` | % gain/loss over the period |
| `ann_volatility_pct` | `returns.std() × √252 × 100` | Annualized daily return std dev |
| `sharpe_ratio` | `(mean_daily × 252) / (std × √252)` | Return per unit of risk (risk-free rate assumed 0) |
| `max_drawdown_pct` | `min((cumulative - rolling_max) / rolling_max) × 100` | Worst peak-to-trough decline |
| `mean_daily_return` | `returns.mean() × 100` | Average daily return as % |
| `skewness` | `returns.skew()` | Negative = left tail risk (crash prone) |
| `kurtosis` | `returns.kurtosis()` | High = fat tails (extreme events more likely) |
| `pe_ratio` | Last non-NaN from Yahoo data | Price-to-Earnings multiple |
| `beta` | Last non-NaN from Yahoo data | Market sensitivity (1.0 = moves with S&P 500) |
| `market_cap_usd_m` | Last non-NaN from Yahoo data | Market capitalization in USD millions |

**Max Drawdown calculation detail:**
```python
cumulative  = (1 + returns).cumprod()   # compounded growth from day 1
rolling_max = cumulative.cummax()        # highest point reached so far
drawdown    = (cumulative - rolling_max) / rolling_max
max_dd      = drawdown.min()             # worst single trough
```

---

### Part 2 — Risk Scoring (`compute_risk_scores`)

Computes 5 scores per ticker (all on 1–10 scale, higher = more risk).

#### Internal Scaling Function
```python
def _scale(series, invert=False):
    scaled = 1 + 9 × (series - min) / (max - min)
    return (10 - scaled + 1) if invert else scaled
```
`invert=True` is used for sentiment (positive sentiment = lower risk, so high compound score → low risk score).

---

#### Volatility Risk
```
volatility_risk = _scale(ann_volatility_pct)

High volatility → high score → riskier
Low volatility  → low score  → safer

Example:
    TSLA ann_vol = 65%  → score ~9.2  (extreme)
    MSFT ann_vol = 22%  → score ~3.1  (normal)
```

---

#### Fundamental Risk
```
pe_score   = _scale(pe_ratio, fill missing with median)
beta_score = _scale(beta, fill missing with 1.0)
dd_score   = _scale(abs(max_drawdown))

fundamental_risk = (pe_score × 0.4) + (beta_score × 0.3) + (dd_score × 0.3)
                   clipped to [1, 10]

Weights reasoning:
  P/E (40%)    — valuation risk, most directly tied to overprice
  Beta (30%)   — market sensitivity
  Drawdown (30%) — realized historical loss behavior
```

---

#### Sentiment Risk
```
If news data available:
    sentiment_by_ticker = news_df.groupby("ticker")["sentiment_mean"].mean()
    sentiment_risk = _scale(sentiment, invert=True)
    
    Positive sentiment → low risk (market is favorable)
    Negative sentiment → high risk (bad press = sell pressure)

If no news data:
    sentiment_risk = 5.0 (neutral/unknown)
```

---

#### Macro Risk
```
Step 1: Load data/silver/silver_macro.csv
Step 2: Get latest row (most recent macro readings)

Fed Funds Rate score:
    fed_score = clip((fed_funds_rate / 6.0) × 10, 1, 10)
    Logic: 6% fed rate maps to max risk (10/10)
           0% fed rate maps to min risk (1/10)

CPI score:
    cpi_z = |latest_cpi - mean_cpi| / std_cpi
    cpi_score = clip(1 + cpi_z × 3, 1, 10)
    Logic: CPI far from historical norm = higher risk

macro_risk = (fed_score + cpi_score) / 2
```

---

#### Composite Risk (Final Score)
```
composite_risk = (fundamental_risk × 0.30)
               + (volatility_risk  × 0.30)
               + (sentiment_risk   × 0.20)
               + (macro_risk       × 0.20)

Risk Label:
    composite ≤ 3.5  → "LOW"
    composite ≤ 6.0  → "MODERATE"
    composite > 6.0  → "HIGH"
```

---

### Part 3 — Visualizations (7 Charts)

All charts use headless matplotlib (`matplotlib.use("Agg")`) so they render without a display.

| Chart | File | What it shows |
|-------|------|--------------|
| `plot_price_trends` | `01_price_trends.png` | All tickers rebased to 100 — relative performance |
| `plot_volatility` | `02_volatility.png` | 30-day rolling annualized vol per ticker over time |
| `plot_return_distribution` | `03_return_distributions.png` | Histogram + KDE of daily returns per ticker |
| `plot_correlation_matrix` | `04_correlation_matrix.png` | Pearson correlation heatmap of returns |
| `plot_risk_scores` | `05_risk_scores.png` | Grouped bar (4 dimensions) + horizontal composite bar |
| `plot_macro_overlay` | `06_macro_overlay_{TICKER}.png` | Price vs. interest rates vs. CPI |
| `plot_sentiment_trend` | `07_sentiment_trend.png` | Daily VADER score per ticker over time |

---

### Part 4 — Auto-Generated Insights (`generate_insights`)

Automatically flags threshold-breaching conditions. Each insight has a category, ticker, text, and severity (HIGH / MEDIUM / LOW).

| Condition | Threshold | Severity |
|-----------|-----------|---------|
| Annualized volatility | > 50% | HIGH |
| Annualized volatility | > 30% | MEDIUM |
| YTD return | < -20% | HIGH |
| YTD return | > +50% | MEDIUM |
| Max drawdown | < -30% | HIGH |
| P/E ratio | > 60× | HIGH |
| P/E ratio | < 10× | MEDIUM |
| Sharpe ratio | < 0 | HIGH |
| Return skewness | < -1 | MEDIUM |
| Composite risk score | > 6.0 | HIGH |

---

### Part 5 — Gold Export (`export_gold`)

```
data/gold/gold_risk_scores.csv
    → stats_df joined with risk_df
    → Columns: all statistical metrics + 4 risk dimensions + composite + risk_label + as_of_date
    → One row per ticker
    → BigQuery-importable directly

data/gold/gold_insights.csv
    → DataFrame of all auto-generated insight dicts
    → Columns: category, ticker, insight, severity, as_of_date
```

---

## 8. pipeline_large_scale.py — Orchestrator

**File:** `pipeline_large_scale.py`  
**Purpose:** Run the complete Bronze → Silver → Gold pipeline for all S&P 500 + NASDAQ-100 companies (~600 tickers), sector by sector, with progress tracking and resume capability.

### What makes it "large-scale"

The single-ticker pipeline (`ingest.py` → `clean.py` → `eda.py` run manually) can only handle a handful of tickers at a time. The orchestrator adds:
- Automatic ticker discovery from Wikipedia
- Sector-by-sector organization (11 GICS sectors)
- Batching within sectors (50 tickers per batch)
- Resume capability (skips completed tickers)
- JSON progress file written after every batch
- Merged Gold table across all sectors

### Progress Tracking

```python
# pipeline_progress.json structure
{
    "completed_tickers": ["AAPL", "MSFT", ...],
    "failed_tickers":    ["SOME_DELISTED_TICKER"],
    "completed_sectors": ["Information Technology", "Health Care"]
}
```

After every batch: progress is saved. If the run crashes at batch 23 of 40, rerunning with `--resume` skips all 1,100 already-completed tickers.

### Batch Runner — `run_sector_batch()`

For each batch of ≤50 tickers:

```
1. Filter out already-done tickers (if --resume)

2. ingest.run_ingestion(tickers)
   → raw_data dict

3. clean.run_cleaning(raw_data)
   → clean_data dict

4. eda.compute_statistics(price_data)
   + eda.compute_risk_scores(stats_df, news_df)
   → risk_df
   NOTE: heavy plots (charts 01–07) are SKIPPED at scale
         Only statistics and risk scores computed

5. Tag risk_df with sector, as_of_date, batch label

6. Save: data/gold/gold_risk_{Sector}_batch{N}.csv

7. Optionally upload to GCS:
   gs://financial-risk-pipeline/gold/sectors/{Sector}/batch{N}.csv
```

### Merge and BigQuery Load — `merge_and_load_bigquery()`

After all batches complete:
```
1. glob all data/gold/gold_risk_*.csv files

2. pd.concat all sector batch files

3. Deduplicate by ticker index (keep last occurrence)
   → Handles cases where same ticker appears in multiple batches

4. Save master: data/gold/gold_risk_scores_ALL.csv

5. Load to BigQuery:
   Table: sacred-catfish-488122-u7.financial_risk.risk_scores
   Mode: WRITE_TRUNCATE (full refresh — overwrites previous run)
   Schema: autodetect
```

### CLI Arguments

```bash
python pipeline_large_scale.py                          # full run, all 600 companies
python pipeline_large_scale.py --test                   # 10 companies per sector
python pipeline_large_scale.py --sectors "Health Care"  # one sector only
python pipeline_large_scale.py --gcs                    # upload to GCS + BigQuery
python pipeline_large_scale.py --resume --gcs           # resume + upload
python pipeline_large_scale.py --batch-size 25          # smaller batches
python pipeline_large_scale.py --list-sectors           # print sector list and exit
```

### Timing Delays

| Delay | Where | Reason |
|-------|-------|--------|
| 0.5s per ticker | Yahoo Finance | Rate limit |
| 0.6s per ticker | SEC EDGAR | SEC policy (10 req/s) |
| 0.3s per series | FRED API | Politeness |
| 0.5s per ticker | NewsAPI | Rate limit |
| 3.0s per batch | Between batches | Yahoo/SEC cooldown |

---

## 9. split.py — Train/Val/Test Split

**File:** `split.py`  
**Purpose:** Divide the Gold dataset into train (70%), validation (15%), and test (15%) sets while preserving the distribution of risk labels.

### Why Stratified Split

The three risk classes (HIGH / MODERATE / LOW) are not equally distributed. A random split could put all HIGH-risk companies in test and leave training with only LOW. Stratified split preserves the ratio in each subset.

```python
from sklearn.model_selection import train_test_split

# First split: 70% train, 30% temp
train, temp = train_test_split(df, test_size=0.30, random_state=42,
                               stratify=df["risk_label"])

# Second split: 50% of temp = val, 50% = test (both 15% of total)
val, test = train_test_split(temp, test_size=0.50, random_state=42,
                             stratify=temp["risk_label"])
```

### Current Split Sizes (123 companies)
```
Train : 86 companies  (70%)
Val   : 18 companies  (15%)
Test  : 19 companies  (15%)
```

### Output Files
```
data/gold/split_train.csv    — 86 companies, all risk score columns
data/gold/split_val.csv      — 18 companies
data/gold/split_test.csv     — 19 companies
data/reports/08_class_balance.png    — bar chart: HIGH/MODERATE/LOW counts per split
data/reports/09_sector_distribution.png  — horizontal bar: companies per GICS sector
```

### Note on Temporal Split
```
NOTE: Single-snapshot data (collected 2026-03-04).
Temporal split will be applied once multi-date collection is complete.
Stratified split used here to preserve HIGH/MODERATE/LOW class balance.
```
The dataset currently represents one point-in-time snapshot per company. A temporal split (train on older data, test on newer) will replace this in Phase 2 when rolling collection is in place.

---

## 10. stats_summary.py — Reporting

**File:** `stats_summary.py`  
**Purpose:** Generate a data statistics summary table showing row counts and descriptions for each pipeline layer.

Scans the actual files on disk, counts rows in each CSV, and produces:

```
data/reports/data_statistics_summary.csv   — machine-readable stats table
data/reports/10_data_statistics_summary.png — color-coded table image for reports
```

Color coding:
- Bronze rows → yellow background
- Silver rows → blue background
- Gold rows   → green background
- Split rows  → purple background

---

## 11. GCP Integration — Storage and BigQuery

### How Files Get to GCS

Every save function has an optional `upload_gcs=True` parameter. When enabled:

```python
def _upload_to_gcs(local_path, gcs_path):
    client = storage.Client(project=config.GCP_PROJECT_ID)
    bucket = client.bucket(config.GCS_BUCKET)
    bucket.blob(gcs_path).upload_from_filename(local_path)
```

This uses Application Default Credentials (ADC). The machine must be authenticated via:
```bash
gcloud auth application-default login
```

### GCS Bucket Structure

```
gs://financial-risk-pipeline/
├── bronze/
│   ├── yahoo/
│   │   ├── yahoo_AAPL_2024-05-06_to_2025-05-06.csv
│   │   ├── yahoo_MSFT_2024-05-06_to_2025-05-06.csv
│   │   └── ...
│   ├── edgar/
│   │   ├── edgar_AAPL_raw_long.csv
│   │   ├── edgar_AAPL_annual_financials.csv
│   │   └── ...
│   ├── fred/
│   │   └── fred_macro_2024-05-06_to_2025-05-06.csv
│   └── news/
│       └── newsapi_headlines_raw.csv
│
├── silver/
│   ├── prices/
│   │   ├── silver_prices_AAPL.csv
│   │   └── ...
│   ├── edgar/
│   │   └── silver_edgar_AAPL.csv
│   ├── macro/
│   │   └── silver_macro.csv
│   └── news/
│       └── silver_news_sentiment.csv
│
└── gold/
    ├── gold_risk_scores.csv
    ├── gold_insights.csv
    └── sectors/
        ├── Information_Technology/
        │   ├── batch1.csv
        │   └── batch2.csv
        └── Health_Care/
            └── batch1.csv
```

### BigQuery Table

```
Project:  sacred-catfish-488122-u7
Dataset:  financial_risk
Table:    risk_scores

Schema (autodetected):
    ticker              STRING
    current_price       FLOAT64
    ytd_return_pct      FLOAT64
    ann_volatility_pct  FLOAT64
    sharpe_ratio        FLOAT64
    max_drawdown_pct    FLOAT64
    pe_ratio            FLOAT64
    beta                FLOAT64
    market_cap_usd_m    FLOAT64
    fundamental_risk    FLOAT64
    volatility_risk     FLOAT64
    sentiment_risk      FLOAT64
    macro_risk          FLOAT64
    composite_risk      FLOAT64
    risk_label          STRING
    sector              STRING
    as_of_date          DATE
    batch               STRING

Write mode: WRITE_TRUNCATE (full refresh on each run)
```

---

## 12. Data Flow: End to End

```
[Wikipedia]
    │
    └──► tickers.py: get_all_tickers()
         → 600 unique companies grouped into 11 sectors
         → 40+ batches of ≤50 tickers each

                         ↓ per batch

[Yahoo Finance] ─────────────────────────────────────────────────────┐
                                                                      │
[SEC EDGAR] ──────────────► ingest.run_ingestion(tickers)            │
                              ↓                                       │
[FRED API] ────────────────► raw_data = {                            │  BRONZE
                                 "prices": {AAPL: df, MSFT: df, ...} │  data/bronze/*.csv
[NewsAPI] ─────────────────►     "edgar":  {AAPL: df, ...}          │  + GCS /bronze
                                 "macro":  fred_df,                  │
                                 "news":   news_df                   │
                             }                                        │
                                                                      │
                         ↓                                           ┘

clean.run_cleaning(raw_data)
    │
    ├── clean_prices()  → ffill → Z-score cap → Min-Max norm → feature engineering
    ├── clean_edgar()   → standardize → deduplicate → sort by year
    ├── clean_macro()   → ffill → resample daily
    └── clean_news()    → drop blanks → dedup URLs → VADER sentiment → daily aggregate
    │
    └──► clean_data = { "prices": {}, "edgar": {}, "macro": df, "news": df }
                                                                          │  SILVER
                                                                          │  data/silver/*.csv
                         ↓                                               │  + GCS /silver
                                                                          ┘

eda.compute_statistics(clean_data["prices"])
    → per-ticker: price, ytd_return, ann_vol, sharpe, max_drawdown, beta, pe

eda.compute_risk_scores(stats_df, news_df)
    → per-ticker: volatility_risk, fundamental_risk, sentiment_risk, macro_risk, composite_risk

eda.export_gold()
    → data/gold/gold_risk_{Sector}_batch{N}.csv                           │  GOLD
                                                                           │  data/gold/*.csv
                         ↓                                                │  + GCS /gold

merge_and_load_bigquery()
    → concat all sector gold files → deduplicate
    → data/gold/gold_risk_scores_ALL.csv                                  │  BIGQUERY
    → BigQuery: financial_risk.risk_scores (WRITE_TRUNCATE)               │  risk_scores table

                         ↓

split.run_split()
    → stratified split by risk_label (70/15/15)
    → data/gold/split_train.csv  (86 rows)
    → data/gold/split_val.csv    (18 rows)
    → data/gold/split_test.csv   (19 rows)

                         ↓

Phase 2: Multi-Agent AI Model Training
    (Feed Gold split files into specialized risk agents)
```

---

## 13. Risk Scoring Model

### Dimension Weights

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| Fundamental Risk | 30% | P/E, beta, drawdown are core quantitative risk indicators |
| Volatility Risk | 30% | Price movement uncertainty directly affects portfolio risk |
| Sentiment Risk | 20% | News sentiment leads price action; earlier signal than fundamentals |
| Macro Risk | 20% | Systematic risk affects all companies but is not company-specific |

### Sub-Component Weights (Fundamental Risk)

| Sub-Component | Weight within Fundamental |
|---------------|--------------------------|
| P/E Ratio | 40% |
| Beta | 30% |
| Max Drawdown | 30% |

### Risk Label Thresholds

```
Score 1.0 – 3.5  →  LOW       (green)   — safe for conservative portfolios
Score 3.5 – 6.0  →  MODERATE  (yellow)  — monitor, acceptable for balanced portfolios
Score 6.0 – 10.0 →  HIGH      (red)     — elevated risk, requires due diligence
```

### Scoring Formula Summary

```
composite = 0.30 × fundamental
          + 0.30 × volatility
          + 0.20 × sentiment
          + 0.20 × macro

where:

fundamental = 0.40 × scale(pe_ratio)
            + 0.30 × scale(beta)
            + 0.30 × scale(|max_drawdown|)

volatility  = scale(ann_volatility_pct)

sentiment   = scale(avg_vader_compound, invert=True)

macro       = (fed_funds_score + cpi_zscore_score) / 2
```

---

## 14. File Output Reference

### Bronze Layer — `data/bronze/`

| Filename Pattern | Source | Content |
|-----------------|--------|---------|
| `yahoo_{TICKER}_{start}_to_{end}.csv` | Yahoo Finance | ~252 rows of daily OHLCV + 11 fundamentals |
| `edgar_{TICKER}_raw_long.csv` | SEC EDGAR | Long format: one row per (concept, year) |
| `edgar_{TICKER}_annual_financials.csv` | SEC EDGAR | Wide format: one row per fiscal year |
| `fred_macro_{start}_to_{end}.csv` | FRED | Monthly macro series merged on date |
| `newsapi_headlines_raw.csv` | NewsAPI | All articles across all tickers |

### Silver Layer — `data/silver/`

| Filename Pattern | Source | Content |
|-----------------|--------|---------|
| `silver_prices_{TICKER}.csv` | clean_prices() | Cleaned prices + engineered features |
| `silver_edgar_{TICKER}.csv` | clean_edgar() | Cleaned annual financials in USD millions |
| `silver_macro.csv` | clean_macro() | Daily macro indicators (forward-filled) |
| `silver_news_sentiment.csv` | clean_news() | Daily VADER scores per ticker |

### Gold Layer — `data/gold/`

| Filename | Content |
|----------|---------|
| `gold_risk_scores.csv` | All stats + risk scores per ticker (small-scale run) |
| `gold_insights.csv` | Auto-generated risk insights per ticker |
| `gold_risk_{Sector}_batch{N}.csv` | Per-sector batch output (large-scale) |
| `gold_risk_scores_ALL.csv` | Merged master table (all 600 companies) |
| `ticker_universe.csv` | Full ticker list with sector labels |
| `run_report.json` | Pipeline run summary: success/fail counts, timing |
| `split_train.csv` | 70% training set (stratified by risk_label) |
| `split_val.csv` | 15% validation set |
| `split_test.csv` | 15% test set |

### Reports — `data/reports/`

| Filename | Chart Type |
|----------|-----------|
| `01_price_trends.png` | Line chart — normalized price performance |
| `02_volatility.png` | Line chart — 30-day rolling volatility |
| `03_return_distributions.png` | Histogram + KDE — daily return distributions |
| `04_correlation_matrix.png` | Heatmap — Pearson correlation of returns |
| `05_risk_scores.png` | Grouped bar + horizontal bar — risk dimensions |
| `06_macro_overlay_{TICKER}.png` | Multi-panel — price vs. rates vs. CPI |
| `07_sentiment_trend.png` | Line chart — daily VADER sentiment |
| `08_class_balance.png` | Bar chart — class balance per split |
| `09_sector_distribution.png` | Horizontal bar — companies per sector |
| `10_data_statistics_summary.png` | Table image — row counts per layer |

---

## 15. Configuration Reference

All settings are in `config.py`. Override any value by setting the corresponding environment variable.

| Setting | Default | Override Env Var | Purpose |
|---------|---------|-----------------|---------|
| `DEFAULT_TICKERS` | `["AAPL","MSFT","TSLA","NVDA","JPM"]` | — | Tickers for manual single run |
| `START_DATE` | today − 365 days | — | Price history start date |
| `END_DATE` | today | — | Price history end date |
| `FRED_API_KEY` | placeholder | `FRED_API_KEY` | FRED API authentication |
| `NEWS_API_KEY` | placeholder | `NEWS_API_KEY` | NewsAPI authentication |
| `GCP_PROJECT_ID` | `sacred-catfish-488122-u7` | `GCP_PROJECT_ID` | GCP project for GCS and BigQuery |
| `GCS_BUCKET` | `financial-risk-pipeline` | `GCS_BUCKET` | GCS bucket name |
| `ZSCORE_THRESHOLD` | `3.0` | — | Daily return outlier cap threshold |
| `MAX_NAN_PCT_ROW` | `0.20` | — | Max % missing per row before drop |
| `FFILL_LIMIT` | `2` | — | Max consecutive NaN forward-fill |
| `RISK_SCORE_LOW` | `3.5` | — | Composite score threshold for LOW label |
| `RISK_SCORE_HIGH` | `6.0` | — | Composite score threshold for HIGH label |
| `FIGURE_DPI` | `150` | — | Chart resolution |

---

*This document covers the complete data pipeline as of DATA 298A Phase 1. The multi-agent AI model training (Phase 2) builds on the Gold layer outputs produced here.*
