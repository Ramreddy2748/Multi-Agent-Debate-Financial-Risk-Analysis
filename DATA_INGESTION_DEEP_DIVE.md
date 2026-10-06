# Data Ingestion Layer — Complete Step-by-Step Breakdown

**File:** `ingest.py`  
**Layer:** Bronze (Raw Data)  
**Output:** `data/bronze/*.csv`

---

## Table of Contents

1. [How the HTTP Session Works](#1-how-the-http-session-works)
2. [Helper Utilities](#2-helper-utilities)
3. [Source 1 — Yahoo Finance](#3-source-1--yahoo-finance)
4. [Source 2 — SEC EDGAR](#4-source-2--sec-edgar)
5. [Source 3 — FRED API](#5-source-3--fred-api)
6. [Source 4 — NewsAPI](#6-source-4--newsapi)
7. [Main Entry Point — run_ingestion()](#7-main-entry-point--run_ingestion)
8. [Error Handling Summary Table](#8-error-handling-summary-table)
9. [Rate Limit Summary Table](#9-rate-limit-summary-table)
10. [What the Bronze Layer Looks Like](#10-what-the-bronze-layer-looks-like)

---

## 1. How the HTTP Session Works

Before any data is fetched, a shared `requests.Session` is created once and reused by all 4 sources.

```python
def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,               # retry up to 3 times
        backoff_factor=1.5,    # wait 1.5s, 3s, 4.5s between retries
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session

SESSION = _make_session()   # module-level singleton
```

### Why a shared session?

- **Connection pooling** — TCP connections are reused across requests instead of opening a new connection every time. Faster and less overhead.
- **Consistent retry policy** — Every HTTP call from any source automatically retries on failure without repeating that logic.
- **Automatic backoff** — If a server returns 429 (rate limited) or 5xx (server error), the session waits before retrying:
  - 1st retry: wait 1.5 seconds
  - 2nd retry: wait 3.0 seconds  
  - 3rd retry: wait 4.5 seconds
- After 3 failures the exception is raised normally.

### Status codes that trigger a retry

| Code | Meaning | Why we retry |
|------|---------|-------------|
| 429 | Too Many Requests | We hit the rate limit — wait and retry |
| 500 | Internal Server Error | Server-side error, often transient |
| 502 | Bad Gateway | Proxy/load balancer issue, usually temporary |
| 503 | Service Unavailable | Server overloaded, retry after backoff |
| 504 | Gateway Timeout | Request took too long, retry |

---

## 2. Helper Utilities

### `_ensure_dirs()`
```python
def _ensure_dirs():
    Path(config.LOCAL_BRONZE).mkdir(parents=True, exist_ok=True)
```
Creates `data/bronze/` directory if it doesn't exist. `parents=True` creates intermediate directories too. `exist_ok=True` does not raise an error if it already exists. Called at the start of every save operation.

---

### `_save_bronze(df, filename)`
```python
def _save_bronze(df: pd.DataFrame, filename: str) -> str:
    _ensure_dirs()
    path = os.path.join(config.LOCAL_BRONZE, filename)
    df.to_csv(path, index=True)
    log.info(f"  ✔ Bronze saved: {path}  ({len(df):,} rows x {df.shape[1]} cols)")
    return path
```
- Saves `index=True` — the DatetimeIndex (for price data) becomes the first column in the CSV. This preserves the date as the row identifier.
- Logs row count and column count so you can immediately verify the fetch size.
- Returns the path so the caller can pass it to `_upload_to_gcs()`.

---

### `_upload_to_gcs(local_path, gcs_path)`
```python
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
```
- `ImportError` is caught separately so the pipeline runs fine locally even if `google-cloud-storage` is not installed.
- Any other GCS failure (auth error, bucket not found, network) logs an error but **does not crash the pipeline** — the local file is already saved.
- Uses Application Default Credentials (ADC): `gcloud auth application-default login` must be run once on the machine.

---

### `_load_api_key(env_var, env_file=".env")`
```python
def _load_api_key(env_var: str, env_file: str = ".env") -> Optional[str]:
    # 1. Check environment variable
    key = os.getenv(env_var)
    if key and "YOUR_" not in key:
        return key

    # 2. Fall back to .env file
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
```

**Two-step key loading:**

Step 1 — Check `os.environ`:
- If the env var is set AND does not contain the placeholder text `"YOUR_"`, return it.
- The `"YOUR_"` check prevents accidentally using the placeholder from `config.py` as a real key.

Step 2 — Parse `.env` file:
- Opens `.env` in the project root line by line.
- Finds the line starting with `FRED_API_KEY=` or `NEWS_API_KEY=`.
- Strips surrounding quotes (handles `KEY="value"`, `KEY='value'`, `KEY=value`).
- If neither source has a key, returns `None` → the source is skipped with a warning.

---

## 3. Source 1 — Yahoo Finance

### What we get

Daily OHLCV price history for each ticker plus snapshot fundamental metrics.

### Full Step-by-Step

```
fetch_yahoo_finance(tickers, start_date, end_date)
```

**Step 1: Loop each ticker**
```python
for ticker in tickers:
```
Each ticker is fetched independently. If one fails, others continue unaffected.

---

**Step 2: Fetch price history**
```python
stock = yf.Ticker(ticker)
hist  = stock.history(start=start_date, end=end_date)
```
- `yf.Ticker()` creates a Yahoo Finance object for that ticker symbol.
- `.history()` calls Yahoo Finance's internal API (no API key needed — uses a session cookie approach internally).
- Returns a DataFrame indexed by trading date with columns: `Open, High, Low, Close, Volume, Dividends, Stock Splits`.
- Only trading days are returned — weekends and market holidays are absent.

---

**Step 3: Check for empty response**
```python
if hist.empty:
    log.warning(f"    No data for {ticker}")
    continue
```
Some tickers may be delisted, incorrectly spelled, or have no data in the requested date range. Rather than crashing, we log and skip to the next ticker.

---

**Step 4: Normalize the DatetimeIndex**
```python
hist.index = pd.to_datetime(hist.index).tz_localize(None)
hist.index.name = "date"
```
Yahoo Finance returns timestamps with timezone info (`2024-01-02 00:00:00-05:00`). We strip the timezone with `tz_localize(None)` to get plain `datetime64[ns]`.

**Why?** Downstream joins with FRED and EDGAR data (which have no timezone) would fail with a `TypeError: Cannot join tz-naive with tz-aware DatetimeIndex` if we kept the timezone.

---

**Step 5: Add ticker column**
```python
hist["ticker"] = ticker
```
When multiple ticker CSVs are later concatenated (in the large-scale pipeline), this column identifies which company each row belongs to.

---

**Step 6: Attach fundamental metrics from stock.info**
```python
info = stock.info
for col, key in [
    ("pe_ratio",       "trailingPE"),
    ("eps",            "trailingEps"),
    ("market_cap",     "marketCap"),
    ("revenue_ttm",    "totalRevenue"),
    ("debt_to_equity", "debtToEquity"),
    ("beta",           "beta"),
    ("52w_high",       "fiftyTwoWeekHigh"),
    ("52w_low",        "fiftyTwoWeekLow"),
    ("sector",         "sector"),
    ("industry",       "industry"),
    ("long_name",      "longName"),
]:
    hist[col] = info.get(key)
```
- `stock.info` is a separate API call that returns a large dict of company metadata.
- `.get(key)` returns `None` if the key is missing (some small companies lack P/E or beta).
- These are **scalar values** — the same value is broadcast to every row in the DataFrame.
- This means: every row for AAPL has the same `pe_ratio` (today's trailing P/E). It's not historical — it's a snapshot. Historical P/E would require EDGAR data.

**The 11 fundamentals added:**

| Column | Yahoo Key | What it is |
|--------|-----------|-----------|
| `pe_ratio` | `trailingPE` | Trailing 12-month P/E ratio |
| `eps` | `trailingEps` | Earnings per share (trailing 12m) |
| `market_cap` | `marketCap` | Total market capitalization (USD) |
| `revenue_ttm` | `totalRevenue` | Revenue trailing 12 months (USD) |
| `debt_to_equity` | `debtToEquity` | Total debt / shareholder equity |
| `beta` | `beta` | Price sensitivity vs S&P 500 |
| `52w_high` | `fiftyTwoWeekHigh` | Highest price in last 52 weeks |
| `52w_low` | `fiftyTwoWeekLow` | Lowest price in last 52 weeks |
| `sector` | `sector` | GICS sector name |
| `industry` | `industry` | GICS sub-industry |
| `long_name` | `longName` | Full company name |

---

**Step 7: Save and optionally upload**
```python
filename = f"yahoo_{ticker}_{start_date}_to_{end_date}.csv"
path = _save_bronze(hist, filename)
if upload_gcs:
    _upload_to_gcs(path, f"bronze/yahoo/{filename}")
```

---

**Step 8: Rate limit sleep**
```python
time.sleep(0.5)
```
Yahoo Finance does not publish official rate limits but blocks IPs that make too many rapid requests. 0.5 seconds between tickers (2 requests/second) is conservative enough to avoid blocks in practice.

---

**Step 9: Add to results dict**
```python
results[ticker] = hist
```
The returned dict `{ticker: DataFrame}` is passed directly into `clean.clean_prices()`.

---

### Error Handling

```python
except Exception as e:
    log.error(f"    FAILED {ticker}: {e}")
```
Any exception (network error, parse error, unexpected data format) is caught at the ticker level. The loop continues to the next ticker. The failed ticker is simply absent from `results`.

---

### Output Example

File: `data/bronze/yahoo_AAPL_2024-05-06_to_2025-05-06.csv`

```
date,Open,High,Low,Close,Volume,Dividends,Stock Splits,ticker,pe_ratio,eps,market_cap,...
2024-05-06,183.42,184.70,181.21,182.01,52341200,AAPL,0,0,AAPL,28.5,6.42,2800000000000,...
2024-05-07,182.30,183.10,180.90,183.50,48920100,AAPL,0,0,AAPL,28.5,6.42,2800000000000,...
...
```

~252 rows (one per trading day), ~20 columns.

---

## 4. Source 2 — SEC EDGAR

### What we get

Annual financial statement data from 10-K filings (public company annual reports) via the SEC's free XBRL API. No API key needed.

### The Two-Step EDGAR Approach

EDGAR works differently from the other sources. Instead of one call per data point, it uses:
1. A **CIK map** (company identifier lookup) fetched once
2. A **company facts JSON** (all XBRL data ever filed) fetched once per company

### Full Step-by-Step

```
fetch_sec_edgar(tickers)
```

---

**Step 1: Download the CIK map (done once)**

```python
def _get_cik_map() -> dict:
    resp = SESSION.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=EDGAR_HEADERS,
        timeout=20
    )
    resp.raise_for_status()
    data = resp.json()
    cik_map = {
        v["ticker"].upper(): str(v["cik_str"]).zfill(10)
        for v in data.values()
    }
    return cik_map
```

The SEC maintains a public JSON file listing all ~10,000 registered companies with their ticker symbols and CIK numbers.

**What is a CIK?** The Central Index Key is the SEC's unique identifier for every registered company. AAPL's CIK is `0000320193`. All EDGAR API calls use CIK, not ticker symbols.

**`zfill(10)`** — pads the CIK to 10 digits with leading zeros: `320193` → `0000320193`. The EDGAR URL requires exactly 10 digits.

**Result:** `{"AAPL": "0000320193", "MSFT": "0000789019", "TSLA": "0001318605", ...}`

If this call fails (network error), the entire EDGAR source is skipped with an error log — we cannot proceed without CIKs.

---

**Step 2: Look up CIK for each ticker**
```python
cik = cik_map.get(ticker.upper())
if not cik:
    log.warning(f"  -> {ticker}: CIK not found — skipping")
    continue
```
Some tickers are not in the SEC map (foreign companies, ETFs, special vehicles). These are skipped cleanly.

---

**Step 3: Fetch the company facts JSON**
```python
url  = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
resp = SESSION.get(url, headers=EDGAR_HEADERS, timeout=30)

if resp.status_code == 404:
    log.warning(f"    No XBRL data for {ticker}")
    continue
resp.raise_for_status()
```

The SEC's XBRL API returns a single large JSON (often 1–10 MB) containing **every financial fact ever reported** by this company in XBRL format, across all filing types (10-K, 10-Q, 8-K, 20-F, etc.) and all years.

The `404` check is important — some companies file with the SEC but have not submitted XBRL-tagged data (older filings or foreign filers using Form 20-F with different standards). We handle this gracefully.

**Required headers:**
```python
EDGAR_HEADERS = {
    "User-Agent": "FinancialRiskPipeline venkatasiddarth.gullipalli@sjsu.edu",
    "Accept":     "application/json",
}
```
The SEC **requires** a `User-Agent` header identifying who you are and a contact email. Requests without it return `403 Forbidden`. This is an SEC policy, not a technical restriction.

---

**Step 4: Navigate the JSON structure**
```python
facts   = resp.json()
us_gaap = facts.get("facts", {}).get("us-gaap", {})
```

The JSON structure looks like:
```json
{
  "cik": 320193,
  "entityName": "Apple Inc.",
  "facts": {
    "us-gaap": {
      "Revenues": {
        "label": "Revenues",
        "description": "...",
        "units": {
          "USD": [
            {"end": "2023-09-30", "val": 383285000000, "form": "10-K", "filed": "2023-11-03"},
            {"end": "2022-09-24", "val": 394328000000, "form": "10-K", "filed": "2022-10-28"},
            {"end": "2023-06-30", "val": 81797000000,  "form": "10-Q", "filed": "2023-08-04"},
            ...
          ]
        }
      },
      "NetIncomeLoss": { ... },
      ...
    },
    "dei": { ... }
  }
}
```

We navigate to `facts → us-gaap` which contains all US GAAP accounting concepts.

---

**Step 5: Extract the 12 XBRL concepts we care about**

```python
EDGAR_CONCEPTS = {
    "Revenues":                                             "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax":  "revenue_alt",
    "NetIncomeLoss":                                        "net_income",
    "Assets":                                               "total_assets",
    "Liabilities":                                          "total_liabilities",
    "StockholdersEquity":                                   "stockholders_equity",
    "EarningsPerShareBasic":                                "eps_basic",
    "EarningsPerShareDiluted":                              "eps_diluted",
    "OperatingIncomeLoss":                                  "operating_income",
    "CashAndCashEquivalentsAtCarryingValue":                "cash",
    "LongTermDebt":                                         "long_term_debt",
    "CommonStockSharesOutstanding":                         "shares_outstanding",
}
```

**Why two revenue concepts?**  
Some companies (especially tech and services) report under `RevenueFromContractWithCustomerExcludingAssessedTax` (the newer ASC 606 standard) instead of the older `Revenues` tag. Having both ensures we capture revenue for all companies regardless of which accounting standard they follow.

For each concept:
```python
for xbrl_tag, col_name in EDGAR_CONCEPTS.items():
    concept = us_gaap.get(xbrl_tag, {})
    units   = concept.get("units", {})
    points  = units.get("USD", units.get("shares", []))

    # Filter to annual 10-K filings only
    annual = [
        d for d in points
        if d.get("form") in ("10-K", "20-F")   # annual reports only
        and d.get("val") is not None             # exclude null values
    ]
```

**`units.get("USD", units.get("shares", []))`** — Financial dollar amounts are under `"USD"`, but `CommonStockSharesOutstanding` is reported as a count of shares (not dollars), so it's under `"shares"`. We check USD first, fall back to shares.

**`form in ("10-K", "20-F")`** — 10-K is the US annual report. 20-F is the equivalent for foreign private issuers. We exclude 10-Q (quarterly) and 8-K (current events) to get annual figures only.

---

**Step 6: Build the long-format DataFrame**
```python
for d in annual:
    rows.append({
        "ticker":   ticker,
        "company":  entity_name,
        "concept":  col_name,       # e.g., "revenue"
        "value":    d.get("val"),   # e.g., 383285000000
        "end_date": d.get("end"),   # e.g., "2023-09-30"
        "filed":    d.get("filed"), # e.g., "2023-11-03"
        "form":     d.get("form"),  # "10-K" or "20-F"
    })

df_long = pd.DataFrame(rows)
```

Example long-format output:
```
ticker | company    | concept    | value          | end_date   | filed      | form
AAPL   | Apple Inc. | revenue    | 383285000000   | 2023-09-30 | 2023-11-03 | 10-K
AAPL   | Apple Inc. | net_income | 96995000000    | 2023-09-30 | 2023-11-03 | 10-K
AAPL   | Apple Inc. | revenue    | 394328000000   | 2022-09-24 | 2022-10-28 | 10-K
AAPL   | Apple Inc. | net_income | 99803000000    | 2022-09-24 | 2022-10-28 | 10-K
```

Saved as: `data/bronze/edgar_AAPL_raw_long.csv`

---

**Step 7: Pivot to wide format**
```python
df_wide = (
    df_long
    .pivot_table(
        index=["ticker", "company", "end_date", "form", "filed"],
        columns="concept",
        values="value",
        aggfunc="last"       # if same concept reported twice, take latest
    )
    .reset_index()
)
df_wide.columns.name = None
df_wide = df_wide.sort_values("end_date", ascending=False)
```

Example wide-format output:
```
ticker | company    | end_date   | cash   | net_income | revenue       | total_assets | ...
AAPL   | Apple Inc. | 2023-09-30 | 29965M | 96995M     | 383285M       | 352583M      | ...
AAPL   | Apple Inc. | 2022-09-24 | 23646M | 99803M     | 394328M       | 352755M      | ...
AAPL   | Apple Inc. | 2021-09-25 | 34940M | 94680M     | 365817M       | 351002M      | ...
```

`aggfunc="last"` handles the case where the same end_date appears in multiple amended filings — we keep only the most recently filed version.

Saved as: `data/bronze/edgar_AAPL_annual_financials.csv`

---

**Step 8: Rate limit sleep**
```python
time.sleep(0.6)
```
The SEC enforces a rate limit of 10 requests per second per IP. With one company facts JSON per ticker, 0.6s sleep (1.67 req/s) keeps us well within limits. Violating this triggers a 429, which the retry session handles, but consistent violations can result in temporary IP bans.

---

### Error Handling

| Error | Handling |
|-------|---------|
| CIK map download fails | Log error, return empty dict, skip entire EDGAR source |
| Ticker not in CIK map | Log warning, `continue` to next ticker |
| 404 — no XBRL data | Log warning, `continue` |
| Timeout (>30s) | Caught as `requests.exceptions.Timeout`, log specific message about SEC being slow |
| Any other exception | `except Exception`, log error, continue |
| Empty rows (no 10-K data found) | Log warning, `continue` |

```python
except requests.exceptions.Timeout:
    log.error(f"    TIMEOUT for {ticker} — SEC may be slow, try again later")
except Exception as e:
    log.error(f"    FAILED {ticker}: {e}")
```

The `Timeout` is caught separately to give a more actionable error message — the SEC EDGAR API is sometimes slow, and a targeted message tells the user to simply retry rather than investigate a bug.

---

## 5. Source 3 — FRED API

### What we get

5 macroeconomic time series from the Federal Reserve Bank of St. Louis Economic Data (FRED). These represent the broader economic environment that affects all stocks.

### Full Step-by-Step

```
fetch_fred(start_date, end_date)
```

---

**Step 1: Load API key**
```python
api_key = _load_api_key("FRED_API_KEY")
if not api_key:
    log.warning("  FRED_API_KEY not set — skipping.")
    return pd.DataFrame()
```
FRED requires a free API key (register at fred.stlouisfed.org). If not configured, FRED is skipped and an empty DataFrame is returned. The pipeline continues without macro data — macro risk defaults to 5.0 (neutral) in the risk scoring step.

---

**Step 2: Fetch each series in a loop**
```python
for col_name, series_id in config.FRED_SERIES.items():
    resp = SESSION.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id":          series_id,
            "observation_start":  start_date,
            "observation_end":    end_date,
            "api_key":            api_key,
            "file_type":          "json"
        },
        timeout=15
    )
    resp.raise_for_status()
```

The FRED REST API endpoint returns observations (date-value pairs) for one series at a time. Parameters:
- `series_id` — FRED's internal code (e.g., `FEDFUNDS`, `CPIAUCSL`)
- `observation_start/end` — date range filter
- `file_type=json` — return JSON not XML

---

**Step 3: Parse the response**
```python
obs = resp.json().get("observations", [])
df  = pd.DataFrame(obs)[["date", "value"]].copy()
df["date"]  = pd.to_datetime(df["date"])
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.rename(columns={"value": col_name}).set_index("date")
```

FRED returns observations like:
```json
{
  "observations": [
    {"date": "2024-05-01", "value": "5.33"},
    {"date": "2024-06-01", "value": "5.33"},
    {"date": "2024-07-01", "value": "5.12"},
    ...
  ]
}
```

Notes:
- All values come as **strings** — `pd.to_numeric(..., errors="coerce")` converts them and turns non-numeric strings (FRED uses `"."` for missing observations) into `NaN`.
- The column is immediately renamed from `"value"` to the series name (`"fed_funds_rate"`, etc.) before appending to `series_dfs`.

---

**Step 4: Per-series sleep**
```python
time.sleep(0.3)
```
FRED's free tier allows 120 API requests per minute. With 5 series, 0.3s sleep is conservative. This also prevents the retry session from being triggered.

---

**Step 5: Merge all series on date index**
```python
macro_df = pd.concat(series_dfs, axis=1).sort_index()
```

`pd.concat(axis=1)` does a **outer join** on the date index — dates present in any series appear in the result. Series with different reporting frequencies will have `NaN` for dates they don't cover.

**Frequency differences:**
| Series | Frequency | Rows per year |
|--------|-----------|--------------|
| `FEDFUNDS` (Fed Funds Rate) | Monthly | 12 |
| `CPIAUCSL` (CPI) | Monthly | 12 |
| `GS10` (10Y Treasury) | Monthly | 12 |
| `UNRATE` (Unemployment) | Monthly | 12 |
| `A191RL1Q225SBEA` (GDP) | Quarterly | 4 |

The Silver cleaning layer resamples this to daily frequency using forward-fill.

---

**Step 6: Save**
```python
filename = f"fred_macro_{start_date}_to_{end_date}.csv"
path = _save_bronze(macro_df, filename)
```

Output: `data/bronze/fred_macro_2024-05-06_to_2025-05-06.csv`

Example:
```
date,fed_funds_rate,cpi,treasury_10y,unemployment,gdp_growth
2024-05-01,5.33,313.548,,3.9,
2024-06-01,5.33,314.175,,4.0,
2024-07-01,5.12,314.540,,4.3,2.8
...
```

---

### Error Handling

```python
except Exception as e:
    log.error(f"  FAILED {series_id}: {e}")
```
Per-series error handling. If one series fails (e.g., GDP endpoint is down), the others still succeed. `series_dfs` will simply have fewer DataFrames to concat.

```python
if not series_dfs:
    return pd.DataFrame()
```
If all series fail, returns empty DataFrame. Downstream risk scoring defaults macro risk to 5.0.

---

## 6. Source 4 — NewsAPI

### What we get

Recent financial news headlines and descriptions for each ticker from over 150,000 news sources.

### Full Step-by-Step

```
fetch_news(tickers, days_back=1)
```

---

**Step 1: Load API key**
```python
api_key = _load_api_key("NEWS_API_KEY")
if not api_key:
    log.warning("  NEWS_API_KEY not set — skipping.")
    log.warning("  Fix: add NEWS_API_KEY=your_key to a .env file")
    log.warning("  Get free key at: https://newsapi.org/register")
    return pd.DataFrame()
```
NewsAPI requires a free API key. Three warning lines are logged with specific actionable instructions — where to get the key, where to put it.

---

**Step 2: Calculate the from_date**
```python
from_date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
```
`days_back=1` by default fetches yesterday's and today's articles. The free NewsAPI tier only allows articles from the last 30 days. The `days_back` parameter makes this configurable.

---

**Step 3: Build search query per ticker**
```python
stock     = yf.Ticker(ticker)
long_name = stock.info.get("longName", ticker)
query     = f'"{ticker}" stock OR "{long_name}" earnings OR revenue'
```

Example queries:
- AAPL → `"AAPL" stock OR "Apple Inc." earnings OR revenue`
- MSFT → `"MSFT" stock OR "Microsoft Corporation" earnings OR revenue`

**Why both ticker AND company name?**  
News articles rarely mention "AAPL" in the body — they use "Apple" or "Apple Inc." Using both increases article coverage significantly.

**The OR logic:**  
Matches articles about the stock (`"AAPL" stock`) OR financial events (`"Apple Inc." earnings OR revenue`). This filters out unrelated articles (Apple the fruit, Apple Records, etc.).

---

**Step 4: Fetch articles**
```python
resp = SESSION.get(
    "https://newsapi.org/v2/everything",
    params={
        "q":        query,
        "from":     from_date,
        "sortBy":   "publishedAt",   # newest first
        "language": "en",            # English only
        "pageSize": 100,             # max articles per request
        "apiKey":   api_key
    },
    timeout=15
)
```

`pageSize=100` is the maximum NewsAPI allows per request on the free tier. For high-profile tickers (AAPL, TSLA) there may be thousands of articles — we take only the 100 most recent.

---

**Step 5: Handle specific error codes**

```python
if resp.status_code == 401:
    log.error("  INVALID NewsAPI key — check your key at newsapi.org/account")
    return pd.DataFrame()

if resp.status_code == 426:
    log.error("  NewsAPI 426 error — free tier only allows today's news.")
    log.error("  Fix: upgrade at newsapi.org OR the pipeline will skip news.")
    log.warning("  Continuing without news data...")
    return pd.DataFrame()
```

These two codes are handled **before** `resp.raise_for_status()` because they require specific user-facing guidance:

- **401** — Wrong API key. No point retrying. Return immediately.
- **426 Upgrade Required** — NewsAPI free tier limitation: if `from_date` is more than ~1 day ago, the free tier blocks access. This is a tier restriction, not an error to retry.

Both immediately return an empty DataFrame — not just skip the current ticker — because the error is pipeline-wide (the key is wrong for all tickers).

---

**Step 6: Extract article fields**
```python
articles = resp.json().get("articles", [])

for art in articles:
    all_rows.append({
        "ticker":       ticker,
        "company":      long_name,
        "published_at": art.get("publishedAt"),   # ISO 8601 timestamp
        "title":        art.get("title", ""),
        "description":  art.get("description", ""),
        "source":       art.get("source", {}).get("name", ""),
        "author":       art.get("author", ""),
        "url":          art.get("url", ""),
    })
```

`art.get("source", {}).get("name", "")` — double `.get()` because `source` is itself a nested dict `{"id": "reuters", "name": "Reuters"}`. The outer `.get("source", {})` returns an empty dict if missing, preventing a `KeyError` on the inner `.get("name")`.

---

**Step 7: Rate limit sleep**
```python
time.sleep(0.5)
```
NewsAPI free tier: 100 requests/day. With 5 default tickers that's 5 requests, well within limits. Sleep is still included for politeness and to avoid burst detection.

---

**Step 8: Combine, deduplicate, save**
```python
news_df = pd.DataFrame(all_rows)
news_df["published_at"] = pd.to_datetime(news_df["published_at"])
news_df = news_df.drop_duplicates(subset=["url"])

path = _save_bronze(news_df, "newsapi_headlines_raw.csv")
```

`drop_duplicates(subset=["url"])` — the same article can appear in results for multiple tickers (e.g., "Tech stocks rally" mentions both AAPL and MSFT). Deduplication by URL removes exact duplicates across tickers.

Note: we do **not** deduplicate by ticker — the same article can legitimately appear for multiple tickers.

---

### Error Handling

```python
except Exception as e:
    log.error(f"  FAILED {ticker}: {e}")
```

```python
if not all_rows:
    log.warning("  No articles fetched")
    return pd.DataFrame()
```

Per-ticker exception handling so one broken ticker doesn't stop news fetching for others. Final empty check before building the DataFrame.

---

## 7. Main Entry Point — `run_ingestion()`

```python
def run_ingestion(
    tickers:    List[str] = config.DEFAULT_TICKERS,
    upload_gcs: bool = False,
) -> dict:

    log.info(f"Tickers : {tickers}")
    log.info(f"Period  : {config.START_DATE} -> {config.END_DATE}")

    return {
        "prices": fetch_yahoo_finance(tickers, upload_gcs=upload_gcs),
        "edgar":  fetch_sec_edgar(tickers,     upload_gcs=upload_gcs),
        "macro":  fetch_fred(                  upload_gcs=upload_gcs),
        "news":   fetch_news(tickers,          upload_gcs=upload_gcs),
    }
```

**Key design decisions:**
- All 4 sources run **sequentially**, not in parallel. This is intentional — parallel fetching would multiply the rate limit burden on each API simultaneously.
- Returns a nested dict. Every downstream module (`clean.py`, `eda.py`) expects exactly this structure. If a source fails completely, its key maps to `{}` or an empty DataFrame rather than being absent — so downstream code always gets the expected keys.

**Running from CLI:**
```bash
# Default tickers (AAPL MSFT TSLA NVDA JPM)
python ingest.py

# Custom tickers
python ingest.py AAPL GOOGL AMZN META NFLX
```

```python
if __name__ == "__main__":
    import sys
    tickers = [t.upper() for t in sys.argv[1:]] if len(sys.argv) > 1 else config.DEFAULT_TICKERS
    run_ingestion(tickers=tickers)
```

`.upper()` normalizes input so `python ingest.py aapl msft` works the same as `python ingest.py AAPL MSFT`.

---

## 8. Error Handling Summary Table

| Source | Error Type | What happens |
|--------|-----------|-------------|
| Yahoo Finance | Empty response | Log warning, skip ticker, continue |
| Yahoo Finance | Any exception | Log error with ticker name, continue loop |
| EDGAR | CIK map download fails | Log error, return `{}`, skip entire EDGAR |
| EDGAR | Ticker not in CIK map | Log warning, skip ticker |
| EDGAR | HTTP 404 (no XBRL) | Log warning, skip ticker |
| EDGAR | Request timeout | Log specific timeout message, skip ticker |
| EDGAR | No 10-K data in JSON | Log warning, skip ticker |
| EDGAR | Any other exception | Log error, continue |
| FRED | Missing API key | Log warning with fix instructions, return empty DataFrame |
| FRED | Series-level exception | Log error, skip that series, continue others |
| FRED | All series fail | Return empty DataFrame |
| NewsAPI | Missing API key | Log warning with registration link, return empty DataFrame |
| NewsAPI | HTTP 401 Invalid key | Log error with link to check key, return empty DataFrame |
| NewsAPI | HTTP 426 Tier limit | Log specific upgrade message, return empty DataFrame |
| NewsAPI | Ticker-level exception | Log error, continue next ticker |
| NewsAPI | No articles fetched | Log warning, return empty DataFrame |
| GCS Upload | `google-cloud-storage` not installed | Log warning, skip — local file already saved |
| GCS Upload | Auth or network error | Log error, skip — does not crash pipeline |

---

## 9. Rate Limit Summary Table

| Source | Sleep | Rate | Why |
|--------|-------|------|-----|
| Yahoo Finance | 0.5s per ticker | 2 req/s | Undocumented limit; 0.5s is empirically safe |
| SEC EDGAR | 0.6s per ticker | ~1.7 req/s | SEC policy: max 10 req/s; 0.6s = very conservative |
| FRED API | 0.3s per series | ~3 req/s | 120 req/min limit; 0.3s = 3.3 req/s, well within |
| NewsAPI | 0.5s per ticker | 2 req/s | 100 req/day total; sleep is for burst protection |
| Between batches | 3s (in orchestrator) | — | Cooldown after 50-ticker batch across all sources |

Additionally, the shared `requests.Session` adds automatic **exponential backoff** when any source returns a 429, so actual sleep time can be longer if a rate limit is hit.

---

## 10. What the Bronze Layer Looks Like

After running `python ingest.py AAPL MSFT TSLA NVDA JPM`, the `data/bronze/` directory contains:

```
data/bronze/
│
│  ── Yahoo Finance (one file per ticker) ──
├── yahoo_AAPL_2024-05-06_to_2025-05-06.csv      (~252 rows, 20 cols)
├── yahoo_MSFT_2024-05-06_to_2025-05-06.csv      (~252 rows, 20 cols)
├── yahoo_TSLA_2024-05-06_to_2025-05-06.csv      (~252 rows, 20 cols)
├── yahoo_NVDA_2024-05-06_to_2025-05-06.csv      (~252 rows, 20 cols)
├── yahoo_JPM_2024-05-06_to_2025-05-06.csv       (~252 rows, 20 cols)
│
│  ── SEC EDGAR (two files per ticker) ──
├── edgar_AAPL_raw_long.csv                       (long format: ~100+ rows)
├── edgar_AAPL_annual_financials.csv              (wide format: ~5-10 rows = years)
├── edgar_MSFT_raw_long.csv
├── edgar_MSFT_annual_financials.csv
├── ...
│
│  ── FRED (one combined file) ──
├── fred_macro_2024-05-06_to_2025-05-06.csv       (~12 rows, 5 cols)
│
│  ── NewsAPI (one combined file) ──
└── newsapi_headlines_raw.csv                     (varies: 0–500 rows, 8 cols)
```

For the full S&P 500 + NASDAQ-100 run (~600 tickers):
- Yahoo files: ~600 CSV files
- EDGAR files: ~1,200 CSV files (long + wide per ticker)
- FRED files: 1 CSV file
- News files: 1 CSV file (all tickers combined)
- **Total: ~1,800+ Bronze CSV files**

---

*The Bronze layer is intentionally kept completely raw — no cleaning, no transformation, no column renames (except adding `ticker`). Every subsequent problem can be diagnosed by looking at the Bronze files, which represent exactly what each external API returned.*
