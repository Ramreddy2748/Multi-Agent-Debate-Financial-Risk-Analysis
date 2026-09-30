# Data298A — Project
## Debate-Based Multi-Agent Reasoning for Financial Risk Modeling
Team 3 -  DATA 298A — MSDA Project I 

**Team Members**
- Harshitha Boinepally
- Kalyan Mamidi
- Venkata Ramireddy Seelam
- Venkata Siddarth Gullipalli

---

## Project Overview

This project builds a multi-agent AI system that performs structured investment
risk assessment for publicly traded companies. Specialised AI agents each analyse
one dimension of financial risk — fundamentals, market volatility, news sentiment,
and macroeconomic conditions — then debate their findings before producing a
final explainable risk classification.

---

## Repository Structure
```text
Data298A--Masters-Project/
├── ingest.py                  # Data collection — Yahoo Finance, SEC EDGAR, FRED, NewsAPI
├── clean.py                   # Data cleaning — Bronze to Silver layer
├── eda.py                     # Risk scoring and visualisations — Silver to Gold layer
├── split.py                   # Train/val/test stratified split
├── pipeline_large_scale.py    # Full S&P 500 + NASDAQ-100 pipeline runner
├── tickers.py                 # Live ticker list fetcher (Wikipedia)
├── config.py                  # Central configuration — paths, thresholds, API keys
├── demo_pipeline.ipynb        # End-to-end pipeline demo notebook
├── requirements.txt           # Python dependencies
├── data/
│   ├── bronze/                # Raw data as collected (CSV)
│   ├── silver/                # Cleaned and feature-engineered data (CSV)
│   ├── silver_parquet/        # Cleaned and feature-engineered data (Parquet)
│   ├── gold/                  # Risk scores, splits, verdict JSON outputs
│   ├── verdicts/              # Auditable PDF/JSON risk verdict scorecards
│   └── reports/               # Generated visualisation charts (PNG)
└── GCP_SETUP.md               # Google Cloud setup instructions
```

---


## Requirements

- Python 3.10 or higher
- API keys for FRED and NewsAPI (Yahoo Finance and SEC EDGAR require no key)

---

## Setup

### Step 1 — Clone the repository
```bash
git clone https://github.com/Ramreddy2748/Data298A--Masters-Project.git
cd Data298A-Project
```

### Step 2 — Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

```bash
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
pip install scikit-learn jupyter ipykernel nbformat
```

### Step 4 — Set API keys

Create a `.env` file in the project root:
FRED_API_KEY=your_fred_api_key_here
NEWS_API_KEY=your_newsapi_key_here


Yahoo Finance and SEC EDGAR require no API key.

---

## Running the Pipeline

Run each step in order. Each step depends on the output of the previous one.

### Step 1 — Data ingestion (Bronze layer)

Collects raw data from all four sources and saves to `data/bronze/`.
```bash
python3 ingest.py
```

Expected output:
- `data/bronze/yahoo_TICKER_*.csv` — stock price history per ticker
- `data/bronze/edgar_TICKER_annual_financials.csv` — SEC annual filings
- `data/bronze/fred_macro_*.csv` — macroeconomic indicators
- `data/bronze/newsapi_headlines_raw.csv` — news headlines

FRED macro coverage includes fed funds, CPI, 10Y Treasury, unemployment, GDP
growth, 10Y-2Y yield spread, VIX, and WTI oil.

To run for specific tickers:
```bash
python3 ingest.py AAPL MSFT TSLA NVDA JPM
```

### Step 2 — Data cleaning (Silver layer)

Cleans and transforms Bronze data. Saves to `data/silver/`.
```bash
python3 clean.py
```

Expected output:
- `data/silver/silver_prices_TICKER.csv` — cleaned price data with engineered features
- `data/silver/silver_edgar_TICKER.csv` — cleaned financial statements
- `data/silver/silver_macro.csv` — cleaned macroeconomic data
- `data/silver/silver_news_sentiment.csv` — VADER sentiment scores
- `data/silver_parquet/*.parquet` — parquet copies for analytics and BigQuery-ready workflows

### Step 3 — EDA and risk scoring (Gold layer)

Computes multi-dimensional risk scores and generates visualisation charts.
Saves to `data/gold/` and `data/reports/`.
```bash
python3 eda.py
```

Expected output:
- `data/gold/gold_risk_scores.csv` — risk scores per ticker (composite 1-10)
- `data/gold/gold_insights.csv` — auto-generated risk insights
- `data/reports/01_price_trends.png` through `data/reports/07_sentiment_trend.png`

### Step 4 — Train/val/test split

Splits the Gold dataset into train (70%), validation (15%), and test (15%) sets
using stratified sampling to preserve class balance.
```bash
python3 split.py
```

Expected output:
- `data/gold/split_train.csv` — 86 companies
- `data/gold/split_val.csv` — 18 companies
- `data/gold/split_test.csv` — 19 companies
- `data/reports/08_class_balance.png` — class balance chart
- `data/reports/09_sector_distribution.png` — sector distribution chart

### Step 5 — Agent critic verdict and PDF scorecard

Runs the fundamental, market/sentiment, macro, and critic layers for one ticker,
then saves an auditable JSON verdict and PDF scorecard with Policy A-H evidence.
```bash
python3 agents/verdict_report.py AAPL --company "Apple Inc." --sector "Information Technology"
```

Expected output:
- `data/verdicts/AAPL_risk_verdict.json` — full agent/critic response
- `data/verdicts/AAPL_risk_verdict.pdf` — PDF scorecard and Policy A-H evidence trail

To log lineage and monitoring metadata to MLflow when available, or to
`data/gold/monitoring_runs.jsonl` otherwise:
```bash
python3 agents/verdict_report.py AAPL \
  --company "Apple Inc." \
  --sector "Information Technology" \
  --log-monitoring
```

### Step 6 — Explicit debate state machine

Runs the same specialist agents through an enforced protocol:
collect evidence, critique contradictions, revise positions, then synthesize a
final verdict. Uses LangGraph when installed and a local state-machine fallback
otherwise.
```bash
python3 agents/debate_state_machine.py AAPL \
  --company "Apple Inc." \
  --sector "Information Technology" \
  --out data/verdicts/AAPL_debate_verdict.json \
  --log-monitoring
```

The debate output includes contradiction records, revision notes, human-review
flags, confidence, and the same Policy A-H evidence trail.

---

## Running the Large-Scale Pipeline (S&P 500 + NASDAQ-100)

To run the full pipeline across all sectors:
```bash
python3 pipeline_large_scale.py
```

To run specific sectors only:
```bash
python3 pipeline_large_scale.py --sectors "Information Technology" "Health Care"
```

To test with 10 companies per sector:
```bash
python3 pipeline_large_scale.py --test
```

To resume an interrupted run:
```bash
python3 pipeline_large_scale.py --resume
```

To upload outputs to Google Cloud Storage:
```bash
python3 pipeline_large_scale.py --gcs
```

---

## Running the Demo Notebook

Opens an end-to-end walkthrough of the pipeline with inline outputs and charts.
```bash
jupyter notebook demo_pipeline.ipynb
```

Or open `demo_pipeline.ipynb` directly in VS Code with the Jupyter extension installed.

---

## Running the Stock Risk App

The project now includes a production-style FastAPI layer plus a browser
dashboard. The API reads the Gold/Silver data artifacts, serves saved JSON/PDF
verdicts, generates new critic/debate verdicts on demand, and exposes monitoring
lineage records to the frontend.

Install the API server dependencies:
```bash
pip install -r requirements.txt
```

Run the app:
```bash
uvicorn src.api.server:app --reload --host 127.0.0.1 --port 8766
```

Compatibility launcher:

```bash
python3 frontend/server.py 8766
```

Then open:
```text
http://127.0.0.1:8766
```

The app includes ticker search, risk filters, investment Q&A, direct verdict
generation, debate review, company comparison, price traces, agent scorecards,
Policy A-H evidence, contradiction/revision status, monitoring records, and
links to generated JSON/PDF artifacts.

Useful API endpoints:
- `GET /api/health` — app/data readiness
- `GET /api/companies` — Gold risk universe
- `GET /api/verdicts/AAPL` — saved verdict plus fallback risk answer
- `POST /api/verdict` — generate a critic or debate verdict
- `GET /api/monitoring?ticker=AAPL` — lineage records

---

## Configuration

All pipeline settings are in `config.py`:

| Setting | Default | Description |
|---|---|---|
| DEFAULT_TICKERS | AAPL, MSFT, TSLA, NVDA, JPM | Tickers for single-run mode |
| ZSCORE_THRESHOLD | 3.0 | Outlier capping threshold on daily returns |
| MAX_NAN_PCT_ROW | 0.20 | Drop rows with more than 20% missing values |
| FFILL_LIMIT | 2 | Forward-fill up to N consecutive NaN values |
| RISK_SCORE_LOW | 3.5 | Composite score threshold for LOW risk label |
| RISK_SCORE_HIGH | 6.0 | Composite score threshold for HIGH risk label |

---

## Data Pipeline Architecture
```text
Raw APIs
   |
   v
Bronze layer (data/bronze/)     -- raw data, no changes
   |
   v
Silver layer (data/silver/)     -- cleaned, normalised, feature-engineered
   |
   v
Gold layer (data/gold/)         -- risk scores, analytics-ready
   |
   v
Train / Val / Test splits       -- stratified by risk label
   |
   v
Model training and evaluation   -- coming in Phase 2
```  

---

## Risk Score Dimensions

Each company receives four risk scores on a 1-10 scale:

| Dimension | Method | Data Source |
|---|---|---|
| Fundamental risk | P/E ratio + beta + max drawdown (weighted) | Yahoo Finance, SEC EDGAR |
| Volatility risk | Annualised standard deviation of daily returns | Yahoo Finance |
| Sentiment risk | VADER compound score from news headlines (inverted) | NewsAPI |
| Macro risk | Fed Funds Rate severity + CPI z-score | FRED |
| Composite risk | 30% fundamental + 30% volatility + 20% sentiment + 20% macro | All sources |

Risk labels: HIGH (above 6.0) · MODERATE (3.5 to 6.0) · LOW (below 3.5)

---

## Current Dataset

- Companies: 123 (S&P 500 + NASDAQ-100)
- Sectors: 11 GICS sectors
- Collection date: 2026-03-04
- Bronze files: 361 CSVs
- Silver files: 243 CSVs
- Gold files: 17 CSVs
- Report charts: 9 PNGs
- Train set: 86 companies
- Validation set: 18 companies
- Test set: 19 companies

---
