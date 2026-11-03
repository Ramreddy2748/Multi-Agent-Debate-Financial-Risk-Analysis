# Financial Risk Multi-Agent System

## End-to-End Technical Documentation

Project: Debate-Based Multi-Agent Reasoning for Financial Risk Modeling  
Repository: Data298A--Masters-Project  
Scope: current implementation from company-universe discovery and API extraction through preprocessing, feature engineering, model inference, debate, monitoring, reports, and the browser application.

This document describes what the repository actually does today. Optional model paths and every fallback path are called out explicitly.

---

## 1. System Purpose

The application answers questions such as:

- What is the financial status of Apple?
- Can I invest in Microsoft?
- Compare Apple and Amazon.

It produces research-oriented risk guidance, not guaranteed investment advice. Independent components analyze:

1. Fundamental risk: cash, debt, equity, revenue, profitability, and operating margin.
2. Market risk: price behavior, volatility, beta, drawdown, trend, and distance from the 52-week high.
3. News sentiment risk: recent financial headlines classified as positive, neutral, or negative.
4. Macro-economic risk: rates, inflation, employment, GDP, yield curve, VIX, and oil.
5. Critic/orchestrator reasoning: comparison, contradiction detection, synthesis, and human-review flags.

The final output contains a 1-10 score, LOW/MODERATE/HIGH label, confidence, evidence, risk drivers, offsets, provenance, freshness, and fallback disclosures.

---

## 2. End-to-End Architecture

~~~text
Company universe
      |
      v
Yahoo Finance + SEC EDGAR + FRED + NewsAPI
      |
      v
Bronze: immutable/raw source files
      |
      v
Silver: cleaned, standardized, feature-engineered files
      |
      v
Gold: analytics-ready features, scores, labels, insights, reports
      |
      +--> Fundamental Agent
      +--> Market + Volatility Agent
      +--> Sentiment Agent
      +--> Macro-Economic Agent
                     |
                     v
       Debate state machine: collect -> critique -> revise -> synthesize
                     |
                     v
       Critic: Claude when configured, weighted fallback otherwise
                     |
      +--------------+------------------+
      |              |                  |
   JSON verdict   PDF scorecard     monitoring lineage
      |
      v
FastAPI -> browser Risk Console
~~~

The project follows a medallion architecture. Raw data is kept separate from cleaned data and model-ready analytics so a result can be traced back to what was received and how it was transformed.

---

## 3. Technology Stack

| Area | Technology | Main location |
|---|---|---|
| Language | Python 3.10+ | backend, ingestion, agents, analytics |
| Data processing | pandas, NumPy, SciPy | data_preprocessing |
| HTTP | requests, urllib, retry adapters | ingestion and remote inference |
| Market data | yfinance / Yahoo Finance | data_preprocessing/ingest.py |
| Regulatory data | SEC EDGAR company facts API | data_preprocessing/ingest.py |
| Macro data | FRED API | data_preprocessing/ingest.py |
| News | NewsAPI and optional GDELT history | data_preprocessing |
| Sentiment baseline | VADER | data_preprocessing/clean.py |
| Fundamental model | fine-tuned ProsusAI/FinBERT | agents/fundamental/finbert.py |
| News model | ProsusAI/finbert | agents/sentiment/finbert.py |
| Market model | Qwen LoRA, local or remote Colab | agents/market and critic_agent.py |
| Macro model | Phi-4-mini LoRA Macro v2 | agents/macro/macro_v2 |
| Critic | Claude/Anthropic or OpenAI-compatible endpoint | agents/orchestrator/critic_agent.py |
| Workflow | LangGraph when installed; local equivalent otherwise | agents/orchestrator/debate_state_machine.py |
| Retrieval | curated JSONL rules and optional Chroma | data/rag and agents/shared |
| API | FastAPI and Uvicorn | src/api/server.py |
| Frontend | HTML, CSS, JavaScript, Chart.js | frontend |
| Monitoring | MLflow or local JSONL | agents/orchestrator/monitoring.py |
| Reports | matplotlib and PDF backend | data_preprocessing and verdict_report.py |
| Tests | Python unittest | tests |
| Cloud | Google Cloud Storage and optional BigQuery | config.py and pipeline_large_scale.py |

---

## 4. Configuration and Secrets

config.py defines:

- default example tickers;
- a one-year default date range;
- local Bronze, Silver, Parquet, Gold, reports, and verdict paths;
- GCS bucket and BigQuery table names;
- cleaning thresholds;
- risk-label thresholds;
- eight FRED series.

Secrets are loaded from environment variables or the project-root .env file. They must never be committed.

~~~text
FRED_API_KEY=...
NEWS_API_KEY=...
ANTHROPIC_API_KEY=...
CLAUDE_MODEL=...
CRITIC_LLM_ENABLED=1
MARKET_LORA_ENABLED=1
MARKET_LORA_REMOTE_URL=...
FUNDAMENTAL_FINBERT_ENABLED=1
FUNDAMENTAL_FINBERT_MODEL_DIR=models/fundamental_agent_finbert
SENTIMENT_FINBERT_MODEL=ProsusAI/finbert
~~~

The code checks environment variables first and can read .env as a fallback. Placeholder values such as YOUR_FRED_API_KEY_HERE are treated as missing.

Shared risk-label convention:

~~~text
score <= 3.5         LOW
3.5 < score <= 6.0   MODERATE
score > 6.0          HIGH
~~~

Scores are clamped to 1.0 through 10.0.

---

## 5. Company Universe

Main files:

- data_preprocessing/tickers.py
- root compatibility wrapper tickers.py
- data/gold/ticker_universe.csv

The universe builder:

1. Fetches the current S&P 500 table from Wikipedia.
2. Fetches the current NASDAQ-100 table from Wikipedia.
3. Normalizes symbols, including Yahoo-compatible hyphens.
4. Combines both lists.
5. Deduplicates by ticker.
6. Marks overlap as index BOTH.
7. Groups companies by sector.
8. Splits sectors into rate-limit-friendly batches.

Fallback order:

1. Live S&P 500 and NASDAQ-100 tables.
2. Hardcoded NASDAQ fallback list if its table cannot be parsed.
3. Cached data/gold/ticker_universe.csv if live fetching fails.
4. An empty result if no source or cache is available.

The exact company count changes as index membership changes. The cached universe is therefore an artifact of the run date, not a permanent list.

---

## 6. Bronze Layer: Data Extraction

Main files:

- data_preprocessing/ingest.py
- ingest.py compatibility wrapper
- data_preprocessing/pipeline_large_scale.py
- data_preprocessing/newsapi_only.py
- data_preprocessing/historical_news_backfill.py

A shared requests.Session uses connection pooling and retries for HTTP 429 and common 5xx errors. Each ticker and source is processed independently.

### 6.1 Yahoo Finance

For each ticker, the extractor:

1. requests daily OHLCV history for the configured date range;
2. removes timezone information from the date index;
3. adds the ticker;
4. reads current stock.info metadata;
5. attaches P/E, EPS, market cap, revenue, debt-to-equity, beta, 52-week high/low, sector, industry, and company name;
6. saves a ticker-specific CSV.

Output:

~~~text
data/bronze/yahoo_<TICKER>_<start>_to_<end>.csv
~~~

The stock.info values are current snapshots broadcast onto historical rows. They are not historical values for each trading date.

### 6.2 SEC EDGAR

For each ticker, the extractor:

1. downloads the SEC CIK map;
2. maps ticker to CIK;
3. requests SEC company facts;
4. selects annual facts across configured accounting concepts;
5. retains dates, form, filing metadata, units, and values;
6. saves raw long-form and annual financial files.

Outputs:

~~~text
data/bronze/edgar_<TICKER>_raw_long.csv
data/bronze/edgar_<TICKER>_annual_financials.csv
~~~

SEC requests use a delay to respect rate limits. Missing CIKs, timeouts, and malformed facts are logged per ticker.

### 6.3 FRED

The configured series are:

| Field | Series |
|---|---|
| fed_funds_rate | FEDFUNDS |
| cpi | CPIAUCSL |
| treasury_10y | GS10 |
| unemployment | UNRATE |
| gdp_growth | A191RL1Q225SBEA |
| yield_curve_10y2y | T10Y2Y |
| vix | VIXCLS |
| wti_oil | DCOILWTICO |

The API key is loaded from the environment or .env. Missing credentials skip FRED rather than stopping all other sources. A dated macro CSV is written to Bronze.

### 6.4 NewsAPI

NewsAPI retrieves recent headlines using ticker, company name, and financial-language terms such as earnings or revenue. Raw rows retain ticker, company, title, description, source, URL, and published timestamp.

Output:

~~~text
data/bronze/newsapi_headlines_raw.csv
~~~

The source handles invalid keys, free-tier date restrictions, and HTTP 429 quota responses. newsapi_only.py resumes from the existing raw file, skips completed tickers, saves partial results, and stops cleanly on quota exhaustion.

NewsAPI history is limited on free plans. historical_news_backfill.py can use GDELT to add older normalized headlines.

### 6.5 Optional cloud outputs

When Google credentials and packages are available:

- Bronze, Silver, and Gold files can be uploaded to GCS.
- Gold risk scores can be loaded into BigQuery.
- Local files remain available if cloud upload fails.

---

## 7. Silver Layer: Cleaning and Feature Engineering

Main files:

- data_preprocessing/clean.py
- clean.py compatibility wrapper
- data/silver
- data/silver_parquet

Bronze is never cleaned in place. Silver is a separate output layer.

### 7.1 Price cleaning

For every Yahoo price file, the cleaner:

1. standardizes names and dates;
2. forward-fills short gaps up to the configured limit;
3. drops rows with excessive missing numeric values;
4. calculates daily returns;
5. calculates rolling 30-day and 60-day volatility;
6. calculates moving averages and trend gaps;
7. calculates running-peak drawdown;
8. preserves beta, 52-week values, valuation, and company metadata;
9. caps extreme return outliers using the z-score threshold;
10. writes CSV and Parquet.

Typical outputs:

~~~text
data/silver/silver_prices_<TICKER>.csv
data/silver_parquet/silver_prices_<TICKER>.parquet
~~~

### 7.2 SEC cleaning

SEC annual records are standardized into fields used by the fundamental agent and retrieval index. Runtime helpers defensively normalize values that may be raw dollars or USD millions.

~~~text
data/silver/silver_edgar_<TICKER>.csv
data/silver_parquet/silver_edgar_<TICKER>.parquet
~~~

### 7.3 Macro cleaning

Macro observations are parsed, sorted, and aligned into:

~~~text
data/silver/silver_macro.csv
~~~

The macro agent reads the latest valid observations and derives CPI year-over-year change, yield-curve state, VIX context, and other features.

### 7.4 News cleaning

News cleaning:

1. removes blank titles;
2. deduplicates by URL;
3. parses mixed ISO8601 and timezone strings with UTC-aware conversion;
4. calculates VADER compound sentiment;
5. aggregates by ticker and date;
6. writes mean sentiment, standard deviation, and article count;
7. saves CSV and Parquet.

Outputs:

~~~text
data/silver/silver_news_sentiment.csv
data/silver_parquet/silver_news_sentiment.parquet
~~~

VADER is a preprocessing baseline. The runtime Sentiment Agent prefers FinBERT.

---

## 8. Gold Analytics and Risk Scoring

Main files:

- data_preprocessing/eda.py
- data_preprocessing/build_gold_from_silver.py
- data/gold/gold_risk_scores.csv
- data/gold/gold_risk_scores_ALL.csv
- data/gold/gold_insights.csv
- data/gold/gold_risk_full_*.csv

Gold is generated either by the full EDA stage or by rebuilding from existing Silver files. build_gold_from_silver.py does not call Yahoo, SEC, FRED, or NewsAPI; it recomputes analytics from data already present.

### 8.1 Market features

Gold contains features such as:

- latest close;
- return mean and standard deviation;
- annualized and rolling volatility;
- beta;
- maximum drawdown;
- distance from 52-week high;
- SMA/trend gaps;
- YTD return;
- price-row count;
- sector and company metadata.

### 8.2 Fundamental risk

The fundamental calculation derives:

- debt-to-cash;
- debt-to-equity;
- profit margin;
- operating margin.

The deterministic score uses four equally weighted pressure groups:

~~~text
debt-to-cash pressure       25%
debt-to-equity pressure     25%
profit-margin pressure      25%
operating-margin pressure   25%
~~~

Evidence rules explain leverage, cash coverage, profitability, and operating cushion.

### 8.3 Market risk

The deterministic market score uses:

~~~text
annualized volatility        up to 3 points
beta                         up to 2 points
maximum drawdown             up to 2 points
distance from 52-week high   up to 1 point
negative YTD return          up to 1 point
base                         1 point
~~~

Components are capped and combined into a 1-10 score. News is not scored here; it is handled by the dedicated Sentiment Agent.

### 8.4 Macro risk

The local macro score adds pressure for:

- high federal funds rate;
- CPI year-over-year inflation;
- high unemployment;
- negative GDP growth;
- high 10-year Treasury yield;
- inverted yield curve;
- elevated VIX;
- elevated oil.

Sector adjustments add rate pressure for Real Estate and Utilities, add growth/discount-rate pressure for Information Technology and Consumer Discretionary, and provide a defensive offset for Consumer Staples and Health Care.

### 8.5 Gold outputs

Gold stores features, component scores, composite score, label, dates, freshness fields, and generated explanations. EDA also produces price, volatility, returns, correlation, risk, macro, and sentiment reports in data/reports.

---

## 9. Feature Priority and Freshness

File: agents/evaluation/feature_priority.py

This layer ranks existing evidence fields for each ticker and agent. It combines financial importance, agent-specific rules, observed values, risk direction, and source age.

Outputs:

~~~text
data/gold/feature_priority_scores.csv
data/gold/feature_priority_summary.csv
data/gold/feature_freshness.csv
data/gold/feature_priority_report.md
~~~

Each row records the feature, source, source date, age, priority score, high-priority flag, and freshness status. The API exposes these records through:

~~~text
GET /api/companies/{ticker}/feature-priority
~~~

This layer ranks existing data; it does not invent a new financial measurement.

---

## 10. RAG and Retrieval

The project has a transparent curated RAG layer and an optional vector retrieval layer.

### 10.1 Curated financial-term RAG

Files:

~~~text
data/rag/financial_terms.jsonl
data/rag/agent_retrieval_config.json
data/rag/README.md
agents/shared/financial_rag.py
~~~

The JSONL contains definitions, formulas, thresholds, risk rules, agent scope, and evidence fields.

Retrieval logic:

1. tokenize query and term metadata;
2. match agent scope;
3. include configured required terms;
4. add points for required-term match, agent match, priority, and token overlap;
5. sort by score;
6. return the top terms;
7. format definitions and thresholds as prompt context.

critic_agent.py attaches the selected context to specialist outputs. It records source, status, term IDs, and text. RAG grounds explanations and feature priority; it does not directly replace a model score.

### 10.2 Optional Chroma index

agents/shared/retrieval.py can build:

~~~text
data/gold/vector_store/
~~~

Chunks are created from SEC filing rows, the latest price snapshot, recent price trend, and Silver news rows. Chroma retrieves the top-k ticker-specific evidence chunks by embedding similarity. Silver remains the source of truth and the vector store is rebuildable.

---

## 11. Specialist Agents

All specialists return structured output with agent, ticker, score, label, confidence, claim type, evidence, positive signals, and negative signals.

### 11.1 Fundamental Agent

Files: agents/orchestrator/critic_agent.py and agents/fundamental/finbert.py

Input: latest data/silver/silver_edgar_<TICKER>.csv row.

Primary model: locally saved fine-tuned ProsusAI/FinBERT sequence classifier when enabled.

Fallback: deterministic leverage/profitability calculation.

FinBERT input text contains company, filing period, cash, long-term debt, net income, operating income, revenue, and equity. The classifier predicts LOW, MODERATE, or HIGH. Confidence is converted to a score in the predicted label band. The output also includes raw class probabilities and independently calculated financial signals.

Enable it with:

~~~text
FUNDAMENTAL_FINBERT_ENABLED=1
~~~

Missing model files, missing dependencies, or inference errors return the deterministic result. Missing SEC data returns a moderate low-confidence fallback.

### 11.2 Market + Volatility Agent

Files: agents/orchestrator/critic_agent.py, agents/market/rule_based.py, agents/market/infer_lora.py.

Input: Silver Yahoo features and Gold return context.

Primary model: Qwen LoRA adapter, locally or through a Colab endpoint.

Fallback: deterministic market score.

The model payload includes annualized volatility, beta, maximum drawdown, price versus 52-week high, YTD return, and rolling volatility. Remote inference is attempted first when MARKET_LORA_REMOTE_URL exists. Local Transformer inference is attempted when enabled. Generated JSON is parsed and normalized.

If LoRA fails, the deterministic score is returned with the reason, such as offline ngrok endpoint or missing adapter. The Market Agent does not consume headline text.

### 11.3 Sentiment Agent

File: agents/sentiment/finbert.py

Input: raw NewsAPI rows and Silver VADER aggregates.

Primary model: ProsusAI/finbert.

Fallbacks: VADER, then no-news fallback.

Headline selection uses exact ticker matching, company-name matching from Silver metadata, and a financial-language relevance gate. FinBERT predicts positive, neutral, or negative for each selected headline.

~~~text
sentiment_value = +1 positive
                   0 neutral
                  -1 negative

risk_score = 5 - 3 * average_sentiment
             + 2 * negative_headline_fraction
~~~

The output includes article count, sentiment mix, average sentiment, confidence, and top positive and negative headlines.

If FinBERT cannot load but VADER rows exist, VADER is used. If no raw or Silver news exists, the result is NO_NEWS_FALLBACK with low confidence and an explicit coverage warning.

### 11.4 Macro-Economic Agent

Files: agents/macro/agent.py and agents/macro/macro_v2.

Input: Silver FRED data plus company sector.

Primary path: Macro v2 sector forecast when its snapshot is available.

Other path: legacy DeepSeek reasoning when configured.

Fallback: local FRED rules.

Macro v2 maps a company sector to a sector ETF, uses macro indicators and sector sensitivities, and forecasts future annualized volatility. It is a sector systematic-risk signal, not a complete company-specific stock picker.

If the model or snapshot is unavailable, local rules use FRED indicators and sector sensitivity. Missing macro data returns a low-confidence fallback instead of stopping the verdict.

### 11.5 Critic / Orchestrator

File: agents/orchestrator/critic_agent.py

Primary model: Claude/Anthropic or an OpenAI-compatible endpoint when enabled.

Fallback: weighted critic.

Default weights:

~~~text
Fundamental Agent             0.35
Market + Volatility Agent    0.25
Sentiment Agent               0.15
Macro-Economic Agent          0.25
~~~

The weighted critic calculates a weighted mean, derives the final label, collects negative and positive signals, and marks disagreement when labels differ. Confidence is lowered for disagreement.

Claude is called only when the critic feature is enabled and a supported key/provider is configured. The prompt requests strict JSON with final score, label, confidence, disagreement, review flag, scorecard, drivers, offsets, and decision. Failed or invalid Claude output falls back to the weighted critic and preserves the error.

---

## 12. Debate State Machine

File: agents/orchestrator/debate_state_machine.py

Workflow:

~~~text
collect_agent_positions
        -> critique_contradictions
        -> revise_positions when needed
        -> critique_contradictions again
        -> synthesize_verdict
~~~

Collection runs Fundamental, Market, Sentiment, and Macro for one ticker.

Contradictions are recorded when:

- labels differ;
- score spread is at least two points;
- confidence is low;
- a fallback claim type was used.

LOW versus HIGH disagreement is high severity.

For revision, each output's support strength uses evidence count and confidence. Weakly supported scores move 25% toward the peer median. The original score is stored as pre_revision_score and a revision note records the calculation.

High-severity contradictions cap final confidence at 0.55 and set requires_human_review=true.

LangGraph is used when installed. If missing, the same ordered protocol executes through the local fallback implementation.

---

## 13. Policy A-H Evidence

agents/orchestrator/policy_evidence.py creates:

| Policy | Check |
|---|---|
| A | Data provenance and captured specialist outputs |
| B | Data freshness and as-of dates |
| C | Fundamental evidence |
| D | Market volatility evidence |
| E | News sentiment evidence |
| F | Macro evidence |
| G | Agent disagreement review |
| H | Confidence and fallback disclosure |

Each check is PASS, WARN, or REVIEW with evidence text. The trail appears in JSON, PDF, and the frontend.

---

## 14. Natural-Language Chatbot

Route: POST /api/chat.

The API normalizes the ticker, resolves company and sector from Gold, and classifies the query into:

- fundamental;
- market;
- sentiment;
- macro;
- investment.

Specialist keywords are checked before broad investment words. This keeps “financial risk of Apple” focused on fundamentals even though it contains the word risk.

Focused questions run only the relevant specialist and return a focused summary. Investment questions run all four specialists and the critic. Each chat request is written to a local candidate JSONL file with query, intent, agents run, outputs, answer, and human-review status.

Route POST /api/chat/feedback stores a rating and correction for future evaluation and training-data curation.

---

## 15. Company Comparison

Routes:

~~~text
GET  /api/compare?tickers=AAPL,AMZN
POST /api/compare
~~~

A POST request can set run_debate=true. The API then:

1. normalizes and deduplicates both tickers;
2. resolves each company independently;
3. runs a separate debate for each ticker;
4. saves each debate artifact;
5. logs each run;
6. returns both final outputs for comparison.

The comparison is independent of the ticker previously selected in the browser.

---

## 16. FastAPI Backend

File: src/api/server.py

| Route | Purpose |
|---|---|
| GET / | redirect to frontend |
| GET /api/health | data and company-count health check |
| GET /api/models/training | training curves and model registry |
| POST /api/agents/market-lora/predict | direct market model test |
| POST /api/chat | natural-language answer |
| POST /api/chat/feedback | save human feedback |
| GET /api/companies | Gold-backed company list |
| GET /api/companies/{ticker} | company metadata |
| GET /api/companies/{ticker}/prices | chart price data |
| GET /api/companies/{ticker}/feature-priority | priority/freshness |
| GET /api/verdicts/{ticker} | saved verdict or Gold fallback |
| POST /api/verdict | critic/debate generation |
| GET /api/monitoring | monitoring records |
| GET /api/artifacts/{ticker}/pdf | PDF download |
| GET /docs | Swagger/OpenAPI documentation |

The API reads saved verdicts first. If no JSON or debate verdict exists, it returns a Gold-backed fallback and labels the active type as fallback. The local data directory is mounted for reports and saved artifacts.

---

## 17. Frontend Risk Console

Files:

- frontend/index.html
- frontend/app.js
- frontend/styles.css
- frontend/vendor/chart.umd.min.js

The dashboard provides:

- ticker/company search;
- All, High, Moderate, and Low filters;
- company cards;
- natural-language investment question;
- focused Ask action;
- full-agent verdict;
- debate review;
- agent scorecards;
- risk drivers and offsets;
- price/risk trace chart;
- final verdict and contradiction state;
- company comparison;
- Policy A-H evidence;
- feature priority and freshness;
- monitoring lineage;
- model-training curves;
- JSON/PDF artifact links.

The typed company is resolved before an action runs so a stale selected card cannot silently replace the user's ticker. Loading, disabled, and error states are shown in the UI.

---

## 18. Verdict and Analytics Artifacts

verdict_report.py writes:

~~~text
data/verdicts/<TICKER>_risk_verdict.json
data/verdicts/<TICKER>_risk_verdict.pdf
~~~

The JSON stores agent outputs, critic output, contradictions, confidence, evidence, and Policy A-H checks. The PDF contains a human-readable scorecard and policy evidence pages. A plain PDF writer is available when matplotlib is unavailable.

EDA creates numbered PNG reports in data/reports, including:

1. price trends;
2. volatility;
3. return distributions;
4. correlation matrix;
5. risk scores;
6. macro overlays;
7. sentiment trend.

Training and model-comparison helpers can add training-loss and model-metric charts.

---

## 19. Monitoring and Lineage

File: agents/orchestrator/monitoring.py

Each logged run records:

- timestamp;
- run type and ticker;
- final score and label;
- confidence;
- human-review flag;
- contradiction count;
- agent count;
- minimum and maximum specialist scores;
- Policy A-H statuses.

With MLflow installed, runs are logged under the financial-risk-multi-agent experiment with parameters, metrics, and artifacts. Without MLflow, records are appended to:

~~~text
data/gold/monitoring_runs.jsonl
~~~

This keeps local development auditable without requiring an MLflow server.

---

## 20. Training and Evaluation

### Fundamental FinBERT

agents/fundamental/finbert.py reads JSONL chat examples, converts SEC metrics into financial text, performs a grouped train/validation split, and trains a Hugging Face sequence classifier. Evaluation computes accuracy and macro F1. The best checkpoint is selected by validation macro F1 and saved with tokenizer and training_metrics.json.

### Market Qwen LoRA

agents/market prepares generative JSON prompts from Yahoo volatility features and Gold labels. Because the model is generative rather than a native classifier, training loss alone is not accuracy. evaluate_lora.py runs test prompts, parses generated JSON, extracts risk labels, and computes accuracy and macro F1.

The adapter can run locally or through a Colab/Ngrok endpoint. A closed endpoint activates the deterministic market fallback.

### Macro v2

agents/macro/macro_v2 contains point-in-time panel creation, ETF sensitivities, labels, model configuration, inference code, metrics, and training logs. It forecasts sector-level forward volatility.

### Training dashboard

The API reads trainer state and metrics when available. For rule-based or API-only components, it explicitly reports that no local training loss curve exists.

---

## 21. Runtime Sequence: “Can I Invest in AAPL?”

1. Browser sends ticker and question to the API.
2. API normalizes AAPL and resolves company/sector from Gold.
3. Query is classified as investment intent.
4. API loads SEC Silver, Yahoo Silver, news, Silver sentiment, and macro inputs.
5. Fundamental FinBERT runs when enabled; otherwise fundamental rules run.
6. Market LoRA is attempted remotely, then locally; rules run if unavailable.
7. Sentiment filters relevant headlines and runs ProsusAI/FinBERT.
8. VADER is used if FinBERT is unavailable; no-news fallback is used if coverage is absent.
9. Macro v2 runs when its snapshot exists; local FRED rules run otherwise.
10. Curated RAG retrieves prioritized financial definitions and thresholds.
11. Debate collects all four specialist outputs.
12. Contradictions are checked.
13. Weakly supported outputs receive one revision pass toward the peer median.
14. Claude synthesizes when configured; weighted critic synthesizes otherwise.
15. High-severity disagreement forces human review.
16. Policy A-H evidence is attached.
17. JSON is returned to the browser.
18. JSON/PDF and monitoring artifacts can be saved.
19. Frontend renders score, label, confidence, evidence, drivers, offsets, chart, and review state.

---

## 22. Reproducible Commands

~~~bash
cd /Users/varunreddyseelam/Documents/Data298A--Masters-Project
source .venv/bin/activate
python -m unittest discover -s tests -q
~~~

Small pipeline run:

~~~bash
python ingest.py AAPL MSFT NVDA
python clean.py
python eda.py
~~~

Gold rebuild without external APIs:

~~~bash
python data_preprocessing/build_gold_from_silver.py
python -m data_preprocessing.eda
~~~

One verdict:

~~~bash
python agents/verdict_report.py AAPL \
  --company "Apple Inc." \
  --sector "Information Technology" \
  --log-monitoring
~~~

Explicit debate:

~~~bash
python agents/debate_state_machine.py AAPL \
  --company "Apple Inc." \
  --sector "Information Technology" \
  --out data/verdicts/AAPL_debate_verdict.json \
  --log-monitoring
~~~

Start the application:

~~~bash
uvicorn src.api.server:app --host 127.0.0.1 --port 8766
~~~

Open:

~~~text
http://127.0.0.1:8766/frontend/index.html
http://127.0.0.1:8766/docs
~~~

News-only refresh:

~~~bash
python data_preprocessing/newsapi_only.py \
  --tickers AAPL MSFT \
  --days 3 \
  --page-size 10 \
  --sleep 5 \
  --clean
~~~

Optional Chroma index:

~~~bash
python agents/retrieval.py --build-all
python agents/retrieval.py AAPL --query "debt and leverage risk"
~~~

---

## 23. Failure and Fallback Matrix

| Failure | Result |
|---|---|
| Yahoo has no rows | skip ticker or return unavailable market output |
| SEC CIK/facts unavailable | low-confidence fundamental fallback |
| FRED key/data missing | local macro fallback or unavailable macro output |
| NewsAPI key invalid | other sources continue; news coverage warning |
| NewsAPI 429 | partial raw data saved and run stops cleanly |
| No relevant headlines | NO_NEWS_FALLBACK |
| Sentiment FinBERT unavailable | VADER baseline |
| Fundamental FinBERT unavailable | deterministic fundamental score |
| Market Colab endpoint offline | deterministic market score with reason |
| Macro v2 unavailable | local FRED rules |
| Claude unavailable | weighted critic |
| LangGraph unavailable | equivalent local state machine |
| MLflow unavailable | local JSONL monitoring |
| Matplotlib unavailable | plain PDF writer |
| Saved verdict absent | Gold-backed API fallback |

The claim_type and evidence fields disclose whether an output is model inference, derived estimate, rule estimate, or fallback.

---

## 24. Current Limitations

1. News coverage depends on provider quota and is thinner than price or SEC coverage.
2. Yahoo stock.info fundamentals on historical rows are current snapshots, not point-in-time history.
3. Market Qwen LoRA is generative JSON and requires a separate label evaluation.
4. Macro v2 is a sector systematic-risk forecast, not a full company-specific model.
5. Curated RAG is a transparent definitions/rules layer, not a substitute for validated financial literature.
6. Scores are analytical signals, not personal financial advice.
7. Providers can revise or delay observations even when an as-of date exists.
8. Local model loading can be slow on CPU/MPS; remote Colab inference depends on tunnel availability.

---

## 25. File Map

~~~text
config.py                                  configuration and paths
data_preprocessing/tickers.py              company universe
data_preprocessing/ingest.py               Yahoo, SEC, FRED, NewsAPI
data_preprocessing/clean.py                Bronze -> Silver
data_preprocessing/eda.py                  Silver -> Gold and charts
data_preprocessing/build_gold_from_silver.py Silver-only Gold rebuild
data_preprocessing/pipeline_large_scale.py sector batches and cloud load

agents/fundamental/finbert.py              Fundamental FinBERT train/inference
agents/fundamental/rule_based.py           Fundamental deterministic logic
agents/market/rule_based.py                Market deterministic logic
agents/market/finetune_lora.py             Market LoRA training
agents/market/infer_lora.py                Market LoRA inference
agents/market/evaluate_lora.py             Market LoRA evaluation
agents/sentiment/finbert.py                News FinBERT agent
agents/macro/agent.py                      Macro runtime adapter
agents/macro/macro_v2/                     Macro v2 model and artifacts

agents/shared/financial_rag.py             curated financial retrieval
agents/shared/retrieval.py                 optional Chroma retrieval
agents/evaluation/feature_priority.py      priority and freshness

agents/orchestrator/critic_agent.py        agents, critic, Claude/fallback
agents/orchestrator/debate_state_machine.py explicit debate
agents/orchestrator/policy_evidence.py     Policy A-H
agents/orchestrator/monitoring.py          MLflow/local lineage
agents/orchestrator/verdict_report.py      JSON/PDF scorecards

src/api/server.py                          FastAPI backend
frontend/index.html                        dashboard markup
frontend/app.js                            frontend state and API calls
frontend/styles.css                        dashboard styling
tests/                                     contract and priority tests
~~~

---

## 26. How to Read a Result

1. Confirm ticker and as-of dates.
2. Read final score and label.
3. Check confidence and human-review state.
4. Inspect each specialist scorecard.
5. Read claim_type to distinguish model inference from fallback.
6. Trace evidence to Silver, Gold, or headline records.
7. Review RAG terms and thresholds.
8. Read Policy A-H statuses.
9. Use JSON, PDF, MLflow, or JSONL records to reproduce the run.

The central engineering principle is traceability: every result remains connected to its source data, transformation, model or rule, evidence, confidence, freshness, and fallback path.

