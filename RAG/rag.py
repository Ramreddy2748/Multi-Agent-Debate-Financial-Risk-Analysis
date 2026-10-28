"""
RAG REQUIREMENTS FILE
=====================
Debate-Based Multi-Agent Financial Risk System
Team 3 | DATA 298B | SJSU 2026

This file defines everything the RAG system needs:
1. Knowledge base content (financial terms, rules, guidelines)
2. Data sources to index (Silver layer files)
3. Collection schema (what goes where)
4. Retrieval configuration (how to query)
5. Agent-to-collection mapping (who queries what)
6. Embedding model config
7. ChromaDB setup

Run:  python rag_requirements.py
Output: ./chroma_db/ — fully built RAG knowledge base
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import glob
import json
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# 1. EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────────────────────
# Use FinBERT — domain-specific financial embeddings
# General embedders miss financial synonyms
# "elevated leverage" and "high debt-to-equity" = same signal in FinBERT
EMBEDDING_MODEL = "ProsusAI/finbert"
EMBEDDING_DIM   = 768

# ─────────────────────────────────────────────────────────────────────────────
# 2. CHROMADB CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"

# Collections — one per purpose
COLLECTIONS = {
    "financial_kb":   "Financial terms, ratios, rules, guidelines",
    "fundamental":    "EDGAR financial statement data per ticker",
    "volatility":     "Yahoo Finance price and market risk data per ticker",
    "sentiment":      "NewsAPI headlines per ticker",
    "macro":          "FRED macroeconomic time series",
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA SOURCE PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATHS = {
    "edgar":   "data/silver/silver_edgar_*.csv",
    "prices":  "data/silver/silver_prices_*.csv",
    "macro":   "data/silver/silver_macro.csv",
    "news":    "data/bronze/newsapi_headlines_raw.csv",
    "gold":    "data/gold/gold_risk_scores_ALL.csv",
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. HIGH PRIORITY COLUMNS PER DATA SOURCE
#    These are the columns with highest predictive value for risk
#    Determined by XGBoost feature importance + mutual information
# ─────────────────────────────────────────────────────────────────────────────
HIGH_PRIORITY_COLUMNS = {

    "edgar": [
        "net_income_usd_m",      # Core profitability
        "lt_debt_usd_m",         # Long-term leverage risk
        "cash_usd_m",            # Liquidity buffer
        "total_assets_usd_m",    # Company scale
        "op_income_usd_m",       # Operating efficiency
        "revenue_ttm_usd_m",     # Top line growth
    ],

    "prices": [
        "pe_ratio",              # Valuation risk
        "beta",                  # Systematic market risk
        "rolling_vol_30d",       # Current volatility
        "debt_to_equity",        # Leverage signal
        "price_vs_52w_high_pct", # Momentum signal
        "daily_return",          # Recent performance
    ],

    "macro": [
        "fed_funds_rate",        # Cost of capital
        "treasury_10y",          # Risk-free rate benchmark
        "cpi",                   # Inflation pressure
        "unemployment",          # Economic health
        "gdp_growth",            # Economic momentum
    ],

    "news": [
        "title",                 # Headline sentiment
        "description",           # Context
        "ticker",                # Company link
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. RETRIEVAL CONFIG PER AGENT
#    Each agent queries specific collections with specific queries
# ─────────────────────────────────────────────────────────────────────────────
AGENT_RETRIEVAL_CONFIG = {

    "fundamental_agent": {
        "collections":  ["fundamental", "financial_kb"],
        "query_template": "{ticker} net income debt revenue profit margin leverage",
        "top_k":        5,
        "use_metadata_filter": True,
        "filter_field": "ticker",
        "why": "Agent 1 needs EDGAR financials + ratio interpretation rules"
    },

    "volatility_agent": {
        "collections":  ["volatility", "financial_kb"],
        "query_template": "{ticker} volatility beta drawdown Sharpe ratio market risk",
        "top_k":        5,
        "use_metadata_filter": True,
        "filter_field": "ticker",
        "why": "Agent 2 needs price features + volatility term definitions"
    },

    "sentiment_agent": {
        "collections":  ["sentiment", "financial_kb"],
        "query_template": "{ticker} news headlines sentiment positive negative",
        "top_k":        5,
        "use_metadata_filter": True,
        "filter_field": "ticker",
        "why": "Agent 3 needs headlines + fixes NO_NEWS_FALLBACK for missing tickers"
    },

    "macro_agent": {
        "collections":  ["macro", "financial_kb"],
        "query_template": "Fed rate CPI Treasury yield curve VIX unemployment recession",
        "top_k":        5,
        "use_metadata_filter": False,
        "why": "Agent 4 needs FRED data + macro risk interpretation guidelines"
    },

    "critic_agent": {
        "collections":  ["financial_kb", "fundamental", "volatility"],
        "query_template": "{ticker} confidence evidence priority policy disagreement verification",
        "top_k":        5,
        "use_metadata_filter": False,
        "why": "Critic needs Policy A-H rules + source data to verify agent claims"
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 6. FINANCIAL KNOWLEDGE BASE CONTENT
#    All terms, rules, thresholds that agents need to interpret data
# ─────────────────────────────────────────────────────────────────────────────
FINANCIAL_KB = {

    # ── FINANCIAL RATIOS ─────────────────────────────────────────────────────
    "pe_ratio": {
        "term":       "P/E ratio",
        "definition": "Price divided by earnings per share. Valuation multiple.",
        "formula":    "P/E = Stock Price / EPS",
        "thresholds": {
            "LOW":      "P/E below 15 — undervalued or low growth",
            "MODERATE": "P/E 15 to 25 — fairly valued",
            "HIGH":     "P/E above 25 — growth premium or overvalued",
            "EXTREME":  "P/E above 50 — speculative"
        },
        "risk_signal": "HIGH P/E means higher fundamental risk if earnings disappoint",
        "category":   "ratio"
    },

    "pb_ratio": {
        "term":       "P/B ratio",
        "definition": "Market cap divided by book value of equity.",
        "formula":    "P/B = Market Cap / Total Equity",
        "thresholds": {
            "LOW":      "P/B below 1 — possible undervaluation or distress",
            "MODERATE": "P/B 1 to 3 — normal range",
            "HIGH":     "P/B above 3 — premium valuation",
        },
        "risk_signal": "P/B below 1 with losses signals HIGH fundamental risk",
        "category":   "ratio"
    },

    "debt_to_equity": {
        "term":       "Debt-to-equity ratio",
        "definition": "Total debt divided by shareholder equity. Leverage measure.",
        "formula":    "D/E = Total Liabilities / Shareholders Equity",
        "thresholds": {
            "LOW":      "D/E below 0.5 — conservative leverage",
            "MODERATE": "D/E 0.5 to 1.5 — acceptable for most sectors",
            "HIGH":     "D/E 1.5 to 3.0 — high leverage",
            "EXTREME":  "D/E above 3.0 — dangerous leverage"
        },
        "risk_signal": "D/E above 2 with falling revenue is HIGH risk",
        "category":   "leverage"
    },

    "current_ratio": {
        "term":       "Current ratio",
        "definition": "Current assets divided by current liabilities. Short-term liquidity.",
        "formula":    "Current Ratio = Current Assets / Current Liabilities",
        "thresholds": {
            "HIGH":     "Below 1.0 — cannot cover short-term obligations",
            "MODERATE": "1.0 to 1.5 — tight liquidity",
            "LOW":      "1.5 to 3.0 — healthy liquidity",
        },
        "risk_signal": "Current ratio below 1 is immediate HIGH risk flag",
        "category":   "leverage"
    },

    "interest_coverage": {
        "term":       "Interest coverage ratio",
        "definition": "EBIT divided by interest expense. Ability to pay interest.",
        "formula":    "ICR = EBIT / Interest Expense",
        "thresholds": {
            "HIGH":     "ICR below 1.5 — cannot cover interest comfortably",
            "MODERATE": "ICR 1.5 to 3.0 — thin coverage",
            "LOW":      "ICR above 5.0 — strong coverage"
        },
        "risk_signal": "ICR below 2 with rising debt is HIGH risk",
        "category":   "leverage"
    },

    "debt_to_ebitda": {
        "term":       "Debt-to-EBITDA",
        "definition": "Total debt divided by EBITDA. Years of earnings to repay debt.",
        "formula":    "Debt/EBITDA = Total Debt / EBITDA",
        "thresholds": {
            "LOW":      "Below 2 — conservative",
            "MODERATE": "2 to 4 — standard investment grade",
            "HIGH":     "4 to 6 — high yield territory",
            "EXTREME":  "Above 6 — distressed"
        },
        "category":   "leverage"
    },

    # ── PROFITABILITY ─────────────────────────────────────────────────────────
    "gross_margin": {
        "term":       "Gross profit margin",
        "definition": "Revenue minus COGS divided by revenue.",
        "formula":    "Gross Margin = (Revenue - COGS) / Revenue × 100",
        "thresholds": {
            "LOW":      "Below 20% — commodity business thin margins",
            "MODERATE": "20 to 40% — typical manufacturing",
            "HIGH":     "40 to 60% — software pharma brands",
            "EXTREME":  "Above 60% — exceptional moat"
        },
        "risk_signal": "Declining gross margin 3 consecutive quarters signals pricing power loss",
        "category":   "profitability"
    },

    "operating_margin": {
        "term":       "Operating margin",
        "definition": "Operating income divided by revenue.",
        "formula":    "Op Margin = Operating Income / Revenue × 100",
        "thresholds": {
            "LOW":      "Below 5% — vulnerable to cost shocks HIGH risk",
            "MODERATE": "5 to 15% — typical retail auto",
            "HIGH":     "15 to 25% — strong tech consumer brands",
            "EXTREME":  "Above 25% — exceptional competitive position"
        },
        "risk_signal": "Operating margin below 5% with rising debt is HIGH risk",
        "category":   "profitability"
    },

    "net_margin": {
        "term":       "Net profit margin",
        "definition": "Net income divided by revenue. Bottom line profitability.",
        "formula":    "Net Margin = Net Income / Revenue × 100",
        "risk_signal": "Negative net margin 2+ years without growth story is HIGH risk",
        "category":   "profitability"
    },

    "roe": {
        "term":       "Return on Equity (ROE)",
        "definition": "Net income divided by shareholder equity.",
        "formula":    "ROE = Net Income / Shareholders Equity × 100",
        "thresholds": {
            "LOW":      "Below 5% — poor returns",
            "MODERATE": "5 to 15% — acceptable",
            "HIGH":     "15 to 25% — strong competitive advantage",
            "EXTREME":  "Above 25% — exceptional but verify not leverage-driven"
        },
        "risk_signal": "ROE declining 3 consecutive years is MODERATE to HIGH risk",
        "category":   "profitability"
    },

    "fcf": {
        "term":       "Free Cash Flow (FCF)",
        "definition": "Operating cash flow minus capital expenditures.",
        "formula":    "FCF = Operating Cash Flow - CapEx",
        "thresholds": {
            "HIGH":    "Negative FCF — burning cash",
            "MODERATE": "Positive but thin",
            "LOW":     "Strong positive FCF — LOW fundamental risk"
        },
        "risk_signal": "Negative FCF 3+ years with no profitability path is HIGH risk",
        "category":   "profitability"
    },

    # ── VOLATILITY ────────────────────────────────────────────────────────────
    "ann_volatility": {
        "term":       "Annualized volatility",
        "definition": "Standard deviation of daily returns scaled to annual frequency.",
        "formula":    "Ann Vol = std(daily_returns) × sqrt(252) × 100",
        "thresholds": {
            "LOW":      "Below 15% — utilities consumer staples LOW risk",
            "MODERATE": "15 to 25% — typical S&P 500 stock",
            "HIGH":     "25 to 40% — tech biotech elevated",
            "EXTREME":  "Above 40% — speculative HIGH risk",
            "CRISIS":   "Above 60% — extreme penny stock behavior"
        },
        "risk_signal": "Annualized volatility above 40% signals HIGH volatility risk",
        "category":   "volatility"
    },

    "beta": {
        "term":       "Beta",
        "definition": "Sensitivity of stock to market returns. Systematic risk.",
        "formula":    "Beta = Cov(stock, market) / Var(market)",
        "thresholds": {
            "DEFENSIVE": "Below 0.5 — moves less than market utilities",
            "LOW":       "0.5 to 1.0 — below market sensitivity",
            "NEUTRAL":   "1.0 — moves exactly with market",
            "MODERATE":  "1.0 to 1.5 — slightly amplified",
            "HIGH":      "Above 1.5 — high systematic risk",
            "EXTREME":   "Above 2.0 — highly speculative"
        },
        "risk_signal": "Beta above 1.5 means stock falls 50% more than market in crash",
        "category":   "volatility"
    },

    "max_drawdown": {
        "term":       "Maximum drawdown",
        "definition": "Largest peak-to-trough decline over the period.",
        "formula":    "Max DD = min((price - rolling_max) / rolling_max) × 100",
        "thresholds": {
            "LOW":      "Above -10% — mild correction normal",
            "MODERATE": "-10 to -30% — significant drawdown",
            "HIGH":     "-30 to -50% — severe HIGH risk",
            "EXTREME":  "Below -50% — catastrophic speculative"
        },
        "risk_signal": "Max drawdown below -40% with high beta is HIGH risk",
        "category":   "volatility"
    },

    "sharpe_ratio": {
        "term":       "Sharpe ratio",
        "definition": "Risk-adjusted return. Excess return per unit of volatility.",
        "formula":    "Sharpe = (Ann Return - Risk Free Rate) / Ann Volatility",
        "thresholds": {
            "HIGH":     "Below 0 — losing money risk-adjusted",
            "MODERATE": "0 to 0.5 — poor risk-adjusted returns",
            "LOW":      "0.5 to 1.0 — acceptable",
            "GOOD":     "1.0 to 2.0 — good",
            "EXCELLENT": "Above 2.0 — excellent rare in practice"
        },
        "risk_signal": "Negative Sharpe means not compensated for risk taken",
        "category":   "volatility"
    },

    "vix": {
        "term":       "VIX",
        "definition": "CBOE Volatility Index. Market 30-day implied volatility expectation.",
        "thresholds": {
            "CALM":     "Below 15 — calm market LOW macro risk",
            "NORMAL":   "15 to 20 — normal range",
            "ELEVATED": "20 to 30 — elevated uncertainty MODERATE risk",
            "HIGH":     "30 to 40 — high fear HIGH risk environment",
            "CRISIS":   "Above 40 — extreme fear crisis conditions"
        },
        "risk_signal": "VIX above 30 historically precedes continued market stress",
        "category":   "volatility"
    },

    # ── MARKET RISK ───────────────────────────────────────────────────────────
    "yield_curve": {
        "term":       "Yield curve",
        "definition": "Interest rates across maturities. Normal = upward sloping.",
        "thresholds": {
            "NORMAL":   "Short rates below long — healthy economy",
            "FLAT":     "Short and long equal — slowing growth signal",
            "INVERTED": "Short above long — recession predictor"
        },
        "risk_signal": "Inverted yield curve 10Y-2Y negative preceded every US recession since 1955",
        "category":   "macro"
    },

    "credit_spread": {
        "term":       "Credit spread",
        "definition": "Yield difference between corporate and Treasury bonds.",
        "thresholds": {
            "TIGHT":  "Below 100bps — LOW credit risk environment",
            "NORMAL": "100 to 300bps — normal range",
            "WIDE":   "300 to 500bps — elevated default risk",
            "CRISIS": "Above 500bps — systemic HIGH risk"
        },
        "risk_signal": "Widening credit spreads precede equity corrections by 2-4 weeks",
        "category":   "macro"
    },

    "tail_risk": {
        "term":       "Tail risk",
        "definition": "Risk of extreme outcomes. Fat tails beyond normal distribution.",
        "thresholds": {
            "HIGH": "Kurtosis above 3 with negative skew — HIGH tail risk"
        },
        "risk_signal": "High kurtosis with negative skew signals HIGH tail risk",
        "category":   "volatility"
    },

    # ── MACRO INDICATORS ─────────────────────────────────────────────────────
    "fed_funds_rate": {
        "term":       "Federal Funds Rate",
        "definition": "Overnight lending rate set by the Federal Reserve.",
        "thresholds": {
            "ACCOMMODATIVE": "Below 2% — LOW macro risk supports growth",
            "NEUTRAL":       "2 to 3.5% — balanced",
            "RESTRICTIVE":   "3.5 to 5% — tightening MODERATE macro risk",
            "TIGHT":         "Above 5% — highly restrictive HIGH macro risk"
        },
        "risk_signal": "Fed rate above 4.5% increases cost of capital for all companies",
        "category":   "macro"
    },

    "cpi": {
        "term":       "Consumer Price Index (CPI)",
        "definition": "Measures inflation. Fed target is 2% annual growth.",
        "thresholds": {
            "LOW":      "CPI YoY below 2% — deflationary risk",
            "TARGET":   "CPI YoY 2 to 3% — Fed target range",
            "ELEVATED": "CPI YoY 3 to 5% — inflation concern",
            "HIGH":     "CPI YoY above 5% — HIGH macro risk forces rate hikes"
        },
        "risk_signal": "CPI above 5% forces Fed to raise rates which increases equity risk",
        "category":   "macro"
    },

    "unemployment": {
        "term":       "Unemployment rate",
        "definition": "Percentage of labor force without jobs. NAIRU around 4%.",
        "thresholds": {
            "LOW":      "Below 4% — tight labor market mild inflation pressure",
            "NATURAL":  "4 to 5% — near natural rate",
            "ELEVATED": "5 to 7% — weakening economy",
            "HIGH":     "Above 7% — recession territory HIGH macro risk"
        },
        "risk_signal": "Rising unemployment above 5% combined with credit spread widening signals HIGH macro risk",
        "category":   "macro"
    },

    # ── CONFIDENCE RULES ─────────────────────────────────────────────────────
    "confidence_high": {
        "term":       "High confidence rule",
        "definition": "Confidence above 0.75 means strong evidence for claim.",
        "rule":       "Report as OBSERVED_FACT when confidence above 0.75 and primary data source used",
        "example":    "FinBERT confidence 0.89 on EDGAR data → OBSERVED_FACT",
        "category":   "confidence"
    },

    "confidence_moderate": {
        "term":       "Moderate confidence rule",
        "definition": "Confidence 0.50 to 0.75 means reasonable but not conclusive evidence.",
        "rule":       "Report as DERIVED_ESTIMATE when confidence 0.50 to 0.75",
        "example":    "Macro agent 0.70 on FRED forecast → DERIVED_ESTIMATE",
        "category":   "confidence"
    },

    "confidence_low": {
        "term":       "Low confidence rule",
        "definition": "Confidence below 0.50 means uncertain — flag for review.",
        "rule":       "Flag for human review when confidence below 0.50",
        "example":    "Sentiment agent 0 headlines → confidence 0.30 → NO_NEWS_FALLBACK",
        "category":   "confidence"
    },

    "confidence_fallback": {
        "term":       "Fallback confidence penalty",
        "definition": "Confidence reduces when fallback mode is active.",
        "rule":       "Deterministic fallback → reduce confidence by 0.20. No-data fallback → reduce by 0.40",
        "example":    "Volatility LoRA offline → rule-based → confidence 0.80 - 0.20 = 0.60",
        "category":   "confidence"
    },

    "confidence_disagreement": {
        "term":       "Agent disagreement penalty",
        "definition": "When agents disagree final confidence is reduced.",
        "rule":       "Two agents differ by more than 2 points → disagreement flag → reduce final confidence by 0.15",
        "example":    "Fundamental 2.80 vs Macro 4.90 → gap 2.10 → Policy G REVIEW",
        "category":   "confidence"
    },

    # ── EVIDENCE PRIORITY RULES (Policy A-H) ─────────────────────────────────
    "policy_a": {
        "term":       "Policy A — Data provenance",
        "definition": "Every verdict must use persisted Silver or Gold layer data.",
        "rule":       "PASS if verdict uses Silver/Gold tables. FAIL if using raw Bronze or external API directly.",
        "category":   "evidence"
    },

    "policy_b": {
        "term":       "Policy B — Data freshness",
        "definition": "Verdict must show generation date within acceptable window.",
        "rule":       "PASS if verdict generated within 30 days of latest data collection.",
        "category":   "evidence"
    },

    "policy_c": {
        "term":       "Policy C — Fundamental evidence",
        "definition": "SEC EDGAR takes priority over Yahoo Finance for fundamental claims.",
        "rule":       "If EDGAR and Yahoo Finance disagree use EDGAR as primary source.",
        "category":   "evidence"
    },

    "policy_d": {
        "term":       "Policy D — Market volatility evidence",
        "definition": "30-day rolling volatility is primary. Beta is secondary.",
        "rule":       "Report rolling_vol_30d as primary. Beta as supporting. Max drawdown as amplifier.",
        "category":   "evidence"
    },

    "policy_e": {
        "term":       "Policy E — News sentiment evidence",
        "definition": "FinBERT headline scoring takes priority over VADER when headlines available.",
        "rule":       "FinBERT when raw headlines exist → VADER Silver fallback → NO_NEWS_FALLBACK",
        "category":   "evidence"
    },

    "policy_f": {
        "term":       "Policy F — Macro evidence",
        "definition": "FRED point-in-time values only. No look-ahead bias.",
        "rule":       "Use only FRED values available on publication date. Label thresholds from training years.",
        "category":   "evidence"
    },

    "policy_g": {
        "term":       "Policy G — Agent disagreement review",
        "definition": "When agents disagree by more than 2 points flag human review.",
        "rule":       "REVIEW when any two agent labels differ. Watchlist recommendation when flagged.",
        "category":   "evidence"
    },

    "policy_h": {
        "term":       "Policy H — Confidence and fallback disclosure",
        "definition": "Final verdict must disclose confidence and any fallback mode used.",
        "rule":       "PASS if confidence stated and fallback disclosed. FAIL if fallback not disclosed.",
        "category":   "evidence"
    },

    # ── INTERPRETATION GUIDELINES ─────────────────────────────────────────────
    "low_risk": {
        "term":       "LOW risk interpretation",
        "definition": "Composite score below 3.5. Stable fundamentals low volatility positive sentiment.",
        "action":     "Research guidance: suitable for conservative portfolios. Monitor quarterly.",
        "example":    "GOOGLE 3.48 LOW — strong fundamentals dominate despite moderate macro",
        "category":   "interpretation"
    },

    "moderate_risk": {
        "term":       "MODERATE risk interpretation",
        "definition": "Composite score 3.5 to 6.0. Mixed signals. Requires further analysis.",
        "action":     "Research guidance: Watchlist. Balanced portfolios with position sizing.",
        "example":    "AAPL fundamental 4.80 MODERATE — LT debt elevated relative to cash",
        "category":   "interpretation"
    },

    "high_risk": {
        "term":       "HIGH risk interpretation",
        "definition": "Composite score above 6.0. Elevated risk in one or more dimensions.",
        "action":     "Research guidance: Caution. Not for conservative portfolios. High conviction required.",
        "example":    "TSLA volatility 7.62 HIGH — annualized volatility 55% beta 2.1",
        "category":   "interpretation"
    },

    "disagreement_flag": {
        "term":       "Disagreement flag interpretation",
        "definition": "Fundamental says LOW but volatility says HIGH — value vs risk tension.",
        "action":     "Policy G REVIEW. Human analyst must reconcile before acting.",
        "example":    "GOOGLE fundamental 2.80 LOW vs macro 4.11 MODERATE → Watchlist",
        "category":   "interpretation"
    },

    "not_financial_advice": {
        "term":       "Research guidance disclaimer",
        "definition": "All verdicts are research guidance only. Not financial advice.",
        "rule":       "Every API response must include disclaimer. Trading excluded from system boundary.",
        "category":   "interpretation"
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 7. BUILD FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def setup():
    """Initialize ChromaDB and embedding model."""
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    client   = chromadb.PersistentClient(path=CHROMA_PATH)
    cols     = {
        name: client.get_or_create_collection(name)
        for name in COLLECTIONS
    }
    print(f"ChromaDB ready at {CHROMA_PATH}")
    print(f"Collections: {list(COLLECTIONS.keys())}")
    return embedder, client, cols


def build_financial_kb(embedder, col):
    """Index all financial terms, rules, and guidelines."""
    docs, ids, metas = [], [], []
    for key, item in FINANCIAL_KB.items():
        text = f"{item['term']}: {item['definition']}"
        if 'formula' in item:
            text += f" Formula: {item['formula']}"
        if 'thresholds' in item:
            for k, v in item['thresholds'].items():
                text += f" {k}: {v}."
        if 'risk_signal' in item:
            text += f" Risk signal: {item['risk_signal']}"
        if 'rule' in item:
            text += f" Rule: {item['rule']}"
        if 'action' in item:
            text += f" Action: {item['action']}"
        if 'example' in item:
            text += f" Example: {item['example']}"
        docs.append(text)
        ids.append(key)
        metas.append({"category": item.get("category","general"), "term": item["term"]})

    embeddings = embedder.encode(docs).tolist()
    col.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
    print(f"Knowledge base: {len(docs)} financial terms indexed")


def build_fundamental_index(embedder, col):
    """Index EDGAR financial data — high priority columns only."""
    files   = glob.glob(DATA_PATHS["edgar"])
    indexed = 0
    cols    = HIGH_PRIORITY_COLUMNS["edgar"]

    for f in files:
        df     = pd.read_csv(f)
        ticker = os.path.basename(f).replace("silver_edgar_","").replace(".csv","")
        avail  = [c for c in cols if c in df.columns]
        if not avail:
            continue

        for i, (_, row) in enumerate(df.iterrows()):
            ni  = row.get("net_income_usd_m", 0)
            ltd = row.get("lt_debt_usd_m", 0)
            csh = row.get("cash_usd_m", 0)
            ta  = row.get("total_assets_usd_m", 0)
            oi  = row.get("op_income_usd_m", 0)
            rev = row.get("revenue_ttm_usd_m", oi or 1)

            pm  = ni  / rev * 100 if rev else 0
            opm = oi  / rev * 100 if rev else 0
            dr  = ltd / ta  if ta  else 0
            dc  = csh / ltd if ltd else 999

            text = (
                f"{ticker} fundamentals: "
                f"net income ${ni:.0f}M, "
                f"LT debt ${ltd:.0f}M, "
                f"cash ${csh:.0f}M, "
                f"revenue ${rev:.0f}M, "
                f"profit margin {pm:.1f}%, "
                f"operating margin {opm:.1f}%, "
                f"debt-to-assets {dr:.2f}, "
                f"debt coverage {dc:.2f}x"
            )
            col.add(
                documents=[text],
                embeddings=embedder.encode([text]).tolist(),
                ids=[f"{ticker}_edgar_{i}"],
                metadatas=[{"ticker": ticker, "data_type": "fundamental"}]
            )
            indexed += 1

    print(f"Fundamental index: {indexed} rows from {len(files)} EDGAR files")


def build_volatility_index(embedder, col):
    """Index Yahoo Finance price data — high priority columns only."""
    files   = glob.glob(DATA_PATHS["prices"])
    indexed = 0

    for f in files:
        df     = pd.read_csv(f).dropna(subset=["rolling_vol_30d","beta"])
        ticker = os.path.basename(f).replace("silver_prices_","").replace(".csv","")
        if df.empty:
            continue

        recent  = df.tail(30)
        avg_vol = recent["rolling_vol_30d"].mean() * (252**0.5) * 100
        avg_bet = recent["beta"].mean()
        avg_pe  = recent["pe_ratio"].mean() if "pe_ratio" in df.columns else 0
        avg_de  = recent["debt_to_equity"].mean() if "debt_to_equity" in df.columns else 0
        avg_52  = recent["price_vs_52w_high_pct"].mean() if "price_vs_52w_high_pct" in df.columns else 0

        rets = df["daily_return"].fillna(0)
        cum  = (1 + rets).cumprod()
        mx   = cum.expanding().max()
        dd   = ((cum - mx) / mx).min() * 100

        text = (
            f"{ticker} market risk: "
            f"annualized volatility {avg_vol:.1f}%, "
            f"beta {avg_bet:.2f}, "
            f"P/E ratio {avg_pe:.1f}, "
            f"debt-to-equity {avg_de:.2f}, "
            f"max drawdown {dd:.1f}%, "
            f"price vs 52w high {avg_52:.1f}%"
        )
        col.add(
            documents=[text],
            embeddings=embedder.encode([text]).tolist(),
            ids=[f"{ticker}_vol"],
            metadatas=[{"ticker": ticker, "data_type": "volatility"}]
        )
        indexed += 1

    print(f"Volatility index: {indexed} tickers from {len(files)} price files")


def build_sentiment_index(embedder, col):
    """Index NewsAPI headlines — fixes NO_NEWS_FALLBACK."""
    try:
        df = pd.read_csv(DATA_PATHS["news"])
    except FileNotFoundError:
        print("No news file found — skipping sentiment index")
        return

    df["text"] = (df["title"].fillna("") + " " + df["description"].fillna("")).str.strip()
    indexed    = 0

    for ticker, group in df.groupby("ticker"):
        headlines = group["text"].tolist()
        combined  = f"{ticker} news: " + " | ".join(headlines[:10])
        col.add(
            documents=[combined],
            embeddings=embedder.encode([combined]).tolist(),
            ids=[f"{ticker}_news"],
            metadatas=[{"ticker": ticker, "data_type": "sentiment"}]
        )
        indexed += 1

    print(f"Sentiment index: {indexed} tickers indexed from {len(df)} headlines")


def build_macro_index(embedder, col):
    """Index FRED macro data — high priority columns only."""
    try:
        df = pd.read_csv(DATA_PATHS["macro"])
    except FileNotFoundError:
        print("No macro file found — skipping macro index")
        return

    cols    = HIGH_PRIORITY_COLUMNS["macro"]
    indexed = 0

    for _, row in df.iterrows():
        avail = {c: row.get(c, 0) for c in cols if c in df.columns}
        text  = f"Macro snapshot {row.get('date','')}: "
        text += " ".join([f"{k} {v:.2f}" for k, v in avail.items()])

        col.add(
            documents=[text],
            embeddings=embedder.encode([text]).tolist(),
            ids=[f"macro_{_}"],
            metadatas=[{"data_type": "macro"}]
        )
        indexed += 1

    print(f"Macro index: {indexed} rows from FRED data")


# ─────────────────────────────────────────────────────────────────────────────
# 8. RETRIEVAL FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def retrieve(query, collection, embedder, k=5, ticker=None, data_type=None):
    """
    Retrieve top-k relevant chunks from a collection.

    Args:
        query:       natural language query
        collection:  ChromaDB collection object
        embedder:    SentenceTransformer model
        k:           number of chunks to return
        ticker:      optional — filter to specific ticker only
        data_type:   optional — filter to specific data type

    Returns:
        list of document strings
    """
    embedding = embedder.encode([query]).tolist()
    where     = {}
    if ticker:    where["ticker"]    = ticker
    if data_type: where["data_type"] = data_type

    results = collection.query(
        query_embeddings=embedding,
        n_results=k,
        where=where if where else None
    )
    return results["documents"][0]


# ─────────────────────────────────────────────────────────────────────────────
# 9. AGENT PROMPT BUILDERS
#    Each agent gets KB context + company data
# ─────────────────────────────────────────────────────────────────────────────
def build_fundamental_prompt(ticker, cols, embedder):
    cfg     = AGENT_RETRIEVAL_CONFIG["fundamental_agent"]
    query   = cfg["query_template"].format(ticker=ticker)
    kb_ctx  = "\n".join(retrieve(query, cols["financial_kb"], embedder, k=3))
    dat_ctx = "\n".join(retrieve(query, cols["fundamental"],  embedder, k=5, ticker=ticker))

    return f"""Financial knowledge base:
{kb_ctx}

Company data for {ticker}:
{dat_ctx}

Using the ratio thresholds above classify {ticker} fundamental risk.
Cite which thresholds were crossed.
Return JSON: score 1-10, label LOW/MODERATE/HIGH, confidence, evidence list, claim_type."""


def build_volatility_prompt(ticker, cols, embedder):
    cfg     = AGENT_RETRIEVAL_CONFIG["volatility_agent"]
    query   = cfg["query_template"].format(ticker=ticker)
    kb_ctx  = "\n".join(retrieve(query, cols["financial_kb"], embedder, k=3))
    dat_ctx = "\n".join(retrieve(query, cols["volatility"],   embedder, k=5, ticker=ticker))

    return f"""Volatility knowledge base:
{kb_ctx}

Market data for {ticker}:
{dat_ctx}

Using the volatility thresholds above classify {ticker} market risk.
Return JSON: score 1-10, label, confidence, evidence list, claim_type."""


def build_sentiment_prompt(ticker, cols, embedder):
    cfg     = AGENT_RETRIEVAL_CONFIG["sentiment_agent"]
    query   = cfg["query_template"].format(ticker=ticker)
    kb_ctx  = "\n".join(retrieve(query, cols["financial_kb"], embedder, k=2))
    dat_ctx = "\n".join(retrieve(query, cols["sentiment"],    embedder, k=5, ticker=ticker))

    return f"""Sentiment knowledge base:
{kb_ctx}

News context for {ticker}:
{dat_ctx if dat_ctx else "NO_NEWS_FALLBACK — no headlines available"}

Score {ticker} sentiment risk based on retrieved headlines.
Return JSON: score 1-10, label, confidence, positive_count, negative_count, evidence list."""


def build_macro_prompt(sector, cols, embedder):
    cfg     = AGENT_RETRIEVAL_CONFIG["macro_agent"]
    query   = cfg["query_template"]
    kb_ctx  = "\n".join(retrieve(query, cols["financial_kb"], embedder, k=3))
    dat_ctx = "\n".join(retrieve(query, cols["macro"],        embedder, k=5))

    return f"""Macro knowledge base:
{kb_ctx}

Current macro environment:
{dat_ctx}

Forecast 60-day volatility risk for {sector} sector.
Apply the macro thresholds above.
Return JSON: score 1-10, label, confidence, macro_regime, evidence list."""


def build_critic_prompt(ticker, agent_outputs, cols, embedder):
    cfg    = AGENT_RETRIEVAL_CONFIG["critic_agent"]
    query  = cfg["query_template"].format(ticker=ticker)
    kb_ctx = "\n".join(retrieve(query, cols["financial_kb"], embedder, k=5))

    return f"""Policy A-H rules and confidence thresholds:
{kb_ctx}

Agent outputs for {ticker}:
{json.dumps(agent_outputs, indent=2)}

Apply Policy A-H rules above.
1. Check data provenance (Policy A)
2. Check confidence thresholds
3. Flag any agent disagreement above 2 points (Policy G)
4. Disclose any fallback usage (Policy H)
Return JSON: final_score, label, confidence, policies_passed, policies_flagged, disagreement_flag."""


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN — BUILD EVERYTHING
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RAG REQUIREMENTS — Building knowledge base")
    print("=" * 60)

    embedder, client, cols = setup()

    print("\nStep 1 — Financial knowledge base")
    build_financial_kb(embedder, cols["financial_kb"])

    print("\nStep 2 — Fundamental index (EDGAR)")
    build_fundamental_index(embedder, cols["fundamental"])

    print("\nStep 3 — Volatility index (Yahoo Finance)")
    build_volatility_index(embedder, cols["volatility"])

    print("\nStep 4 — Sentiment index (NewsAPI)")
    build_sentiment_index(embedder, cols["sentiment"])

    print("\nStep 5 — Macro index (FRED)")
    build_macro_index(embedder, cols["macro"])

    print("\n" + "=" * 60)
    print("RAG COMPLETE — All collections ready")
    print("=" * 60)
    for name, desc in COLLECTIONS.items():
        col   = cols[name]
        count = col.count()
        print(f"  {name:20s} → {count:5d} documents — {desc}")

    print("\nUsage:")
    print("  from rag_requirements import setup, retrieve, build_fundamental_prompt")
    print("  embedder, client, cols = setup()")
    print("  prompt = build_fundamental_prompt('AAPL', cols, embedder)")