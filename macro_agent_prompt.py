"""
macro_agent_prompt.py
─────────────────────────────────────────────────────────────────
MACRO-ECONOMIC AGENT — Prompt-Based Version
─────────────────────────────────────────────────────────────────
"""

import os
import json
import time
import logging
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] — %(message)s"
)
log = logging.getLogger("macro_agent")

# ── DeepSeek Client ───────────────────────────────────────────
def get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set in .env file")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

# ── Load FRED Data ────────────────────────────────────────────
def load_macro_data(path: str = "data/silver/silver_macro_enhanced.csv") -> dict:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df.sort_index().ffill(limit=5).dropna()

    latest     = df.iloc[-1]
    prev_year  = df.iloc[-252] if len(df) >= 252 else df.iloc[0]
    prev_month = df.iloc[-22]  if len(df) >= 22  else df.iloc[0]

    def safe_get(col, default=0.0):
        try:
            return float(latest[col])
        except Exception:
            return default

    def yoy_change(col):
        try:
            curr = float(latest[col])
            prev = float(prev_year[col])
            return round((curr - prev) / abs(prev) * 100, 2)
        except Exception:
            return 0.0

    def mom_change(col):
        try:
            return round(float(latest[col]) - float(prev_month[col]), 4)
        except Exception:
            return 0.0

    fed        = safe_get("FEDFUNDS")
    cpi        = safe_get("CPIAUCSL")
    t10y2y     = safe_get("T10Y2Y")
    vix        = safe_get("VIXCLS")
    oil        = safe_get("DCOILWTICO")
    unemp      = safe_get("UNRATE")
    gdp        = safe_get("GDP")
    vix_30d    = float(df["VIXCLS"].tail(30).mean())
    vix_90d    = float(df["VIXCLS"].tail(90).mean())
    fed_6m_ago = float(df["FEDFUNDS"].iloc[-132]) if len(df) >= 132 else fed
    fed_trend  = "rising" if fed > fed_6m_ago else "falling" if fed < fed_6m_ago else "stable"
    yield_trend= "steepening" if mom_change("T10Y2Y") > 0 else "flattening"

    return {
        "as_of_date":       str(df.index[-1].date()),
        "fed_funds_rate":   fed,
        "fed_trend":        fed_trend,
        "fed_6m_change":    round(fed - fed_6m_ago, 2),
        "cpi":              cpi,
        "cpi_yoy_pct":      yoy_change("CPIAUCSL"),
        "t10y2y_spread":    t10y2y,
        "yield_trend":      yield_trend,
        "recession_signal": t10y2y < 0,
        "vix":              vix,
        "vix_30d_avg":      round(vix_30d, 2),
        "vix_90d_avg":      round(vix_90d, 2),
        "vix_vs_avg":       "elevated" if vix > vix_30d * 1.2 else "normal",
        "oil_price":        oil,
        "oil_yoy_pct":      yoy_change("DCOILWTICO"),
        "unemployment":     unemp,
        "gdp":              gdp,
        "gdp_yoy_pct":      yoy_change("GDP"),
    }

# ── Sector Sensitivity ────────────────────────────────────────
SECTOR_SENSITIVITY = {
    "Real Estate":            {"rate_sensitivity": "VERY HIGH",     "reason": "REITs are heavily debt-financed. Rising rates increase refinancing costs and make bond yields competitive vs REIT dividends.", "oil_sensitivity": "LOW",       "vix_sensitivity": "MODERATE"},
    "Utilities":              {"rate_sensitivity": "HIGH",          "reason": "Capital-intensive sector with high debt loads. Rising rates compress margins.", "oil_sensitivity": "MODERATE",  "vix_sensitivity": "LOW"},
    "Financials":             {"rate_sensitivity": "MODERATE",      "reason": "Banks benefit from higher net interest margins but face credit risk if rates rise too fast.", "oil_sensitivity": "LOW",       "vix_sensitivity": "HIGH"},
    "Information Technology": {"rate_sensitivity": "MODERATE",      "reason": "High-growth tech valuations are compressed by rising discount rates.", "oil_sensitivity": "LOW",       "vix_sensitivity": "HIGH"},
    "Consumer Discretionary": {"rate_sensitivity": "MODERATE-HIGH", "reason": "Higher rates reduce consumer borrowing power. Unemployment directly affects spending.", "oil_sensitivity": "MODERATE",  "vix_sensitivity": "MODERATE"},
    "Consumer Staples":       {"rate_sensitivity": "LOW",           "reason": "Defensive sector with inelastic demand.", "oil_sensitivity": "LOW",       "vix_sensitivity": "LOW"},
    "Energy":                 {"rate_sensitivity": "LOW",           "reason": "Energy companies benefit from inflation. Oil price is the primary driver.", "oil_sensitivity": "VERY HIGH", "vix_sensitivity": "MODERATE"},
    "Health Care":            {"rate_sensitivity": "LOW-MODERATE",  "reason": "Defensive sector. Demand is largely inelastic.", "oil_sensitivity": "LOW",       "vix_sensitivity": "LOW"},
    "Industrials":            {"rate_sensitivity": "MODERATE",      "reason": "Sensitive to GDP growth. Higher rates slow capital investment.", "oil_sensitivity": "HIGH",      "vix_sensitivity": "MODERATE"},
    "Materials":              {"rate_sensitivity": "MODERATE",      "reason": "Commodity prices driven more by global demand. Inflation can boost revenue.", "oil_sensitivity": "HIGH",      "vix_sensitivity": "MODERATE"},
    "Communication Services": {"rate_sensitivity": "LOW-MODERATE",  "reason": "Mix of defensive telecom and growth media names. Relatively resilient.", "oil_sensitivity": "LOW",       "vix_sensitivity": "MODERATE"},
    "General":                {"rate_sensitivity": "MODERATE",      "reason": "Standard sensitivity to macro conditions.", "oil_sensitivity": "MODERATE",  "vix_sensitivity": "MODERATE"},
}

def get_sector_context(sector: str) -> str:
    info = SECTOR_SENSITIVITY.get(sector, SECTOR_SENSITIVITY["General"])
    return (
        f"Rate sensitivity  : {info['rate_sensitivity']}\n"
        f"Why               : {info['reason']}\n"
        f"Oil sensitivity   : {info['oil_sensitivity']}\n"
        f"VIX sensitivity   : {info['vix_sensitivity']}"
    )

# ── System Prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are the Macro-Economic Agent in a structured financial
risk debate system. Assess how the current macroeconomic
environment affects the investment risk of a specific company.

Scoring guide:
1.0 - 3.5  : LOW macro risk
3.5 - 6.0  : MODERATE macro risk
6.0 - 10.0 : HIGH macro risk

Rules:
- Every evidence point must cite a specific number
- Be sector-specific
- Return ONLY valid JSON — no extra text, no markdown
"""

# ── Build Prompt ──────────────────────────────────────────────
def build_prompt(ticker: str, sector: str,
                 company_name: str, macro: dict) -> str:

    recession_note = (
        "YIELD CURVE INVERTED — recession signal"
        if macro["recession_signal"] else
        "Yield curve positive — no inversion"
    )
    sector_context = get_sector_context(sector)

    return f"""
Analyse macroeconomic investment risk for:
Company : {company_name} ({ticker})
Sector  : {sector}

MACRO ENVIRONMENT (as of {macro['as_of_date']})
Fed Funds Rate  : {macro['fed_funds_rate']:.2f}% ({macro['fed_trend']}, {macro['fed_6m_change']:+.2f}% over 6 months)
CPI Index       : {macro['cpi']:.3f} (YoY: {macro['cpi_yoy_pct']:+.2f}%)
10Y-2Y Spread   : {macro['t10y2y_spread']:.3f}% — {recession_note}
VIX             : {macro['vix']:.2f} (30d avg: {macro['vix_30d_avg']:.2f})
WTI Oil         : ${macro['oil_price']:.2f}/barrel (YoY: {macro['oil_yoy_pct']:+.2f}%)
Unemployment    : {macro['unemployment']:.1f}%
GDP YoY         : {macro['gdp_yoy_pct']:+.2f}%

SECTOR SENSITIVITY: {sector}
{sector_context}

Return ONLY this exact JSON:
{{
  "agent": "Macro-Economic Agent",
  "ticker": "{ticker}",
  "company": "{company_name}",
  "sector": "{sector}",
  "as_of_date": "{macro['as_of_date']}",
  "macro_risk_score": <number 1.0 to 10.0>,
  "risk_label": "<LOW or MODERATE or HIGH>",
  "claim_type": "DERIVED_ESTIMATE",
  "recession_signal": <true or false>,
  "evidence": ["<fact with number>", "<fact with number>", "<fact with number>"],
  "top_risk_factors": ["<factor 1>", "<factor 2>", "<factor 3>"],
  "sector_sensitivity": "<one sentence>",
  "indicator_interactions": "<one sentence>",
  "justification": "<2 sentences maximum>",
  "confidence": <0.0 to 1.0>
}}
"""

# ── Parse Response ────────────────────────────────────────────
def parse_response(raw: str, ticker: str, sector: str) -> dict:
    try:
        cleaned = raw.strip()
        if "```" in cleaned:
            for part in cleaned.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    cleaned = part
                    break
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.warning(f"  JSON parse failed: {e}")
        return _fallback(ticker, sector, error=str(e))

def _fallback(ticker: str, sector: str, error: str = "") -> dict:
    return {
        "agent":                  "Macro-Economic Agent",
        "ticker":                 ticker,
        "sector":                 sector,
        "macro_risk_score":       5.85,
        "risk_label":             "MODERATE",
        "claim_type":             "DERIVED_ESTIMATE",
        "recession_signal":       False,
        "evidence":               [f"Fallback: {error}"],
        "top_risk_factors":       ["API unavailable"],
        "sector_sensitivity":     "Unknown",
        "indicator_interactions": "Not computed",
        "justification":          "Fallback score used.",
        "confidence":             0.0,
        "error":                  error,
    }

# ── Single Company Agent ──────────────────────────────────────
def run_macro_agent(
    ticker:       str,
    sector:       str,
    company_name: str = None,
    macro_path:   str = "data/silver/silver_macro_enhanced.csv",
    max_retries:  int = 3
) -> dict:
    # Fix NaN sector
    if not isinstance(sector, str) or sector != sector or sector == "nan":
        sector = "General"

    name = company_name or ticker
    log.info(f"[Macro Agent] Analysing {name} ({ticker}) | {sector}")

    try:
        macro = load_macro_data(macro_path)
        log.info(f"  Macro data loaded — as of {macro['as_of_date']}")
    except Exception as e:
        log.error(f"  Failed to load macro data: {e}")
        return _fallback(ticker, sector, error=str(e))

    prompt = build_prompt(ticker, sector, name, macro)
    client = get_client()

    for attempt in range(1, max_retries + 1):
        try:
            log.info(f"  API call attempt {attempt}/{max_retries}...")
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            raw    = response.choices[0].message.content
            result = parse_response(raw, ticker, sector)

            # Clamp score
            score = float(result.get("macro_risk_score", 5.0))
            score = max(1.0, min(10.0, score))
            result["macro_risk_score"] = round(score, 2)

            # Fix label
            result["risk_label"] = (
                "HIGH"     if score > 6.0 else
                "MODERATE" if score > 3.5 else
                "LOW"
            )

            log.info(
                f"  ✅ {ticker} → "
                f"score={result['macro_risk_score']}/10 | "
                f"label={result['risk_label']} | "
                f"recession={result.get('recession_signal', False)} | "
                f"confidence={result.get('confidence', 0):.2f}"
            )
            return result

        except Exception as e:
            log.warning(f"  Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    return _fallback(ticker, sector, error="Max retries exceeded")

# ── Batch Runner ──────────────────────────────────────────────
def run_macro_agent_batch(
    companies:     list,
    macro_path:    str   = "data/silver/silver_macro_enhanced.csv",
    delay_seconds: float = 1.5,
    save_path:     str   = "data/gold/macro_agent_outputs.json"
) -> list:

    # Resume from checkpoint
    existing_tickers = set()
    if os.path.exists(save_path):
        with open(save_path) as f:
            existing = json.load(f)
        existing_tickers = {r["ticker"] for r in existing}
        results = existing
        log.info(f"Resuming — {len(existing_tickers)} already done")
    else:
        results = []

    remaining = [c for c in companies
                 if c["ticker"] not in existing_tickers]
    log.info(f"Remaining: {len(remaining)} companies")

    if not remaining:
        log.info("All companies already processed!")
        return results

    succeeded = 0
    failed    = 0

    log.info(f"[Macro Agent] Batch starting — {len(remaining)} companies")
    log.info(f"  Estimated time: {len(remaining) * delay_seconds / 60:.1f} minutes")

    for i, company in enumerate(remaining, 1):
        ticker = company["ticker"]
        sector = company["sector"]
        name   = company.get("company_name", ticker)

        log.info(f"\n  [{i}/{len(remaining)}] {ticker} — {sector}")

        result = run_macro_agent(
            ticker=ticker,
            sector=sector,
            company_name=name,
            macro_path=macro_path
        )
        results.append(result)

        if "error" not in result:
            succeeded += 1
        else:
            failed += 1

        # Save checkpoint every 10 companies
        if i % 10 == 0:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)
            log.info(f"  Checkpoint saved — {len(results)} total")

        if i < len(remaining):
            time.sleep(delay_seconds)

    # Final save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    scores = [r["macro_risk_score"] for r in results if "error" not in r]
    labels = [r["risk_label"]       for r in results if "error" not in r]

    log.info("\n" + "="*55)
    log.info("  BATCH COMPLETE")
    log.info("="*55)
    log.info(f"  Total saved  : {len(results)}")
    log.info(f"  Succeeded    : {succeeded}/{len(remaining)}")
    log.info(f"  Failed       : {failed}/{len(remaining)}")
    if scores:
        log.info(f"  Avg score    : {np.mean(scores):.2f}/10")
        log.info(f"  HIGH         : {labels.count('HIGH')}")
        log.info(f"  MODERATE     : {labels.count('MODERATE')}")
        log.info(f"  LOW          : {labels.count('LOW')}")
    log.info(f"  Saved to     : {save_path}")

    return results

# ── Print Result ──────────────────────────────────────────────
def print_result(result: dict):
    print("\n" + "="*55)
    print(f"  MACRO AGENT — {result.get('ticker')} ({result.get('sector')})")
    print("="*55)
    print(f"  Risk Score   : {result.get('macro_risk_score')}/10")
    print(f"  Risk Label   : {result.get('risk_label')}")
    print(f"  Recession    : {result.get('recession_signal')}")
    print(f"  Confidence   : {result.get('confidence', 0):.2f}")
    print(f"\n  Evidence:")
    for ev in result.get("evidence", []):
        print(f"    • {ev}")
    print(f"\n  Justification:")
    print(f"    {result.get('justification', '')}")
    print("="*55)

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "="*55)
    print("  MACRO AGENT — BATCH RUN")
    print("="*55 + "\n")

    gold_df = pd.read_csv(
        "data/gold/gold_risk_scores_ALL.csv",
        index_col=0
    )
    gold_df["sector"] = gold_df["sector"].replace(
        "Technology", "Information Technology"
    )

    companies = [
        {
            "ticker":       ticker,
            "sector":       str(row["sector"])
                            if str(row["sector"]) != "nan"
                            else "General",
            "company_name": ticker
        }
        for ticker, row in gold_df.iterrows()
    ]

    print(f"Total companies : {len(companies)}")
    print("Starting batch (resumes from checkpoint automatically)...\n")

    results = run_macro_agent_batch(companies)

    print(f"\nDone! {len(results)} results saved to "
          f"data/gold/macro_agent_outputs.json")