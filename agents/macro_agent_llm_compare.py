"""
macro_agent_llm_compare.py

LLM-based Macro-Economic Agent — multi-model comparison variant.

This is separate from the *live* agents/macro_agent.py, which is imported
directly by agents/critic_agent.py and already has real DeepSeek-produced
output on disk (data/gold/macro_agent_outputs.json) — that file is
untouched by this one. This module reuses its FRED-data loading and sector-
sensitivity context (load_macro_data, get_sector_context) but swaps the
single hardcoded DeepSeek call for the shared open-source multi-LLM
comparison harness in agents/llm_comparison.py (shared with
fundamental_agent_llm.py and market_sentiment_agent_llm.py).

Usage:
  python agents/macro_agent_llm_compare.py AAPL MSFT JPM
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from agents import llm_comparison
from agents.critic_agent import _latest_row, _read_csv
from agents.macro_agent import get_sector_context, load_macro_data
from agents.retrieval import retrieve

AGENT_LABEL = "macro"

SYSTEM_PROMPT = """
You are the Macro-Economic Agent in a structured financial risk debate
system. Assess how the current macroeconomic environment affects the
investment risk of a specific company, given its sector's sensitivity to
rates, oil, and volatility.

Scoring guide:
1.0 - 3.5  : LOW macro risk
3.5 - 6.0  : MODERATE macro risk
6.0 - 10.0 : HIGH macro risk

Rules:
- Every evidence point must cite a specific number from the data provided
- Be sector-specific
- Return ONLY valid JSON — no extra text, no markdown
"""

JUDGE_SYSTEM_PROMPT = """
You are a neutral evaluator for a financial risk debate system. You judge
whether another agent's macro-risk assessment is actually grounded in the
retrieved source evidence and macro data provided, and whether its stated
score is internally consistent with its own evidence. You are not judging
writing quality.

Return ONLY valid JSON — no extra text, no markdown.
"""


def _company_and_sector(ticker: str) -> tuple[str, str]:
    rows = _read_csv(SILVER_DIR / f"silver_prices_{ticker}.csv")
    row = _latest_row(rows, ("date",))
    if not row:
        return ticker, "General"
    sector = row.get("sector") or "General"
    if sector == "Technology":
        sector = "Information Technology"
    return row.get("long_name") or ticker, sector


def _format_chunks(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "(no retrieved evidence available)"
    return "\n".join(f"- {chunk['text']}" for chunk in chunks)


def context_loader(ticker: str) -> dict[str, Any]:
    company, sector = _company_and_sector(ticker)
    macro = load_macro_data(str(SILVER_DIR / "silver_macro.csv"))
    sector_context = get_sector_context(sector)
    chunks = retrieve(ticker, "macro economic exposure and sector risk", k=5)
    return {
        "company": company,
        "sector": sector,
        "macro": macro,
        "sector_context": sector_context,
        "chunks": chunks,
    }


def prompt_builder(ticker: str, context: dict[str, Any]) -> str:
    macro = context["macro"]
    recession_note = (
        "YIELD CURVE INVERTED — recession signal"
        if macro["recession_signal"] else
        "Yield curve positive — no inversion"
    )
    return f"""
Analyse macroeconomic investment risk for:
Company : {context['company']} ({ticker})
Sector  : {context['sector']}

MACRO ENVIRONMENT (as of {macro['as_of_date']})
Fed Funds Rate  : {macro['fed_funds_rate']:.2f}% ({macro['fed_trend']}, {macro['fed_6m_change']:+.2f}% over 6 months)
CPI Index       : {macro['cpi']:.3f} (YoY: {macro['cpi_yoy_pct']:+.2f}%)
10Y-2Y Spread   : {macro['t10y2y_spread']:.3f}% — {recession_note}
VIX             : {macro['vix']:.2f} (30d avg: {macro['vix_30d_avg']:.2f})
WTI Oil         : ${macro['oil_price']:.2f}/barrel (YoY: {macro['oil_yoy_pct']:+.2f}%)
Unemployment    : {macro['unemployment']:.1f}%
GDP YoY         : {macro['gdp_yoy_pct']:+.2f}%

SECTOR SENSITIVITY: {context['sector']}
{context['sector_context']}

RETRIEVED EVIDENCE (top matches for "macro economic exposure and sector risk")
{_format_chunks(context['chunks'])}

Return ONLY this exact JSON:
{{
  "agent": "Macro-Economic Agent",
  "ticker": "{ticker}",
  "company": "{context['company']}",
  "sector": "{context['sector']}",
  "risk_score": <number 1.0 to 10.0>,
  "risk_label": "<LOW or MODERATE or HIGH>",
  "confidence": "<LOW or MEDIUM or HIGH>",
  "recession_signal": <true or false>,
  "evidence": ["<fact with number>", "<fact with number>", "<fact with number>"],
  "top_risk_factors": ["<factor 1>", "<factor 2>", "<factor 3>"],
  "overall_assessment": "<2 sentences maximum>"
}}
"""


def judge_prompt_builder_factory(ticker: str, context: dict[str, Any]):
    chunks = context["chunks"]

    def _judge_prompt(candidate: dict[str, Any]) -> str:
        return f"""
Evaluate this macro-risk assessment for {ticker} against the retrieved
source evidence below. Judge groundedness (is every claim actually supported
by the retrieved evidence or macro data?) and consistency (does the evidence
support the stated risk_score and risk_label?).

CANDIDATE ASSESSMENT
{json.dumps(candidate, indent=2)}

RETRIEVED SOURCE EVIDENCE
{_format_chunks(chunks)}

Return ONLY this exact JSON:
{{
  "groundedness": <number 1.0 to 10.0>,
  "consistency": <number 1.0 to 10.0>,
  "overall": <number 1.0 to 10.0>,
  "rationale": "<2 sentences maximum>"
}}
"""

    return _judge_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the Macro-Economic Agent across open-source LLMs.")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols, for example AAPL MSFT JPM")
    parser.add_argument("--models", nargs="+", default=list(llm_comparison.MODELS.keys()),
                         choices=list(llm_comparison.MODELS.keys()),
                         help="Model keys to compare (default: all 4).")
    parser.add_argument("--out", default=str(GOLD_DIR / "macro_llm_comparison.json"),
                         help="Output JSON path.")
    args = parser.parse_args()

    llm_comparison.compare_models(
        AGENT_LABEL,
        args.tickers,
        context_loader,
        prompt_builder,
        judge_prompt_builder_factory,
        SYSTEM_PROMPT,
        JUDGE_SYSTEM_PROMPT,
        models=args.models,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
