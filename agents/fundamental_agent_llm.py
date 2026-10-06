"""
fundamental_agent_llm.py

LLM-based Fundamental Agent — comparison variant.

The production Fundamental Agent (agents/fundamental_agent.py) is fully
deterministic. This module is a *separate* experimental agent that answers
the same question (fundamental financial risk for a ticker) using an LLM
instead of hand-tuned ratio math, so open-weight models can be run side by
side and compared.

Provider/client/retry/judge-panel plumbing lives in agents/llm_comparison.py
and is shared with market_sentiment_agent_llm.py and
macro_agent_llm_compare.py. This file only supplies what's fundamental-
specific: EDGAR context loading, the candidate prompt, and the judge prompt.

Usage:
  python agents/fundamental_agent_llm.py AAPL MSFT JPM
  python agents/fundamental_agent_llm.py AAPL MSFT JPM --models groq_gptoss120b openrouter_qwen27b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from agents import llm_comparison
from agents.critic_agent import SILVER_DIR, _latest_row, _read_csv, _safe_float, _usd_m
from agents.retrieval import retrieve

AGENT_LABEL = "fundamental"

SYSTEM_PROMPT = """
You are the Fundamental Analysis Agent in a structured financial risk debate
system. Assess a company's fundamental financial risk (leverage, liquidity,
profitability) using only the filing data and retrieved evidence provided.

Scoring guide:
1.0 - 3.5  : LOW fundamental risk
3.5 - 6.0  : MODERATE fundamental risk
6.0 - 10.0 : HIGH fundamental risk

Rules:
- Every evidence point must cite a specific number from the data provided
- Do not invent numbers that are not present in the data
- Return ONLY valid JSON — no extra text, no markdown
"""

JUDGE_SYSTEM_PROMPT = """
You are a neutral evaluator for a financial risk debate system. You judge
whether another agent's fundamental-risk assessment is actually grounded in
the retrieved source evidence, and whether its stated score is internally
consistent with its own evidence. You are not judging writing quality.

Return ONLY valid JSON — no extra text, no markdown.
"""


def _load_edgar_context(ticker: str) -> dict[str, Any]:
    rows = _read_csv(SILVER_DIR / f"silver_edgar_{ticker}.csv")
    row = _latest_row(rows, ("end_date", "filed"))
    if not row:
        return {}
    return {
        "end_date": row.get("end_date"),
        "form": row.get("form"),
        "filed": row.get("filed"),
        "company": row.get("company"),
        "cash_usd_m": _usd_m(row.get("cash_usd_m")),
        "long_term_debt_usd_m": _usd_m(row.get("long_term_debt")),
        "net_income_usd_m": _usd_m(row.get("net_income_usd_m")),
        "operating_income_usd_m": _usd_m(row.get("operating_income_usd_m")),
        "total_assets_usd_m": _safe_float(row.get("total_assets_usd_m"), 0.0) or 0.0,
        "total_liabilities_usd_m": _safe_float(row.get("total_liabilities_usd_m"), 0.0) or 0.0,
        "stockholders_equity_usd_m": _usd_m(row.get("stockholders_equity_usd_m")),
    }


def _format_chunks(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "(no retrieved evidence available)"
    return "\n".join(f"- {chunk['text']}" for chunk in chunks)


def context_loader(ticker: str) -> dict[str, Any]:
    edgar = _load_edgar_context(ticker)
    chunks = retrieve(ticker, "leverage and profitability risk", k=5)
    return {
        "company": edgar.get("company") or ticker,
        "edgar": edgar,
        "chunks": chunks,
    }


def prompt_builder(ticker: str, context: dict[str, Any]) -> str:
    edgar = context["edgar"]
    filing_block = (
        "\n".join(f"{key}: {value}" for key, value in edgar.items())
        if edgar
        else "(no EDGAR filing data available for this ticker)"
    )
    return f"""
Analyse fundamental financial risk for:
Company : {context['company']} ({ticker})

LATEST FILING DATA
{filing_block}

RETRIEVED EVIDENCE (top matches for "leverage and profitability risk")
{_format_chunks(context['chunks'])}

Return ONLY this exact JSON:
{{
  "agent": "Fundamental Agent",
  "ticker": "{ticker}",
  "company": "{context['company']}",
  "risk_score": <number 1.0 to 10.0>,
  "risk_label": "<LOW or MODERATE or HIGH>",
  "confidence": "<LOW or MEDIUM or HIGH>",
  "evidence": ["<fact with number>", "<fact with number>", "<fact with number>"],
  "positive_signals": ["<signal>"],
  "negative_signals": ["<signal>"],
  "overall_assessment": "<2 sentences maximum>"
}}
"""


def judge_prompt_builder_factory(ticker: str, context: dict[str, Any]):
    chunks = context["chunks"]

    def _judge_prompt(candidate: dict[str, Any]) -> str:
        return f"""
Evaluate this fundamental-risk assessment for {ticker} against the retrieved
source evidence below. Judge groundedness (is every claim actually supported
by the retrieved evidence?) and consistency (does the evidence support the
stated risk_score and risk_label?).

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
    parser = argparse.ArgumentParser(description="Compare the Fundamental Agent across open-source LLMs.")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols, for example AAPL MSFT JPM")
    parser.add_argument("--models", nargs="+", default=list(llm_comparison.MODELS.keys()),
                         choices=list(llm_comparison.MODELS.keys()),
                         help="Model keys to compare (default: all 4).")
    parser.add_argument("--out", default=str(GOLD_DIR / "fundamental_llm_comparison.json"),
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
