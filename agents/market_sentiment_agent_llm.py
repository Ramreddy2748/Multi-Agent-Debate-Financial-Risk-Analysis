"""
market_sentiment_agent_llm.py

LLM-based Market / Volatility / Sentiment Agent — comparison variant.

The production agent (agents/market_sentiment_agent.py) is fully
deterministic. This module reuses its exact computed features (annualized
volatility, beta, max drawdown, weighted news sentiment, etc. — via
MarketSentimentAgent's own loader/scoring methods, not reimplemented) but
hands them to an LLM to reason about instead of the hand-tuned weighted
formula, so open-weight models can be run side by side and compared.

Provider/client/retry/judge-panel plumbing lives in agents/llm_comparison.py
(shared with fundamental_agent_llm.py and macro_agent_llm_compare.py).

Usage:
  python agents/market_sentiment_agent_llm.py AAPL MSFT JPM
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
from agents.market_sentiment_agent import MarketSentimentAgent
from agents.retrieval import retrieve

AGENT_LABEL = "market_sentiment"

SYSTEM_PROMPT = """
You are the Market / Volatility / Sentiment Agent in a structured financial
risk debate system. Assess a company's market risk using price volatility,
drawdown, beta, and news sentiment signals provided below.

Scoring guide:
1.0 - 3.5  : LOW market/sentiment risk
3.5 - 6.0  : MODERATE market/sentiment risk
6.0 - 10.0 : HIGH market/sentiment risk

Rules:
- Every evidence point must cite a specific number from the data provided
- Do not invent numbers that are not present in the data
- Return ONLY valid JSON — no extra text, no markdown
"""

JUDGE_SYSTEM_PROMPT = """
You are a neutral evaluator for a financial risk debate system. You judge
whether another agent's market/sentiment-risk assessment is actually
grounded in the retrieved source evidence, and whether its stated score is
internally consistent with its own evidence. You are not judging writing
quality.

Return ONLY valid JSON — no extra text, no markdown.
"""


def _company_name(price_rows: list[dict[str, str]], ticker: str) -> str:
    if not price_rows:
        return ticker
    return price_rows[-1].get("long_name") or ticker


def context_loader(ticker: str) -> dict[str, Any]:
    agent = MarketSentimentAgent()
    price_rows = agent.load_prices(ticker)
    market_score, market_features, market_evidence = agent._market_score(price_rows)
    sentiment_score, sentiment_features, sentiment_evidence, sentiment_confidence = agent._sentiment_score(ticker)
    chunks = retrieve(ticker, "volatility and sentiment risk", k=5)

    return {
        "company": _company_name(price_rows, ticker),
        "market_features": market_features,
        "sentiment_features": sentiment_features,
        "sentiment_confidence": sentiment_confidence,
        "chunks": chunks,
    }


def _format_features(features: dict[str, Any]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in features.items() if value is not None)


def _format_chunks(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "(no retrieved evidence available)"
    return "\n".join(f"- {chunk['text']}" for chunk in chunks)


def prompt_builder(ticker: str, context: dict[str, Any]) -> str:
    return f"""
Analyse market/volatility and sentiment risk for:
Company : {context['company']} ({ticker})

MARKET / VOLATILITY FEATURES
{_format_features(context['market_features'])}

NEWS SENTIMENT FEATURES (confidence: {context['sentiment_confidence']})
{_format_features(context['sentiment_features'])}

RETRIEVED EVIDENCE (top matches for "volatility and sentiment risk")
{_format_chunks(context['chunks'])}

Return ONLY this exact JSON:
{{
  "agent": "Market/Volatility/Sentiment Agent",
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
Evaluate this market/sentiment-risk assessment for {ticker} against the
retrieved source evidence below. Judge groundedness (is every claim actually
supported by the retrieved evidence?) and consistency (does the evidence
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
    parser = argparse.ArgumentParser(description="Compare the Market/Sentiment Agent across open-source LLMs.")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols, for example AAPL MSFT JPM")
    parser.add_argument("--models", nargs="+", default=list(llm_comparison.MODELS.keys()),
                         choices=list(llm_comparison.MODELS.keys()),
                         help="Model keys to compare (default: all 4).")
    parser.add_argument("--out", default=str(GOLD_DIR / "market_sentiment_llm_comparison.json"),
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
