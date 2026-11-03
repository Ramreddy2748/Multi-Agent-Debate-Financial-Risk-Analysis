"""Generate reproducible evidence that RAG and self-healing are enabled.

This is a controlled verification run. The RAG check uses the real local AAPL
vector index and financial_terms.jsonl. The self-healing check uses a test
double that returns one invalid specialist response followed by a valid one,
so the retry metadata can be demonstrated without calling external models.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestrator.critic_agent import run_specialist_agents
from agents.shared.financial_rag import build_rag_context
from agents.shared.retrieval import retrieve


REPORT_PATH = PROJECT_ROOT / "data" / "reports" / "rag_self_healing_verification.json"


def _valid(agent: str) -> dict:
    return {
        "agent": agent,
        "risk_score": 4.0,
        "risk_label": "MODERATE",
        "confidence": 0.8,
        "evidence": ["Controlled verification response."],
    }


def main() -> None:
    rag_chunks = retrieve("AAPL", "leverage profitability volatility risk", k=5)
    curated = build_rag_context(
        "fundamental",
        "leverage liquidity profitability risk",
        limit=5,
    )

    invalid_fundamental = {"agent": "Fundamental Agent"}
    with patch.dict(os.environ, {"SYSTEM_MODE": "self_healing"}, clear=False), \
         patch("agents.orchestrator.critic_agent.run_fundamental_agent",
               side_effect=[invalid_fundamental, _valid("Fundamental Agent")]), \
         patch("agents.orchestrator.critic_agent.run_market_sentiment_agent",
               return_value=_valid("Market + Volatility Agent")), \
         patch("agents.orchestrator.critic_agent.run_sentiment_agent",
               return_value=_valid("Sentiment Agent")), \
         patch("agents.orchestrator.critic_agent.run_macro_agent_safe",
               return_value=_valid("Macro-Economic Agent")):
        specialist_outputs = run_specialist_agents("AAPL", "Apple Inc.", "Information Technology")

    report = {
        "verification_type": "controlled_local_verification",
        "rag": {
            "ticker": "AAPL",
            "vector_index": "data/gold/vector_store",
            "retrieved_chunk_count": len(rag_chunks),
            "retrieved_sources": sorted({chunk.get("source") for chunk in rag_chunks}),
            "retrieved_evidence": [chunk.get("text") for chunk in rag_chunks],
            "curated_terms_source": curated["source"],
            "curated_term_count": len(curated["terms"]),
            "curated_terms_context": curated["context"],
        },
        "self_healing": {
            "system_mode": "self_healing",
            "specialist_outputs": [
                {
                    "agent": output.get("agent"),
                    "attempts": output.get("attempts"),
                    "execution_mode": output.get("execution_mode"),
                    "healing_actions": output.get("healing_actions", []),
                    "data_quality_warnings": output.get("data_quality_warnings", []),
                }
                for output in specialist_outputs
            ],
            "retry_observed": specialist_outputs[0].get("attempts") == 2,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(REPORT_PATH),
        "rag_chunks": report["rag"]["retrieved_chunk_count"],
        "curated_terms": report["rag"]["curated_term_count"],
        "retry_observed": report["self_healing"]["retry_observed"],
    }, indent=2))


if __name__ == "__main__":
    main()
