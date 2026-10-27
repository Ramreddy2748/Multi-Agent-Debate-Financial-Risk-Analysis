"""Lightweight retrieval over the curated financial RAG data layer.

This module intentionally does not build or load Chroma. It reads the small
Git-friendly files in data/rag/ and returns the highest-priority financial
rules for a given agent/query. Vector retrieval over Silver evidence remains in
agents/shared/retrieval.py.
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = PROJECT_ROOT / "data" / "rag"
TERMS_PATH = RAG_DIR / "financial_terms.jsonl"
CONFIG_PATH = RAG_DIR / "agent_retrieval_config.json"


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_]+", text.lower())
        if len(token) > 2
    }


def _term_search_text(term: dict[str, Any]) -> str:
    pieces: list[str] = [
        str(term.get("id", "")),
        str(term.get("term", "")),
        str(term.get("category", "")),
        str(term.get("definition", "")),
        str(term.get("formula", "")),
        str(term.get("risk_rule", "")),
    ]
    pieces.extend(str(item) for item in term.get("agent_scope", []) or [])
    pieces.extend(str(item) for item in term.get("evidence_fields", []) or [])
    thresholds = term.get("thresholds") or {}
    if isinstance(thresholds, dict):
        pieces.extend(str(key) for key in thresholds)
        pieces.extend(str(value) for value in thresholds.values())
    return " ".join(pieces)


@lru_cache(maxsize=1)
def load_financial_terms() -> list[dict[str, Any]]:
    """Load curated financial terms from data/rag/financial_terms.jsonl."""
    if not TERMS_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(TERMS_PATH.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {TERMS_PATH}:{line_no}: {exc}") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


@lru_cache(maxsize=1)
def load_retrieval_config() -> dict[str, Any]:
    """Load the agent-to-RAG source mapping."""
    if not CONFIG_PATH.exists():
        return {"agents": {}}
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def agent_config(agent: str) -> dict[str, Any]:
    """Return retrieval config for an agent id or display name."""
    config = load_retrieval_config()
    agents = config.get("agents", {}) if isinstance(config, dict) else {}
    normalized = agent.lower().replace(" ", "_").replace("/", "_")
    if agent in agents:
        return agents[agent]
    for agent_id, cfg in agents.items():
        names = {
            agent_id.lower(),
            str(cfg.get("display_name", "")).lower(),
            str(cfg.get("display_name", "")).lower().replace(" ", "_").replace("/", "_"),
        }
        if agent.lower() in names or normalized in names:
            return cfg
    return {}


def retrieve_financial_terms(
    agent: str,
    query: str = "",
    limit: int = 5,
    include_required: bool = True,
) -> list[dict[str, Any]]:
    """Return the most relevant financial rules for an agent and query.

    Ranking combines three signals:
    - agent scope match
    - required term ids from agent_retrieval_config.json
    - token overlap with the user/model query
    """
    terms = load_financial_terms()
    cfg = agent_config(agent)
    required_ids = set(cfg.get("required_terms", []) or []) if include_required else set()
    agent_key = agent.lower().replace("_agent", "")
    query_tokens = _tokenize(query)

    ranked: list[tuple[float, dict[str, Any]]] = []
    for term in terms:
        scopes = {str(scope).lower() for scope in term.get("agent_scope", []) or []}
        term_id = str(term.get("id", ""))
        scope_match = agent_key in scopes or agent.lower() in scopes or term_id in required_ids
        if not scope_match and required_ids:
            continue
        if not scope_match and not query_tokens:
            continue

        term_tokens = _tokenize(_term_search_text(term))
        overlap = len(query_tokens & term_tokens)
        priority = int(term.get("priority") or 3)
        score = 0.0
        if term_id in required_ids:
            score += 100.0
        if scope_match:
            score += 25.0
        score += max(0, 5 - priority) * 4.0
        score += overlap * 3.0
        if score > 0:
            ranked.append((score, term))

    ranked.sort(key=lambda item: (-item[0], int(item[1].get("priority") or 3), str(item[1].get("id", ""))))
    return [term for _, term in ranked[:limit]]


def format_financial_terms_context(terms: list[dict[str, Any]]) -> str:
    """Format retrieved terms as compact context for an agent prompt/evidence."""
    blocks = []
    for term in terms:
        thresholds = term.get("thresholds") or {}
        threshold_text = "; ".join(f"{key}: {value}" for key, value in thresholds.items()) if isinstance(thresholds, dict) else ""
        pieces = [
            f"{term.get('term', term.get('id', 'Financial term'))}",
            f"Definition: {term.get('definition', '')}",
        ]
        if threshold_text:
            pieces.append(f"Thresholds: {threshold_text}")
        if term.get("risk_rule"):
            pieces.append(f"Priority rule: {term['risk_rule']}")
        blocks.append(" | ".join(piece for piece in pieces if piece))
    return "\n".join(f"- {block}" for block in blocks)


def build_rag_context(agent: str, query: str = "", limit: int = 5) -> dict[str, Any]:
    """Return structured RAG rules plus prompt-ready context."""
    terms = retrieve_financial_terms(agent=agent, query=query, limit=limit)
    return {
        "agent": agent,
        "query": query,
        "source": str(TERMS_PATH.relative_to(PROJECT_ROOT)),
        "terms": terms,
        "context": format_financial_terms_context(terms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Query curated financial RAG rules.")
    parser.add_argument("--agent", default="critic_agent", help="Agent id, for example fundamental_agent.")
    parser.add_argument("--query", default="", help="Natural-language query or evidence text.")
    parser.add_argument("--limit", type=int, default=5, help="Number of financial terms to return.")
    parser.add_argument("--json", action="store_true", help="Print structured JSON instead of text context.")
    args = parser.parse_args()

    context = build_rag_context(agent=args.agent, query=args.query, limit=args.limit)
    if args.json:
        print(json.dumps(context, indent=2))
        return
    print(f"RAG context for {args.agent}")
    print(f"Source: {context['source']}")
    print(context["context"] or "No matching financial terms found.")


if __name__ == "__main__":
    main()
