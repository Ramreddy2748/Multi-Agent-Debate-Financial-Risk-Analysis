"""
Explicit debate state machine for the financial risk agents.

The workflow uses LangGraph when installed and falls back to the same ordered
state-machine steps when it is not. This keeps the project runnable while making
the intended debate protocol concrete:

collect evidence -> critique contradictions -> revise positions -> synthesize.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Literal, TypedDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.critic_agent import (
    _label_from_score,
    _weighted_critic,
    run_fundamental_agent,
    run_macro_agent_safe,
    run_market_sentiment_agent,
)
from agents.monitoring import log_verdict_run
from agents.policy_evidence import build_policy_evidence_trail


class DebateState(TypedDict, total=False):
    ticker: str
    company_name: str | None
    sector: str | None
    query: str
    agent_outputs: list[dict[str, Any]]
    contradictions: list[dict[str, Any]]
    revision_round: int
    max_revision_rounds: int
    final_output: dict[str, Any]


def collect_agent_positions(state: DebateState) -> DebateState:
    ticker = state["ticker"].upper()
    company_name = state.get("company_name") or ticker
    sector = state.get("sector") or "General"
    outputs = [
        run_fundamental_agent(ticker, company_name=company_name),
        run_market_sentiment_agent(ticker, company_name=company_name),
        run_macro_agent_safe(ticker, sector=sector, company_name=company_name),
    ]
    return {"agent_outputs": outputs, "revision_round": 0}


def critique_contradictions(state: DebateState) -> DebateState:
    outputs = state.get("agent_outputs", [])
    labels = [o.get("risk_label") for o in outputs if o.get("risk_label")]
    scores = [
        float(score)
        for score in (o.get("risk_score", o.get("macro_risk_score")) for o in outputs)
        if score not in (None, "")
    ]
    contradictions = []

    if len(set(labels)) > 1:
        contradictions.append({
            "type": "label_disagreement",
            "severity": "HIGH" if "HIGH" in labels and "LOW" in labels else "MEDIUM",
            "detail": f"Agent labels disagree: {', '.join(labels)}.",
        })

    if len(scores) >= 2 and max(scores) - min(scores) >= 2.0:
        contradictions.append({
            "type": "score_dispersion",
            "severity": "HIGH",
            "detail": f"Agent score spread is {max(scores) - min(scores):.2f} points.",
        })

    for output in outputs:
        confidence = output.get("confidence")
        if isinstance(confidence, str):
            low_confidence = confidence.upper() == "LOW"
        else:
            low_confidence = confidence is not None and float(confidence) < 0.50
        if low_confidence:
            contradictions.append({
                "type": "low_confidence_agent",
                "severity": "MEDIUM",
                "detail": f"{output.get('agent')} confidence is {confidence}.",
            })
        if str(output.get("claim_type", "")).startswith("FALLBACK"):
            contradictions.append({
                "type": "fallback_claim",
                "severity": "MEDIUM",
                "detail": f"{output.get('agent')} used fallback claim type {output.get('claim_type')}.",
            })

    return {"contradictions": contradictions}


def revise_positions(state: DebateState) -> DebateState:
    outputs = [dict(output) for output in state.get("agent_outputs", [])]
    contradictions = state.get("contradictions", [])
    revision_round = state.get("revision_round", 0) + 1

    if not contradictions:
        return {"agent_outputs": outputs, "revision_round": revision_round}

    scores = [
        float(score)
        for score in (o.get("risk_score", o.get("macro_risk_score")) for o in outputs)
        if score not in (None, "")
    ]
    median_score = statistics.median(scores) if scores else 5.0

    for output in outputs:
        score_key = "risk_score" if "risk_score" in output else "macro_risk_score"
        raw_score = output.get(score_key)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue

        evidence_count = len(output.get("evidence", []))
        confidence = output.get("confidence")
        numeric_confidence = 0.65
        if isinstance(confidence, (int, float)):
            numeric_confidence = float(confidence)
        elif isinstance(confidence, str):
            numeric_confidence = {"LOW": 0.35, "MEDIUM": 0.60, "HIGH": 0.82}.get(confidence.upper(), 0.60)

        support_strength = min(1.0, evidence_count / 3.0) * numeric_confidence
        if support_strength < 0.55:
            revised_score = score * 0.75 + median_score * 0.25
        else:
            revised_score = score

        output["pre_revision_score"] = round(score, 2)
        output[score_key] = round(max(1.0, min(10.0, revised_score)), 2)
        if score_key == "macro_risk_score":
            output["risk_score"] = output[score_key]
        output["risk_label"] = _label_from_score(float(output[score_key]))
        output["revision_note"] = (
            "Contradiction pass reviewed this agent against peer outputs; "
            f"support_strength={support_strength:.2f}, median_peer_score={median_score:.2f}."
        )

    return {"agent_outputs": outputs, "revision_round": revision_round}


def synthesize_verdict(state: DebateState) -> DebateState:
    outputs = state.get("agent_outputs", [])
    contradictions = state.get("contradictions", [])
    final = _weighted_critic(outputs, query=state.get("query", ""))
    final["contradictions"] = contradictions
    final["revision_rounds"] = state.get("revision_round", 0)

    high_contradictions = [c for c in contradictions if c.get("severity") == "HIGH"]
    if high_contradictions:
        final["confidence"] = min(float(final.get("confidence", 0.60)), 0.55)
        final["requires_human_review"] = True
        final["final_decision"] += " Contradictions remain material, so this verdict requires human review."
    else:
        final["requires_human_review"] = False

    report = {
        "ticker": state["ticker"].upper(),
        "query": state.get("query", ""),
        "agent_outputs": outputs,
        "final_output": final,
    }
    final["policy_evidence_trail"] = build_policy_evidence_trail(report)
    return {"final_output": final}


def _run_fallback_state_machine(state: DebateState) -> DebateState:
    state.update(collect_agent_positions(state))
    state.update(critique_contradictions(state))
    if state.get("contradictions") and state.get("revision_round", 0) < state.get("max_revision_rounds", 1):
        state.update(revise_positions(state))
        state.update(critique_contradictions(state))
    state.update(synthesize_verdict(state))
    return state


def build_debate_graph():
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return None

    def route_after_critique(state: DebateState) -> Literal["revise_positions", "synthesize_verdict"]:
        if state.get("contradictions") and state.get("revision_round", 0) < state.get("max_revision_rounds", 1):
            return "revise_positions"
        return "synthesize_verdict"

    builder = StateGraph(DebateState)
    builder.add_node("collect_agent_positions", collect_agent_positions)
    builder.add_node("critique_contradictions", critique_contradictions)
    builder.add_node("revise_positions", revise_positions)
    builder.add_node("synthesize_verdict", synthesize_verdict)
    builder.add_edge(START, "collect_agent_positions")
    builder.add_edge("collect_agent_positions", "critique_contradictions")
    builder.add_conditional_edges("critique_contradictions", route_after_critique)
    builder.add_edge("revise_positions", "critique_contradictions")
    builder.add_edge("synthesize_verdict", END)
    return builder.compile()


def run_debate_state_machine(
    ticker: str,
    query: str = "",
    company_name: str | None = None,
    sector: str | None = None,
    max_revision_rounds: int = 1,
) -> dict[str, Any]:
    state: DebateState = {
        "ticker": ticker.upper(),
        "query": query,
        "company_name": company_name,
        "sector": sector,
        "max_revision_rounds": max_revision_rounds,
    }
    graph = build_debate_graph()
    if graph is None:
        final_state = _run_fallback_state_machine(state)
    else:
        # The graph declares the protocol explicitly; the fallback above keeps
        # runtime behavior identical when LangGraph is not installed.
        final_state = graph.invoke(state)
    return {
        "ticker": final_state["ticker"],
        "query": final_state.get("query", ""),
        "agent_outputs": final_state.get("agent_outputs", []),
        "contradictions": final_state.get("contradictions", []),
        "final_output": final_state.get("final_output", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the explicit debate state machine for one ticker.")
    parser.add_argument("ticker")
    parser.add_argument("--query", default="")
    parser.add_argument("--company", default=None)
    parser.add_argument("--sector", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--log-monitoring", action="store_true")
    args = parser.parse_args()

    report = run_debate_state_machine(
        ticker=args.ticker,
        query=args.query,
        company_name=args.company,
        sector=args.sector,
    )
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
        print(f"Saved debate verdict to {path}")
    if args.log_monitoring:
        tracking = log_verdict_run(report, run_type="debate_state_machine", artifact_paths=[args.out] if args.out else [])
        print(f"Logged monitoring via {tracking['tracking_backend']}")


if __name__ == "__main__":
    main()
