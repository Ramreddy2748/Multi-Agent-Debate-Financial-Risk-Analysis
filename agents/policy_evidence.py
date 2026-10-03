"""
Policy A-H evidence trail for agent verdicts.

The trail is intentionally model-agnostic: it records which evidence each
specialist contributed, whether the final verdict had enough support, and which
audit checks were satisfied.
"""

from __future__ import annotations

from datetime import date
from typing import Any


POLICIES = {
    "A": "Data provenance",
    "B": "Data freshness",
    "C": "Fundamental evidence",
    "D": "Market volatility evidence",
    "E": "News sentiment evidence",
    "F": "Macro environment evidence",
    "G": "Agent disagreement review",
    "H": "Confidence and fallback disclosure",
}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def _score(output: dict[str, Any]) -> float | None:
    value = output.get("risk_score", output.get("macro_risk_score"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_agent(outputs: list[dict[str, Any]], name_fragment: str) -> bool:
    return any(name_fragment.lower() in str(o.get("agent", "")).lower() for o in outputs)


def build_policy_evidence_trail(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Build auditable Policy A-H checks from a critic report."""
    outputs = report.get("agent_outputs", [])
    final = report.get("final_output", {})
    labels = [o.get("risk_label") for o in outputs if o.get("risk_label")]
    scores = [s for s in (_score(o) for o in outputs) if s is not None]
    confidence = final.get("confidence")

    trail = [
        {
            "policy_id": "A",
            "policy_name": POLICIES["A"],
            "status": "PASS" if outputs else "WARN",
            "evidence": [
                "Verdict uses persisted Silver/Gold project tables.",
                f"{len(outputs)} specialist agent output(s) captured.",
            ],
        },
        {
            "policy_id": "B",
            "policy_name": POLICIES["B"],
            "status": "PASS" if any(o.get("as_of_date") for o in outputs) else "WARN",
            "evidence": [
                f"Verdict generated on {date.today().isoformat()}.",
                *[f"{o.get('agent')}: as_of_date={o.get('as_of_date')}" for o in outputs if o.get("as_of_date")],
            ],
        },
        {
            "policy_id": "C",
            "policy_name": POLICIES["C"],
            "status": "PASS" if _has_agent(outputs, "Fundamental") else "WARN",
            "evidence": [
                item
                for o in outputs
                if "fundamental" in str(o.get("agent", "")).lower()
                for item in _as_list(o.get("evidence"))
            ],
        },
        {
            "policy_id": "D",
            "policy_name": POLICIES["D"],
            "status": "PASS" if _has_agent(outputs, "Market") else "WARN",
            "evidence": [
                item
                for o in outputs
                if "market" in str(o.get("agent", "")).lower()
                for item in _as_list(o.get("evidence"))
            ],
        },
        {
            "policy_id": "E",
            "policy_name": POLICIES["E"],
            "status": "PASS" if any("sentiment" in str(o.get("agent", "")).lower() for o in outputs) else "WARN",
            "evidence": [
                item
                for o in outputs
                if "sentiment" in str(o.get("agent", "")).lower()
                for item in _as_list(o.get("evidence"))
                if "sentiment" in item.lower() or "article" in item.lower() or "news" in item.lower()
            ],
        },
        {
            "policy_id": "F",
            "policy_name": POLICIES["F"],
            "status": "PASS" if _has_agent(outputs, "Macro") else "WARN",
            "evidence": [
                item
                for o in outputs
                if "macro" in str(o.get("agent", "")).lower()
                for item in _as_list(o.get("evidence"))
            ],
        },
        {
            "policy_id": "G",
            "policy_name": POLICIES["G"],
            "status": "PASS" if len(set(labels)) <= 1 else "REVIEW",
            "evidence": [
                f"Agent labels: {', '.join(labels) if labels else 'none'}",
                f"Agent score range: {min(scores):.2f}-{max(scores):.2f}" if scores else "No numeric agent scores found.",
                f"Final disagreement flag: {final.get('disagreement_detected', False)}",
            ],
        },
        {
            "policy_id": "H",
            "policy_name": POLICIES["H"],
            "status": "PASS" if confidence is not None and float(confidence) >= 0.60 else "WARN",
            "evidence": [
                f"Final confidence: {confidence}",
                *[
                    f"{o.get('agent')}: {o.get('claim_type')}"
                    for o in outputs
                    if o.get("claim_type")
                ],
            ],
        },
    ]

    for item in trail:
        if not item["evidence"]:
            item["evidence"] = ["No direct evidence captured for this policy."]
    return trail
