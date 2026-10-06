"""
MLflow lineage and local monitoring helpers.

When MLflow is installed, verdict runs are logged to an MLflow experiment. When
it is not installed, the same summary is appended to a local JSONL file so the
project keeps an auditable run history without extra setup.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


import config


DEFAULT_EXPERIMENT = "financial-risk-multi-agent"


def _score(output: dict[str, Any]) -> float | None:
    value = output.get("risk_score", output.get("macro_risk_score"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary_record(report: dict[str, Any], run_type: str) -> dict[str, Any]:
    final = report.get("final_output", {})
    outputs = report.get("agent_outputs", [])
    contradictions = report.get("contradictions", final.get("contradictions", []))
    scores = [score for score in (_score(output) for output in outputs) if score is not None]
    return {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run_type": run_type,
        "ticker": report.get("ticker"),
        "final_risk_score": final.get("final_risk_score", final.get("risk_score")),
        "final_risk_label": final.get("final_risk_label", final.get("risk_label")),
        "confidence": final.get("confidence"),
        "requires_human_review": final.get("requires_human_review", False),
        "contradiction_count": len(contradictions),
        "agent_count": len(outputs),
        "agent_score_min": min(scores) if scores else None,
        "agent_score_max": max(scores) if scores else None,
        "policy_statuses": {
            item.get("policy_id"): item.get("status")
            for item in final.get("policy_evidence_trail", [])
        },
    }


def log_verdict_run(
    report: dict[str, Any],
    run_type: str = "critic_verdict",
    artifact_paths: list[str] | None = None,
    tracking_uri: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> dict[str, Any]:
    """Log verdict lineage to MLflow when available, otherwise local JSONL."""
    record = _summary_record(report, run_type=run_type)
    artifact_paths = artifact_paths or []

    try:
        import mlflow
    except ImportError:
        Path(config.LOCAL_GOLD).mkdir(parents=True, exist_ok=True)
        path = Path(config.LOCAL_GOLD) / "monitoring_runs.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        record["tracking_backend"] = "local_jsonl"
        record["tracking_path"] = str(path)
        return record

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{run_type}:{record.get('ticker')}"):
        mlflow.log_param("run_type", run_type)
        mlflow.log_param("ticker", record.get("ticker"))
        mlflow.log_param("final_risk_label", record.get("final_risk_label"))
        mlflow.log_param("requires_human_review", record.get("requires_human_review"))
        mlflow.log_metric("final_risk_score", float(record.get("final_risk_score") or 0.0))
        mlflow.log_metric("confidence", float(record.get("confidence") or 0.0))
        mlflow.log_metric("contradiction_count", float(record.get("contradiction_count") or 0))
        mlflow.log_metric("agent_count", float(record.get("agent_count") or 0))
        if record.get("agent_score_min") is not None:
            mlflow.log_metric("agent_score_min", float(record["agent_score_min"]))
        if record.get("agent_score_max") is not None:
            mlflow.log_metric("agent_score_max", float(record["agent_score_max"]))

        temp_dir = Path(config.LOCAL_GOLD) / "_mlflow_artifacts"
        temp_dir.mkdir(parents=True, exist_ok=True)
        summary_path = temp_dir / f"{record.get('ticker')}_{run_type}_summary.json"
        summary_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(summary_path))
        for artifact in artifact_paths:
            if artifact and Path(artifact).exists():
                mlflow.log_artifact(artifact)

        run = mlflow.active_run()
        record["tracking_backend"] = "mlflow"
        record["mlflow_run_id"] = run.info.run_id if run else None
        record["mlflow_tracking_uri"] = mlflow.get_tracking_uri()
    return record
