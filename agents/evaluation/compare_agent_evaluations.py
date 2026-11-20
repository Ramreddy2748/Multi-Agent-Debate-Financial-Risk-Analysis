"""Build one comparison report from saved evaluation artifacts.

Run the model evaluations first, then run this script. Missing artifacts are
reported as pending rather than being replaced with invented metrics.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def model_row(name: str, path: Path, metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {"model": name, "status": "PENDING", "artifact": str(path)}
    return {
        "model": name,
        "status": "COMPLETE",
        "artifact": str(path),
        "examples": metrics.get("eval_examples_parsed", metrics.get("eval_examples")),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1", metrics.get("eval_macro_f1")),
        "parse_success_rate": metrics.get("parse_success_rate", metrics.get("valid_rate")),
        "eval_loss": metrics.get("eval_loss"),
        "score_mae": metrics.get("score_mae", metrics.get("vol_mae_pp")),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    qwen_path = Path(args.qwen_metrics)
    mistral_path = Path(args.mistral_metrics)
    rule_path = Path(args.rule_metrics)
    fundamental_path = Path(args.fundamental_metrics)
    macro_path = Path(args.macro_metrics)

    qwen = load_json(qwen_path)
    mistral = load_json(mistral_path)
    rule = load_json(rule_path)
    fundamental = load_json(fundamental_path)
    macro = load_json(macro_path)

    rows = [
        model_row("Market - Mistral 7B QLoRA", mistral_path, mistral),
        model_row("Market - Qwen 1.5B QLoRA", qwen_path, qwen),
        model_row("Market - deterministic rule baseline", rule_path, rule),
    ]
    if fundamental:
        rows.append({
            "model": "Fundamental - fine-tuned FinBERT",
            "status": "COMPLETE",
            "artifact": str(fundamental_path),
            "accuracy": fundamental.get("eval_accuracy"),
            "macro_f1": fundamental.get("eval_macro_f1"),
            "eval_loss": fundamental.get("eval_loss"),
        })
    else:
        rows.append(model_row("Fundamental - fine-tuned FinBERT", fundamental_path, None))

    if macro and isinstance(macro.get("rows"), dict):
        test = macro["rows"].get("test", {})
        rows.append({
            "model": "Macro - Phi-4-mini LoRA",
            "status": "COMPLETE",
            "artifact": str(macro_path),
            "examples": test.get("n"),
            "macro_f1": test.get("macro_f1"),
            "valid_rate": test.get("valid_rate"),
            "spearman_score": test.get("spearman_score"),
            "vol_mae_pp": test.get("vol_mae_pp"),
        })
    else:
        rows.append(model_row("Macro - Phi-4-mini LoRA", macro_path, None))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": {
            "market": "Mistral and Qwen use the saved market evaluation artifacts.",
            "fundamental": "FinBERT uses its saved classifier evaluation metrics.",
            "macro": "Phi-4-mini uses its saved point-in-time test metrics.",
            "sentiment": "Sentiment requires a labeled headline holdout and is not assigned fabricated metrics.",
        },
        "models": rows,
        "limitations": [
            "Market models must be evaluated on the same held-out examples for a fair comparison.",
            "Generative models require JSON validity, label accuracy, macro F1, and score MAE.",
            "The current Qwen artifact may use a remote endpoint; record that backend in the report.",
            "Sentiment needs labeled positive/neutral/negative headlines before accuracy and macro F1 can be reported.",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Multi-Agent Model Evaluation Comparison",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "| Agent/model | Status | Examples | Accuracy | Macro F1 | Parse/valid rate | Eval loss | Score MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["models"]:
        lines.append(
            "| {model} | {status} | {examples} | {accuracy} | {macro_f1} | {parse_success_rate} | {eval_loss} | {score_mae} |".format(
                model=row.get("model", "-"),
                status=row.get("status", "-"),
                examples=row.get("examples", row.get("n", "-")),
                accuracy=row.get("accuracy", "-"),
                macro_f1=row.get("macro_f1", "-"),
                parse_success_rate=row.get("parse_success_rate", row.get("valid_rate", "-")),
                eval_loss=row.get("eval_loss", "-"),
                score_mae=row.get("score_mae", row.get("vol_mae_pp", "-")),
            )
        )
    lines += ["", "## Limitations", ""]
    lines.extend(f"- {item}" for item in report["limitations"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a multi-agent model evaluation report.")
    parser.add_argument("--qwen-metrics", default=str(PROJECT_ROOT / "models/market_sentiment_lora/evaluation_metrics.json"))
    parser.add_argument("--mistral-metrics", default=str(PROJECT_ROOT / "models/market_mistral_7b_lora/evaluation_metrics.json"))
    parser.add_argument("--rule-metrics", default=str(PROJECT_ROOT / "models/market_rule/evaluation_metrics.json"))
    parser.add_argument("--fundamental-metrics", default=str(PROJECT_ROOT / "models/fundamental_agent_finbert/training_metrics.json"))
    parser.add_argument("--macro-metrics", default=str(PROJECT_ROOT / "agents/macro/macro_v2/metrics.json"))
    parser.add_argument("--output-json", default=str(PROJECT_ROOT / "data/reports/multi_agent_model_evaluation.json"))
    parser.add_argument("--output-md", default=str(PROJECT_ROOT / "data/reports/multi_agent_model_evaluation.md"))
    args = parser.parse_args()

    report = build_report(args)
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.output_md))
    print(f"Saved JSON report: {json_path}")
    print(f"Saved Markdown report: {args.output_md}")
    for row in report["models"]:
        print(f"{row['model']}: {row['status']}")


if __name__ == "__main__":
    main()
