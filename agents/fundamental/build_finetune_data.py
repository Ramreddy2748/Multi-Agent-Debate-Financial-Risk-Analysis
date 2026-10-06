"""
Build Fundamental Agent FinBERT training JSONL from SEC Silver files.

Input:
  data/silver/silver_edgar_<TICKER>.csv

Output:
  data/gold/fundamental_finetune_data.jsonl

The output uses the same chat-style JSONL shape consumed by
agents/fundamental_finbert.py train.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from agents.fundamental.rule_based import FundamentalAnalysisAgent


DEFAULT_SILVER_DIR = PROJECT_ROOT / config.LOCAL_SILVER
DEFAULT_OUT_PATH = PROJECT_ROOT / config.LOCAL_GOLD / "fundamental_finetune_data.jsonl"


def _json_default(value: Any) -> Any:
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return value


def build_example(ticker: str, agent: FundamentalAnalysisAgent) -> dict[str, Any] | None:
    try:
        result = asdict(agent.analyze(ticker))
    except Exception as exc:
        print(f"Skipping {ticker}: {exc}")
        return None

    user_payload = {
        "ticker": ticker,
        "task": "Assess company fundamental risk from SEC-derived financial features.",
        "features": result["features"],
        "evidence": result["evidence"],
    }
    assistant_payload = {
        "fundamental_risk_score": round(float(result["risk_score"]), 2),
        "fundamental_risk_label": result["risk_label"],
        "explanation": "Assess leverage, profitability, asset strength, cash position, and net income trend.",
    }
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a financial fundamental risk analyst. Return strict JSON.",
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, default=_json_default),
            },
            {
                "role": "assistant",
                "content": json.dumps(assistant_payload),
            },
        ]
    }


def ticker_from_path(path: Path) -> str:
    return path.stem.replace("silver_edgar_", "").upper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Fundamental FinBERT JSONL from SEC Silver files.")
    parser.add_argument("--silver-dir", default=str(DEFAULT_SILVER_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    silver_dir = Path(args.silver_dir)
    out_path = Path(args.out)
    files = sorted(silver_dir.glob("silver_edgar_*.csv"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No SEC Silver files found in {silver_dir}")

    agent = FundamentalAnalysisAgent(silver_dir=str(silver_dir))
    examples = []
    labels = Counter()
    for path in files:
        ticker = ticker_from_path(path)
        example = build_example(ticker, agent)
        if example is None:
            continue
        assistant_payload = json.loads(example["messages"][-1]["content"])
        labels[assistant_payload["fundamental_risk_label"]] += 1
        examples.append(example)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {out_path}")
    print(f"Label counts: {dict(labels)}")


if __name__ == "__main__":
    main()
