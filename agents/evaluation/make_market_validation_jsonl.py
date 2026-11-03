"""Recreate the deterministic 15% market validation split used by training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Full market training JSONL")
    parser.add_argument("--output", required=True, help="Validation JSONL output")
    args = parser.parse_args()

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit("Install datasets first: pip install datasets") from exc

    input_path = Path(args.input)
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset = Dataset.from_dict({"row": rows})
    split = dataset.train_test_split(test_size=0.15, seed=42)
    validation_rows = split["test"]["row"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in validation_rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(validation_rows)} validation examples to {output_path}")


if __name__ == "__main__":
    main()
