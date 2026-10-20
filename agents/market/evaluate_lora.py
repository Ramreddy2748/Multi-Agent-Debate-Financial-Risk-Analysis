"""Evaluate a market-volatility LoRA adapter on held-out JSONL examples.

This script treats the LoRA as a generative JSON model:
  prompt -> generated JSON -> parsed risk label -> accuracy / macro-F1

Recommended in Colab/GPU:
  python3 agents/market/evaluate_lora.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter-dir models/market_sentiment_lora \
    --test-jsonl data/gold/market_sentiment_finetune_data.jsonl \
    --output-json models/market_sentiment_lora/evaluation_metrics.json \
    --limit 50
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = ["LOW", "MODERATE", "HIGH"]


def _extract_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def _label_from_payload(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    for key in ("market_risk_label", "market_sentiment_risk_label", "risk_label", "label"):
        value = payload.get(key)
        if isinstance(value, str) and value.upper() in LABELS:
            return value.upper()
    return None


def _score_from_payload(payload: dict[str, Any] | None) -> float | None:
    if not payload:
        return None
    for key in ("market_risk_score", "market_sentiment_risk_score", "risk_score", "score"):
        try:
            return float(payload.get(key))
        except (TypeError, ValueError):
            continue
    return None


def _label_from_text(text: str) -> str | None:
    upper = text.upper()
    for label in LABELS:
        if re.search(rf"\b{label}\b", upper):
            return label
    return None


def _load_examples(path: Path) -> list[dict[str, Any]]:
    examples = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = item.get("messages") or []
            system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
            user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            assistant = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
            target_payload = _extract_json(assistant)
            target_label = _label_from_payload(target_payload)
            if not user or not target_label:
                continue
            ticker = ""
            features = {}
            user_payload = _extract_json(user)
            if user_payload:
                ticker = str(user_payload.get("ticker") or "")
                features = user_payload.get("features") or {}
            examples.append({
                "line_no": line_no,
                "ticker": ticker,
                "system": system or "You are a financial risk agent. Return strict JSON.",
                "user": user,
                "features": features,
                "target_label": target_label,
                "target_score": _score_from_payload(target_payload),
            })
    return examples


def _remote_prediction_payload(response_payload: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(response_payload.get("prediction"), dict):
        return response_payload["prediction"]
    if isinstance(response_payload.get("raw_output"), dict):
        return response_payload["raw_output"]
    if isinstance(response_payload, dict):
        return response_payload
    return None


def _run_remote_prediction(example: dict[str, Any], remote_url: str, timeout: int, max_new_tokens: int) -> tuple[dict[str, Any] | None, str]:
    features = example.get("features") or {}
    payload = {
        "ticker": example.get("ticker") or "UNKNOWN",
        "features": {
            "annualized_volatility": features.get("annualized_volatility", 0.0),
            "beta": features.get("beta", 1.0),
            "max_drawdown": features.get("max_drawdown", 0.0),
            "price_vs_52w_high_pct": features.get("price_vs_52w_high_pct", 0.0),
            "ytd_return_pct": features.get("ytd_return_pct", 0.0),
            "rolling_vol_30d": features.get("rolling_vol_30d", 0.0),
            "rolling_vol_60d": features.get("rolling_vol_60d", 0.0),
        },
        "max_new_tokens": max_new_tokens,
    }
    request = urllib.request.Request(
        remote_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        parsed = json.loads(body)
        prediction = _remote_prediction_payload(parsed)
        return prediction, json.dumps(parsed)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return None, f"HTTP {exc.code}: {detail[:500]}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    f1_scores = []
    for label in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores)


def _confusion_matrix(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, int]]:
    matrix = {label: {pred: 0 for pred in LABELS} for label in LABELS}
    for target, pred in zip(y_true, y_pred):
        if target in LABELS and pred in LABELS:
            matrix[target][pred] += 1
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a market-volatility LoRA JSON generator.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--predictions-jsonl", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--remote-url", default="", help="Optional Colab/ngrok /predict endpoint. Skips local model loading.")
    parser.add_argument("--remote-timeout", type=int, default=120)
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir).expanduser()
    if not args.remote_url and adapter_dir.exists() and not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"Missing adapter_config.json in {adapter_dir}")

    examples = _load_examples(Path(args.test_jsonl))
    if not examples:
        raise SystemExit(f"No valid examples found in {args.test_jsonl}")

    random.Random(args.seed).shuffle(examples)
    if args.limit:
        examples = examples[:args.limit]

    tokenizer = None
    model = None
    torch = None
    if not args.remote_url:
        try:
            import torch as torch_mod
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit(
                "Missing dependencies. Install with:\n"
                "  pip install torch transformers peft accelerate bitsandbytes scikit-learn\n"
                f"Original error: {exc}"
            )

        torch = torch_mod
        use_4bit = torch.cuda.is_available() and not args.no_4bit
        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            if use_4bit else None
        )

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir) if adapter_dir.exists() else args.adapter_dir)
        model.eval()

    predictions = []
    y_true = []
    y_pred = []
    parse_success = 0

    for idx, example in enumerate(examples, start=1):
        if args.remote_url:
            payload, generated = _run_remote_prediction(
                example,
                args.remote_url,
                timeout=args.remote_timeout,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            messages = [
                {"role": "system", "content": example["system"]},
                {"role": "user", "content": example["user"]},
            ]
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = f"SYSTEM: {messages[0]['content']}\n\nUSER: {messages[1]['content']}\n\nASSISTANT:"

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            payload = _extract_json(generated)
        pred_label = _label_from_payload(payload) or _label_from_text(generated) or "PARSE_ERROR"
        pred_score = _score_from_payload(payload)
        parsed = pred_label in LABELS
        parse_success += int(parsed)

        target_label = example["target_label"]
        if parsed:
            y_true.append(target_label)
            y_pred.append(pred_label)

        predictions.append({
            "index": idx,
            "line_no": example["line_no"],
            "ticker": example["ticker"],
            "target_label": target_label,
            "target_score": example["target_score"],
            "pred_label": pred_label,
            "pred_score": pred_score,
            "parsed": parsed,
            "raw_output": generated,
        })
        print(f"[{idx}/{len(examples)}] {example['ticker'] or '-'} true={target_label} pred={pred_label}")

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    metrics = {
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "inference_backend": "remote_endpoint" if args.remote_url else "local_transformers",
        "remote_url_configured": bool(args.remote_url),
        "test_jsonl": args.test_jsonl,
        "eval_examples_requested": len(examples),
        "eval_examples_parsed": len(y_true),
        "parse_success_rate": round(parse_success / len(examples), 4) if examples else 0.0,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(_macro_f1(y_true, y_pred), 4) if y_true else 0.0,
        "label_counts_true": dict(Counter(y_true)),
        "label_counts_pred": dict(Counter(y_pred)),
        "confusion_matrix": _confusion_matrix(y_true, y_pred),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    predictions_path = Path(args.predictions_jsonl) if args.predictions_jsonl else output_path.with_name("evaluation_predictions.jsonl")
    with predictions_path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row) + "\n")

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {output_path}")
    print(f"Saved predictions: {predictions_path}")


if __name__ == "__main__":
    main()
