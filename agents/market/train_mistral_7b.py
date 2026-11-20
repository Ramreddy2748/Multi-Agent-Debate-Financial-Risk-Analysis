"""Colab-ready QLoRA trainer for the Market/Volatility Agent.

The input JSONL must contain chat-style rows with a ``messages`` field. The
saved adapter can be loaded by the existing market inference/remote endpoint.

Typical Colab run:
  python3 agents/market/train_mistral_7b.py \
    --train-jsonl data/gold/market_sentiment_finetune_data.jsonl \
    --output-dir models/market_mistral_7b_lora
"""

from __future__ import annotations

import argparse
import json
import inspect
from pathlib import Path


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def load_messages(path: Path, limit: int | None = None) -> list[list[dict[str, str]]]:
    rows: list[list[dict[str, str]]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            messages = item.get("messages")
            if isinstance(messages, list) and messages:
                rows.append(messages)
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Mistral 7B with QLoRA for market risk JSON output.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install Colab dependencies first: pip install -U torch transformers datasets "
            f"peft accelerate bitsandbytes\nOriginal error: {exc}"
        ) from exc

    if not torch.cuda.is_available() and not args.no_4bit:
        raise SystemExit("CUDA GPU not detected. Select a Colab GPU or pass --no-4bit for CPU testing.")

    train_path = Path(args.train_jsonl)
    output_dir = Path(args.output_dir)
    messages = load_messages(train_path, args.limit)
    if len(messages) < 20:
        raise SystemExit(f"Expected at least 20 training examples, found {len(messages)} in {train_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_messages(row: list[dict[str, str]]) -> str:
        if tokenizer.chat_template:
            return tokenizer.apply_chat_template(row, tokenize=False, add_generation_prompt=False)
        return "\n\n".join(f"{item['role'].upper()}: {item['content']}" for item in row) + tokenizer.eos_token

    dataset = Dataset.from_dict({"text": [format_messages(row) for row in messages]})
    split = dataset.train_test_split(test_size=args.eval_fraction, seed=42)

    def tokenize(batch):
        encoded = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
        )
        encoded["labels"] = [ids[:] for ids in encoded["input_ids"]]
        return encoded

    tokenized = split.map(tokenize, batched=True, remove_columns=["text"])
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_4bit = torch.cuda.is_available() and not args.no_4bit
    quantization = None
    if use_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        quantization_config=quantization,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()

    training_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available() and dtype == torch.float16,
        bf16=torch.cuda.is_available() and dtype == torch.bfloat16,
        remove_unused_columns=False,
    )
    # Transformers renamed this argument; support both Colab versions.
    parameter_name = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    training_kwargs[parameter_name] = "epoch"

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_kwargs),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics = {
        "base_model": args.base_model,
        "train_examples": len(tokenized["train"]),
        "eval_examples": len(tokenized["test"]),
        "epochs": args.epochs,
        "max_length": args.max_length,
        "qlora_4bit": use_4bit,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metrics, indent=2, default=str))
    print(f"Saved Mistral QLoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
