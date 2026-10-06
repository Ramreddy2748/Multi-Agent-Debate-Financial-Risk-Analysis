"""
LoRA fine-tuning runner for the project agent JSONL files.

Use this for:
  1. Fundamental Agent fine-tuning data:
     data/gold/fundamental_finetune_data.jsonl

  2. Market/Sentiment Agent fine-tuning data:
     data/gold/market_sentiment_finetune_data.jsonl

Expected JSONL format:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Install requirements in a GPU environment:
  pip install torch transformers datasets peft accelerate bitsandbytes

Example:
  python3 agents/finetune_lora.py \
    --train-jsonl data/gold/fundamental_finetune_data.jsonl \
    --base-model mistralai/Mistral-7B-Instruct-v0.3 \
    --output-dir models/fundamental_lora

For Mistral Small 4 specifically, replace --base-model with the exact Hugging
Face model id you are allowed to use.

By default, loads the base model in 4-bit (QLoRA) when a CUDA GPU is
available — a 7B model at fp16 needs ~14GB just for weights, too tight for
a free Colab T4 (16GB shared); 4-bit drops that to ~4-5GB. Pass --no-4bit to
load at full precision instead (needs a bigger GPU).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_messages(path: Path) -> list[list[dict[str, str]]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append(item["messages"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune an agent model with LoRA.")
    parser.add_argument("--train-jsonl", required=True, help="Path to JSONL file with chat messages.")
    parser.add_argument("--base-model", required=True, help="Base HF model id or local path.")
    parser.add_argument("--output-dir", required=True, help="Where to save the LoRA adapter.")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-4bit", action="store_true",
                         help="Load the base model at full precision instead of 4-bit QLoRA.")
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
            "Missing fine-tuning dependencies. Install with:\n"
            "  pip install torch transformers datasets peft accelerate bitsandbytes\n"
            f"Original error: {exc}"
        )

    train_path = Path(args.train_jsonl)
    output_dir = Path(args.output_dir)
    messages = load_messages(train_path)
    if not messages:
        raise SystemExit(f"No training examples found in {train_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_messages(chat_messages: list[dict[str, str]]) -> str:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        chunks = []
        for msg in chat_messages:
            chunks.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n\n".join(chunks) + tokenizer.eos_token

    texts = [format_messages(row) for row in messages]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        tokenized = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    use_4bit = torch.cuda.is_available() and not args.no_4bit
    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        if use_4bit else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if use_4bit:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available() and dtype == torch.float16,
        bf16=torch.cuda.is_available() and dtype == torch.bfloat16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to: {output_dir}")


if __name__ == "__main__":
    main()
