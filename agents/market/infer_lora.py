"""
Run inference with a base model plus a saved LoRA adapter.

Example:
  python3 agents/infer_lora.py \
    --base-model mistralai/Mistral-7B-Instruct-v0.3 \
    --adapter-dir models/fundamental_lora \
    --prompt '{"ticker":"AAPL","features":{"debt_to_assets":0.25}}'
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with a LoRA fine-tuned agent.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-4bit", action="store_true",
                         help="Load the base model at full precision instead of 4-bit.")
    args = parser.parse_args()
    adapter_dir = Path(args.adapter_dir).expanduser()
    if adapter_dir.exists():
        adapter_config = adapter_dir / "adapter_config.json"
        if not adapter_config.exists():
            raise SystemExit(
                f"LoRA adapter folder exists but is missing {adapter_config}.\n"
                "Use the exact --output-dir produced by agents/finetune_lora.py, "
                "or rerun fine-tuning to create the adapter files."
            )
        adapter_source = str(adapter_dir)
    elif "/" not in args.adapter_dir:
        raise SystemExit(
            f"LoRA adapter path not found: {adapter_dir}\n"
            "PEFT would otherwise treat this as a Hugging Face repo id and fail with 404.\n"
            "Pass the local adapter folder created by fine-tuning, for example:\n"
            "  --adapter-dir models/fundamental_lora\n"
            "In Colab/Drive, use the full path, for example:\n"
            "  --adapter-dir /content/drive/MyDrive/298A/models/fundamental_lora"
        )
    else:
        adapter_source = args.adapter_dir

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit(
            "Missing inference dependencies. Install with:\n"
            "  pip install torch transformers peft accelerate bitsandbytes\n"
            f"Original error: {exc}"
        )

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
    model = PeftModel.from_pretrained(model, adapter_source)

    messages = [
        {"role": "system", "content": "You are a financial risk agent. Return strict JSON."},
        {"role": "user", "content": args.prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"SYSTEM: {messages[0]['content']}\n\nUSER: {messages[1]['content']}\n\nASSISTANT:"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
