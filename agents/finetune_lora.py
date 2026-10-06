"""Compatibility wrapper for LoRA fine-tuning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.market.finetune_lora import *  # noqa: F401,F403


if __name__ == "__main__":
    from agents.market.finetune_lora import main

    main()
