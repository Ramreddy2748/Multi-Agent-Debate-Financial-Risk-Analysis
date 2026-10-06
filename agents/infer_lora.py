"""Compatibility wrapper for LoRA inference."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.market.infer_lora import *  # noqa: F401,F403


if __name__ == "__main__":
    from agents.market.infer_lora import main

    main()
