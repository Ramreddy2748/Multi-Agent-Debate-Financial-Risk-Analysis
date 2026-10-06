"""Compatibility wrapper for the market/news fine-tune data builder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.market.build_finetune_data import *  # noqa: F401,F403


if __name__ == "__main__":
    from agents.market.build_finetune_data import main

    main()
