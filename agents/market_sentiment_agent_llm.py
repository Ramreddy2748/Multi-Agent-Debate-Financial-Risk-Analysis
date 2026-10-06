"""Compatibility wrapper for market/news LLM comparison experiments."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.market.llm_compare import *  # noqa: F401,F403


if __name__ == "__main__":
    from agents.market.llm_compare import main

    main()
