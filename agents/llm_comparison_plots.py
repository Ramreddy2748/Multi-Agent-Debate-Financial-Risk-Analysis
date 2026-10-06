"""Compatibility wrapper for LLM comparison plots."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.shared.llm_comparison_plots import *  # noqa: F401,F403


if __name__ == "__main__":
    from agents.shared.llm_comparison_plots import main

    main()
