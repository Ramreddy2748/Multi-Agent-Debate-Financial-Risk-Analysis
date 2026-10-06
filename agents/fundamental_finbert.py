"""Compatibility wrapper for the Fundamental FinBERT agent.

New code should import from agents.fundamental.finbert.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.fundamental.finbert import *  # noqa: F401,F403


if __name__ == "__main__":
    from agents.fundamental.finbert import main

    main()
