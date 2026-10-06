"""Compatibility wrapper for the fundamental rule-based agent.

New code should import from agents.fundamental.rule_based.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.fundamental.rule_based import *  # noqa: F401,F403


if __name__ == "__main__":
    from agents.fundamental.rule_based import main

    main()
