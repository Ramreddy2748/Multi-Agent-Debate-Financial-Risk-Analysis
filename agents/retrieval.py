"""Compatibility wrapper for retrieval utilities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.shared.retrieval import *  # noqa: F401,F403


if __name__ == "__main__":
    from agents.shared.retrieval import main

    main()
