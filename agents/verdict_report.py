"""Compatibility wrapper for verdict report generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.orchestrator.verdict_report import *  # noqa: F401,F403


if __name__ == "__main__":
    from agents.orchestrator.verdict_report import main

    main()
