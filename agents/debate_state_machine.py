"""Compatibility wrapper for the debate state machine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.orchestrator.debate_state_machine import *  # noqa: F401,F403


if __name__ == "__main__":
    from agents.orchestrator.debate_state_machine import main

    main()
