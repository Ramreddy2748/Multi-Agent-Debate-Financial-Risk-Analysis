"""Compatibility wrapper for the critic/orchestrator agent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.orchestrator.critic_agent import *  # noqa: F401,F403


if __name__ == "__main__":
    from agents.orchestrator.critic_agent import main

    main()
