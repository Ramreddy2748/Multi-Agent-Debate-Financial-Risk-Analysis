"""Compatibility wrapper for the macro-economic agent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.macro.agent import *  # noqa: F401,F403
