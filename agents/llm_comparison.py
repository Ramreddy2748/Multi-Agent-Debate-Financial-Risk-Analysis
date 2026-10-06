"""Compatibility wrapper for shared LLM comparison utilities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.shared.llm_comparison import *  # noqa: F401,F403
