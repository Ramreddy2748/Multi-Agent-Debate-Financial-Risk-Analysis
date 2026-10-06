"""Compatibility wrapper for Policy A-H evidence helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.orchestrator.policy_evidence import *  # noqa: F401,F403
