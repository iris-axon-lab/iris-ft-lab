#!/usr/bin/env python3
"""
Entry point for the collab_eval demo.

Run from the collab-eval/ directory:
  python scripts/run_demo.py

Or from the repo root:
  python collab-eval/scripts/run_demo.py

No API credentials required for deterministic-only mode.
Set ANTHROPIC_API_KEY to enable LLM-based grading.
"""

import sys
from pathlib import Path

# Ensure collab_eval is importable when running from scripts/ or the repo root.
_root = Path(__file__).parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from collab_eval.demo import run_demo

if __name__ == "__main__":
    run_demo()
