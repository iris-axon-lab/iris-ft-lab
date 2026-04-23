"""
pytest configuration for collab-eval.

Adds the collab-eval/ directory to sys.path so that `collab_eval` is importable
when running tests from either the collab-eval/ directory or the repo root:

  # From repo root:
  pytest collab-eval/tests/ -v

  # From collab-eval/:
  pytest tests/ -v
"""

import sys
from pathlib import Path

# Insert collab-eval/ at the front of sys.path so collab_eval takes priority
# over any same-named package elsewhere on the path.
sys.path.insert(0, str(Path(__file__).parent))
