"""
collab_eval — task environment and grader harness for virtual-collaborator RL scenarios.

Tasks, documents, and failure cases in this package are fully synthetic and derived
from public evaluation patterns. No proprietary data or internal workflows are referenced.
"""

from collab_eval.base import TaskSpec, Episode, TaskEnv

__all__ = ["TaskSpec", "Episode", "TaskEnv"]
