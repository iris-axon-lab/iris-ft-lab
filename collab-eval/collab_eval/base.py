"""
Base types and abstract environment interface for collab_eval.

Tasks, documents, and failure cases in this artifact are fully synthetic and derived
from public evaluation patterns. No proprietary data or internal workflows are referenced.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TaskSpec:
    """
    Specification for a single evaluation task.

    task_id:      Unique identifier, e.g. "doc_revision_001".
    task_type:    One of "doc_revision", "spreadsheet_clean", "citation_ground".
    input_doc:    The raw text (or CSV string) the agent must process.
    intent:       Natural-language description of what a correct output achieves.
    constraints:  Explicit rules the agent output must satisfy.
    source_docs:  Reference material available to the agent (e.g. citation sources).
                  Empty list if not applicable.
    """

    task_id: str
    task_type: str
    input_doc: str
    intent: str
    constraints: list[str]
    source_docs: list[str] = field(default_factory=list)


@dataclass
class Episode:
    """
    Result of one agent step through a task environment.

    spec:              The TaskSpec that generated this episode.
    agent_output:      The agent's raw text response.
    grader_scores:     Per-dimension scores, all in [0.0, 1.0].
    composite_score:   Weighted aggregate of grader_scores, possibly capped by hard-fail.
    flags:             List of triggered hard-fail conditions (empty if none).
    grader_reasoning:  Optional dict of dimension -> short explanation from LLM judge.
                       None when running in deterministic-only mode.
    """

    spec: TaskSpec
    agent_output: str
    grader_scores: dict[str, float]
    composite_score: float
    flags: list[str]
    grader_reasoning: dict[str, str] | None = None


class TaskEnv(ABC):
    """
    Abstract base for task environments.

    A TaskEnv encapsulates one task type. It generates TaskSpec instances,
    accepts agent outputs, runs grading, and returns Episodes.

    Subclasses implement:
      reset() -> TaskSpec        — return a fresh task specification
      grade(spec, output) -> dict — return per-dimension scores
      step(output) -> Episode    — grade and package into an Episode

    Design note: keeping reset() and grade() separate makes it easy to swap
    graders without touching task logic, and to test graders in isolation.
    """

    def __init__(self) -> None:
        self._current_spec: TaskSpec | None = None

    @abstractmethod
    def reset(self) -> TaskSpec:
        """Return a TaskSpec for a fresh task instance."""
        ...

    @abstractmethod
    def grade(self, spec: TaskSpec, output: str) -> dict[str, float]:
        """
        Score agent output against spec.

        Returns a dict mapping dimension name -> score in [0.0, 1.0].
        """
        ...

    def step(self, agent_output: str) -> Episode:
        """
        Accept an agent output and return a graded Episode.

        Calls grade() using the spec from the most recent reset().
        Raises RuntimeError if reset() has not been called.
        """
        if self._current_spec is None:
            raise RuntimeError("Call reset() before step().")
        spec = self._current_spec
        scores = self.grade(spec, agent_output)
        composite = sum(scores.values()) / len(scores) if scores else 0.0
        return Episode(
            spec=spec,
            agent_output=agent_output,
            grader_scores=scores,
            composite_score=composite,
            flags=[],
        )
