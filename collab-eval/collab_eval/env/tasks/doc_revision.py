"""
Document revision task environment.

The agent receives a synthetic project-update document with three seeded problems:
  1. A factual inconsistency (metric in body contradicts the stated target).
  2. A structural issue (most important result is buried in paragraph 3).
  3. An instruction violation (passive voice despite active-voice constraint).

Tasks, documents, and failure cases in this artifact are fully synthetic and derived
from public evaluation patterns. No proprietary data or internal workflows are referenced.
"""

from __future__ import annotations

from pathlib import Path

from collab_eval.base import Episode, TaskEnv, TaskSpec
from collab_eval.graders import composite as composite_grader
from collab_eval.graders import deterministic as det


# ── Default task spec ─────────────────────────────────────────────────────────

_DEFAULT_TASK_ID = "doc_revision_001"
_DEFAULT_INTENT = (
    "Revise this project update to be clearer and more accurate. "
    "Fix the structural issue so the most important result appears first. "
    "Correct any factual inconsistency."
)
_DEFAULT_CONSTRAINTS = [
    "Use active voice throughout.",
    "Keep the total output under 300 words.",
    "Do not add new claims that are not present in the original.",
]

# Grading weights for this task type.
# Design note: over_editing is included because agents tend to rewrite far more
# than instructed, which changes meaning and is hard to detect with a single score.
_WEIGHTS: dict[str, float] = {
    "instruction_following": 0.30,
    "faithfulness": 0.30,
    "over_editing": 0.20,
    "quality_delta": 0.20,
}

# Hard-fail conditions. If triggered, composite score is capped at 0.3.
# Design note: capping rather than zeroing allows partial credit while ensuring
# catastrophic failures are never rewarded above a minimal threshold.
_HARD_FAIL_CONDITIONS = [
    "passive_voice_dominant",   # >50% sentences contain passive constructions
    "word_count_exceeded",      # output exceeds 300 words by >20%
]


def load_default_task(data_dir: Path | None = None) -> TaskSpec:
    """
    Load the default document revision task from the sample data directory.

    If data_dir is None, resolves relative to this file's location so the
    task can be loaded from any working directory.
    """
    if data_dir is None:
        data_dir = Path(__file__).parents[3] / "data" / "sample_docs"
    input_doc = (data_dir / "project_update_draft.txt").read_text()
    return TaskSpec(
        task_id=_DEFAULT_TASK_ID,
        task_type="doc_revision",
        input_doc=input_doc,
        intent=_DEFAULT_INTENT,
        constraints=_DEFAULT_CONSTRAINTS,
        source_docs=[],
    )


class DocRevisionEnv(TaskEnv):
    """
    Task environment for document revision.

    Grading uses deterministic heuristics for instruction-following dimensions
    (word count, passive voice ratio) and produces placeholder scores for
    dimensions that require judgment (faithfulness, quality_delta).

    In production use with an LLM judge configured, the composite grader
    replaces the placeholder scores with model-based assessments.
    """

    def __init__(self, task: TaskSpec | None = None) -> None:
        super().__init__()
        self._task = task or load_default_task()

    def reset(self) -> TaskSpec:
        """Return the document revision task spec."""
        self._current_spec = self._task
        return self._current_spec

    def grade(self, spec: TaskSpec, output: str) -> dict[str, float]:
        """
        Score output on four dimensions using deterministic heuristics.

        instruction_following: penalizes passive voice and word-count violations.
        faithfulness:          placeholder — requires LLM judge in full mode.
        over_editing:          ratio of output length to input length, penalizing extremes.
        quality_delta:         placeholder — requires LLM judge in full mode.
        """
        words = det.word_count(output)
        passive_ratio = det.passive_voice_ratio(output)

        # instruction_following: average of word-count compliance and active-voice compliance.
        wc_score = det.word_count_check(output, max_words=300)
        active_score = 1.0 - passive_ratio
        instruction_score = (wc_score + active_score) / 2.0

        # over_editing: penalize outputs that are >2x the input length.
        # Design note: this heuristic is gameable by shortening the input quote,
        # but catches the most common failure mode (agents rewriting everything).
        input_words = det.word_count(spec.input_doc)
        ratio = words / max(input_words, 1)
        if ratio <= 1.5:
            over_edit_score = 1.0
        elif ratio <= 2.0:
            over_edit_score = 0.5
        else:
            over_edit_score = 0.0

        # faithfulness and quality_delta: set to neutral 0.5 in deterministic-only mode.
        # These dimensions genuinely require judgment and should be replaced by LLM scores
        # when a judge is available. 0.5 signals "unassessed" rather than "passing".
        return {
            "instruction_following": round(instruction_score, 3),
            "faithfulness": 0.5,
            "over_editing": round(over_edit_score, 3),
            "quality_delta": 0.5,
        }

    def step(self, agent_output: str) -> Episode:
        """
        Grade agent output and return an Episode with hard-fail flags applied.

        Flags:
          passive_voice_dominant  — passive ratio > 0.5
          word_count_exceeded     — output > 360 words (300 * 1.2)
        """
        if self._current_spec is None:
            raise RuntimeError("Call reset() before step().")
        spec = self._current_spec
        scores = self.grade(spec, agent_output)
        flags: list[str] = []

        if det.passive_voice_ratio(agent_output) > 0.5:
            flags.append("passive_voice_dominant")
        if det.word_count(agent_output) > 360:
            flags.append("word_count_exceeded")

        composite = composite_grader.weighted_score(scores, _WEIGHTS)
        if flags:
            composite = composite_grader.apply_hard_fail_cap(composite, cap=0.3)

        return Episode(
            spec=spec,
            agent_output=agent_output,
            grader_scores=scores,
            composite_score=round(composite, 3),
            flags=flags,
        )
