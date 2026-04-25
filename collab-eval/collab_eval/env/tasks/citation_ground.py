"""
Citation-grounded editing task environment.

The agent receives a synthetic research brief with two claims that lack citation support.
It is given three source paragraphs and must add inline citations grounding each claim
to a specific sentence in one of the sources.

Source A clearly supports claim 1 (reward shaping).
Source B loosely supports claim 2 (format vs. semantic bias) — loose enough to create
a hallucination trap if the agent fabricates a stronger claim than the source makes.
Source C is relevant background but does not directly support either claim.

Tasks, documents, and failure cases in this artifact are fully synthetic and derived
from public evaluation patterns. No proprietary data or internal workflows are referenced.
"""

from __future__ import annotations

from pathlib import Path

from collab_eval.base import Episode, TaskEnv, TaskSpec
from collab_eval.graders import composite as composite_grader
from collab_eval.graders import deterministic as det


_DEFAULT_TASK_ID = "citation_ground_001"
_DEFAULT_INTENT = (
    "Edit the research brief to add inline citations grounding each major claim "
    "to a specific source paragraph."
)
_DEFAULT_CONSTRAINTS = [
    "Only cite sources provided in the source_docs list.",
    "Do not alter the core argument of the brief.",
    "Citations must be traceable to a specific sentence in the source.",
]

_WEIGHTS: dict[str, float] = {
    "citation_present": 0.25,
    "citation_accurate": 0.30,
    "hallucination_flag": 0.30,
    "argument_preservation": 0.15,
}

# The two claims that must be cited. These strings are used to check
# whether the agent produced some citation marker near each claim.
_REQUIRED_CITATIONS = 2

# Anchor phrases from the brief that must survive editing.
# Design note: checking for anchor phrases is a weak but auditable proxy for
# "argument preserved". LLM judgment is more reliable here.
_ANCHOR_PHRASES = [
    "sparse and delayed",
    "decompose reward",
]


def load_default_task(data_dir: Path | None = None) -> TaskSpec:
    """
    Load the default citation-grounding task from the sample data directory.

    Resolves data_dir relative to this file if not provided.
    """
    if data_dir is None:
        data_dir = Path(__file__).parents[3] / "data" / "sample_docs"
    input_doc = (data_dir / "research_brief.txt").read_text()
    sources_raw = (data_dir / "sources.txt").read_text()
    # Split sources on the separator line.
    source_docs = [
        block.strip()
        for block in sources_raw.split("--- SOURCE")
        if block.strip() and not block.strip().startswith("---")
    ]
    return TaskSpec(
        task_id=_DEFAULT_TASK_ID,
        task_type="citation_ground",
        input_doc=input_doc,
        intent=_DEFAULT_INTENT,
        constraints=_DEFAULT_CONSTRAINTS,
        source_docs=source_docs,
    )


class CitationGroundEnv(TaskEnv):
    """
    Task environment for citation-grounded editing.

    Grading mixes deterministic checks (citation markers present, anchor phrases
    preserved) with placeholder scores for dimensions that require judgment
    (citation accuracy, hallucination detection).

    Hallucination detection is intentionally hard: Source B supports claim 2
    only weakly, and an agent that strengthens the claim beyond what the source
    says will produce output that looks cited but is actually hallucinated.
    This is the core adversarial case for this task type.
    """

    def __init__(self, task: TaskSpec | None = None) -> None:
        super().__init__()
        self._task = task or load_default_task()

    def reset(self) -> TaskSpec:
        """Return the citation-grounding task spec."""
        self._current_spec = self._task
        return self._current_spec

    def grade(self, spec: TaskSpec, output: str) -> dict[str, float]:
        """
        Score output on four dimensions.

        citation_present:      Deterministic — does the output contain enough citation markers?
        citation_accurate:     Placeholder — requires LLM to verify source traceability.
        hallucination_flag:    Placeholder — requires LLM to check claim strength vs. source.
        argument_preservation: Partial deterministic — anchor phrases present in output.
        """
        # citation_present: look for citation-like markers.
        # Design note: citation_present uses a simple bracket/parenthesis heuristic.
        # It is gameable by inserting "(Source X)" anywhere without grounding.
        # citation_accurate (LLM dimension) is the real check.
        citation_score = det.citation_present(output, required_count=_REQUIRED_CITATIONS)

        # argument_preservation: check that key phrases from the original brief survive.
        preserved = sum(
            1 for phrase in _ANCHOR_PHRASES if phrase.lower() in output.lower()
        )
        arg_score = preserved / len(_ANCHOR_PHRASES)

        # citation_accurate and hallucination_flag require LLM judgment.
        # Set to 0.5 (neutral / unassessed) in deterministic-only mode.
        return {
            "citation_present": round(citation_score, 3),
            "citation_accurate": 0.5,
            "hallucination_flag": 0.5,
            "argument_preservation": round(arg_score, 3),
        }

    def step(self, agent_output: str) -> Episode:
        """
        Grade agent output and apply hard-fail for zero citations.

        Flag: no_citations_found — output contains no citation markers at all.
        """
        if self._current_spec is None:
            raise RuntimeError("Call reset() before step().")
        spec = self._current_spec
        scores = self.grade(spec, agent_output)
        flags: list[str] = []

        if det.citation_present(agent_output, required_count=1) < 0.5:
            flags.append("no_citations_found")

        composite = composite_grader.weighted_score(scores, _WEIGHTS)
        if flags:
            composite = composite_grader.apply_hard_fail_cap(composite, cap=0.2)

        return Episode(
            spec=spec,
            agent_output=agent_output,
            grader_scores=scores,
            composite_score=round(composite, 3),
            flags=flags,
        )
