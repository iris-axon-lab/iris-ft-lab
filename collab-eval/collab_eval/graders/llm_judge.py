"""
LLM-as-judge grader using the Anthropic SDK.

Usage is optional. If ANTHROPIC_API_KEY is not set, all functions raise
LLMJudgeUnavailableError with a clear message explaining how to enable LLM grading.
The rest of the harness continues to function in deterministic-only mode.

Design note on per-dimension vs. holistic grading:
  Holistic one-shot scoring ("rate this output 1-10") is easy to game because
  the agent can optimize a single soft signal. Per-dimension scoring forces the
  judge to attend to each criterion independently, making it harder for an agent
  to satisfy one dimension while quietly failing another. It also makes failure
  analysis more informative: you can see *which* dimension failed rather than
  just a combined score drop.

Tasks, documents, and failure cases in this artifact are fully synthetic and derived
from public evaluation patterns. No proprietary data or internal workflows are referenced.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Design note: model name is read from an environment variable. This avoids
# hardcoding a model that may be deprecated, and lets users choose cheaper
# or faster alternatives without editing source code.
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_MODEL_ENV_VAR = "COLLAB_EVAL_JUDGE_MODEL"


class LLMJudgeUnavailableError(RuntimeError):
    """Raised when the Anthropic API key is not configured."""
    pass


def _get_client():  # type: ignore[return]
    """
    Return an Anthropic client, or raise LLMJudgeUnavailableError.

    Importing anthropic at call time (not module load time) ensures the rest
    of the harness can be imported even if anthropic is not installed.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise LLMJudgeUnavailableError(
            "ANTHROPIC_API_KEY is not set. Set it to enable LLM-based grading, "
            "or run the demo without it for deterministic-only mode."
        )
    try:
        import anthropic
    except ImportError as exc:
        raise LLMJudgeUnavailableError(
            "anthropic package is not installed. Run: pip install anthropic"
        ) from exc
    return anthropic.Anthropic(api_key=api_key)


def _judge_model() -> str:
    """Return the model ID to use for judging."""
    return os.environ.get(_MODEL_ENV_VAR, _DEFAULT_MODEL)


@dataclass
class DimensionScore:
    """Result of scoring one dimension."""

    dimension: str
    score: float          # in [0.0, 1.0]
    reasoning: str        # short explanation from the judge


# ── Per-dimension rubrics ─────────────────────────────────────────────────────

_RUBRICS: dict[str, str] = {
    "faithfulness": (
        "Does the output contain only information that was present in the input document? "
        "Score 1.0 if no new claims are introduced. Score 0.5 if minor additions are present. "
        "Score 0.0 if the output introduces significant unsupported claims."
    ),
    "quality_delta": (
        "Is the output meaningfully clearer and more accurate than the input? "
        "Score 1.0 if the structural issue and factual inconsistency are both resolved. "
        "Score 0.5 if only one problem is addressed. Score 0.0 if neither is addressed."
    ),
    "citation_accurate": (
        "For each citation marker in the output, is the cited claim traceable to a specific "
        "sentence in the provided sources? Score 1.0 if all citations are traceable. "
        "Score 0.5 if some are traceable. Score 0.0 if no citation can be verified."
    ),
    "hallucination_flag": (
        "Does the output make any claim that is stronger than what the provided sources support? "
        "Score 1.0 if all claims stay within source bounds. Score 0.5 if borderline. "
        "Score 0.0 if the output clearly overstates what a source says."
    ),
}

_SCORE_PROMPT_TEMPLATE = """\
You are grading one dimension of an agent's output on a document task.

DIMENSION: {dimension}

RUBRIC:
{rubric}

TASK INTENT: {intent}

INPUT DOCUMENT:
{input_doc}

SOURCE DOCS (if any):
{source_docs}

AGENT OUTPUT:
{agent_output}

Respond with ONLY the following format, no additional text:
SCORE: <float between 0.0 and 1.0>
REASONING: <one sentence explaining the score>
"""


def score_dimension(
    dimension: str,
    spec_intent: str,
    input_doc: str,
    agent_output: str,
    source_docs: list[str] | None = None,
) -> DimensionScore:
    """
    Ask the LLM judge to score one dimension of an agent output.

    Raises LLMJudgeUnavailableError if API key is not configured.

    Design note: we score one dimension per call rather than asking for all
    dimensions at once. This is more expensive but harder to game: the judge
    cannot trade off across dimensions in a single attention pass, and the
    prompt stays short enough that the judge can focus on the rubric.
    """
    if dimension not in _RUBRICS:
        raise ValueError(
            f"Unknown dimension: {dimension!r}. Known: {list(_RUBRICS)}"
        )

    client = _get_client()
    rubric = _RUBRICS[dimension]
    source_text = "\n\n".join(source_docs) if source_docs else "(none)"

    prompt = _SCORE_PROMPT_TEMPLATE.format(
        dimension=dimension,
        rubric=rubric,
        intent=spec_intent,
        input_doc=input_doc,
        source_docs=source_text,
        agent_output=agent_output,
    )

    response = client.messages.create(
        model=_judge_model(),
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    score, reasoning = _parse_judge_response(raw, dimension)
    return DimensionScore(dimension=dimension, score=score, reasoning=reasoning)


def _parse_judge_response(raw: str, dimension: str) -> tuple[float, str]:
    """
    Parse the judge's SCORE / REASONING response.

    Returns (score, reasoning). On parse failure, returns (0.5, error message).
    Design note: returning 0.5 on parse failure rather than raising keeps the
    composite grader functional and signals "unassessed" rather than "failing".
    """
    score_line = ""
    reasoning_line = ""
    for line in raw.splitlines():
        if line.startswith("SCORE:"):
            score_line = line.replace("SCORE:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning_line = line.replace("REASONING:", "").strip()

    try:
        score = float(score_line)
        score = max(0.0, min(1.0, score))
    except ValueError:
        return 0.5, f"Parse error for {dimension}: {raw[:80]}"

    return score, reasoning_line or "(no reasoning provided)"


def score_dimensions_batch(
    dimensions: list[str],
    spec_intent: str,
    input_doc: str,
    agent_output: str,
    source_docs: list[str] | None = None,
) -> dict[str, DimensionScore]:
    """
    Score multiple dimensions, returning a dict keyed by dimension name.

    Calls score_dimension() once per dimension. Propagates LLMJudgeUnavailableError
    if the API key is not configured.
    """
    return {
        dim: score_dimension(
            dimension=dim,
            spec_intent=spec_intent,
            input_doc=input_doc,
            agent_output=agent_output,
            source_docs=source_docs,
        )
        for dim in dimensions
    }
