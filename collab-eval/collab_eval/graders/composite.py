"""
Composite grader: combines deterministic and optional LLM-based scores.

Design note on hard-fail caps:
  A naive weighted average allows a catastrophically bad output to still score
  well if it happens to satisfy the cheaply-gameable dimensions. Hard-fail caps
  break this: if the output violates a fundamental constraint (e.g., drops rows
  silently, contains no citations, or is not parseable CSV), the composite score
  is capped regardless of how well it scores on other dimensions.

  This is an explicit anti-reward-hacking design choice. The cap is set low
  enough that a hard-failing output cannot be confused with a marginal pass,
  but high enough that partial credit is preserved for debugging.

Usage:
  # Deterministic-only mode (default):
  score = weighted_score(det_scores, weights)
  score = apply_hard_fail_cap(score, cap=0.3)  # if any flag triggered

  # With LLM judge:
  llm_scores = score_dimensions_batch(llm_dimensions, ...)
  merged = merge_scores(det_scores, llm_scores)
  score = weighted_score(merged, weights)

Tasks, documents, and failure cases in this artifact are fully synthetic and derived
from public evaluation patterns. No proprietary data or internal workflows are referenced.
"""

from __future__ import annotations

from collab_eval.base import Episode, TaskSpec


def weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    """
    Compute a weighted average of dimension scores.

    Dimensions present in scores but absent from weights are skipped.
    Dimensions present in weights but absent from scores are also skipped
    (they do not contribute to the denominator).

    Returns 0.0 if no overlapping dimensions exist.

    Design note: skipping missing dimensions rather than raising allows the
    composite grader to function in both deterministic-only mode (where some
    LLM dimensions are absent) and full mode (where all dimensions are present).
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for dim, weight in weights.items():
        if dim in scores:
            weighted_sum += scores[dim] * weight
            total_weight += weight
    if total_weight == 0.0:
        return 0.0
    return round(weighted_sum / total_weight, 4)


def apply_hard_fail_cap(score: float, cap: float = 0.3) -> float:
    """
    Cap score at `cap` if a hard-fail condition was triggered.

    Design note: this function is intentionally simple. The caller (task env)
    is responsible for deciding which flags constitute hard fails and what the
    cap should be. Keeping the cap logic here and the flag logic in the task env
    maintains separation between "what failed" and "how to penalize it".
    """
    return round(min(score, cap), 4)


def merge_scores(
    det_scores: dict[str, float],
    llm_scores: dict[str, float],
) -> dict[str, float]:
    """
    Merge deterministic and LLM dimension scores.

    LLM scores override deterministic placeholder scores (0.5) for the same
    dimension name. Deterministic scores for dimensions not in llm_scores are
    kept as-is.

    Design note: deterministic placeholders are set to 0.5 to signal "unassessed".
    Merging on dimension name means the caller does not need to know which grader
    produced which score — the composite grader sees a unified dict.
    """
    merged = dict(det_scores)
    merged.update(llm_scores)
    return merged


def grade_episode_with_llm(
    episode: Episode,
    llm_dimensions: list[str],
    weights: dict[str, float],
    hard_fail_cap: float = 0.3,
) -> Episode:
    """
    Re-grade an existing Episode by adding LLM scores for specified dimensions.

    Imports llm_judge at call time so that deterministic-only mode never
    requires the anthropic package to be installed.

    Returns a new Episode with updated grader_scores, composite_score,
    and grader_reasoning. Flags from the original episode are preserved.

    Raises collab_eval.graders.llm_judge.LLMJudgeUnavailableError if
    ANTHROPIC_API_KEY is not set.
    """
    from collab_eval.graders import llm_judge

    spec: TaskSpec = episode.spec
    llm_results = llm_judge.score_dimensions_batch(
        dimensions=llm_dimensions,
        spec_intent=spec.intent,
        input_doc=spec.input_doc,
        agent_output=episode.agent_output,
        source_docs=spec.source_docs if spec.source_docs else None,
    )

    llm_score_values = {dim: result.score for dim, result in llm_results.items()}
    reasoning = {dim: result.reasoning for dim, result in llm_results.items()}
    merged = merge_scores(episode.grader_scores, llm_score_values)
    composite = weighted_score(merged, weights)

    if episode.flags:
        composite = apply_hard_fail_cap(composite, cap=hard_fail_cap)

    return Episode(
        spec=episode.spec,
        agent_output=episode.agent_output,
        grader_scores=merged,
        composite_score=composite,
        flags=episode.flags,
        grader_reasoning=reasoning,
    )
