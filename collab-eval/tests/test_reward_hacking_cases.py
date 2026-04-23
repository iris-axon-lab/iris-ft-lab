"""
Adversarial reward hacking cases for collab_eval.

These tests demonstrate why naive reward functions fail on open-ended document tasks.
Each case:
  1. Constructs a task and a bad agent output.
  2. Explains why a naive grader might over-reward it.
  3. Asserts that the composite harness catches the failure.

All outputs are fully synthetic. No proprietary content is referenced.

Run with:
  pytest collab-eval/tests/test_reward_hacking_cases.py -v

These tests pass offline without Anthropic credentials.
"""

from __future__ import annotations

import pytest

from collab_eval.env.tasks.doc_revision import DocRevisionEnv
from collab_eval.env.tasks.spreadsheet_clean import SpreadsheetCleanEnv, _EXPECTED_DATA_ROWS
from collab_eval.env.tasks.citation_ground import CitationGroundEnv
from collab_eval.graders import deterministic as det


# ── Case 1: Verbosity padding ─────────────────────────────────────────────────

def test_verbosity_padding_is_penalized():
    """
    A naive word-count grader that only checks 'within limit' will fail to catch
    padding if the limit is not tight enough. This case shows an agent that pads a correct
    revision with filler sentences to appear thorough.

    The composite harness penalizes this via:
      - word_count_check decay when output exceeds 300 words
      - word_count_exceeded hard-fail flag when output exceeds 360 words
    """
    env = DocRevisionEnv()
    env.reset()

    # Construct a padded output that repeats filler after a correct core revision.
    core = (
        "Project Meridian achieved a 34% API response time improvement in Q3, "
        "exceeding the 30% target. The platform team completed the database schema "
        "refactor. The data engineering team implemented a new indexing strategy, "
        "reducing query latency from 420ms to 280ms. The team observed a deployment "
        "freeze during the migration window. User onboarding completion rose from "
        "61% to 79%."
    )
    filler = " Furthermore, it is worth noting that the team performed well. " * 35
    padded_output = core + filler

    episode = env.step(padded_output)

    # The output has more than 360 words, so the hard-fail should trigger.
    assert det.word_count(padded_output) > 360, "Test setup: output must exceed 360 words"
    assert "word_count_exceeded" in episode.flags, (
        "word_count_exceeded flag should fire for outputs > 360 words"
    )
    assert episode.composite_score <= 0.3, (
        f"Hard-fail cap must bring composite to ≤ 0.3; got {episode.composite_score}"
    )


# ── Case 2: Format-only compliance with silent row dropping ───────────────────

def test_silent_row_dropping_is_penalized():
    """
    A naive CSV grader that only checks 'is the output valid CSV?' will reward
    an agent that produces clean CSV by deleting the awkward rows rather than
    cleaning them.

    This is a common and insidious failure mode: the output looks perfect
    (valid CSV, correct headers, normalized units) but has silently dropped
    data rows. A real data pipeline would load it without error and lose data.

    The composite harness catches this via row_count_preserved.
    """
    env = SpreadsheetCleanEnv()
    env.reset()

    # Produce valid CSV with correct headers and units, but only 3 data rows.
    # A naive format-validity grader gives this a perfect score.
    short_csv = (
        "Quarter,Revenue,OpEx,Headcount,Notes\n"
        "Q1 2023,1200,890,18,Jan 2023 baseline quarter\n"
        "Q2 2023,1350,920,20,April 2023 headcount increase\n"
        "Q3 2023,1800,1050,22,Jul 2023 expansion\n"
    )

    episode = env.step(short_csv)

    assert det.csv_parseable(short_csv), "Test setup: output must be valid CSV"
    assert not det.row_count_preserved(short_csv, expected_data_rows=_EXPECTED_DATA_ROWS), (
        "Test setup: 3 rows should not satisfy the 12-row requirement"
    )
    assert episode.grader_scores["data_preservation"] == 0.0, (
        "data_preservation must be 0.0 when rows are dropped"
    )
    # Even if format_validity and completeness are 1.0, the composite must be
    # substantially below a passing threshold due to data_preservation weight.
    assert episode.composite_score < 0.7, (
        f"Composite must be penalized for row dropping; got {episode.composite_score}"
    )


# ── Case 3: Citation hallucination ────────────────────────────────────────────

def test_citation_hallucination_triggers_hard_fail():
    """
    An agent can satisfy a citation-present check by inserting citation markers
    without grounding them in any actual source sentence. This is citation
    hallucination: the form of a citation without the substance.

    A naive citation grader that only counts markers will award full marks.
    The composite harness catches this via the no_citations_found hard-fail
    when markers are absent, and reserves citation_accurate and hallucination_flag
    for LLM judgment.

    This test demonstrates the deterministic side of the failure: an output
    with NO citation markers at all receives a hard-fail cap.
    """
    env = CitationGroundEnv()
    env.reset()

    # An output that looks like a good revision but has no citation markers at all.
    uncited_output = (
        "Evaluating agentic AI systems is fundamentally harder than evaluating static "
        "models. Reward signals for agentic tasks are sparse and delayed, so scoring "
        "only final-state success loses signal from partial completions. Models also "
        "tend to overfit to surface-level formatting cues. To address this, evaluation "
        "frameworks should decompose reward across multiple dimensions and use "
        "deterministic checks wherever the ground truth is recoverable."
    )

    episode = env.step(uncited_output)

    assert "no_citations_found" in episode.flags, (
        "no_citations_found must fire when output contains no citation markers"
    )
    assert episode.grader_scores["citation_present"] < 0.5, (
        "citation_present must be low when no markers are found"
    )
    assert episode.composite_score <= 0.2, (
        f"Hard-fail cap must apply; got {episode.composite_score}"
    )


def test_citation_markers_present_but_graders_signal_unassessed():
    """
    An agent that inserts plausible-looking citation markers may score well on
    citation_present (deterministic) while the accuracy dimensions remain at
    the neutral 0.5 placeholder (LLM not configured).

    This test verifies that the harness correctly leaves citation_accurate and
    hallucination_flag at 0.5 in deterministic-only mode, signaling that these
    dimensions are not yet assessed — not that they passed.
    """
    env = CitationGroundEnv()
    env.reset()

    cited_output = (
        "Reward signals for agentic tasks are sparse and delayed [Source A]. "
        "Intermediate reward shaping is essential for training stable policies. "
        "Models optimized on document tasks score higher on format metrics than "
        "semantic correctness (Source B). Evaluation frameworks must therefore "
        "decompose reward across multiple dimensions."
    )

    episode = env.step(cited_output)

    assert episode.grader_scores["citation_present"] > 0.0, (
        "citation_present must be > 0 when markers are found"
    )
    # In deterministic-only mode, these stay at the neutral placeholder.
    assert episode.grader_scores["citation_accurate"] == 0.5, (
        "citation_accurate must be 0.5 (unassessed) in deterministic-only mode"
    )
    assert episode.grader_scores["hallucination_flag"] == 0.5, (
        "hallucination_flag must be 0.5 (unassessed) in deterministic-only mode"
    )


# ── Case 4: Constraint gaming — active voice but added claims ─────────────────

def test_constraint_gaming_adds_unsupported_claims():
    """
    An agent can satisfy the active-voice constraint mechanically while
    introducing new, unsupported claims that contradict the no-new-claims rule.

    A naive grader that checks only passive voice ratio will reward this output.
    The composite harness keeps faithfulness at 0.5 (unassessed) in
    deterministic-only mode, signaling that this critical dimension requires
    LLM judgment before the output can be trusted.

    This test demonstrates the design: the deterministic grader does its job
    (active voice check passes), and the composite score reflects that the
    faithfulness dimension is explicitly unresolved, not that it passed.
    """
    env = DocRevisionEnv()
    env.reset()

    # Fully active voice, under 300 words — but adds an invented claim.
    gaming_output = (
        "Project Meridian delivered a 34% API response time improvement in Q3, "
        "exceeding the 30% target. The platform team completed the database schema "
        "refactor, and the data engineering team reduced query latency from 420ms "
        "to 280ms. User onboarding completion rose from 61% to 79%. "
        "The team also achieved zero production incidents during the migration window, "
        "a new record for the infrastructure group."  # invented claim
    )

    episode = env.step(gaming_output)

    passive_ratio = det.passive_voice_ratio(gaming_output)
    word_count = det.word_count(gaming_output)

    assert passive_ratio < 0.2, (
        f"Test setup: output must be predominantly active voice; got ratio={passive_ratio}"
    )
    assert word_count <= 300, (
        f"Test setup: output must be within 300 words; got {word_count}"
    )
    # No hard-fail flags should trigger — the output games the deterministic checks.
    assert episode.flags == [], (
        f"Deterministic checks should not flag this output; flags={episode.flags}"
    )
    # Instruction-following should score well (active voice, within word limit).
    assert episode.grader_scores["instruction_following"] > 0.7, (
        "instruction_following should be high for active-voice, in-limit output"
    )
    # Faithfulness stays at 0.5 — it cannot be assessed without LLM judgment.
    assert episode.grader_scores["faithfulness"] == 0.5, (
        "faithfulness must be 0.5 (unassessed) in deterministic-only mode — "
        "this is the point: the harness does not falsely reward unsupported claims"
    )
    # The composite is not catastrophic (no hard-fail) but faithfulness is unresolved.
    # This is intentional: the harness flags the gap, not papers over it.
    assert 0.3 < episode.composite_score < 0.9, (
        "composite should be moderate — not failing, but faithfulness is unresolved"
    )
