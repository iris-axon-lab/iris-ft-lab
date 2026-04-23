"""
Demo runner for collab_eval.

For each task type, shows two agent outputs side by side:
  - A good output that scores well on the deterministic dimensions.
  - A bad output that games a naive grader but is caught by the harness.

Designed to be runnable offline with no API credentials.

Usage:
  python scripts/run_demo.py
  ANTHROPIC_API_KEY=sk-... python scripts/run_demo.py
"""

from __future__ import annotations

import os
import textwrap

from collab_eval.base import Episode

# Dimensions that are placeholders in deterministic-only mode.
# These are set to 0.5 to signal "not yet assessed" rather than "passing".
_PLACEHOLDER_DIMENSIONS = frozenset(
    {"faithfulness", "quality_delta", "citation_accurate", "hallucination_flag"}
)


def _section(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def _subsection(label: str) -> None:
    trailing = "─" * max(2, 54 - len(label))
    print(f"\n  ── {label} {trailing}")


def _print_episode(episode: Episode) -> None:
    """Print episode scores, flags, and placeholder legend."""
    print(f"  Composite: {episode.composite_score:.3f}")
    if episode.flags:
        print(f"  Flags:     {', '.join(episode.flags)}")
    else:
        print("  Flags:     (none)")
    print("  Scores:")
    for dim, score in episode.grader_scores.items():
        needs_llm = dim in _PLACEHOLDER_DIMENSIONS and episode.grader_reasoning is None
        marker = "  *" if needs_llm else ""
        print(f"    {dim:<28} {score:.3f}{marker}")
    if episode.grader_reasoning:
        print("  Reasoning (LLM):")
        for dim, reason in episode.grader_reasoning.items():
            wrapped = textwrap.fill(
                reason, width=54, initial_indent="    ", subsequent_indent="    "
            )
            print(f"    [{dim}]")
            print(wrapped)
    else:
        has_placeholders = any(
            d in _PLACEHOLDER_DIMENSIONS for d in episode.grader_scores
        )
        if has_placeholders:
            print("  * placeholder (0.5) — requires LLM judge for real assessment")


def run_doc_revision_demo(use_llm: bool = False) -> None:
    """
    Demo the document revision task.

    Good agent: active voice, within word limit, fixes the structural issue.
    Bad agent:  correct core content buried under padding — games a naive
                word-count check but triggers the word_count_exceeded hard-fail.
    """
    from collab_eval.env.tasks.doc_revision import DocRevisionEnv, _WEIGHTS
    from collab_eval.graders import composite as comp

    _section("Task 1: Document Revision")

    env = DocRevisionEnv()
    spec = env.reset()
    print(f"\n  Intent:      {spec.intent}")
    print(f"  Constraints: {'; '.join(spec.constraints)}")

    # ── Good agent ────────────────────────────────────────────────────────────
    # Active voice, under 300 words, buries nothing — the headline result leads.
    good_output = (
        "Project Meridian delivered a 34% API response time improvement in Q3, "
        "exceeding the 30% target. "
        "The platform team completed the database schema refactor, and the data "
        "engineering team implemented a new indexing strategy that reduced query "
        "latency from 420ms to 280ms. The team held a deployment freeze during "
        "the migration window. "
        "User onboarding completion rose from 61% to 79%, driven by the redesigned "
        "flow the product team shipped in early July. Headcount expanded from 24 "
        "to 31 through infrastructure hires. "
        "Q4 priorities include observability tooling, two new partner API endpoints, "
        "and a 40% MTTR reduction target for P1 incidents. The data lake migration "
        "has been descoped to Q1 due to budget constraints."
    )
    good_episode = env.step(good_output)

    if use_llm:
        from collab_eval.graders.llm_judge import LLMJudgeUnavailableError
        try:
            good_episode = comp.grade_episode_with_llm(
                episode=good_episode,
                llm_dimensions=["faithfulness", "quality_delta"],
                weights=_WEIGHTS,
            )
        except LLMJudgeUnavailableError as e:
            print(f"\n  [LLM judge skipped: {e}]")

    _subsection("good agent")
    _print_episode(good_episode)

    # ── Bad agent: verbosity padding ──────────────────────────────────────────
    # Correct facts, but padded with filler to appear thorough.
    # A naive grader that checks only "is output non-empty?" rewards this.
    # The harness triggers word_count_exceeded and caps the composite at 0.3.
    core = (
        "Project Meridian achieved a 34% API response time improvement in Q3, "
        "exceeding the 30% target. The platform team completed the database schema "
        "refactor. The data engineering team reduced query latency from 420ms to 280ms. "
        "User onboarding completion rose from 61% to 79%."
    )
    filler = " Furthermore, it is worth noting that the team performed well. " * 35
    bad_output = core + filler

    bad_episode = env.step(bad_output)
    _subsection("bad agent — verbosity padding (word_count_exceeded)")
    _print_episode(bad_episode)


def run_spreadsheet_demo(use_llm: bool = False) -> None:
    """
    Demo the spreadsheet cleanup task.

    Good agent: clean CSV, all 12 rows preserved, units normalized to $K.
    Bad agent:  clean CSV with only 3 rows — silently drops 9 rows of data.
                A format-only grader sees a perfect CSV; data_preservation catches it.
    """
    from collab_eval.env.tasks.spreadsheet_clean import SpreadsheetCleanEnv

    _section("Task 2: Spreadsheet Cleanup")

    env = SpreadsheetCleanEnv()
    spec = env.reset()
    print(f"\n  Intent:      {spec.intent}")
    print(f"  Constraints: {'; '.join(spec.constraints)}")

    # ── Good agent ────────────────────────────────────────────────────────────
    # All 12 data rows, $M values converted to $K, blank rows removed,
    # merged-header annotation removed, dates normalised to ISO format.
    good_output = (
        "Quarter,Revenue,OpEx,Headcount,Notes\n"
        "Q1 2023,1200,890,18,2023-01 baseline quarter\n"
        "Q2 2023,1350,920,20,2023-04 headcount increase approved\n"
        "Q3 2023,1800,1050,22,2023-07 expansion into APAC\n"
        "Q4 2023,2100,1140,24,2023-10 office lease renewed\n"
        "Q1 2024,2300,1200,26,2024-01 new sales team onboarded\n"
        "Q2 2024,2450,1250,28,2024-04 platform migration started\n"
        "Q3 2024,2800,1380,31,2024-07 infra hires completed\n"
        "Q4 2024,3100,1500,33,2024-10 partner integrations launched\n"
        "Q1 2025,3400,1620,35,2025-01 observability tooling shipped\n"
        "Q2 2025,3700,1700,37,2025-04 data lake descoped\n"
        "Q3 2025,4100,1820,40,2025-07 Series B closed\n"
        "Q4 2025,4500,1950,42,2025-10 hiring freeze lifted\n"
    )
    good_episode = env.step(good_output)
    _subsection("good agent")
    _print_episode(good_episode)

    # ── Bad agent: silent row dropping ────────────────────────────────────────
    # Produces perfectly valid CSV with correct headers and normalised units —
    # but keeps only 3 of the 12 data rows, silently discarding the rest.
    # A naive format-validity grader scores this perfectly.
    # data_preservation catches it: expected 12 rows, found 3.
    bad_output = (
        "Quarter,Revenue,OpEx,Headcount,Notes\n"
        "Q1 2023,1200,890,18,2023-01 baseline quarter\n"
        "Q2 2023,1350,920,20,2023-04 headcount increase approved\n"
        "Q3 2023,1800,1050,22,2023-07 expansion into APAC\n"
    )
    bad_episode = env.step(bad_output)
    _subsection("bad agent — silent row dropping (data_preservation = 0.0)")
    _print_episode(bad_episode)


def run_citation_demo(use_llm: bool = False) -> None:
    """
    Demo the citation-grounded editing task.

    Good agent: two traceable inline citations, argument preserved.
    Bad agent:  well-written revision with zero citation markers.
                A length or fluency grader sees nothing wrong.
                The harness fires no_citations_found and caps composite at 0.2.
    """
    from collab_eval.env.tasks.citation_ground import CitationGroundEnv, _WEIGHTS
    from collab_eval.graders import composite as comp

    _section("Task 3: Citation-Grounded Editing")

    env = CitationGroundEnv()
    spec = env.reset()
    print(f"\n  Intent:      {spec.intent}")
    print(f"  Constraints: {'; '.join(spec.constraints)}")

    # ── Good agent ────────────────────────────────────────────────────────────
    # Two citation markers, each traceable to a source sentence.
    # Argument preserved: both anchor phrases survive.
    good_output = (
        "Evaluating agentic AI systems presents fundamentally different challenges "
        "than evaluating static language models.\n\n"
        "One key challenge is that reward signals for agentic tasks are often sparse "
        "and delayed. Scoring only final-state success fails to distinguish a near-miss "
        "from a complete failure, making intermediate reward shaping essential for "
        "stable policy training [Source A].\n\n"
        "A second challenge is that agents operating on document tasks tend to overfit "
        "to surface-level formatting cues rather than semantic correctness. Studies have "
        "found that models optimised for human preference ratings improve on formatting "
        "dimensions while showing no significant gain on factual accuracy (Source B).\n\n"
        "Addressing these challenges requires frameworks that decompose reward across "
        "multiple dimensions, applying deterministic checks wherever ground truth is "
        "recoverable and reserving model-based judgment where no algorithmic check suffices."
    )
    good_episode = env.step(good_output)

    if use_llm:
        from collab_eval.graders.llm_judge import LLMJudgeUnavailableError
        try:
            good_episode = comp.grade_episode_with_llm(
                episode=good_episode,
                llm_dimensions=["citation_accurate", "hallucination_flag"],
                weights=_WEIGHTS,
            )
        except LLMJudgeUnavailableError as e:
            print(f"\n  [LLM judge skipped: {e}]")

    _subsection("good agent")
    _print_episode(good_episode)

    # ── Bad agent: no citations ───────────────────────────────────────────────
    # Fluent, coherent revision — but zero citation markers.
    # A fluency or word-count grader sees nothing wrong.
    # citation_present = 0.0, no_citations_found flag fires, composite capped at 0.2.
    bad_output = (
        "Evaluating agentic AI systems is fundamentally harder than evaluating static "
        "models. Reward signals for agentic tasks are sparse and delayed, so scoring "
        "only final-state success discards signal from partial completions. Models also "
        "tend to overfit to formatting cues over semantic correctness. To address this, "
        "evaluation frameworks should decompose reward across multiple dimensions, "
        "using deterministic checks wherever the ground truth is recoverable."
    )
    bad_episode = env.step(bad_output)
    _subsection("bad agent — no citations (no_citations_found)")
    _print_episode(bad_episode)


def run_demo() -> None:
    """Run all three task demos, each with a good and a bad agent output."""
    use_llm = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())

    print("\ncollab_eval — task environment and grader harness demo")
    print("Each task shows a good agent output and a bad one that games a naive grader.")
    if use_llm:
        print("LLM grading: ENABLED (ANTHROPIC_API_KEY detected)")
    else:
        print("LLM grading: DISABLED — running in deterministic-only mode")
        print("  Set ANTHROPIC_API_KEY to enable LLM-based dimension scoring.")

    run_doc_revision_demo(use_llm=use_llm)
    run_spreadsheet_demo(use_llm=use_llm)
    run_citation_demo(use_llm=use_llm)

    print(f"\n{'═' * 60}")
    print("  Demo complete.")
    print(f"{'═' * 60}\n")
