"""
Demo runner for collab_eval.

Loads one example task of each type, runs deterministic grading on representative
agent outputs, and optionally runs LLM grading if ANTHROPIC_API_KEY is set.

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
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _print_episode(episode: Episode) -> None:
    """Print episode results in a readable format."""
    print(f"  Task:      {episode.spec.task_id}")
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
            wrapped = textwrap.fill(reason, width=56, initial_indent="    ", subsequent_indent="    ")
            print(f"    [{dim}]")
            print(wrapped)
    else:
        has_placeholders = any(
            d in _PLACEHOLDER_DIMENSIONS for d in episode.grader_scores
        )
        if has_placeholders:
            print("  * placeholder (0.5) — requires LLM judge for real assessment")


def run_doc_revision_demo(use_llm: bool = False) -> None:
    """Demo the document revision task."""
    from collab_eval.env.tasks.doc_revision import DocRevisionEnv
    from collab_eval.graders import composite as comp

    _section("Task 1: Document Revision")

    env = DocRevisionEnv()
    spec = env.reset()
    print(f"\n  Intent: {spec.intent}\n")
    print(f"  Constraints: {spec.constraints}\n")

    # A representative agent output: active voice, within limit, fixes structural
    # issue but still contains one unsupported addition.
    agent_output = (
        "Project Meridian delivered a 34% API response time improvement in Q3, "
        "exceeding the 30% target — the headline result for the quarter. "
        "The platform team completed the database schema refactor. "
        "The data engineering team implemented a new indexing strategy, "
        "reducing query latency from 420ms to 280ms. "
        "The team observed a deployment freeze during the migration window. "
        "User onboarding completion rose from 61% to 79%, driven by the "
        "redesigned flow the product team shipped in early July. "
        "Headcount expanded from 24 to 31, primarily through infrastructure hires. "
        "Q4 priorities include observability tooling, two new partner API endpoints, "
        "and a 40% MTTR reduction target for P1 incidents. "
        "The planned data lake migration has been descoped to Q1 due to budget constraints."
    )

    episode = env.step(agent_output)

    if use_llm:
        from collab_eval.graders.llm_judge import LLMJudgeUnavailableError
        from collab_eval.env.tasks.doc_revision import _WEIGHTS
        try:
            episode = comp.grade_episode_with_llm(
                episode=episode,
                llm_dimensions=["faithfulness", "quality_delta"],
                weights=_WEIGHTS,
            )
        except LLMJudgeUnavailableError as e:
            print(f"\n  [LLM judge skipped: {e}]")

    _print_episode(episode)


def run_spreadsheet_demo(use_llm: bool = False) -> None:
    """Demo the spreadsheet cleanup task."""
    from collab_eval.env.tasks.spreadsheet_clean import SpreadsheetCleanEnv

    _section("Task 2: Spreadsheet Cleanup")

    env = SpreadsheetCleanEnv()
    spec = env.reset()
    print(f"\n  Intent: {spec.intent}\n")
    print(f"  Constraints: {spec.constraints}\n")

    # A good agent output: clean CSV, normalized units, no blank rows, no header artifact.
    agent_output = (
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

    episode = env.step(agent_output)
    _print_episode(episode)


def run_citation_demo(use_llm: bool = False) -> None:
    """Demo the citation-grounded editing task."""
    from collab_eval.env.tasks.citation_ground import CitationGroundEnv
    from collab_eval.graders import composite as comp

    _section("Task 3: Citation-Grounded Editing")

    env = CitationGroundEnv()
    spec = env.reset()
    print(f"\n  Intent: {spec.intent}\n")
    print(f"  Constraints: {spec.constraints}\n")

    # A good agent output: two citation markers, argument preserved.
    agent_output = (
        "Evaluating agentic AI systems presents fundamentally different challenges "
        "than evaluating static language models.\n\n"
        "One key challenge is that reward signals for agentic tasks are often sparse "
        "and delayed. Scoring only final-state success fails to distinguish a near-miss "
        "from a complete failure, making intermediate reward shaping essential for "
        "stable policy training [Source A].\n\n"
        "A second challenge is that agents operating in document-manipulation tasks "
        "tend to overfit to surface-level formatting cues rather than semantic "
        "correctness. Studies have found that models optimized for human preference "
        "ratings improve on formatting dimensions while showing no significant gain "
        "on factual accuracy (Source B).\n\n"
        "Addressing these challenges requires evaluation frameworks that decompose "
        "reward across multiple dimensions, applying deterministic checks wherever "
        "the ground truth is recoverable and reserving model-based judgment for "
        "dimensions where no algorithmic check is feasible."
    )

    episode = env.step(agent_output)

    if use_llm:
        from collab_eval.graders.llm_judge import LLMJudgeUnavailableError
        from collab_eval.env.tasks.citation_ground import _WEIGHTS
        try:
            episode = comp.grade_episode_with_llm(
                episode=episode,
                llm_dimensions=["citation_accurate", "hallucination_flag"],
                weights=_WEIGHTS,
            )
        except LLMJudgeUnavailableError as e:
            print(f"\n  [LLM judge skipped: {e}]")

    _print_episode(episode)


def run_demo() -> None:
    """Run the full demo across all three task types."""
    use_llm = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())

    print("\ncollab_eval — virtual-collaborator RL task harness demo")
    if use_llm:
        print("LLM grading: ENABLED (ANTHROPIC_API_KEY detected)")
    else:
        print("LLM grading: DISABLED (running in deterministic-only mode)")
        print("  Set ANTHROPIC_API_KEY to enable LLM-based dimension scoring.")

    run_doc_revision_demo(use_llm=use_llm)
    run_spreadsheet_demo(use_llm=use_llm)
    run_citation_demo(use_llm=use_llm)

    print(f"\n{'─' * 60}")
    print("  Demo complete.")
    print(f"{'─' * 60}\n")
