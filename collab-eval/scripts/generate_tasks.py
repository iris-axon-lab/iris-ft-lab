"""
Generate synthetic task cases for collab_eval.

Usage:
    python scripts/generate_tasks.py \\
        --task spreadsheet_clean --n 80 --seed 42 \\
        --output data/generated/spreadsheet_clean_v1.jsonl

Only spreadsheet_clean is fully implemented in this cycle.
doc_revision and citation_ground are stub placeholders.

Coverage targets verified post-generation (not enforced during generation):
  - >= 20 cases: primary_dimension = data_preservation
  - >= 20 cases: primary_dimension = unit_consistency
  - >= 20 cases: primary_dimension = format_validity or completeness
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

# Allow running from repo root or from collab-eval/
_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))

from collab_eval.generation.spreadsheet_generator import generate_cases, cases_to_jsonl


def _verify_coverage(cases: list, n: int) -> list[str]:
    """Return a list of coverage warnings (empty if all targets met)."""
    dim_counts: dict[str, int] = collections.Counter(c.primary_dimension for c in cases)
    warnings = []

    if dim_counts.get("data_preservation", 0) < 20:
        warnings.append(
            f"data_preservation: {dim_counts.get('data_preservation', 0)} cases (target ≥ 20)"
        )
    if dim_counts.get("unit_consistency", 0) < 20:
        warnings.append(
            f"unit_consistency: {dim_counts.get('unit_consistency', 0)} cases (target ≥ 20)"
        )
    fv_completeness = dim_counts.get("format_validity", 0) + dim_counts.get("completeness", 0)
    if fv_completeness < 20:
        warnings.append(
            f"format_validity + completeness: {fv_completeness} cases (target ≥ 20)"
        )
    return warnings


def run(task: str, n: int, seed: int, output: str) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if task == "spreadsheet_clean":
        print(f"Generating {n} spreadsheet_clean cases (seed={seed})...")
        cases = generate_cases(n=n, seed=seed)

        warnings = _verify_coverage(cases, n)
        if warnings:
            print("Coverage warnings:")
            for w in warnings:
                print(f"  ! {w}")
        else:
            print("Coverage targets met.")

        dim_counts: dict[str, int] = collections.Counter(c.primary_dimension for c in cases)
        diff_counts: dict[str, int] = collections.Counter(c.difficulty for c in cases)
        print(f"Dimension distribution: {dict(dim_counts)}")
        print(f"Difficulty distribution: {dict(diff_counts)}")

        jsonl = cases_to_jsonl(cases)
        output_path.write_text(jsonl, encoding="utf-8")
        print(f"Wrote {len(cases)} cases → {output_path}")

    elif task == "doc_revision":
        print("doc_revision generation: STUB — not implemented in training cycle v0.")
        print("This task type requires an LLM judge for its key dimensions.")
        print("Skipping.")

    elif task == "citation_ground":
        print("citation_ground generation: STUB — not implemented in training cycle v0.")
        print("This task type requires an LLM judge for its key dimensions.")
        print("Skipping.")

    else:
        print(f"Unknown task type: {task!r}")
        print("Supported: spreadsheet_clean, doc_revision (stub), citation_ground (stub)")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic eval task cases.")
    parser.add_argument(
        "--task",
        required=True,
        choices=["spreadsheet_clean", "doc_revision", "citation_ground"],
        help="Task type to generate.",
    )
    parser.add_argument("--n", type=int, default=80, help="Number of cases to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path.",
    )
    args = parser.parse_args()
    run(task=args.task, n=args.n, seed=args.seed, output=args.output)


if __name__ == "__main__":
    main()
