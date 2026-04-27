"""
Generate synthetic task cases for collab_eval.

Usage:
    python scripts/generate_tasks.py \\
        --task spreadsheet_clean --n 240 --seed 100 \\
        --output data/generated/spreadsheet_train_v1.jsonl

    python scripts/generate_tasks.py \\
        --task spreadsheet_clean --n 80 --seed 200 \\
        --output data/generated/spreadsheet_heldout_v1.jsonl

    python scripts/generate_tasks.py \\
        --task spreadsheet_clean_stress --n 80 --seed 400 \\
        --output data/generated/spreadsheet_train_stress_v1.jsonl

The `spreadsheet_clean_stress` task type produces preservation-stress cases:
15–25 row tables where every row is real data and the gold preserves all rows
even when they look droppable ($M cells, blank Notes, marker-style notes).
Used as supplemental SFT v1 training data and as a separate held-out eval.

Only spreadsheet_clean is fully implemented in this cycle.
doc_revision and citation_ground are stub placeholders.

Coverage targets verified post-generation (not enforced during generation):
  - >= 20 cases: primary_dimension = data_preservation
  - >= 20 cases: primary_dimension = unit_consistency
  - >= 20 cases: primary_dimension = format_validity or completeness

Validation (--validate-against FILE):
  Checks that two generated files have:
  - no overlap by case_id
  - no overlap by input hash
  - expected metadata fields present in each
  - all deterministic grading dimensions covered
  - every record gradeable by the existing deterministic grader
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import sys
from pathlib import Path

# Allow running from repo root or from collab-eval/
_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))

from collab_eval.generation.spreadsheet_generator import (
    generate_cases,
    generate_preservation_stress_cases,
    cases_to_jsonl,
    load_jsonl,
    GeneratedCase,
)
from collab_eval.graders import deterministic as det


_REQUIRED_METADATA_KEYS = {
    "expected_row_count", "required_columns", "unit_normalization", "allowed_missing_fields"
}
_VALID_DIMENSIONS = {
    "data_preservation", "unit_consistency", "format_validity", "completeness"
}


def _input_hash(input_str: str) -> str:
    return hashlib.sha256(input_str.encode("utf-8")).hexdigest()


def validate_cases(cases: list[GeneratedCase]) -> list[str]:
    """
    Validate a list of GeneratedCase objects.

    Checks:
    - No duplicate case_ids within the set
    - No duplicate input hashes within the set
    - All expected metadata fields present
    - All required grading dimensions covered
    - Every case is gradeable by the deterministic grader (gold output parseable)

    Returns a list of error strings (empty if all checks pass).
    """
    errors: list[str] = []
    seen_ids: dict[str, int] = {}
    seen_hashes: dict[str, int] = {}
    covered_dims: set[str] = set()

    for i, case in enumerate(cases):
        # case_id uniqueness
        if case.case_id in seen_ids:
            errors.append(
                f"case[{i}]: duplicate case_id '{case.case_id}' "
                f"(also at index {seen_ids[case.case_id]})"
            )
        seen_ids[case.case_id] = i

        # input hash uniqueness
        h = _input_hash(case.input)
        if h in seen_hashes:
            errors.append(
                f"case[{i}] ({case.case_id}): duplicate input hash "
                f"(same as index {seen_hashes[h]})"
            )
        seen_hashes[h] = i

        # metadata fields
        meta = case.expected_metadata
        missing_keys = _REQUIRED_METADATA_KEYS - meta.keys()
        if missing_keys:
            errors.append(
                f"case[{i}] ({case.case_id}): missing metadata keys: {sorted(missing_keys)}"
            )

        # dimension validity
        if case.primary_dimension not in _VALID_DIMENSIONS:
            errors.append(
                f"case[{i}] ({case.case_id}): invalid primary_dimension "
                f"'{case.primary_dimension}'"
            )
        covered_dims.add(case.primary_dimension)

        # gradeability: gold output must be parseable CSV
        if not det.csv_parseable(case.gold_or_reference_output):
            errors.append(
                f"case[{i}] ({case.case_id}): gold output is not parseable CSV"
            )
        else:
            # row count must match metadata
            expected_rows = meta.get("expected_row_count")
            if expected_rows is not None:
                if not det.row_count_preserved(
                    case.gold_or_reference_output, expected_rows
                ):
                    errors.append(
                        f"case[{i}] ({case.case_id}): gold row count does not match "
                        f"expected_row_count={expected_rows}"
                    )

    # dimension coverage
    missing_dims = _VALID_DIMENSIONS - covered_dims
    if missing_dims:
        errors.append(
            f"Dataset missing primary_dimension coverage for: {sorted(missing_dims)}"
        )

    return errors


def _validate_stress_cases(cases: list[GeneratedCase]) -> list[str]:
    """
    Stress-specific integrity check. All preservation-stress cases share
    primary_dimension=data_preservation by design, so multi-dim coverage is
    intentionally not enforced. Per-case integrity matches validate_cases().
    """
    errors: list[str] = []
    seen_ids: dict[str, int] = {}
    seen_hashes: dict[str, int] = {}

    for i, case in enumerate(cases):
        if case.case_id in seen_ids:
            errors.append(
                f"case[{i}]: duplicate case_id '{case.case_id}' "
                f"(also at index {seen_ids[case.case_id]})"
            )
        seen_ids[case.case_id] = i

        h = _input_hash(case.input)
        if h in seen_hashes:
            errors.append(
                f"case[{i}] ({case.case_id}): duplicate input hash "
                f"(same as index {seen_hashes[h]})"
            )
        seen_hashes[h] = i

        meta = case.expected_metadata
        missing_keys = _REQUIRED_METADATA_KEYS - meta.keys()
        if missing_keys:
            errors.append(
                f"case[{i}] ({case.case_id}): missing metadata keys: "
                f"{sorted(missing_keys)}"
            )
        if not meta.get("preservation_stress"):
            errors.append(
                f"case[{i}] ({case.case_id}): missing preservation_stress flag"
            )
        if not meta.get("rows_must_preserve"):
            errors.append(
                f"case[{i}] ({case.case_id}): missing rows_must_preserve list"
            )

        if not det.csv_parseable(case.gold_or_reference_output):
            errors.append(
                f"case[{i}] ({case.case_id}): gold output is not parseable CSV"
            )
        else:
            expected_rows = meta.get("expected_row_count")
            if expected_rows is not None:
                if not det.row_count_preserved(
                    case.gold_or_reference_output, expected_rows
                ):
                    errors.append(
                        f"case[{i}] ({case.case_id}): gold row count != "
                        f"expected_row_count={expected_rows}"
                    )

    return errors


def validate_cases_from_dicts(records: list[dict]) -> list[str]:
    """
    Validate a list of raw JSONL dicts (as loaded by load_jsonl).
    Mirrors validate_cases but works on plain dicts.
    """
    errors: list[str] = []
    seen_ids: dict[str, int] = {}
    seen_hashes: dict[str, int] = {}
    covered_dims: set[str] = set()

    for i, rec in enumerate(records):
        case_id = rec.get("case_id", f"<index {i}>")

        # case_id uniqueness
        if case_id in seen_ids:
            errors.append(
                f"record[{i}]: duplicate case_id '{case_id}' "
                f"(also at index {seen_ids[case_id]})"
            )
        seen_ids[case_id] = i

        # input hash uniqueness
        input_str = rec.get("input", "")
        h = _input_hash(input_str)
        if h in seen_hashes:
            errors.append(
                f"record[{i}] ({case_id}): duplicate input hash "
                f"(same as index {seen_hashes[h]})"
            )
        seen_hashes[h] = i

        # metadata fields
        meta = rec.get("expected_metadata", {})
        missing_keys = _REQUIRED_METADATA_KEYS - meta.keys()
        if missing_keys:
            errors.append(
                f"record[{i}] ({case_id}): missing metadata keys: {sorted(missing_keys)}"
            )

        # dimension validity
        dim = rec.get("primary_dimension", "")
        if dim not in _VALID_DIMENSIONS:
            errors.append(
                f"record[{i}] ({case_id}): invalid primary_dimension '{dim}'"
            )
        covered_dims.add(dim)

        # gradeability
        gold = rec.get("gold_or_reference_output", "")
        if not det.csv_parseable(gold):
            errors.append(f"record[{i}] ({case_id}): gold output is not parseable CSV")
        else:
            expected_rows = meta.get("expected_row_count")
            if expected_rows is not None:
                if not det.row_count_preserved(gold, expected_rows):
                    errors.append(
                        f"record[{i}] ({case_id}): gold row count != expected_row_count={expected_rows}"
                    )

    missing_dims = _VALID_DIMENSIONS - covered_dims
    if missing_dims:
        errors.append(
            f"Dataset missing primary_dimension coverage for: {sorted(missing_dims)}"
        )

    return errors


def validate_no_overlap(
    records_a: list[dict], records_b: list[dict]
) -> list[str]:
    """
    Check that two record sets have no overlap by case_id or input hash.
    Returns a list of error strings (empty if no overlap).
    """
    errors: list[str] = []

    ids_a = {r["case_id"] for r in records_a if "case_id" in r}
    ids_b = {r["case_id"] for r in records_b if "case_id" in r}
    overlap_ids = ids_a & ids_b
    if overlap_ids:
        sample = sorted(overlap_ids)[:5]
        errors.append(
            f"case_id overlap ({len(overlap_ids)} records): {sample}"
            + (" ..." if len(overlap_ids) > 5 else "")
        )

    hashes_a = {_input_hash(r["input"]) for r in records_a if "input" in r}
    hashes_b = {_input_hash(r["input"]) for r in records_b if "input" in r}
    overlap_hashes = hashes_a & hashes_b
    if overlap_hashes:
        errors.append(
            f"input_hash overlap ({len(overlap_hashes)} records)"
        )

    return errors


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


def run(
    task: str, n: int, seed: int, output: str, validate_against: str | None = None
) -> None:
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

        # Validate generated cases
        errors = validate_cases(cases)
        if errors:
            print("Validation errors:")
            for e in errors:
                print(f"  ERROR: {e}")
            sys.exit(1)
        else:
            print("Single-file validation passed.")

        jsonl = cases_to_jsonl(cases)
        output_path.write_text(jsonl, encoding="utf-8")
        print(f"Wrote {len(cases)} cases → {output_path}")

        # Cross-file overlap validation
        if validate_against:
            against_path = Path(validate_against)
            if not against_path.exists():
                print(f"--validate-against: file not found: {against_path}")
                sys.exit(1)
            other_records = load_jsonl(str(against_path))
            new_records = load_jsonl(str(output_path))
            overlap_errors = validate_no_overlap(new_records, other_records)
            if overlap_errors:
                print("Overlap validation errors:")
                for e in overlap_errors:
                    print(f"  ERROR: {e}")
                sys.exit(1)
            else:
                print(f"No overlap with {validate_against}.")

    elif task == "spreadsheet_clean_stress":
        print(f"Generating {n} preservation-stress cases (seed={seed})...")
        cases = generate_preservation_stress_cases(n=n, seed=seed)

        dim_counts: dict[str, int] = collections.Counter(c.primary_dimension for c in cases)
        diff_counts: dict[str, int] = collections.Counter(c.difficulty for c in cases)
        print(f"Dimension distribution: {dict(dim_counts)}")
        print(f"Difficulty distribution: {dict(diff_counts)}")

        # Stress cases are all primary_dimension=data_preservation by design,
        # so the multi-dimension coverage check from regular spreadsheet_clean
        # does not apply. Validate only per-case integrity (parseability, row
        # counts, no duplicate ids/inputs).
        errors = _validate_stress_cases(cases)
        if errors:
            print("Validation errors:")
            for e in errors:
                print(f"  ERROR: {e}")
            sys.exit(1)
        else:
            print("Stress-case validation passed.")

        jsonl = cases_to_jsonl(cases)
        output_path.write_text(jsonl, encoding="utf-8")
        print(f"Wrote {len(cases)} stress cases → {output_path}")

        if validate_against:
            against_path = Path(validate_against)
            if not against_path.exists():
                print(f"--validate-against: file not found: {against_path}")
                sys.exit(1)
            other_records = load_jsonl(str(against_path))
            new_records = load_jsonl(str(output_path))
            overlap_errors = validate_no_overlap(new_records, other_records)
            if overlap_errors:
                print("Overlap validation errors:")
                for e in overlap_errors:
                    print(f"  ERROR: {e}")
                sys.exit(1)
            else:
                print(f"No overlap with {validate_against}.")

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
        print(
            "Supported: spreadsheet_clean, spreadsheet_clean_stress, "
            "doc_revision (stub), citation_ground (stub)"
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic eval task cases.")
    parser.add_argument(
        "--task",
        required=True,
        choices=[
            "spreadsheet_clean",
            "spreadsheet_clean_stress",
            "doc_revision",
            "citation_ground",
        ],
        help="Task type to generate.",
    )
    parser.add_argument("--n", type=int, default=80, help="Number of cases to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--validate-against",
        metavar="FILE",
        default=None,
        help="If set, validate no case_id or input_hash overlap with this existing JSONL file.",
    )
    args = parser.parse_args()
    run(
        task=args.task,
        n=args.n,
        seed=args.seed,
        output=args.output,
        validate_against=args.validate_against,
    )


if __name__ == "__main__":
    main()
