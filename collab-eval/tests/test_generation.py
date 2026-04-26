"""
Tests for the synthetic spreadsheet task generator.

All tests run offline without Anthropic credentials.
Tests verify:
  - Determinism (same seed → same output)
  - Meaningful variation (different seed → different output)
  - Schema validity of generated cases
  - Coverage targets (post-hoc check for n=80)
  - expected_metadata sufficient for deterministic grading
  - No private/proprietary/internal strings in generated content
  - Reward-hacking probes present in generated set
"""

from __future__ import annotations

import json
import re

import pytest

from collab_eval.generation.spreadsheet_generator import (
    generate_cases,
    cases_to_jsonl,
    load_jsonl,
    REQUIRED_COLUMNS,
    GeneratedCase,
)
from collab_eval.graders import deterministic as det


_FORBIDDEN_STRINGS = [
    "microsoft", "teams", "sharepoint", "azure", "office 365",
    "internal", "confidential", "proprietary",
]

VALID_DIMENSIONS = {
    "data_preservation", "unit_consistency", "format_validity", "completeness"
}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
REQUIRED_METADATA_KEYS = {
    "expected_row_count", "required_columns", "unit_normalization", "allowed_missing_fields"
}


class TestDeterminism:
    def test_same_seed_same_output(self):
        cases_a = generate_cases(n=20, seed=42)
        cases_b = generate_cases(n=20, seed=42)
        assert len(cases_a) == len(cases_b)
        for a, b in zip(cases_a, cases_b):
            assert a.case_id == b.case_id
            assert a.input == b.input
            assert a.gold_or_reference_output == b.gold_or_reference_output
            assert a.expected_metadata == b.expected_metadata
            assert a.difficulty == b.difficulty
            assert a.primary_dimension == b.primary_dimension

    def test_different_seed_different_output(self):
        cases_42 = generate_cases(n=20, seed=42)
        cases_99 = generate_cases(n=20, seed=99)
        inputs_42 = {c.input for c in cases_42}
        inputs_99 = {c.input for c in cases_99}
        # Different seeds should produce meaningfully different inputs
        overlap = inputs_42 & inputs_99
        assert len(overlap) < len(inputs_42) // 2, (
            "Too many identical inputs across different seeds — "
            "generation is not sufficiently seed-sensitive"
        )

    def test_n_controls_case_count(self):
        for n in (10, 40, 80):
            cases = generate_cases(n=n, seed=1)
            assert len(cases) == n


class TestSchema:
    def test_all_required_fields_present(self):
        cases = generate_cases(n=10, seed=42)
        for case in cases:
            assert case.case_id
            assert case.task_type == "spreadsheet_clean"
            assert case.input
            assert case.gold_or_reference_output
            assert case.generation_seed == 42
            assert case.difficulty in VALID_DIFFICULTIES
            assert case.primary_dimension in VALID_DIMENSIONS
            assert isinstance(case.known_failure_modes, list)

    def test_expected_metadata_keys(self):
        cases = generate_cases(n=10, seed=42)
        for case in cases:
            meta = case.expected_metadata
            assert REQUIRED_METADATA_KEYS.issubset(meta.keys()), (
                f"{case.case_id}: missing metadata keys {REQUIRED_METADATA_KEYS - meta.keys()}"
            )

    def test_expected_metadata_types(self):
        cases = generate_cases(n=10, seed=42)
        for case in cases:
            meta = case.expected_metadata
            assert isinstance(meta["expected_row_count"], int)
            assert meta["expected_row_count"] >= 1
            assert isinstance(meta["required_columns"], list)
            assert len(meta["required_columns"]) > 0
            assert isinstance(meta["unit_normalization"], dict)
            assert "forbidden_pattern" in meta["unit_normalization"]

    def test_required_columns_correct(self):
        cases = generate_cases(n=10, seed=42)
        for case in cases:
            assert case.expected_metadata["required_columns"] == REQUIRED_COLUMNS

    def test_case_ids_unique(self):
        cases = generate_cases(n=80, seed=42)
        ids = [c.case_id for c in cases]
        assert len(ids) == len(set(ids)), "Case IDs must be unique"

    def test_jsonl_roundtrip(self, tmp_path):
        cases = generate_cases(n=5, seed=42)
        jsonl = cases_to_jsonl(cases)
        path = tmp_path / "test.jsonl"
        path.write_text(jsonl, encoding="utf-8")
        loaded = load_jsonl(str(path))
        assert len(loaded) == 5
        for orig, loaded_rec in zip(cases, loaded):
            assert orig.case_id == loaded_rec["case_id"]
            assert orig.input == loaded_rec["input"]


class TestGradability:
    def test_gold_output_parseable(self):
        """Gold outputs must be parseable CSV."""
        cases = generate_cases(n=40, seed=42)
        for case in cases:
            assert det.csv_parseable(case.gold_or_reference_output), (
                f"{case.case_id}: gold output is not parseable CSV"
            )

    def test_gold_output_row_count_matches_metadata(self):
        """Gold output must have exactly expected_row_count data rows."""
        cases = generate_cases(n=40, seed=42)
        for case in cases:
            expected = case.expected_metadata["expected_row_count"]
            assert det.row_count_preserved(case.gold_or_reference_output, expected), (
                f"{case.case_id}: gold output row count does not match "
                f"expected_row_count={expected}"
            )

    def test_gold_output_has_required_headers(self):
        """Gold outputs must have all required columns."""
        cases = generate_cases(n=40, seed=42)
        for case in cases:
            required = set(case.expected_metadata["required_columns"])
            score = det.headers_preserved(case.gold_or_reference_output, required)
            assert score == 1.0, (
                f"{case.case_id}: gold output missing required headers "
                f"(completeness={score})"
            )

    def test_gold_output_units_clean(self):
        """Gold outputs must pass the unit_normalized check."""
        cases = generate_cases(n=40, seed=42)
        for case in cases:
            pattern = case.expected_metadata["unit_normalization"]["forbidden_pattern"]
            assert det.unit_normalized(case.gold_or_reference_output, pattern), (
                f"{case.case_id}: gold output contains forbidden unit pattern '{pattern}'"
            )

    def test_input_differs_from_gold(self):
        """At least for non-easy cases, the input should differ from the gold output."""
        cases = generate_cases(n=40, seed=42)
        non_trivial = [c for c in cases if c.difficulty in ("medium", "hard")]
        different = sum(1 for c in non_trivial if c.input != c.gold_or_reference_output)
        assert different > len(non_trivial) * 0.8, (
            "Expected most medium/hard inputs to differ from gold output"
        )


class TestCoverage:
    def test_coverage_targets_met_n80(self):
        """Post-hoc coverage check for n=80."""
        cases = generate_cases(n=80, seed=42)
        from collections import Counter
        dim_counts = Counter(c.primary_dimension for c in cases)

        assert dim_counts["data_preservation"] >= 20, (
            f"data_preservation: {dim_counts['data_preservation']} < 20"
        )
        assert dim_counts["unit_consistency"] >= 20, (
            f"unit_consistency: {dim_counts['unit_consistency']} < 20"
        )
        fv_c = dim_counts["format_validity"] + dim_counts["completeness"]
        assert fv_c >= 20, (
            f"format_validity + completeness: {fv_c} < 20"
        )

    def test_all_difficulties_present_n80(self):
        cases = generate_cases(n=80, seed=42)
        from collections import Counter
        diff_counts = Counter(c.difficulty for c in cases)
        assert diff_counts["easy"] >= 1
        assert diff_counts["medium"] >= 1
        assert diff_counts["hard"] >= 1

    def test_reward_hacking_probes_present_n80(self):
        """Generated set must include explicit reward-hacking probe cases."""
        cases = generate_cases(n=80, seed=42)
        all_modes = set()
        for c in cases:
            all_modes.update(c.known_failure_modes)
        required_probes = {"duplicated_rows", "units_hidden_in_notes"}
        for probe in required_probes:
            assert probe in all_modes, (
                f"Reward-hacking probe '{probe}' not found in any generated case"
            )


class TestNoPrivateData:
    def test_no_forbidden_strings(self):
        """Generated cases must not contain private/proprietary/internal strings."""
        cases = generate_cases(n=80, seed=42)
        for case in cases:
            content = (case.input + case.gold_or_reference_output).lower()
            for forbidden in _FORBIDDEN_STRINGS:
                assert forbidden not in content, (
                    f"{case.case_id}: contains forbidden string '{forbidden}'"
                )
