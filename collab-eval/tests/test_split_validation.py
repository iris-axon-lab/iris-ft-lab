"""
Tests for train/held-out split validation.

All tests use small fixtures (n <= 10) and run offline without GPU, API keys,
or generating the full 320-case dataset.

Covers:
- No overlap by case_id across splits
- No overlap by input hash across splits
- Expected metadata fields present
- All deterministic grading dimensions covered
- Every record gradeable by the existing deterministic grader
- No train/eval leakage
"""

from __future__ import annotations

import json

import pytest

from collab_eval.generation.spreadsheet_generator import generate_cases
from scripts.generate_tasks import (
    validate_cases,
    validate_cases_from_dicts,
    validate_no_overlap,
    _input_hash,
)
from collab_eval.graders import deterministic as det


# Small fixture sets — different seeds, different sizes
_TRAIN_CASES = generate_cases(n=8, seed=100)
_HELDOUT_CASES = generate_cases(n=4, seed=200)


class TestSplitNoOverlap:
    def test_case_ids_differ_between_splits(self):
        """Train and held-out splits must have no case_id overlap."""
        train_ids = {c.case_id for c in _TRAIN_CASES}
        heldout_ids = {c.case_id for c in _HELDOUT_CASES}
        assert train_ids.isdisjoint(heldout_ids), (
            f"case_id overlap: {train_ids & heldout_ids}"
        )

    def test_input_hashes_differ_between_splits(self):
        """Train and held-out splits must have no input hash overlap."""
        train_hashes = {_input_hash(c.input) for c in _TRAIN_CASES}
        heldout_hashes = {_input_hash(c.input) for c in _HELDOUT_CASES}
        assert train_hashes.isdisjoint(heldout_hashes), (
            "Input hash overlap between train and held-out splits"
        )

    def test_validate_no_overlap_passes_for_different_seeds(self):
        """validate_no_overlap should return no errors for splits from different seeds."""
        import dataclasses
        train_dicts = [dataclasses.asdict(c) for c in _TRAIN_CASES]
        heldout_dicts = [dataclasses.asdict(c) for c in _HELDOUT_CASES]
        errors = validate_no_overlap(train_dicts, heldout_dicts)
        assert errors == [], f"Unexpected overlap errors: {errors}"

    def test_validate_no_overlap_detects_same_seed_overlap(self):
        """validate_no_overlap should catch overlap when both sets come from same seed."""
        import dataclasses
        cases_a = generate_cases(n=4, seed=42)
        cases_b = generate_cases(n=4, seed=42)  # identical
        dicts_a = [dataclasses.asdict(c) for c in cases_a]
        dicts_b = [dataclasses.asdict(c) for c in cases_b]
        errors = validate_no_overlap(dicts_a, dicts_b)
        assert len(errors) > 0, "Expected overlap errors but got none"
        # Should report both case_id and input_hash overlap
        assert any("case_id" in e for e in errors)
        assert any("input_hash" in e for e in errors)


class TestMetadataPresent:
    def test_all_required_metadata_keys_present(self):
        for case in _TRAIN_CASES + _HELDOUT_CASES:
            meta = case.expected_metadata
            required = {"expected_row_count", "required_columns", "unit_normalization", "allowed_missing_fields"}
            missing = required - meta.keys()
            assert missing == set(), f"{case.case_id}: missing metadata keys {missing}"

    def test_metadata_types_valid(self):
        for case in _TRAIN_CASES + _HELDOUT_CASES:
            meta = case.expected_metadata
            assert isinstance(meta["expected_row_count"], int)
            assert meta["expected_row_count"] > 0
            assert isinstance(meta["required_columns"], list)
            assert len(meta["required_columns"]) > 0
            assert isinstance(meta["unit_normalization"], dict)
            assert "forbidden_pattern" in meta["unit_normalization"]


class TestDimensionCoverage:
    def test_all_four_dimensions_covered_in_train(self):
        """Train split of size 8 should cover all 4 dimensions (2 per dim with floor(8/4)=2)."""
        dims = {c.primary_dimension for c in _TRAIN_CASES}
        required = {"data_preservation", "unit_consistency", "format_validity", "completeness"}
        assert dims == required, f"Missing dimensions: {required - dims}"

    def test_validate_cases_passes_for_train(self):
        errors = validate_cases(_TRAIN_CASES)
        assert errors == [], f"Validation errors: {errors}"

    def test_validate_cases_passes_for_heldout(self):
        errors = validate_cases(_HELDOUT_CASES)
        assert errors == [], f"Validation errors: {errors}"

    def test_validate_cases_from_dicts_works(self):
        import dataclasses
        dicts = [dataclasses.asdict(c) for c in _TRAIN_CASES]
        errors = validate_cases_from_dicts(dicts)
        assert errors == [], f"Validation errors: {errors}"


class TestGradeability:
    def test_gold_outputs_parseable(self):
        for case in _TRAIN_CASES + _HELDOUT_CASES:
            assert det.csv_parseable(case.gold_or_reference_output), (
                f"{case.case_id}: gold not parseable CSV"
            )

    def test_gold_row_count_matches_metadata(self):
        for case in _TRAIN_CASES + _HELDOUT_CASES:
            expected = case.expected_metadata["expected_row_count"]
            assert det.row_count_preserved(case.gold_or_reference_output, expected), (
                f"{case.case_id}: gold row count != expected_row_count={expected}"
            )

    def test_gold_headers_complete(self):
        for case in _TRAIN_CASES + _HELDOUT_CASES:
            required = set(case.expected_metadata["required_columns"])
            score = det.headers_preserved(case.gold_or_reference_output, required)
            assert score == 1.0, (
                f"{case.case_id}: gold missing required headers (score={score})"
            )

    def test_gold_units_clean(self):
        for case in _TRAIN_CASES + _HELDOUT_CASES:
            pattern = case.expected_metadata["unit_normalization"]["forbidden_pattern"]
            assert det.unit_normalized(case.gold_or_reference_output, pattern), (
                f"{case.case_id}: gold contains forbidden unit pattern"
            )

    def test_grade_case_function_works(self, tmp_path):
        """grade_case in run_model_baseline should work on generated cases."""
        import dataclasses
        from scripts.run_model_baseline import grade_case
        for case in _TRAIN_CASES[:3]:
            case_dict = dataclasses.asdict(case)
            gold = case.gold_or_reference_output
            scores, flags, composite = grade_case(case_dict, gold)
            assert composite >= 0.0
            assert "data_preservation" in scores
            assert "format_validity" in scores


class TestValidateDetectsDuplicates:
    def test_catches_duplicate_case_ids(self):
        import copy, dataclasses
        dicts = [dataclasses.asdict(c) for c in _TRAIN_CASES[:4]]
        dicts_with_dup = dicts + [copy.deepcopy(dicts[0])]  # duplicate first record
        errors = validate_cases_from_dicts(dicts_with_dup)
        assert any("duplicate case_id" in e for e in errors), (
            f"Expected duplicate case_id error, got: {errors}"
        )

    def test_catches_duplicate_input_hashes(self):
        import copy, dataclasses
        dicts = [dataclasses.asdict(c) for c in _TRAIN_CASES[:4]]
        dup = copy.deepcopy(dicts[0])
        dup["case_id"] = "sc_gen_UNIQUE_9999"  # unique ID but same input
        dicts_with_dup = dicts + [dup]
        errors = validate_cases_from_dicts(dicts_with_dup)
        assert any("input hash" in e.lower() or "input_hash" in e for e in errors), (
            f"Expected input hash error, got: {errors}"
        )
