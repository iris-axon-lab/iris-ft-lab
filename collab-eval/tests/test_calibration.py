"""
Tests for the judge/rubric calibration harness.

All tests run offline without Anthropic credentials.
Tests verify:
  - Calibration records parse and validate
  - Schema completeness
  - Band/score consistency (sanity checks)
  - Dimension coverage (all four LLM-required dimensions present)
  - Borderline cases present (not just obvious wins/failures)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_CALIB_PATH = (
    Path(__file__).parents[1] / "data" / "judge_calibration" / "seed_calibration_v1.jsonl"
)

REQUIRED_FIELDS = [
    "case_id", "task_type", "dimension", "input", "output",
    "rubric", "expected_score", "expected_score_band", "rationale", "known_failure_mode",
]
VALID_TASK_TYPES = {"doc_revision", "spreadsheet_clean", "citation_ground"}
VALID_DIMENSIONS = {"faithfulness", "quality_delta", "citation_accurate", "hallucination_flag"}
VALID_BANDS = {"low", "medium", "high"}


def _load_cases() -> list[dict]:
    cases = []
    with open(_CALIB_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


class TestCalibrationSchema:
    def test_file_exists(self):
        assert _CALIB_PATH.exists(), f"Calibration file not found: {_CALIB_PATH}"

    def test_parses_as_jsonl(self):
        cases = _load_cases()
        assert len(cases) > 0, "Calibration file is empty"

    def test_minimum_case_count(self):
        cases = _load_cases()
        assert len(cases) >= 20, f"Need at least 20 calibration cases; got {len(cases)}"

    def test_all_required_fields_present(self):
        cases = _load_cases()
        for case in cases:
            cid = case.get("case_id", "?")
            for field in REQUIRED_FIELDS:
                assert field in case, f"{cid}: missing field '{field}'"

    def test_valid_task_types(self):
        cases = _load_cases()
        for case in cases:
            cid = case.get("case_id", "?")
            assert case["task_type"] in VALID_TASK_TYPES, (
                f"{cid}: invalid task_type '{case['task_type']}'"
            )

    def test_valid_dimensions(self):
        cases = _load_cases()
        for case in cases:
            cid = case.get("case_id", "?")
            assert case["dimension"] in VALID_DIMENSIONS, (
                f"{cid}: invalid dimension '{case['dimension']}'"
            )

    def test_valid_bands(self):
        cases = _load_cases()
        for case in cases:
            cid = case.get("case_id", "?")
            assert case["expected_score_band"] in VALID_BANDS, (
                f"{cid}: invalid expected_score_band '{case['expected_score_band']}'"
            )

    def test_expected_score_in_range(self):
        cases = _load_cases()
        for case in cases:
            cid = case.get("case_id", "?")
            score = float(case["expected_score"])
            assert 0.0 <= score <= 1.0, (
                f"{cid}: expected_score {score} out of [0.0, 1.0]"
            )

    def test_unique_case_ids(self):
        cases = _load_cases()
        ids = [c["case_id"] for c in cases]
        assert len(ids) == len(set(ids)), "Duplicate case_id values found"

    def test_nonempty_input_output_rubric(self):
        cases = _load_cases()
        for case in cases:
            cid = case.get("case_id", "?")
            assert len(case["input"]) > 10, f"{cid}: input is too short"
            assert len(case["output"]) > 10, f"{cid}: output is too short"
            assert len(case["rubric"]) > 20, f"{cid}: rubric is too short"
            assert len(case["rationale"]) > 10, f"{cid}: rationale is too short"


class TestSanityChecks:
    def test_low_band_has_low_score(self):
        cases = _load_cases()
        for case in cases:
            if case["expected_score_band"] == "low":
                score = float(case["expected_score"])
                assert score <= 0.45, (
                    f"{case['case_id']}: band=low but expected_score={score} > 0.45"
                )

    def test_high_band_has_high_score(self):
        cases = _load_cases()
        for case in cases:
            if case["expected_score_band"] == "high":
                score = float(case["expected_score"])
                assert score >= 0.55, (
                    f"{case['case_id']}: band=high but expected_score={score} < 0.55"
                )

    def test_medium_band_in_range(self):
        cases = _load_cases()
        for case in cases:
            if case["expected_score_band"] == "medium":
                score = float(case["expected_score"])
                assert 0.3 <= score <= 0.75, (
                    f"{case['case_id']}: band=medium but expected_score={score} outside [0.3, 0.75]"
                )


class TestCoverage:
    def test_all_four_llm_dimensions_covered(self):
        cases = _load_cases()
        dims = {c["dimension"] for c in cases}
        for required_dim in VALID_DIMENSIONS:
            assert required_dim in dims, (
                f"LLM-required dimension '{required_dim}' not covered in calibration set"
            )

    def test_minimum_per_dimension(self):
        """Each dimension must have at least 3 calibration examples."""
        cases = _load_cases()
        from collections import Counter
        dim_counts = Counter(c["dimension"] for c in cases)
        for dim in VALID_DIMENSIONS:
            assert dim_counts[dim] >= 3, (
                f"Dimension '{dim}' has only {dim_counts[dim]} calibration case(s) (need ≥ 3)"
            )

    def test_borderline_cases_present(self):
        """Each dimension must have at least one medium-band case."""
        cases = _load_cases()
        from collections import defaultdict
        medium_by_dim: dict[str, list] = defaultdict(list)
        for case in cases:
            if case["expected_score_band"] == "medium":
                medium_by_dim[case["dimension"]].append(case)
        for dim in VALID_DIMENSIONS:
            assert len(medium_by_dim[dim]) >= 1, (
                f"Dimension '{dim}' has no borderline (medium) calibration cases"
            )

    def test_low_and_high_bands_present_per_dimension(self):
        """Each dimension must have at least one low and one high case."""
        cases = _load_cases()
        from collections import defaultdict
        by_dim_band: dict[str, set] = defaultdict(set)
        for case in cases:
            by_dim_band[case["dimension"]].add(case["expected_score_band"])
        for dim in VALID_DIMENSIONS:
            assert "low" in by_dim_band[dim], (
                f"Dimension '{dim}' has no 'low' calibration cases"
            )
            assert "high" in by_dim_band[dim], (
                f"Dimension '{dim}' has no 'high' calibration cases"
            )
