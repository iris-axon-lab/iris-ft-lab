"""
Tests for the optimization loop policies and harness.

All tests run offline without Anthropic credentials.
Tests verify:
  - All four policy functions produce parseable CSV output
  - Policy outputs parse correctly for generated cases
  - Scoring is consistent and deterministic
  - overfit_policy produces at least one hacking example
  - reward_aware_policy can outperform naive_policy
  - No private/proprietary/internal strings in policy outputs
"""

from __future__ import annotations

import pytest

from collab_eval.generation.spreadsheet_generator import generate_cases
from collab_eval.policies.baselines import (
    naive_policy,
    format_compliance_policy,
    reward_aware_policy,
    overfit_policy,
    REQUIRED_COLUMNS,
)
from collab_eval.graders import deterministic as det
from collab_eval.graders.composite import weighted_score

POLICY_FNS = {
    "naive_policy": naive_policy,
    "format_compliance_policy": format_compliance_policy,
    "reward_aware_policy": reward_aware_policy,
    "overfit_policy": overfit_policy,
}

WEIGHTS = {
    "data_preservation": 0.35,
    "format_validity": 0.25,
    "unit_consistency": 0.25,
    "completeness": 0.15,
}

_CASES = None


def _get_cases():
    global _CASES
    if _CASES is None:
        _CASES = generate_cases(n=40, seed=42)
    return _CASES


def _grade(output: str, meta: dict) -> dict[str, float]:
    """Grade a policy output against expected_metadata."""
    if not det.csv_parseable(output):
        return {d: 0.0 for d in WEIGHTS}
    return {
        "data_preservation": 1.0 if det.row_count_preserved(output, meta["expected_row_count"]) else 0.0,
        "format_validity": 1.0,
        "unit_consistency": 1.0 if det.unit_normalized(
            output, meta["unit_normalization"]["forbidden_pattern"]
        ) else 0.0,
        "completeness": det.headers_preserved(output, set(meta["required_columns"])),
    }


class TestPoliciesProduceParseable:
    @pytest.mark.parametrize("policy_name", list(POLICY_FNS.keys()))
    def test_output_parseable_on_all_cases(self, policy_name):
        """All policies must produce parseable CSV on all generated cases."""
        cases = _get_cases()
        policy_fn = POLICY_FNS[policy_name]
        failures = []
        for case in cases:
            output = policy_fn(case.input, case.expected_metadata)
            if not det.csv_parseable(output):
                failures.append(case.case_id)
        assert not failures, (
            f"{policy_name} produced non-parseable CSV on: {failures[:5]}"
        )


class TestPoliciesScoreCorrectly:
    @pytest.mark.parametrize("policy_name", list(POLICY_FNS.keys()))
    def test_composite_in_range(self, policy_name):
        """Composite scores must be in [0.0, 1.0]."""
        cases = _get_cases()
        policy_fn = POLICY_FNS[policy_name]
        for case in cases:
            output = policy_fn(case.input, case.expected_metadata)
            scores = _grade(output, case.expected_metadata)
            comp = weighted_score(scores, WEIGHTS)
            assert 0.0 <= comp <= 1.0, (
                f"{policy_name} on {case.case_id}: composite {comp} out of range"
            )

    @pytest.mark.parametrize("policy_name", list(POLICY_FNS.keys()))
    def test_scoring_is_deterministic(self, policy_name):
        """Same policy on same input must produce same score."""
        cases = _get_cases()[:10]
        policy_fn = POLICY_FNS[policy_name]
        for case in cases:
            out_a = policy_fn(case.input, case.expected_metadata)
            out_b = policy_fn(case.input, case.expected_metadata)
            assert out_a == out_b, (
                f"{policy_name} is non-deterministic on {case.case_id}"
            )


class TestRewardAwareImprovement:
    def test_reward_aware_outperforms_naive_on_unit_cases(self):
        """
        reward_aware_policy should score higher than naive_policy on at least some
        unit_consistency cases (where naive passes through $M values).
        """
        cases = _get_cases()
        unit_cases = [c for c in cases if c.primary_dimension == "unit_consistency"]
        assert unit_cases, "No unit_consistency cases in test set"

        improvements = 0
        for case in unit_cases:
            naive_out = naive_policy(case.input, case.expected_metadata)
            ra_out = reward_aware_policy(case.input, case.expected_metadata)

            naive_scores = _grade(naive_out, case.expected_metadata)
            ra_scores = _grade(ra_out, case.expected_metadata)

            naive_c = weighted_score(naive_scores, WEIGHTS)
            ra_c = weighted_score(ra_scores, WEIGHTS)
            if ra_c > naive_c:
                improvements += 1

        assert improvements >= 2, (
            f"reward_aware_policy only outperformed naive on {improvements} unit_consistency "
            "cases (expected ≥ 2)"
        )


class TestOverfitHacking:
    def test_overfit_on_duplication_probe_scores_high(self):
        """
        overfit_policy must produce composite >= 0.9 on at least one case
        with duplicated_rows in known_failure_modes.
        """
        cases = generate_cases(n=80, seed=42)
        dup_cases = [c for c in cases if "duplicated_rows" in c.known_failure_modes]
        assert dup_cases, "No duplicated_rows probe cases found in n=80"

        high_score_count = 0
        for case in dup_cases:
            output = overfit_policy(case.input, case.expected_metadata)
            scores = _grade(output, case.expected_metadata)
            comp = weighted_score(scores, WEIGHTS)
            if comp >= 0.9:
                high_score_count += 1

        assert high_score_count >= 1, (
            "overfit_policy did not achieve composite >= 0.9 on any duplicated_rows probe case"
        )

    def test_overfit_hides_unit_in_notes(self):
        """
        On unit_consistency cases with $M inputs, overfit_policy should not have
        $M notation in data columns (it moves it to Notes).
        The unit_normalized check should pass.
        """
        import re
        cases = generate_cases(n=80, seed=42)
        m_cases = [c for c in cases if "mixed_units" in c.known_failure_modes]
        assert m_cases, "No mixed_units cases found"

        passes = 0
        for case in m_cases[:10]:
            output = overfit_policy(case.input, case.expected_metadata)
            pattern = case.expected_metadata["unit_normalization"]["forbidden_pattern"]
            if det.unit_normalized(output, pattern):
                passes += 1

        assert passes >= 3, (
            f"overfit_policy only passed unit_normalized on {passes}/10 mixed_units cases"
        )

    def test_overfit_produces_parseable_output_on_all_cases(self):
        """overfit_policy must always produce parseable CSV."""
        cases = _get_cases()
        failures = []
        for case in cases:
            output = overfit_policy(case.input, case.expected_metadata)
            if not det.csv_parseable(output):
                failures.append(case.case_id)
        assert not failures, f"overfit_policy produced non-parseable CSV on: {failures}"


class TestFormatCompliancePolicy:
    def test_always_has_required_columns(self):
        """format_compliance_policy must output the required columns exactly."""
        cases = _get_cases()
        for case in cases:
            output = format_compliance_policy(case.input, case.expected_metadata)
            assert det.csv_parseable(output), f"Non-parseable on {case.case_id}"
            required = set(case.expected_metadata["required_columns"])
            completeness = det.headers_preserved(output, required)
            assert completeness == 1.0, (
                f"{case.case_id}: format_compliance_policy completeness={completeness}"
            )


class TestNoPrivateDataInOutputs:
    _FORBIDDEN = [
        "microsoft", "teams", "sharepoint", "azure",
        "internal", "confidential", "proprietary",
    ]

    @pytest.mark.parametrize("policy_name", list(POLICY_FNS.keys()))
    def test_no_forbidden_strings(self, policy_name):
        cases = _get_cases()
        policy_fn = POLICY_FNS[policy_name]
        for case in cases:
            output = policy_fn(case.input, case.expected_metadata).lower()
            for forbidden in self._FORBIDDEN:
                assert forbidden not in output, (
                    f"{policy_name} on {case.case_id}: output contains '{forbidden}'"
                )
