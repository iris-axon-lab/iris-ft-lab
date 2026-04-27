"""Tests for v1 stress generator and 5-condition promotion gate.

All tests are offline-only and fast (< 1s total). No MLX, no model loads,
no API calls.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

# conftest.py adds collab-eval/ to sys.path; guard for direct invocation.
_ROOT = Path(__file__).parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from collab_eval.generation.spreadsheet_generator import (
    generate_preservation_stress_cases,
    cases_to_jsonl,
)
from eval.run_collab_model_eval import _check_promotion_gate


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_agg(
    composite: float = 0.9569,
    dp: float = 0.9750,
    rh: int = 2,
    uc: float = 0.8625,
    fv: float = 1.0,
    comp: float = 1.0,
) -> dict:
    return {
        "composite_mean": composite,
        "hard_fail_rate": 0.0,
        "parseability_rate": 1.0,
        "dim_means": {
            "data_preservation": dp,
            "format_validity": fv,
            "unit_consistency": uc,
            "completeness": comp,
        },
        "reward_hacking_count": rh,
        "n": 80,
    }


def _data_row_count(csv_text: str) -> int:
    """Count non-header data rows in a CSV string using csv.reader."""
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    return sum(1 for r in rows[1:] if r and any(c.strip() for c in r))


# ── Test 1: Stress generator determinism ──────────────────────────────────────

def test_stress_generator_count():
    cases = generate_preservation_stress_cases(n=10, seed=400)
    assert len(cases) == 10


def test_stress_generator_same_seed_byte_identical():
    cases_a = generate_preservation_stress_cases(n=10, seed=400)
    cases_b = generate_preservation_stress_cases(n=10, seed=400)
    assert cases_to_jsonl(cases_a) == cases_to_jsonl(cases_b)


def test_stress_generator_different_seed_different_ids_and_inputs():
    cases_400 = generate_preservation_stress_cases(n=10, seed=400)
    cases_401 = generate_preservation_stress_cases(n=10, seed=401)
    ids_400 = {c.case_id for c in cases_400}
    ids_401 = {c.case_id for c in cases_401}
    assert ids_400 != ids_401
    inputs_400 = {c.input for c in cases_400}
    inputs_401 = {c.input for c in cases_401}
    assert inputs_400 != inputs_401


# ── Test 2: Stress case schema ────────────────────────────────────────────────

def test_stress_case_preservation_stress_flag():
    cases = generate_preservation_stress_cases(n=10, seed=400)
    for c in cases:
        assert c.expected_metadata.get("preservation_stress") is True, c.case_id


def test_stress_case_rows_must_preserve_nonempty():
    cases = generate_preservation_stress_cases(n=10, seed=400)
    for c in cases:
        rmp = c.expected_metadata.get("rows_must_preserve")
        assert isinstance(rmp, list) and len(rmp) > 0, c.case_id


def test_stress_case_primary_dimension_data_preservation():
    cases = generate_preservation_stress_cases(n=10, seed=400)
    for c in cases:
        assert c.primary_dimension == "data_preservation", c.case_id


# ── Test 3: Stress gold preserves all rows ────────────────────────────────────

def test_stress_gold_row_count_matches_expected():
    cases = generate_preservation_stress_cases(n=10, seed=400)
    for c in cases:
        expected = c.expected_metadata["expected_row_count"]
        actual = _data_row_count(c.gold_or_reference_output)
        assert actual == expected, (
            f"{c.case_id}: gold has {actual} data rows, expected {expected}"
        )


# ── Test 4: Gate condition 1 — composite no regression ───────────────────────

def test_gate_cond1_pass_within_tolerance():
    # composite delta = -0.003 >= -0.005 tolerance → PASS
    base = _make_agg(composite=0.9569)
    sft = _make_agg(composite=0.9569 - 0.003, dp=0.9751, uc=0.9125)
    passed, issues, conditions = _check_promotion_gate(base, sft, stress_data_preservation=0.90)
    assert passed is True


def test_gate_cond1_fail_exceeds_tolerance():
    # composite delta = -0.010 < -0.005 tolerance → FAIL
    base = _make_agg(composite=0.9569)
    sft = _make_agg(composite=0.9569 - 0.010, dp=0.9751, uc=0.9125)
    passed, issues, conditions = _check_promotion_gate(base, sft, stress_data_preservation=0.90)
    assert passed is False
    assert any("composite" in issue for issue in issues)


# ── Test 5: Gate condition 2 — data_preservation no regression ───────────────

def test_gate_cond2_fail_dp_regression():
    # dp delta = -0.001 < 0 → FAIL
    base = _make_agg(dp=0.9750)
    sft = _make_agg(dp=0.9750 - 0.001, uc=0.9125)
    passed, issues, conditions = _check_promotion_gate(base, sft, stress_data_preservation=0.90)
    assert passed is False
    assert any("data_preservation" in issue for issue in issues)


def test_gate_cond2_pass_dp_no_regression():
    # dp delta = +0.001 >= 0 → PASS
    base = _make_agg(dp=0.9750)
    sft = _make_agg(dp=0.9750 + 0.001, uc=0.9125)
    passed, issues, conditions = _check_promotion_gate(base, sft, stress_data_preservation=0.90)
    assert passed is True


# ── Test 6: Gate condition 3 — RH-like no increase ───────────────────────────

def test_gate_cond3_fail_rh_increase():
    # rh 2→3, delta +1 > 0 → FAIL
    base = _make_agg(rh=2)
    sft = _make_agg(rh=3, dp=0.9751, uc=0.9125)
    passed, issues, conditions = _check_promotion_gate(base, sft, stress_data_preservation=0.90)
    assert passed is False
    assert any("reward-hacking" in issue for issue in issues)


def test_gate_cond3_pass_rh_no_increase():
    # rh 2→2, delta 0 <= 0 → PASS
    base = _make_agg(rh=2)
    sft = _make_agg(rh=2, dp=0.9751, uc=0.9125)
    passed, issues, conditions = _check_promotion_gate(base, sft, stress_data_preservation=0.90)
    assert passed is True


# ── Test 7: Gate condition 4 — best dimension improvement ────────────────────

def test_gate_cond4_fail_all_dims_below_threshold():
    # every improvement dim delta = +0.01 < +0.02 → FAIL
    base = _make_agg(uc=0.80, fv=0.90, comp=0.90)
    sft = _make_agg(uc=0.81, fv=0.91, comp=0.91)
    passed, issues, conditions = _check_promotion_gate(base, sft, stress_data_preservation=0.90)
    assert passed is False
    assert any("improvement dimension" in issue for issue in issues)


def test_gate_cond4_pass_unit_consistency_at_plus_five():
    # unit_consistency delta = +0.05 >= +0.02 → PASS
    base = _make_agg(uc=0.80, fv=0.90, comp=0.90)
    sft = _make_agg(uc=0.85, fv=0.91, comp=0.91)
    passed, issues, conditions = _check_promotion_gate(base, sft, stress_data_preservation=0.90)
    assert passed is True


# ── Test 8: Gate condition 5 — stress threshold ───────────────────────────────

def test_gate_cond5_fail_below_0_85():
    base = _make_agg()
    sft = _make_agg(dp=0.9751, uc=0.9125)
    passed, issues, conditions = _check_promotion_gate(base, sft, stress_data_preservation=0.84)
    assert passed is False
    assert any("preservation-stress" in issue for issue in issues)


def test_gate_cond5_pass_at_0_85():
    base = _make_agg()
    sft = _make_agg(dp=0.9751, uc=0.9125)
    passed, issues, conditions = _check_promotion_gate(base, sft, stress_data_preservation=0.85)
    assert passed is True


def test_gate_cond5_fail_none_reports_not_evaluated():
    base = _make_agg()
    sft = _make_agg(dp=0.9751, uc=0.9125)
    passed, issues, conditions = _check_promotion_gate(base, sft, stress_data_preservation=None)
    assert passed is False
    assert any("not evaluated" in issue for issue in issues)


# ── Test 9: Gate composition ──────────────────────────────────────────────────

def test_gate_all_conditions_pass():
    base = _make_agg(composite=0.9569, dp=0.9750, rh=2, uc=0.8625, fv=1.0, comp=1.0)
    sft = _make_agg(
        composite=0.9569 - 0.003,    # cond1 PASS: delta -0.003 >= -0.005
        dp=0.9750 + 0.001,           # cond2 PASS: delta +0.001 >= 0
        rh=2,                         # cond3 PASS: delta 0 <= 0
        uc=0.8625 + 0.05,            # cond4 PASS: best delta +0.05 >= +0.02
        fv=1.0,
        comp=1.0,
    )
    passed, issues, conditions = _check_promotion_gate(
        base, sft, stress_data_preservation=0.90   # cond5 PASS: 0.90 >= 0.85
    )
    assert passed is True
    assert issues == []
    assert len(conditions) == 5


def test_gate_all_conditions_fail():
    base = _make_agg(composite=0.9569, dp=0.9750, rh=2, uc=0.8625, fv=1.0, comp=1.0)
    sft = _make_agg(
        composite=0.9569 - 0.010,    # cond1 FAIL: delta -0.010 < -0.005
        dp=0.9750 - 0.001,           # cond2 FAIL: delta -0.001 < 0
        rh=3,                         # cond3 FAIL: delta +1 > 0
        uc=0.8625 + 0.01,            # cond4 FAIL: best delta +0.01 < +0.02
        fv=1.0 + 0.01,               # also +0.01 < +0.02
        comp=1.0 + 0.01,             # also +0.01 < +0.02
    )
    passed, issues, conditions = _check_promotion_gate(
        base, sft, stress_data_preservation=0.50   # cond5 FAIL: 0.50 < 0.85
    )
    assert passed is False
    assert len(issues) == 5
    assert len(conditions) == 5
