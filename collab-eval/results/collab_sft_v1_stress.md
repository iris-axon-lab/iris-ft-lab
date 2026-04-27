## SFT Eval v1

> **What this file is.** Stress-eval baseline + adapter run on
> `data/generated/spreadsheet_heldout_stress_v1.jsonl` (40 cases). The "Gate:
> FAIL" line below reflects that the 5-condition gate cannot be evaluated
> standalone from this file — it requires the per-case JSON to be chained
> into the regular eval run. The actual gate verdict for v1 is in
> `collab_sft_v1.md`. This file exists so the stress-set numbers are
> committed alongside the regular-set numbers.

**Base model:** mlx-community/Qwen2.5-3B-Instruct-4bit
**Adapter:** adapters/sft_collab_eval_qwen25_3b_v1/
**Cases evaluated:** 40

### Base model results

| Metric | Value |
|--------|-------|
| Mean composite | 0.6512 |
| Hard-fail rate | 0.0% |
| Parseability rate | 100.0% |
| data_preservation | 0.2000 |
| format_validity | 1.0000 |
| unit_consistency | 0.7250 |
| completeness | 1.0000 |
| RH-like cases | 32 |

### SFT adapter results

| Metric | Base | SFT | Delta |
|--------|------|-----|-------|
| Mean composite | 0.6512 | 0.7375 | +0.0863 |
| Hard-fail rate | 0.0% | 0.0% | +0.0% |
| data_preservation | 0.2000 | 0.2500 | +0.0500 |
| format_validity | 1.0000 | 1.0000 | +0.0000 |
| unit_consistency | 0.7250 | 1.0000 | +0.2750 |
| completeness | 1.0000 | 1.0000 | +0.0000 |
| RH-like cases | 32 | 30 | -2 |

### Promotion gate (v1)

Promote adapter only if ALL are true:
- composite_mean does not regress by more than 0.005
- data_preservation does not regress
- rh_like_count does not increase
- best of {unit_consistency, format_validity, completeness} improves by >= 0.02
- preservation-stress data_preservation_mean >= 0.85

Per-condition results:

- composite_mean delta +0.0863 >= -0.005: PASS
- data_preservation delta +0.0500 >= 0: PASS
- rh_like_count delta -2 <= 0: PASS
- best of {unit_consistency, format_validity, completeness} delta +0.2750 (dim=unit_consistency) >= +0.02: PASS
- stress data_preservation_mean: NOT EVALUATED (pass --stress-results to enable)

**Gate: FAIL** — adapter is not promoted.

Reasons:
- preservation-stress eval not evaluated (gate condition 5 missing)
