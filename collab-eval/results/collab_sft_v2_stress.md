## SFT Eval v1

**Base model:** mlx-community/Qwen2.5-3B-Instruct-4bit
**Adapter:** adapters/sft_collab_eval_qwen25_3b_v2/
**Cases evaluated:** 40

### Base model results

| Metric | Value |
|--------|-------|
| Mean composite | 0.6450 |
| Hard-fail rate | 0.0% |
| Parseability rate | 100.0% |
| data_preservation | 0.2000 |
| format_validity | 1.0000 |
| unit_consistency | 0.7000 |
| completeness | 1.0000 |
| RH-like cases | 32 |

### SFT adapter results

| Metric | Base | SFT | Delta |
|--------|------|-----|-------|
| Mean composite | 0.6450 | 0.7375 | +0.0925 |
| Hard-fail rate | 0.0% | 0.0% | +0.0% |
| data_preservation | 0.2000 | 0.2500 | +0.0500 |
| format_validity | 1.0000 | 1.0000 | +0.0000 |
| unit_consistency | 0.7000 | 1.0000 | +0.3000 |
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

- composite_mean delta +0.0925 >= -0.005: PASS
- data_preservation delta +0.0500 >= 0: PASS
- rh_like_count delta -2 <= 0: PASS
- best of {unit_consistency, format_validity, completeness} delta +0.3000 (dim=unit_consistency) >= +0.02: PASS
- stress data_preservation_mean: NOT EVALUATED (pass --stress-results to enable)

**Gate: FAIL** — adapter is not promoted.

Reasons:
- preservation-stress eval not evaluated (gate condition 5 missing)
