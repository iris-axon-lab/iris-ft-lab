## SFT Eval v1

**Base model:** mlx-community/Qwen2.5-3B-Instruct-4bit
**Adapter:** adapters/sft_collab_eval_qwen25_3b_v1/
**Cases evaluated:** 80

### Base model results

| Metric | Value |
|--------|-------|
| Mean composite | 0.9569 |
| Hard-fail rate | 0.0% |
| Parseability rate | 100.0% |
| data_preservation | 0.9750 |
| format_validity | 1.0000 |
| unit_consistency | 0.8625 |
| completeness | 1.0000 |
| RH-like cases | 2 |

### SFT adapter results

| Metric | Base | SFT | Delta |
|--------|------|-----|-------|
| Mean composite | 0.9569 | 0.9956 | +0.0387 |
| Hard-fail rate | 0.0% | 0.0% | +0.0% |
| data_preservation | 0.9750 | 0.9875 | +0.0125 |
| format_validity | 1.0000 | 1.0000 | +0.0000 |
| unit_consistency | 0.8625 | 1.0000 | +0.1375 |
| completeness | 1.0000 | 1.0000 | +0.0000 |
| RH-like cases | 2 | 1 | -1 |

### Promotion gate (v1)

Promote adapter only if ALL are true:
- composite_mean does not regress by more than 0.005
- data_preservation does not regress
- rh_like_count does not increase
- best of {unit_consistency, format_validity, completeness} improves by >= 0.02
- preservation-stress data_preservation_mean >= 0.85

Per-condition results:

- composite_mean delta +0.0387 >= -0.005: PASS
- data_preservation delta +0.0125 >= 0: PASS
- rh_like_count delta -1 <= 0: PASS
- best of {unit_consistency, format_validity, completeness} delta +0.1375 (dim=unit_consistency) >= +0.02: PASS
- stress data_preservation_mean 0.2500 >= 0.85: FAIL

Stress eval data_preservation_mean: 0.2500

**Gate: FAIL** — adapter is not promoted.

Reasons:
- preservation-stress data_preservation_mean 0.2500 < 0.85
