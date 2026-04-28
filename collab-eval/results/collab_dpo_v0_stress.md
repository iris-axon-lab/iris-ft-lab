# collab-eval DPO v0 — stress eval (preserve-vs-drop)

This file is the standalone stress-set eval for the DPO v0 adapter. The first generated section below is labeled `## SFT Eval v1` because the shared eval script emits that header; interpret it as "DPO v0 stress eval." The "Gate: FAIL — gate condition 5 not evaluated" line further down is expected for a standalone stress run; the chained promotion verdict that combines stress + regular evals lives in [`collab_dpo_v0.md`](collab_dpo_v0.md).

---

## SFT Eval v1

**Base model:** mlx-community/Qwen2.5-3B-Instruct-4bit
**Adapter:** adapters/dpo_collab_eval_qwen25_3b_v0/
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
| Mean composite | 0.6450 | 0.6325 | -0.0125 |
| Hard-fail rate | 0.0% | 0.0% | +0.0% |
| data_preservation | 0.2000 | 0.2000 | +0.0000 |
| format_validity | 1.0000 | 1.0000 | +0.0000 |
| unit_consistency | 0.7000 | 0.6500 | -0.0500 |
| completeness | 1.0000 | 1.0000 | +0.0000 |
| RH-like cases | 32 | 32 | +0 |

### Promotion gate (v1)

Promote adapter only if ALL are true:
- composite_mean does not regress by more than 0.005
- data_preservation does not regress
- rh_like_count does not increase
- best of {unit_consistency, format_validity, completeness} improves by >= 0.02
- preservation-stress data_preservation_mean >= 0.85

Per-condition results:

- composite_mean delta -0.0125 >= -0.005: FAIL
- data_preservation delta +0.0000 >= 0: PASS
- rh_like_count delta +0 <= 0: PASS
- best of {unit_consistency, format_validity, completeness} delta +0.0000 (dim=format_validity) >= +0.02: FAIL
- stress data_preservation_mean: NOT EVALUATED (pass --stress-results to enable)

**Gate: FAIL** — adapter is not promoted.

Reasons:
- composite regressed: delta -0.0125 below tolerance -0.005
- no improvement dimension reached +0.02 (best: format_validity=+0.0000)
- preservation-stress eval not evaluated (gate condition 5 missing)
