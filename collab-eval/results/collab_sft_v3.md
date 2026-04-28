## SFT Eval v1

**Base model:** mlx-community/Qwen2.5-3B-Instruct-4bit
**Adapter:** adapters/sft_collab_eval_qwen25_3b_v3/
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
| Mean composite | 0.9569 | 0.9912 | +0.0343 |
| Hard-fail rate | 0.0% | 0.0% | +0.0% |
| data_preservation | 0.9750 | 0.9750 | +0.0000 |
| format_validity | 1.0000 | 1.0000 | +0.0000 |
| unit_consistency | 0.8625 | 1.0000 | +0.1375 |
| completeness | 1.0000 | 1.0000 | +0.0000 |
| RH-like cases | 2 | 2 | +0 |

### Promotion gate (v1)

Promote adapter only if ALL are true:
- composite_mean does not regress by more than 0.005
- data_preservation does not regress
- rh_like_count does not increase
- best of {unit_consistency, format_validity, completeness} improves by >= 0.02
- preservation-stress data_preservation_mean >= 0.85

Per-condition results:

- composite_mean delta +0.0343 >= -0.005: PASS
- data_preservation delta +0.0000 >= 0: PASS
- rh_like_count delta +0 <= 0: PASS
- best of {unit_consistency, format_validity, completeness} delta +0.1375 (dim=unit_consistency) >= +0.02: PASS
- stress data_preservation_mean 0.2500 >= 0.85: FAIL

Stress eval data_preservation_mean: 0.2500

**Gate: FAIL** — adapter is not promoted.

Reasons:
- preservation-stress data_preservation_mean 0.2500 < 0.85

---

## v3 Verdict: NOT PROMOTED — hypothesis refuted

**Headline:** stress data_preservation went from v2's 0.25 to v3's 0.25 (delta ±0.00).

### Per-condition summary

| Condition | v1 | v2 | v3 |
|---|---|---|---|
| 1. composite no regression | PASS (+0.039) | PASS (+0.034) | PASS (+0.034) |
| 2. data_preservation no regression | PASS (+0.013) | PASS (+0.000) | PASS (+0.000) |
| 3. rh_like no increase | PASS (−1) | PASS (+0) | PASS (+0) |
| 4. one dim improves ≥ 0.02 | PASS (uc +0.138) | PASS (uc +0.138) | PASS (uc +0.138) |
| 5. stress data_preservation ≥ 0.85 | FAIL (0.25) | FAIL (0.25) | FAIL (0.25) |

### Hypothesis test

The v3 curriculum hypothesis was: separating stress and mixed data into two phases
(stress-only first, then mixed at lower LR) addresses gradient competition (H2) between
preservation and deletion signals. v3 trained 150 iters stress-only at lr=5e-5, then
150 iters mixed at lr=2e-5 (40% of phase 1).

**Hypothesis result:** REFUTED. Stress data_preservation is 0.25 — identical to v1 and
v2. Curriculum provided zero improvement on the preservation-stress dimension.

### Diagnostic — what did each phase contribute?

Phase 1 (stress-only, 150 iters) drove train loss from 0.883 → 0.267 on stress data,
with val loss tracking closely (1.009 → 0.268). Phase 2 resumed correctly from phase 1
weights (iter 1 val 0.388 vs fresh-start 1.109) and refined to 0.245 train / 0.262 val.
Both phases trained successfully. The resume mechanism required adding `resume_adapter_file`
to the phase 2 config (mlx-lm's `adapter_path` is save-only; it does not load existing
weights on its own). Despite correct training, stress data_preservation remained flat at
0.25 across all three SFT versions.

This rules out gradient competition (H2) as the primary bottleneck. The stress-only
phase 1 had no competing deletion gradients and still failed to push preservation above
0.25. The preservation signal is not learnable through standard SFT positive
demonstrations — confirmed H3 (signal asymmetry).

### Likely next stage (only if NOT PROMOTED)

| Bucket | Recommended v4 |
|---|---|
| D (refuted, stress 0.25) | **Pivot to DPO-on-preservation as v4. SFT instrument exhausted for this dimension.** |

Do **not** start v4 in this session.

### Reproducibility footer

| Field | Value |
|---|---|
| Phase 1 config | configs/sft_collab_eval_qwen25_3b_v3_phase1.yaml |
| Phase 2 config | configs/sft_collab_eval_qwen25_3b_v3_phase2.yaml |
| Phase 1 adapter | adapters/sft_collab_eval_qwen25_3b_v3_phase1/ |
| Final adapter (v3) | adapters/sft_collab_eval_qwen25_3b_v3/ |
| Phase 1 data | data/sft_collab_eval_full_v3_phase1/ (regenerate from seed=500) |
| Phase 2 data | data/sft_collab_eval_full_v2/ (regenerate from seeds 100 + 500) |
| Held-out regular | data/generated/spreadsheet_heldout_v1.jsonl (seed=200) |
| Held-out stress | data/generated/spreadsheet_heldout_stress_v1.jsonl (seed=300) |
| Phase 1 train log | results/collab_sft_v3_phase1_train.log |
| Phase 2 train log | results/collab_sft_v3_phase2_train.log |
| Stress eval log | results/collab_sft_v3_stress_eval.log |
| Regular eval log | results/collab_sft_v3_regular_eval.log |
