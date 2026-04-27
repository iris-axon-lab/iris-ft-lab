## SFT Eval v1

**Base model:** mlx-community/Qwen2.5-3B-Instruct-4bit
**Adapter:** adapters/sft_collab_eval_qwen25_3b_v2/
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

## v2 Verdict: NOT PROMOTED — with concern

**Headline:** stress data_preservation went from v1's 0.25 to v2's 0.25 (delta 0.00). No
movement despite 3× more stress training cases (80 → 240).

### Per-condition summary

| Condition | v1 result | v2 result |
|---|---|---|
| 1. composite no regression | PASS (+0.0387) | PASS (+0.0343) |
| 2. data_preservation no regression | PASS (+0.0125) | PASS (+0.0000) |
| 3. rh_like no increase | PASS (−1) | PASS (+0) |
| 4. one dim improves ≥ 0.02 | PASS (uc +0.1375) | PASS (uc +0.1375) |
| 5. stress data_preservation ≥ 0.85 | FAIL (0.25) | FAIL (0.25) |

### Hypothesis test

The v2 rebalance hypothesis was: doubling stress proportion from 25% to 50%
(deletion-event ratio 61.2% → 30.6%, see `docs/sft_v2_data_audit.md`) lifts stress
data_preservation meaningfully above v1's 0.25.

**Hypothesis: REFUTED.** Stress data_preservation stayed at 0.25 (< 0.50 meaningful-progress
threshold). The deletion-event ratio drop confirmed the data rebalance worked, but it did not
translate into preserved rows on the stress eval.

**Key diagnostic finding:** on the stress held-out set, unit_consistency improved sharply
(base 0.70 → v2 SFT 1.00, +0.30), while data_preservation was flat (0.20 → 0.25, +0.05).
Both improvements are identical to v1 (v1 stress uc also went 0.70 → 1.00). This means
the model IS learning from stress demonstrations — specifically the `$M→$K` unit-conversion
signal — but the "preserve all rows" instruction is NOT transferring. The row-dropping reflex
trained by the 147 deletion events in the regular cases appears stronger than the preservation
signal from 240 stress cases, and doubling stress cases from 80 to 240 did not change this.

The flat trajectory (0.25 at v1 with 80 stress cases; 0.25 at v2 with 240 stress cases)
suggests a qualitative problem, not a quantity problem: the deletion reflex and the
preservation signal are not competing on equal terms in the loss landscape.

### Likely next stage

The stress data_preservation score has been flat at 0.25 across two experiments (v1: 80
stress cases at 25%; v2: 240 stress cases at 50%). The unit_consistency signal from stress
data is fully absorbed but the preservation signal is not. Three candidate interventions for v3:

1. **Curriculum training** — train first on stress-only for 150 iters (enough for preservation
   to stick), then fine-tune on the full mixed set for another 150 iters. This separates the
   preservation learning phase from the deletion-learning phase and avoids them competing in
   the same gradient updates.

2. **Aggressive rebalance to 75% stress** — 360 stress + 120 regular = 480 total. Deletion
   ratio drops to ~73/480 ≈ 15% (vs. 61% at v1, 31% at v2). The large regular-case
   reduction protects the unit_consistency and format_validity signals while giving the
   preservation gradient far more headroom. Risk: regular data_preservation may regress if
   the model no longer sees enough deletion-required cases (duplicates, annotation rows).

3. **Row-level reward shaping** — add explicit `PRESERVE_ROW` markers to the stress gold
   outputs so the model sees direct token-level supervision for the preservation decision, not
   just its downstream CSV consequence. This requires generator changes and is the highest
   effort option.

Curriculum training (option 1) is the recommended first experiment: it addresses the root
cause (gradient competition) without changing data ratios or generators, and is reversible —
if it regresses conditions 1–4, the two-phase nature makes it easy to isolate. Do NOT start
v3 in this session.

### Reproducibility footer

| Field | Value |
|---|---|
| Config | configs/sft_collab_eval_qwen25_3b_v2.yaml |
| Adapter | adapters/sft_collab_eval_qwen25_3b_v2/ |
| Training data | data/sft_collab_eval_full_v2/ (regenerate from seeds 100 + 500) |
| Held-out regular | data/generated/spreadsheet_heldout_v1.jsonl (seed=200, n=80) |
| Held-out stress | data/generated/spreadsheet_heldout_stress_v1.jsonl (seed=300, n=40) |
| Training log | results/collab_sft_v2_train.log |
| Stress eval log | results/collab_sft_v2_stress_eval.log |
| Regular eval log | results/collab_sft_v2_regular_eval.log |
