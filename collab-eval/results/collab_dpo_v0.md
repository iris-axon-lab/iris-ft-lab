# collab-eval DPO v0 — preserve-vs-drop preference learning

This report documents the DPO v0 adapter eval. The first generated section below is labeled `## SFT Eval v1` because the shared eval script (`eval/run_collab_model_eval.py`) emits that header for any adapter; interpret the legacy label as "adapter eval" — it is the DPO v0 adapter being evaluated, not an SFT v1 adapter. The authoritative DPO verdict (with bucket classification, hypothesis test, and root cause) starts in the **`v0 Verdict`** section below the metric tables.

---

## SFT Eval v1

**Base model:** mlx-community/Qwen2.5-3B-Instruct-4bit
**Adapter:** adapters/dpo_collab_eval_qwen25_3b_v0/
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
| Mean composite | 0.9569 | 0.9506 | -0.0063 |
| Hard-fail rate | 0.0% | 0.0% | +0.0% |
| data_preservation | 0.9750 | 0.9750 | +0.0000 |
| format_validity | 1.0000 | 1.0000 | +0.0000 |
| unit_consistency | 0.8625 | 0.8375 | -0.0250 |
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

- composite_mean delta -0.0063 >= -0.005: FAIL
- data_preservation delta +0.0000 >= 0: PASS
- rh_like_count delta +0 <= 0: PASS
- best of {unit_consistency, format_validity, completeness} delta +0.0000 (dim=format_validity) >= +0.02: FAIL
- stress data_preservation_mean 0.2000 >= 0.85: FAIL

Stress eval data_preservation_mean: 0.2000

**Gate: FAIL** — adapter is not promoted.

Reasons:
- composite regressed: delta -0.0063 below tolerance -0.005
- no improvement dimension reached +0.02 (best: format_validity=+0.0000)
- preservation-stress data_preservation_mean 0.2000 < 0.85

---

## v0 Verdict: NOT PROMOTED — hypothesis refuted

**Headline:** stress data_preservation went from v3's 0.25 to DPO v0's 0.20 (delta −0.05). DPO made preservation *worse*, not better.

### Per-condition summary

| Condition | v1 | v2 | v3 | DPO v0 |
|---|---|---|---|---|
| 1. composite no regression | PASS (+0.039) | PASS (+0.034) | PASS (+0.034) | FAIL (−0.006) |
| 2. data_preservation no regression | PASS (+0.013) | PASS (+0.000) | PASS (+0.000) | PASS (+0.000) |
| 3. rh_like no increase | PASS (−1) | PASS (+0) | PASS (+0) | PASS (+0) |
| 4. one dim improves ≥ 0.02 | PASS (uc +0.138) | PASS (uc +0.138) | PASS (uc +0.138) | FAIL (+0.000) |
| 5. stress data_preservation ≥ 0.85 | FAIL (0.25) | FAIL (0.25) | FAIL (0.25) | FAIL (0.20) |

### Hypothesis test

The v0 DPO hypothesis: discriminative supervision ("preserve > drop") via 80 preference
pairs lifts stress preservation above v3's 0.25, addressing the signal asymmetry that
SFT positive demonstrations couldn't (FAILURE_MODES.md Mode 13).

**Result: REFUTED.** Stress preservation dropped from 0.25 (v3 SFT) to 0.20 (DPO v0),
matching the base model baseline. DPO not only failed to improve preservation — it
erased v3's modest gain (+0.05 over base) and also regressed unit_consistency (−0.025
on regular eval) and composite (−0.006). The preference-learning instrument did not
transfer discrimination signal into generation-time row preservation.

### Root cause analysis

The training loss collapsed to ~0.001 by iter 30 (train accuracy 1.000 from iter 30
onward; val accuracy 1.000 at iter 150). Critically, val loss at iter 1 was exactly
0.693 = −log(0.5), confirming the reference model loaded correctly — this is not the
FAILURE_MODES.md Mode 2 (fusion defect) signature.

The likely mechanism: the DPO discriminator learned to separate "CSV with N rows" from
"CSV with N−k rows" as a token-count signal rather than a semantic preservation policy.
This discrimination is trivially learnable (loss collapse by iter 30) but does not
generalize to generation: the policy at inference time still drops rows, because the
reward signal never reached the generation layer — it only sharpened the log-ratio.

β=0.1 provided insufficient KL penalty against such a crisp binary signal, allowing
the policy to diverge aggressively from the v3 SFT reference while appearing to learn
the preference. The unit_consistency regression (−0.025) is a symptom of this
drift — v3's $M→$K conversion knowledge was partially overwritten.

### Next stage

**Pivot to RL with grader as reward (v5). Preference learning instrument exhausted.**

Both SFT positive demonstrations (v1/v2/v3, Modes 9/13) and DPO discriminative
supervision (v0, Mode 13 confirmed) have failed to move stress preservation above 0.25.
The signal asymmetry is deeper than preference pairs can address — the model needs
online feedback from a task-grader that measures row count directly, not a pre-computed
preference between fixed outputs.

Do **not** start v5 in this session.

### Reproducibility footer

| Field | Value |
|---|---|
| Config | configs/dpo_collab_eval_qwen25_3b.yaml |
| DPO adapter | adapters/dpo_collab_eval_qwen25_3b_v0/ |
| Fused SFT model | adapters/qwen25_3b_collab_v3_fused/ |
| v3 SFT adapter | adapters/sft_collab_eval_qwen25_3b_v3/ |
| Train data | data/processed/dpo/train.jsonl (regenerate from seed=600) |
| Valid data | data/processed/dpo/valid.jsonl (regenerate from seed=601) |
| Held-out regular | data/generated/spreadsheet_heldout_v1.jsonl (seed=200) |
| Held-out stress | data/generated/spreadsheet_heldout_stress_v1.jsonl (seed=300) |
| Train log | results/collab_dpo_v0_train.log |
| Stress eval log | results/collab_dpo_v0_stress_eval.log |
| Regular eval log | results/collab_dpo_v0_regular_eval.log |
