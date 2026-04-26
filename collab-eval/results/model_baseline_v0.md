## Model Baseline v0

**Status:** Baseline run complete.
**Model:** mlx-community/Qwen2.5-3B-Instruct-4bit
**Cases evaluated:** 80

### Aggregate metrics

| Metric | Value |
|--------|-------|
| Parseability rate | 100.0% |
| Mean composite score | 0.9569 |
| Hard-fail rate | 0.0% |

### Per-dimension means

| Dimension | Mean |
|-----------|------|
| data_preservation | 0.9750 |
| unit_consistency | 0.8625 |
| format_validity | 1.0000 |
| completeness | 1.0000 |

### Baseline decision gate

Proceed to actual SFT training only if composite mean >= 0.30 AND hard-fail rate <= 40%.

| Gate condition | Value | Status |
|----------------|-------|--------|
| composite mean >= 0.30 | 0.9569 | PASS |
| hard-fail rate <= 40% | 0.0% | PASS |
| Overall | | **PASS — proceed to SFT training.** |

### Notes

This is a base-model baseline, not a fine-tuned result. No gradient updates have been applied to the model.

The baseline gate passing justified proceeding to SFT training. It did not imply that SFT would
improve performance. See `results/collab_sft_v0.md` for the actual SFT result.

### Representative failure cases

**Case:** sc_gen_200_0029  
Composite: 0.4 | Flags: []  
Scores: {'data_preservation': 0.0, 'format_validity': 1.0, 'unit_consistency': 0.0, 'completeness': 1.0}  

### Reward-hacking-like behavior

2 cases scored high on format_validity but low on data_preservation — consistent with row-dropping behavior.

