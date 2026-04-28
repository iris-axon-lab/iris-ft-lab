# collab-eval DPO v0 Training Notes

Mirrors `eval/dpo_v1_training_notes.md` format.

## Environment

| Field | Value |
|---|---|
| mlx-lm-lora version | 2.1.0 |
| Backend | mlx_lm_lora.train --train-mode dpo |
| Config | configs/dpo_collab_eval_qwen25_3b.yaml |
| Base model (fused) | adapters/qwen25_3b_collab_v3_fused (Qwen2.5-3B-Instruct dequantized + v3 SFT) |
| Reference model | adapters/qwen25_3b_collab_v3_fused (same — KL anchored at v3 SFT) |
| DPO adapter output | adapters/dpo_collab_eval_qwen25_3b_v0/ |

## Hyperparameters

| Field | Value |
|---|---|
| iters | 150 |
| batch_size | 1 |
| learning_rate | 5.0e-6 |
| max_seq_length | 2048 |
| beta | 0.1 |
| loss_type | sigmoid |
| lora rank | 8 |
| lora alpha | 16 (scale=2.0) |
| lora dropout | 0.05 |
| num_layers | 16 |
| grad_checkpoint | true |

## Training curve

| Iter | Train loss | Train acc | Train margin | Val loss | Val acc | Val margin |
|---|---|---|---|---|---|---|
| 1 | — | — | — | 0.693 | 0.000 | 0.000 |
| 10 | 0.467 | 0.900 | 0.600 | — | — | — |
| 50 | 0.002 | 1.000 | 6.426 | — | — | — |
| 100 | 0.001 | 1.000 | 6.920 | — | — | — |
| 150 | 0.001 | 1.000 | 7.722 | 0.001 | 1.000 | 7.676 |

## Loss curve notes

- **Iter 1 val loss = 0.693 = −log(0.5)**: confirms reference model loaded correctly.
  This rules out the fusion-defect signature (FAILURE_MODES.md Mode 2) — the fused
  model IS being honored as the KL anchor.
- **Loss collapsed to ~0.001 by iter 30**: faster than the healthy trace DPO curve
  (0.693 → 0.3–0.4 over 150 iters). Likely due to the discrimination being very
  clear-cut: chosen and rejected are identical CSVs differing only in row count.
  β=0.1 provided insufficient KL penalty against such a crisp signal.
- **Val loss tracks train loss**: val at iter 150 = 0.001, val accuracy = 1.000.
  The model generalizes the preference discrimination to the validation set.
- **Reward margin growing monotonically** (0.600 at iter 10 → 7.722 at iter 150):
  no reward collapse. The chosen/rejected gap widened throughout training.
- Stop condition "collapse-to-zero by iter 30" technically triggered, but the iter-1
  val loss confirms this was *not* the fusion defect. The adapter is the best available
  for eval. Stress eval will determine if preference discrimination translated to
  generation-time preservation.

## Performance

| Metric | Value |
|---|---|
| Wall time | ~23 min (much shorter than 80–120 min estimate; actual sequences < 2048 tokens) |
| Peak memory | 19.936 GB |
| Final reward margin (train) | 7.722 |
| Final reward margin (val) | 7.676 |
| Peak it/s | ~0.116 it/s avg (iters 10–140) |

## Checkpoints saved

| Iter | Path |
|---|---|
| 50 | adapters/dpo_collab_eval_qwen25_3b_v0/0000050_adapters.safetensors |
| 100 | adapters/dpo_collab_eval_qwen25_3b_v0/0000100_adapters.safetensors |
| 150 | adapters/dpo_collab_eval_qwen25_3b_v0/adapters.safetensors (final) |
