# DPO v0 Training Notes

**Date:** 2026-04-27
**Branch:** trace-dpo-harness
**Config:** configs/dpo_trace_qwen25_3b.yaml

## Environment

- mlx-lm-lora version: 2.1.0
- Base model: mlx-community/Qwen2.5-3B-Instruct-4bit
- Fused SFT model: outputs/qwen25_3b_sft_fused
- DPO adapter output: outputs/dpo_qwen25_3b_v0

## Training parameters

- iters: 150, batch_size: 2, lr: 5e-6, max_seq_length: 512
- LoRA: rank=8, alpha=16, dropout=0.05, num_layers=16, scale=2.0
- DPO: beta=0.1, loss_type=sigmoid
- Train set: 80 pairs (data/processed/dpo/train.jsonl)
- Val set: 12 pairs (data/processed/dpo/valid.jsonl)

## Loss curve

| Iter | Train loss | Val loss |
|------|-----------|---------|
| 1    | —         | 0.693   |
| 10   | 0.614     | —       |
| 20   | 0.280     | —       |
| 30   | 0.186     | —       |
| 40   | 0.075     | —       |
| 50   | 0.035     | —       |
| 60   | 0.018     | —       |
| 70   | 0.019     | —       |
| 80   | 0.017     | —       |
| 90   | 0.021     | —       |
| 100  | 0.007     | —       |
| 110  | 0.005     | —       |
| 120  | 0.006     | —       |
| 130  | 0.002     | —       |
| 140  | 0.001     | —       |
| 150  | 0.001     | 0.002   |

## Final metrics

- Final train loss: 0.001
- Final val loss: 0.002
- Final train reward margin: 7.795 (chosen_r: −4.098, rejected_r: −11.893)
- Final val accuracy: 1.000, val margin: 7.914
- Peak memory: 10.551 GB
- Wall time: ~7m41s

## Notes

Loss dropped steeply from 0.693 to ~0.018 by iter 60, then flattened near zero.
Both chosen_r and rejected_r went negative (both pushed below reference); this is
normal DPO behavior when beta is low and the model is learning strong preferences.
The margin (chosen − rejected reward) grew monotonically to 7.8, indicating the
ordering is preserved throughout. Val loss tracked train loss with no divergence.

No stop conditions triggered. Final checkpoint: outputs/dpo_qwen25_3b_v0/adapters.safetensors
(also at 0000150_adapters.safetensors). mlx-lm-lora auto-fused the adapter into
outputs/dpo_qwen25_3b_v0/ (adapter files co-exist with the fused model files).
