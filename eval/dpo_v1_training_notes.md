# DPO v1 Training Notes

**Date:** 2026-04-27
**Branch:** trace-dpo-harness
**Config:** configs/dpo_trace_qwen25_3b.yaml

## Environment

- mlx-lm-lora version: 2.1.0
- Base model: mlx-community/Qwen2.5-3B-Instruct-4bit
- Fused SFT model: outputs/qwen25_3b_sft_fused (float16, ~5.8 GB, re-fused with --dequantize)
- DPO adapter output: outputs/dpo_qwen25_3b_v1

## Pre-training fix

The stale `outputs/qwen25_3b_sft_fused/model.safetensors` (1.6 GB, 4-bit, from the defective v0 fuse run) was
renamed to `model.safetensors.defective_v0_bak` before training. mlx_lm's weight discovery glob
`model*.safetensors` would otherwise load all three files simultaneously, causing a 506-parameter
conflict between the quantization keys in the stale 4-bit file and the float16 shards.

## Training parameters

- iters: 150, batch_size: 2, lr: 5e-6, max_seq_length: 512
- LoRA: rank=8, alpha=16, dropout=0.05, num_layers=16, scale=2.0
- DPO: beta=0.1, loss_type=sigmoid
- Train set: 80 pairs (data/processed/dpo/train.jsonl)
- Val set: 12 pairs (data/processed/dpo/valid.jsonl)

## Loss curve

| Iter | Train loss | Val loss | Val accuracy | Val margin |
|------|-----------|---------|-------------|-----------|
| 1    | —         | 0.693   | 0.000       | 0.000     |
| 10   | 0.643     | —       | 0.700       | —         |
| 20   | 0.344     | —       | 1.000       | —         |
| 30   | 0.159     | —       | 1.000       | —         |
| 40   | 0.051     | —       | 1.000       | —         |
| 50   | 0.042     | —       | 1.000       | —         |
| 60   | 0.013     | —       | 1.000       | —         |
| 70   | 0.028     | —       | 1.000       | —         |
| 80   | 0.015     | —       | 1.000       | —         |
| 90   | 0.020     | —       | 1.000       | —         |
| 100  | 0.006     | —       | 1.000       | —         |
| 110  | 0.009     | —       | 1.000       | —         |
| 120  | 0.004     | —       | 1.000       | —         |
| 130  | 0.002     | —       | 1.000       | —         |
| 140  | 0.004     | —       | 1.000       | —         |
| 150  | 0.003     | 0.004   | 1.000       | 6.008     |

## Final metrics

- Final train loss: 0.003
- Final val loss: 0.004
- Final train reward margin: 6.714 (chosen_r: −1.238, rejected_r: −7.952)
- Final val accuracy: 1.000, val margin: 6.008
- Peak memory: 19.388 GB
- Wall time: ~6m55s

## Notes

Loss dropped from 0.693 to ~0.013 by iter 60 and remained near zero through iter 150. Val
accuracy reached 1.0 by iter 20 and held throughout. Both chosen_r and rejected_r went
negative (normal DPO behavior at beta=0.1 as both outputs are pushed below reference), but
the margin grew monotonically to ~6.7, confirming clear preference ordering. Peak memory
was 19.4 GB (higher than v0's 10.6 GB, consistent with the float16 dequantized model vs
the defective 4-bit base used in v0).

Final checkpoint: outputs/dpo_qwen25_3b_v1/adapters.safetensors
(also at 0000050, 0000100, 0000150 checkpoints). mlx-lm-lora auto-fused the adapter at
completion.
