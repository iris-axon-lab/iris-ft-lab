# SFT v3 Phase 2 Training Notes

## Summary

| Field | Value |
|---|---|
| Config | configs/sft_collab_eval_qwen25_3b_v3_phase2.yaml |
| Data | data/sft_collab_eval_full_v2/ (mixed, 432 train / 48 valid approx) |
| Resume from | adapters/sft_collab_eval_qwen25_3b_v3_phase1/adapters.safetensors (via resume_adapter_file) |
| Iters | 150 |
| Batch size | 4 |
| LR | 2.0e-5 cosine decay (40% of phase 1 peak) |
| Throughput | ~0.079–0.098 it/sec |
| Wall time | ~28 min (18:00–18:28 PDT) |
| Peak memory | 41.755 GB |
| Final adapter | adapters/sft_collab_eval_qwen25_3b_v3/adapters.safetensors |

## Loss progression

| Iter | Train loss | Val loss | Notes |
|---|---|---|---|
| 1 | — | 0.388 | Resume confirmed (fresh start would be ~1.109) |
| 10 | 0.361 | — | (fresh start was 1.069 — clear confirmation) |
| 20 | 0.292 | — | |
| 30 | 0.264 | — | |
| 40 | 0.269 | — | |
| 50 | 0.260 | 0.267 | Checkpoint saved |
| 60 | 0.252 | — | |
| 70 | 0.257 | — | |
| 80 | 0.260 | — | |
| 90 | 0.259 | — | |
| 100 | 0.260 | 0.262 | Checkpoint saved |
| 110 | 0.257 | — | |
| 120 | 0.254 | — | |
| 130 | 0.246 | — | |
| 140 | 0.255 | — | |
| 150 | 0.245 | 0.262 | Final checkpoint saved |

## Notes

- Resume confirmed by iter 1 val loss 0.388 (vs 1.109 for fresh start on mixed data).
- Phase 1 ended at train loss 0.267; phase 2 entered below that and continued declining.
- Loss stable in 0.245–0.261 range through phase 2, consistent with lower-LR refinement.
- No stop conditions triggered.
- Bug discovered during run: mlx-lm's adapter_path does not load existing weights;
  resume requires the separate resume_adapter_file field. Config updated accordingly.
