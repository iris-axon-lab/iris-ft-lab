# SFT v3 Phase 1 Training Notes

## Summary

| Field | Value |
|---|---|
| Config | configs/sft_collab_eval_qwen25_3b_v3_phase1.yaml |
| Data | data/sft_collab_eval_full_v3_phase1/ (stress-only, 216 train / 24 valid) |
| Iters | 150 |
| Batch size | 4 |
| LR | 5.0e-5 cosine decay |
| Throughput | ~0.077–0.082 it/sec (~12–13 sec/iter; longer stress sequences) |
| Wall time | ~35 min (16:42–17:17 PDT) |
| Peak memory | 41.755 GB |
| Final adapter | adapters/sft_collab_eval_qwen25_3b_v3_phase1/adapters.safetensors |

## Loss progression

| Iter | Train loss | Val loss | Notes |
|---|---|---|---|
| 1 | — | 1.009 | Initial val |
| 10 | 0.883 | — | Warmup (LR 3.0e-5) |
| 20 | 0.495 | — | Sharp drop |
| 30 | 0.298 | — | Plateau beginning |
| 40 | 0.282 | — | |
| 50 | 0.276 | 0.274 | Checkpoint saved |
| 60 | 0.271 | — | |
| 70 | 0.271 | — | |
| 80 | 0.268 | — | |
| 90 | 0.270 | — | |
| 100 | 0.268 | 0.270 | Checkpoint saved |
| 110 | 0.269 | — | |
| 120 | 0.262 | — | |
| 130 | 0.264 | — | |
| 140 | 0.266 | — | |
| 150 | 0.267 | 0.268 | Final checkpoint saved |

## Notes

- Loss dropped sharply from 0.883 (iter 10) to ~0.276 (iter 50), then plateaued around 0.265–0.270
  through iters 50–150. This is consistent with stress-only data converging quickly.
- Val loss closely tracks train loss (no overfitting observed).
- No stop conditions triggered.
- Phase 1 final checkpoint copied to adapters/sft_collab_eval_qwen25_3b_v3/ as phase 2 init weights.
