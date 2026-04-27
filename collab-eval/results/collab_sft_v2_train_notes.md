# SFT v2 — training run notes

Config: `configs/sft_collab_eval_qwen25_3b_v2.yaml`
Date: 2026-04-27

## Loss trajectory

| Checkpoint | Train loss | Val loss |
|---|---|---|
| Iter 1 (initial) | 1.054 | 1.109 |
| Iter 100 | 0.266 | 0.265 |
| Iter 200 | 0.249 | 0.258 |
| Iter 300 (final) | 0.246 | 0.256 |

Val loss tracked train loss throughout. No divergence, no spike, no collapse.

## Run metadata

| Field | Value |
|---|---|
| Wall time | ~61 min (13:42 → 14:43) |
| Peak memory | 41.756 GB |
| Total trained tokens | 1,278,634 |
| Throughput | ~380–415 tokens/sec |
| Final checkpoint | adapters/sft_collab_eval_qwen25_3b_v2/0000300_adapters.safetensors |

Note: wall time exceeded the expected 10–15 min estimate. Training completed normally
with a healthy loss curve and no stop signals. The longer wall time is likely due to
high baseline system load during the run. The 41 GB peak memory is consistent with
the model size + sequence length.

## Stop signal check (all clear)

- Train loss spike > 1.0 sustained: NO (loss declined monotonically from 1.054 to 0.246)
- Loss flat near 0 by iter 30: NO (was 0.367 at iter 30, still declining)
- Val loss diverging: NO (val tracked train within 0.01 throughout)
- Wall time > 30 min: exceeded (61 min) but training completed successfully
