# Trace DPO v0 Results

> **Verdict: NOT PROMOTED — pre-gate failure at §4.1 (fusion defect)**

## Header

| Field | Value |
|---|---|
| Base model | mlx-community/Qwen2.5-3B-Instruct-4bit |
| Fused model | outputs/qwen25_3b_sft_fused |
| DPO adapter | outputs/dpo_qwen25_3b_v0 |
| Eval set | data/eval_gold.jsonl (12 cases) |
| mlx-lm-lora version | 2.1.0 |
| Eval date | 2026-04-27 |

## Summary Table

| Metric | SFT v2 (LoRA-on-base) | Fused-SFT baseline | DPO v0 |
|---|---|---|---|
| Overall tier accuracy | 9/12 (75.0%) | 0/12 (0.0%) ❌ | not evaluated |
| episodic | 2/2 | 0/2 | — |
| semantic | 3/3 | 0/3 | — |
| procedural | 1/1 | 0/1 | — |
| prospective | 3/6 | 0/6 | — |
| parse errors | 0 | 0 | — |
| partial parses (wrong schema) | 0 | 12/12 | — |

DPO eval was not run — §4.1 hard stop triggered (baseline tier total 0/12 vs expected 9/12, delta = 9, threshold = 1).

## Fusion Defect — Root Cause Analysis

**Symptom:** Fused SFT model (`outputs/qwen25_3b_sft_fused`) produces the same `{"memory": ...}` output format as the untuned base model. It scores 0/12 on tier classification — identical to the pre-SFT baseline documented in `eval/results.md`.

**Diagnosis:** The `mlx_lm fuse` command was run *without* `--dequantize` on a 4-bit quantized base model. File-size evidence is conclusive:

| File | Size (bytes) |
|---|---|
| Base model (4-bit, HF cache) | 1,736,293,090 |
| Fused model (no `--dequantize`) | 1,736,292,896 |
| SFT v2 adapter | 26,631,752 |

The fused model is 194 bytes smaller than the base — not 25 MB larger as required if the adapter delta had been merged. MLX cannot add float32 LoRA deltas into a 4-bit quantized tensor representation and produce a meaningful result; the command ran to completion but the merged weights are identical (within quantization error) to the original quantized weights.

**Control confirmation:** `SFT v2 LoRA-on-base` (base model + adapter applied at inference time) still scores 9/12, confirming the eval script is correct and the adapter weights are intact. The defect is solely in the fusion step.

## Per-case Verdicts

Not completed — stopped at §4.1 per hard stop condition.

| case_id | gold | baseline (fused-SFT) | DPO v0 | verdict |
|---|---|---|---|---|
| eval_001 | episodic | partial (schema wrong) | — | not evaluated |
| eval_002 | episodic | partial | — | not evaluated |
| eval_003 | semantic | partial | — | not evaluated |
| eval_004 | procedural | partial | — | not evaluated |
| eval_005 | prospective | partial | — | not evaluated |
| eval_006 | prospective | partial | — | not evaluated |
| eval_007 | semantic | partial | — | not evaluated |
| eval_008 | prospective | partial | — | not evaluated |
| eval_009 | prospective | partial | — | not evaluated |
| eval_010 | prospective | partial | — | not evaluated |
| eval_011 | prospective | partial | — | not evaluated |
| eval_012 | semantic | partial | — | not evaluated |

## Promotion Gate

| Condition | Threshold | Result |
|---|---|---|
| 1. Tier accuracy ≥ baseline | ≥ 9/12 | **FAIL** — baseline itself is 0/12 (fusion defect) |
| 2. Prospective accuracy ≥ 4/6 | ≥ 4/6 | not evaluated |
| 3. Parse errors = 0 | 0/12 | not evaluated |
| 4. No regressions on 9 baseline-correct cases | 0 regressions | not evaluated |

**Verdict: NOT PROMOTED.** Gate condition 1 cannot be evaluated because the baseline itself is broken. The DPO adapter trained on the defective fused model is not usable.

## Likely Failure Mode and Fix for DPO v1

The `fuse_sft.py` script called `mlx_lm fuse` without `--dequantize` on a 4-bit quantized base model. MLX cannot merge float32 LoRA deltas into a 4-bit quantized tensor; the fuse command completed silently but produced a model file of identical size to the quantized base (1.62 GB), confirming no weight data was actually merged. The SFT adapter content is intact (`outputs/sft_qwen25_3b_v2/` is unchanged); only the fusion step needs to be re-run correctly.

**Fix for DPO v1:** Add `--dequantize` to the fuse command in `scripts/fuse_sft.py`. This converts the 4-bit model to float16 (~6 GB) during fusion, allowing the LoRA delta to be merged correctly. The resulting model will be larger and may require more memory during DPO training (estimated ~18–22 GB peak on M-series). Before training, verify the fused model scores ≥ 8/12 on `data/eval_gold.jsonl` before proceeding to DPO training.

`scripts/fuse_sft.py` has been updated with `--dequantize` in this commit (the fix is committed but DPO v1 training is deferred to a new session).

## Reproducibility Footer

| Field | Value |
|---|---|
| Config | configs/dpo_trace_qwen25_3b.yaml |
| DPO adapter (not usable) | outputs/dpo_qwen25_3b_v0 |
| Fused model (defective, 4-bit) | outputs/qwen25_3b_sft_fused |
| SFT v2 adapter (intact) | outputs/sft_qwen25_3b_v2 |
| Seed | 42 |
| Training log | eval/dpo_v0_train.log |
| Baseline eval log | eval/dpo_v0_baseline_fused.log |
