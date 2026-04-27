# Trace DPO v1 Results

> **Verdict: PROMOTED — all 4 gate conditions passed**

## Header

| Field | Value |
|---|---|
| Base model | mlx-community/Qwen2.5-3B-Instruct-4bit |
| Fused model | outputs/qwen25_3b_sft_fused (float16, re-fused with --dequantize) |
| DPO adapter | outputs/dpo_qwen25_3b_v1 |
| Eval set | data/eval_gold.jsonl (12 cases) |
| mlx-lm-lora version | 2.1.0 |
| Eval date | 2026-04-27 |

## Summary Table

| Metric | SFT v2 (LoRA-on-base) | Fused-SFT baseline | DPO v1 |
|---|---|---|---|
| Overall tier accuracy | 9/12 (75.0%) | 9/12 (75.0%) ✓ | **12/12 (100.0%)** |
| episodic | 2/2 | 2/2 | 2/2 |
| semantic | 3/3 | 3/3 | 3/3 |
| procedural | 1/1 | 1/1 | 1/1 |
| prospective | 3/6 | 3/6 | **6/6** |
| parse errors | 0 | 0 | 0 |
| partial parses (wrong schema) | 0 | 0 | 0 |

## Per-case Verdicts

| case_id | gold | baseline (fused-SFT) | DPO v1 | verdict |
|---|---|---|---|---|
| eval_001 | episodic | episodic | episodic | OK — no change |
| eval_002 | episodic | episodic | episodic | OK — no change |
| eval_003 | semantic | semantic | semantic | OK — no change |
| eval_004 | procedural | procedural | procedural | OK — no change |
| eval_005 | prospective | prospective | prospective | OK — no change |
| eval_006 | prospective | prospective | prospective | OK — no change |
| eval_007 | semantic | semantic | semantic | OK — no change |
| eval_008 | prospective | **semantic** ❌ | **prospective** ✓ | FLIP — corrected |
| eval_009 | prospective | **semantic** ❌ | **prospective** ✓ | FLIP — corrected |
| eval_010 | prospective | **episodic** ❌ | **prospective** ✓ | FLIP — corrected |
| eval_011 | prospective | prospective | prospective | OK — no change |
| eval_012 | semantic | semantic | semantic | OK — no change |

DPO v1 corrected all three baseline misses (eval_008, 009, 010) with no regressions on the
9 baseline-correct cases.

## Promotion Gate

| Condition | Threshold | Result |
|---|---|---|
| 1. Tier accuracy ≥ baseline (9/12) | ≥ 9/12 | **PASS** — 12/12 (+3) |
| 2. Prospective accuracy ≥ 4/6 with ≥1 flip in eval_008/009/010 | ≥ 4/6, ≥1 flip | **PASS** — 6/6, all 3 flipped |
| 3. Parse errors = 0 | 0/12 | **PASS** — 0 |
| 4. No regressions on 9 baseline-correct cases | 0 regressions | **PASS** — 0 regressions |

**Verdict: PROMOTED.**

## Analysis

The three baseline misses were the hardest prospective cases:
- **eval_008** (repeated deferral with self-stated intention): baseline predicted semantic; DPO v1 correctly identified the commitment signal ("intention to act this week").
- **eval_009** (contingent commitment — sprint wrap → Friday off): baseline predicted semantic; DPO v1 correctly identified the conditional prospective commitment.
- **eval_010** (post-delivery commitment to customer success): baseline predicted episodic (focused on the past delivery); DPO v1 correctly extracted the forward-facing commitment.

These are exactly the cases targeted by DPO data families C (mis_tier_mixed) and D (conditional_commitment), confirming the training signal was effective.

The clean 12/12 with 0 parse errors and 0 regressions makes this a straightforward promotion decision.

## Reproducibility Footer

| Field | Value |
|---|---|
| Config | configs/dpo_trace_qwen25_3b.yaml |
| DPO adapter | outputs/dpo_qwen25_3b_v1 |
| Fused model | outputs/qwen25_3b_sft_fused (float16) |
| SFT v2 adapter | outputs/sft_qwen25_3b_v2 |
| Seed | 42 |
| Training log | eval/dpo_v1_train.log |
| Baseline eval log | eval/dpo_v1_baseline_fused.log |
| DPO eval log | eval/dpo_v1_eval.log |
