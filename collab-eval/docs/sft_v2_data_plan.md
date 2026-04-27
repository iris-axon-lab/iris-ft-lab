# SFT v2 — data rebalance plan

Generated: 2026-04-27. All judgment calls locked before this session.

---

## Problem: v1 training distribution asymmetry

`docs/sft_v1_data_audit.md` counted **147 deletion events** across 240 v1 regular
training cases with **0 preserve-and-convert demonstrations**. The training signal
rewarded row deletion or Notes stripping in every case that contained noise. The
25% stress proportion in v1 (80 stress / 240 regular = 320 total) was insufficient
to override this deletion-heavy signal — the v1 stress eval `data_preservation` score
was **0.25**, well below the 0.85 gate (`collab_sft_v1_stress.md`).

## v2 rebalance

| Split | Generator | Seed | n |
|---|---|---|---|
| Regular training | `spreadsheet_clean` | 100 | 240 |
| Stress training | `spreadsheet_clean_stress` | 500 | 240 |
| **Total training** | | | **480 (50% stress)** |
| Regular held-out | `spreadsheet_clean` | 200 | 80 |
| Stress held-out | `spreadsheet_clean_stress` | 300 | 40 |

Held-out splits are **unchanged** from v1; `spreadsheet_heldout_v1.jsonl` and
`spreadsheet_heldout_stress_v1.jsonl` remain the eval baselines.

v2 data writes to `data/sft_collab_eval_full_v2/` so v1 artifacts are untouched.
The v1 training config (`configs/sft_collab_eval_qwen25_3b.yaml`) is not modified
in this session.

## Hypothesis

Doubling stress proportion from 25% to 50% will approximately halve the
deletion-event ratio in training data (from ~61% to ~31%), which should lift the
v2 stress-eval `data_preservation` score meaningfully above v1's 0.25. The
preserve-and-convert signal from 240 stress cases (each instructing the model to
keep all rows and convert `$M→$K`) should counterbalance the 147 deletion events
in the regular training cases.

Success criterion for Phase 4 (deferred): `data_preservation` ≥ 0.85 on the
held-out stress set at `seed=300`.

## Deferred work

Phase 3 (MLX training with `--train`) and Phase 4 (eval + promotion gate) are
intentionally deferred to the next session, which requires the Mac with Apple
Silicon. No generator templates were modified; the fix is more stress data,
not new template families.
