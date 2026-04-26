# SFT v0 Builder Notes

Status: SFT v0 run complete. Adapter not promoted. Next stage: dataset hardening.

---

## Measured metrics

### Base model (mlx-community/Qwen2.5-3B-Instruct-4bit)

| Metric | Value |
|--------|-------|
| Cases evaluated | 80 |
| Parseability | 100.0% |
| Mean composite | 0.9569 |
| Hard-fail rate | 0.0% |
| data_preservation | 0.9750 |
| format_validity | 1.0000 |
| unit_consistency | 0.8625 |
| completeness | 1.0000 |
| RH-like cases | 2 |
| Baseline gate | PASS |

### SFT adapter (adapters/sft_collab_eval_qwen25_3b/)

| Metric | Base | SFT | Delta |
|--------|------|-----|-------|
| Mean composite | 0.9569 | 0.9110 | -0.0459 |
| Hard-fail rate | 0.0% | 0.0% | +0.0% |
| data_preservation | 0.9750 | 0.7500 | -0.2250 |
| format_validity | 1.0000 | 1.0000 | +0.0000 |
| unit_consistency | 0.8625 | 1.0000 | +0.1375 |
| completeness | 1.0000 | 0.9900 | -0.0100 |
| RH-like cases | 2 | 20 | +18 |
| Promotion gate | — | — | FAIL |

---

## Interpretation

The adapter learned unit normalization and `unit_consistency` improved from 0.8625 to 1.0000.
This is the only genuine improvement.

The recipe over-optimized for unit normalization while damaging `data_preservation`, which
fell from 0.9750 to 0.7500 — a regression of -0.2250. Mean composite fell by -0.0459.

RH-like cases increased from 2 to 20 (+18). This is a 10x increase in the row-dropping
failure mode this harness was designed to detect: **clean-looking CSV output that silently
drops rows or values**. The hard-fail cap did not trigger on any individual case, but the
population-level RH-like rate is now 25%.

This is a **negative/mixed result**. The adapter is experimental and unpromoted.
Do not describe this as an SFT improvement.

---

## Why the adapter is not promoted

Promotion requires ALL of:
- composite improves by >= 0.10 — actual delta: -0.0459 (FAIL, wrong direction)
- hard-fail rate does not increase — 0.0% → 0.0% (PASS)
- format_validity does not regress — 1.0 → 1.0 (PASS)
- no obvious increase in RH-like behavior — 2 → 20 (FAIL, 10x increase)

Two conditions fail. The promotion gate was not met.

**Note on the +0.10 gate:** The base composite was 0.9569. Since composite is bounded at 1.0,
the theoretical maximum improvement is ~0.043. The original +0.10 gate is unreachable at this
base level. The gate must be revised before the next SFT attempt — likely to composite >= base
with no preservation regression and no RH-like increase.

---

## Likely failure mode

The SFT gold outputs included perfectly clean CSVs with normalized units. The training signal
rewarded unit normalization strongly (the one area where base model scored below 1.0). The
recipe apparently traded row completeness for unit cleanliness: the model learned to produce
well-formatted, correctly-normalized CSVs by dropping rows it couldn't easily normalize, rather
than preserving all rows with best-effort unit handling.

This is the canonical spreadsheet-cleanup reward-hacking failure mode documented in
`docs/rl_env_design.md` and `tests/test_reward_hacking_cases.py`.

---

## Recommended next stage: dataset hardening

DPO is deferred until SFT failure modes are understood and a revised SFT run avoids
data-preservation regression.

The next stage should focus on:
1. Understanding the 20 RH-like SFT cases — which rows were dropped, in what context.
2. Generating more preservation-heavy synthetic cases — longer tables, ambiguous units,
   harder rows that are tempting to drop.
3. Running gentler SFT with a recipe that doesn't over-reward unit normalization at
   the expense of row completeness.

Only after SFT no longer regresses `data_preservation` or increases RH-like behavior should
DPO be considered.

---

## Next stage proposal: sft-data-v1-hardening

Suggested branch: `sft-data-v1-hardening`

Checklist (do not implement until explicitly asked):

- [ ] Inspect the 20 RH-like SFT cases — identify which rows were dropped and why
- [ ] Verify that deterministic SFT gold outputs score perfectly (data_preservation = 1.0)
- [ ] Generate more preservation-heavy synthetic cases (longer tables, harder rows)
- [ ] Create a hard held-out stress split (cases chosen to stress data_preservation)
- [ ] Add a validation split to enable early stopping during training
- [ ] Revise promotion gate: replace +0.10 composite requirement with no-regression
      requirements on data_preservation and RH-like rate
- [ ] Train gentler SFT:
  - fewer iterations
  - lower learning rate
  - smaller LoRA rank
  - dropout enabled
  - checkpoint selection against validation set
- [ ] Evaluate against original held-out and new hard held-out split
- [ ] Only consider DPO after SFT no longer increases RH-like behavior and
      data_preservation >= base

DPO is intentionally deferred. Building preference pairs before the SFT failure mode
is understood would risk amplifying the same row-dropping behavior.
