## SFT Eval v0

**Status:** Training run complete. Adapter is **not promoted**.

**Base model:** mlx-community/Qwen2.5-3B-Instruct-4bit
**Adapter:** adapters/sft_collab_eval_qwen25_3b/
**Cases evaluated:** 80

---

### Base model results

| Metric | Value |
|--------|-------|
| Mean composite | 0.9569 |
| Hard-fail rate | 0.0% |
| Parseability rate | 100.0% |
| data_preservation | 0.9750 |
| format_validity | 1.0000 |
| unit_consistency | 0.8625 |
| completeness | 1.0000 |
| RH-like cases | 2 |

---

### SFT adapter results

| Metric | Base | SFT | Delta |
|--------|------|-----|-------|
| Mean composite | 0.9569 | 0.9110 | **-0.0459** |
| Hard-fail rate | 0.0% | 0.0% | +0.0% |
| data_preservation | 0.9750 | 0.7500 | **-0.2250** |
| format_validity | 1.0000 | 1.0000 | +0.0000 |
| unit_consistency | 0.8625 | 1.0000 | +0.1375 |
| completeness | 1.0000 | 0.9900 | -0.0100 |
| RH-like cases | 2 | 20 | **+18** |

---

### Promotion gate

Promote adapter only if ALL are true:
- composite improves by >= 0.10 over base
- hard-fail rate does not increase
- format_validity does not regress
- no obvious increase in reward-hacking behavior

**Gate: FAIL** — adapter is not promoted.

Reasons for failure:
- Composite delta was -0.0459, not +0.10 (required direction not met, let alone magnitude).
- data_preservation regressed by -0.2250 (0.9750 → 0.7500). This is a hard quality regression.
- RH-like cases increased by +18 (2 → 20). Consistent with the known row-dropping failure mode:
  clean-looking CSV output that silently drops rows or values.

Note on the +0.10 gate: The base composite was already 0.9569. Since composite is capped at 1.0,
the maximum possible improvement from base is ~0.043. The original +0.10 gate is mathematically
unreachable. The gate is documented as written; it must be revised before the next training attempt.

---

### Interpretation

This is a **negative/mixed result**. The SFT adapter is not promoted and remains experimental.

The adapter learned unit normalization — `unit_consistency` improved from 0.8625 to 1.0000 (+0.1375).
This is the one genuine improvement. However, the recipe over-optimized for unit normalization and
damaged the more critical `data_preservation` dimension, which fell from 0.9750 to 0.7500.

The increase in RH-like cases from 2 to 20 is a clear signal: the adapter is producing
well-formatted outputs that silently drop rows or values — exactly the reward-hacking failure
mode this harness was designed to detect. The hard-fail cap did not trigger (no single case scored
below the hard-fail threshold), but the population-level RH-like rate increased by 10x.

**The adapter is experimental and unpromoted. Do not claim SFT improvement.**

Next steps are dataset hardening, not DPO. See `docs/sft_v0_builder_notes.md`.
