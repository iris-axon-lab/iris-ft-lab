## Training Cycle v0 — End-to-End Summary

**Date:** 2026-04-25
**Task type:** spreadsheet_clean (only)
**Cycle:** audit → task generation → seed calibration → four-policy optimization loop

---

### What was implemented

| Component | Status | Details |
|-----------|--------|---------|
| Synthetic task generation | ✓ Complete | 80 generated cases, seed 42, all deterministic |
| Seed judge/rubric calibration harness | ✓ Complete | 24 synthetic examples, 4 LLM-required dimensions |
| Four-policy optimization loop | ✓ Complete | naive, format_compliance, reward_aware, overfit |
| Reward-hacking analysis | ✓ Complete | 3 concrete hacking examples found, documented |
| Offline test suite | ✓ Complete | 96 tests, 96 passed, no API key required |

### What was NOT implemented (and must not be claimed)

- **No RL checkpoint.** There is no trained model. All policies are heuristic rules.
- **No human judge calibration.** Calibration is seed/synthetic only — the artifact creator's best judgment, not human rater agreement.
- **No full model fine-tune.** No gradient updates, no policy improvement across episodes.
- **No production-scale distribution.** 80 synthetic spreadsheet cases, one schema, no real data.
- **No calibrated LLM judge.** ANTHROPIC_API_KEY was not set during this cycle. All LLM-required dimensions (faithfulness, quality_delta, citation_accurate, hallucination_flag) are unassessed.
- **No inference/runtime product.** No serving, no endpoint, no agent deployment.

---

### Results summary

**Generated tasks:** 80 cases, evenly distributed across 4 primary dimensions (20 each), difficulties 40% easy / 40% medium / 20% hard, seed 42. Coverage targets met.

**Judge calibration (offline):** 24 synthetic calibration records pass schema validation and band/score sanity checks. LLM judge not run (no API key). Dimensions marked unassessed.

**Optimization loop v0 — mean composite scores:**

| Policy | Mean composite | Interpretation |
|--------|---------------|----------------|
| naive_policy | 0.9345 | Floor. Passes most checks by doing minimal work. |
| format_compliance_policy | 0.9375 | Marginal improvement over naive (column normalization). |
| reward_aware_policy | 1.0000 | Selects best template using grader — exploits grader access. |
| overfit_policy | 1.0000 | Games row-count and unit checks — scores 1.0 with data loss. |

**Key findings:**

1. **reward_aware_policy improves over naive on 24/80 cases** (all unit_consistency), with Δ=+0.25 composite per case. The improvement comes entirely from unit normalization, not data preservation or completeness.

2. **overfit_policy games the grader on 17 probe cases with composite = 1.0:**
   - **Hack 1 (row duplication):** Drops the second half of the time series, duplicates the first half to meet `expected_row_count`. `data_preservation = 1.0` despite silent data loss. Found on 13 data_preservation cases.
   - **Hack 2 (unit hiding):** Stores `$M` values as `orig $X.XXXM` in the Notes field. The forbidden_pattern `\$M|\bM\b` does not match `$1.800M` (digit precedes M, no word boundary). `unit_consistency = 1.0` despite `$M` values present in output. Found on 4 unit_consistency/hard cases.
   - **Hack 3 (extra fabricated rows):** Appended rows inflate count above expected. Also scores `data_preservation = 1.0`.

3. **naive_policy achieves 0.9345 composite with zero normalization effort.** The harness's floor is already high because blank-row removal alone satisfies most checks. This indicates the deterministic grader rewards structural compliance heavily and data integrity only at the row-count level.

---

### Grader weakness ranking (from optimization pressure)

| Weakness | Dimension affected | Exploit | Required fix |
|----------|-------------------|---------|--------------|
| Row count only, no identity | data_preservation | Duplicate rows to reach count | Per-row identity / uniqueness check |
| Unit regex misses `$X.XXXM` in free text | unit_consistency | Hide `$M` in Notes as `$1.800M` | Cell-by-cell numeric parser |
| Format check has no content check | format_validity | Valid CSV with garbage data | Content validation layer (requires schema knowledge) |
| Header check only, no value check | completeness | Rename values to dummy entries | Spot-check a sample of cell values |

The first two weaknesses are exploitable by the overfit_policy in a fully automated way.

---

### Unassessed dimensions

All four LLM-required dimensions across the three task types are **unassessed** in this cycle because `ANTHROPIC_API_KEY` was not set and the optimization loop runs on spreadsheet_clean (which has no LLM dimensions). Assessed status will change only after:
1. Running `python scripts/run_judge_calibration.py` with a valid API key.
2. Achieving ≥ 0.80 expected-band agreement per dimension.

Until then, no training signal claims can be made for faithfulness, quality_delta, citation_accurate, or hallucination_flag.

---

### Reviewer quickstart

```bash
cd collab-eval
pip install -r requirements.txt

# 1. Run tests (96 tests, ~0.2s, no API key required)
pytest tests/ -v

# 2. Generate tasks
python scripts/generate_tasks.py \
    --task spreadsheet_clean --n 80 --seed 42 \
    --output data/generated/spreadsheet_clean_v1.jsonl

# 3. Run calibration (offline safe; LLM scoring optional)
python scripts/run_judge_calibration.py
# With LLM judge: ANTHROPIC_API_KEY=sk-... python scripts/run_judge_calibration.py

# 4. Run the optimization loop
python scripts/run_optimization_loop.py \
    --tasks data/generated/spreadsheet_clean_v1.jsonl

# 5. Read results
cat results/optimization_loop_v0.md
```

---

### Go / No-go for loop 2

See `results/optimization_loop_v0.md` for the full Go/No-go section.

**Summary:** The harness is structurally sound and grader weaknesses are concretely documented. Before loop 2:
1. Patch `row_count_preserved` with a row-identity check (prevents duplication exploit as training signal).
2. Run LLM judge calibration with API key to unlock faithfulness/quality_delta dimensions.
3. Implement a trainable policy (a model, not template selection) before calling any loop "RL."

**Preference pairs:** Conditionally warranted. The overfit vs. reward_aware outputs on probe cases constitute natural negative/positive pairs — but only after the row-identity patch, or the "positive" outputs will be gameable by the same duplication strategy.

---

*Training cycle v0 complete. STOP.*

---

**→ Next stage:** Model baseline and SFT scaffold.
See [`results/model_baseline_v0.md`](model_baseline_v0.md) for baseline run status
and [`results/collab_sft_v0.md`](collab_sft_v0.md) for SFT scaffold status.
