# collab-eval

A task environment and grader harness for open-ended document manipulation tasks,
built to make reward design explicit. The core problem: document tasks have no binary
oracle, so naive reward functions get gamed. This harness addresses that with decomposed
rewards, deterministic checks where the ground truth is recoverable, LLM judges only
where judgment is genuinely required, and hard-fail caps that prevent catastrophic
outputs from hiding behind good scores on cheap dimensions.
The failure modes in this harness reflect the class of grader failures that surface
repeatedly in multi-step production systems — the design is grounded in what breaks
in practice, not just what is theoretically gameable.

All tasks, documents, and failure cases are fully synthetic.

---

## What this demonstrates

Each task is chosen because it has a specific, non-obvious grader failure mode:

- **Document revision:** An agent can produce fluent, active-voice prose that satisfies
  every deterministic check while silently introducing unsupported claims. Word count and
  passive-voice ratio cannot catch this — only a faithfulness judge can. The demo shows a
  padded agent that exceeds the word limit and gets hard-capped, and explains why the
  faithfulness dimension stays at 0.5 until an LLM assesses it.

- **Spreadsheet cleanup:** An agent can produce perfectly valid, correctly-headed CSV by
  quietly dropping the difficult rows. Format-validity alone gives this a perfect score.
  `data_preservation` catches it by checking row count against a known expected value.

- **Citation-grounded editing:** An agent can insert `[Source B]` next to a claim that
  Source B does not actually support — citation hallucination. The citation-present check
  (deterministic) rewards the form; `citation_accurate` and `hallucination_flag` (LLM)
  are required to catch the substance. The demo shows a fully uncited output triggering
  a hard-fail cap, making the gap between "looks cited" and "is cited" concrete.

The adversarial cases in `tests/test_reward_hacking_cases.py` make each of these
failure modes executable and runnable as regression tests.

---

## What this is

`collab-eval` provides three runnable task environments, deterministic graders for each,
an optional LLM-as-judge layer, a composite grader with hard-fail caps — and as of
training cycle v0, a minimal measured training-cycle artifact:

- **Synthetic task generation** for spreadsheet cleanup (80 seeded cases, deterministic)
- **Seed judge/rubric calibration harness** (24 synthetic examples, 4 LLM-required dimensions)
- **One measured optimization loop** (four policies compared on generated tasks)
- **Reward-hacking analysis** (three concrete hacking examples found under optimization pressure)

The original `training_cycle_v0` artifact below is NOT an RL implementation — it is
policy-search over fixed heuristics, chosen to expose grader weaknesses and establish a
credible baseline. See [`results/training_cycle_v0.md`](results/training_cycle_v0.md) for
what was and was not implemented in that artifact.

Since training cycle v0, the harness has hosted four SFT experiments (v0–v3) targeting
the spreadsheet-cleanup task. All four are documented in §4–§7 below; **all four are
NOT PROMOTED**. The harness's promotion gate has held the line: no adapter that drops a
material fraction of rows on long stress tables has been promoted, even when other
dimensions improved. The flat trajectory across v1/v2/v3 (stress `data_preservation`
stuck at 0.25) ruled out two SFT-shaped hypotheses (data quantity, gradient competition)
and points at preference learning (DPO) as the next instrument — see the "What's next"
pointer in §7.

The design note at [`docs/rl_env_design.md`](docs/rl_env_design.md) covers reward
decomposition strategy, failure modes, and extension paths to a fuller RL setup.

A longer builder's log — covering design decisions, failure modes encountered during
construction, and the reward-hacking probes — is published at
[iris-axon-lab.github.io](https://iris-axon-lab.github.io). Note: the public builder's
log reflects the harness as of training cycle v0, before the SFT experiments. For the
current SFT artifact catalog, see §4–§7 below.

---

## Setup

```bash
cd collab-eval
pip install -r requirements.txt
```

Python 3.11+ required.

---

## Run the demo

```bash
# Deterministic-only mode (no API key required):
python scripts/run_demo.py

# With LLM grading (Anthropic API key required):
ANTHROPIC_API_KEY=sk-... python scripts/run_demo.py

# Optional: use a different model for the LLM judge:
COLLAB_EVAL_JUDGE_MODEL=claude-sonnet-4-6 python scripts/run_demo.py
```

Each task section prints two episodes: a good agent output and a bad one that games
a naive grader, so the harness's purpose is visible in the output itself.

---

## Run the tests

```bash
# From the collab-eval/ directory:
pytest tests/ -v

# Or from the repo root:
pytest collab-eval/tests/ -v
```

Tests pass offline without Anthropic credentials. The adversarial cases are executable
demonstrations of reward hacking failure modes, not just smoke tests.

---

## Validate from scratch

Full end-to-end reproduction from a clean clone. No API key required for any step below.

**1. Clone and install**

```bash
git clone <repo-url>
cd iris-ft-lab/collab-eval
pip install -r requirements.txt
```

**2. Run the test suite (96 tests, ~0.2 s)**

```bash
pytest tests/ -v
```

Expected: `96 passed`. The suite covers five test files:
- `test_reward_hacking_cases.py` — original 4 adversarial probes (the canonical set)
- `test_eval_extended.py` — 35 extended cases: quality-range checks and 9 additional RH probes
- `test_generation.py` — generated task schema, determinism, coverage, gradeability
- `test_calibration.py` — calibration record schema, band/score sanity, dimension coverage
- `test_optimization_loop.py` — policy outputs, overfit hacking probes, reward_aware improvement

**3. Run the demo**

```bash
python scripts/run_demo.py
```

Prints two episodes per task type (good agent / reward-hacking agent). All output is
deterministic — no model inference, no API calls. Each bad-agent episode demonstrates
a specific grader failure mode described in the "What this demonstrates" section above.

**4. Re-run the scorer and inspect results**

```bash
python scripts/score_eval.py
```

Writes (or overwrites) `results/eval_results_v1.md` with per-case composite scores and
dimension breakdowns. The FT v1 row in that file remains a placeholder — it was the
original heuristic-policies eval, separate from the SFT experiments. Up-to-date SFT eval
results are in `collab_sft_v0.md` through `collab_sft_v3.md` (see §4–§7).

```bash
# Quick sanity check — should show mean composite ~0.67:
head -40 results/eval_results_v1.md
```

**5. Run the training cycle (task generation → calibration → optimization loop)**

```bash
# Generate 80 synthetic spreadsheet tasks
python scripts/generate_tasks.py \
    --task spreadsheet_clean --n 80 --seed 42 \
    --output data/generated/spreadsheet_clean_v1.jsonl

# Run calibration harness (offline safe; LLM scoring optional)
python scripts/run_judge_calibration.py
# With LLM judge: ANTHROPIC_API_KEY=sk-... python scripts/run_judge_calibration.py

# Run the four-policy optimization loop
python scripts/run_optimization_loop.py \
    --tasks data/generated/spreadsheet_clean_v1.jsonl

# Read results
cat results/optimization_loop_v0.md
```

**6. Read the design rationale**

```bash
open docs/rl_env_design.md   # or: cat docs/rl_env_design.md
```

Covers reward decomposition strategy, known grader limitations, extension paths
toward a full RL training loop, and training cycle v0 additions.

---

## Reward decomposition

Each task uses 3–4 named dimensions. The table below shows which are deterministic,
which require an LLM judge, and what the main exploit is for each.

**Document revision**

| Dimension | Grader | Gameable by |
|---|---|---|
| `instruction_following` | deterministic | compressing to word limit with contractions |
| `faithfulness` | LLM | adding unsupported claims in active voice |
| `over_editing` | deterministic | shortening the quoted input text |
| `quality_delta` | LLM | surface improvements that miss the real problems |

**Spreadsheet cleanup**

| Dimension | Grader | Gameable by |
|---|---|---|
| `data_preservation` | deterministic | duplicating rows to inflate count |
| `format_validity` | deterministic | valid CSV with wrong schema |
| `unit_consistency` | deterministic | hiding units in free-text Notes field |
| `completeness` | deterministic | renaming columns |

**Citation-grounded editing**

| Dimension | Grader | Gameable by |
|---|---|---|
| `citation_present` | deterministic | inserting empty brackets `[]` |
| `citation_accurate` | LLM | fabricating plausible-sounding quotes |
| `hallucination_flag` | LLM | citing a source that only loosely supports the claim |
| `argument_preservation` | deterministic | rephrasing anchor phrases |

Hard-fail caps: any output that is not parseable CSV, exceeds 120% of the word limit,
or contains zero citation markers is capped at ≤ 0.3 regardless of other dimension scores.

---

## Model Baseline and SFT Scaffold

This section is a reproducibility document. It distinguishes four things that must not
be confused: the heuristic optimization loop, the base-model inference baseline,
the SFT scaffold infrastructure, and actual SFT results.

---

### 1. Heuristic optimization loop v0

**What it is:** A four-policy comparison (`naive`, `format_compliance`, `reward_aware`,
`overfit`) run against 80 synthetically generated spreadsheet-cleaning tasks. All policies
are fixed heuristic rules — no model is trained. The purpose was to establish a grader
baseline and expose concrete reward-hacking failure modes under optimization pressure.

**What it demonstrated:** The harness floor is already high (naive achieves 0.9345 composite
with minimal effort). The `reward_aware` policy reaches 1.0 by selecting templates using
grader access at inference time. The `overfit` policy exposes two exploitable grader
weaknesses: row duplication and unit-hiding in Notes free text.

**Where results live:**
- [`results/optimization_loop_v0.md`](results/optimization_loop_v0.md) — policy comparison
- [`results/training_cycle_v0.md`](results/training_cycle_v0.md) — end-to-end cycle summary

---

### 2. Model baseline

**What it tests:** Whether the default base model (`mlx-community/Qwen2.5-3B-Instruct-4bit`)
can perform spreadsheet-cleaning tasks at a level that justifies SFT training. Scored with
the existing deterministic composite grader only — no LLM judge, no new grader.

**How to run:**

```bash
# Generate held-out split (seed 200, n=80)
python scripts/generate_tasks.py \
    --task spreadsheet_clean --n 80 --seed 200 \
    --output data/generated/spreadsheet_heldout_v1.jsonl

# Smoke test (20 cases, default)
python scripts/run_model_baseline.py

# Full run
python scripts/run_model_baseline.py --limit 0

# Dry run (print config only)
python scripts/run_model_baseline.py --dry-run
```

**Where results live:** [`results/model_baseline_v0.md`](results/model_baseline_v0.md)

**Baseline decision gate:**

Proceed to actual SFT training only if **both** are true:
- composite mean **>= 0.30**
- hard-fail rate **<= 40%**

This is a provisional engineering gate, not a statistical claim. If either condition
fails: document the result; do NOT train; the SFT scaffold may still be committed
as infrastructure; the README must state that training is not yet justified.

**Baseline result (training_cycle_v0):**

- model: `mlx-community/Qwen2.5-3B-Instruct-4bit`
- cases evaluated: 80
- parseability: 100.0%
- mean composite: **0.9569**
- hard-fail rate: 0.0%
- **gate: PASS** — baseline criteria met; SFT training is justified

See [`results/model_baseline_v0.md`](results/model_baseline_v0.md) for full per-dimension breakdown.

---

### 3. SFT scaffold

**What it adds:** Infrastructure for SFT training — train/held-out data splits,
SFT record builder, training config, and a training script with `--dry-run` /
`--check-only` modes. No training has been run.

**How to build SFT data:**

```bash
# Generate training split (seed 100, n=240)
python scripts/generate_tasks.py \
    --task spreadsheet_clean --n 240 --seed 100 \
    --output data/generated/spreadsheet_train_v1.jsonl

# Build SFT records (gold outputs from generator, no model calls)
# Writes to data/sft_collab_eval_full/train.jsonl (MLX-LM directory format)
python scripts/build_sft_data.py

# The committed sample (20 records, seed 999) is at:
# data/sft_collab_eval_sample.jsonl
```

**How to verify setup / dry-run / train:**

```bash
# Verify config, data path, and mlx_lm availability
python scripts/train_collab_sft.py --check-only

# Print exact MLX-LM training command without training
python scripts/train_collab_sft.py --dry-run

# Run actual training (requires mlx-lm installed and train.jsonl built)
python scripts/train_collab_sft.py --train
```

The `--dry-run` command prints:
```
mlx_lm.lora --config configs/sft_collab_eval_qwen25_3b.yaml
```

**Promotion gate (also in `scripts/train_collab_sft.py` and `results/collab_sft_v0.md`):**

Promote the SFT adapter only if **all** are true:
- composite improves by **>= 0.10** over base model
- hard-fail rate does **not increase**
- format_validity does **not regress**
- no obvious increase in reward-hacking behavior

**Important note on the +0.10 gate:** The base-model baseline already reached composite
0.9569. Since composite is capped at 1.0, the maximum possible improvement is ~0.043 —
the original +0.10 gate is mathematically unreachable. The gate is preserved here as
documented; it should be reviewed and revised before running actual training.

To run the comparison eval after training:

```bash
python eval/run_collab_model_eval.py \
    --model-base mlx-community/Qwen2.5-3B-Instruct-4bit \
    --adapter adapters/sft_collab_eval_qwen25_3b/ \
    --tasks data/generated/spreadsheet_heldout_v1.jsonl
```

---

### 4. Actual SFT results

**SFT v0 was run. The adapter is not promoted.**

Initial SFT v0 was run after the base-model baseline passed the provisional training gate.
The adapter is not promoted. It improved `unit_consistency` from 0.8625 to 1.0000, but reduced
mean composite from 0.9569 to 0.9110 and `data_preservation` from 0.9750 to 0.7500.
RH-like cases increased from 2 to 20, consistent with clean-looking outputs that drop rows or
values. This suggests the first SFT recipe over-optimized unit normalization while damaging
preservation behavior.

**Note on the promotion gate:** The base composite was already 0.9569. Since composite is capped
at 1.0, the maximum possible improvement is ~0.043. The original +0.10 composite promotion gate
is mathematically unreachable given the base score. The gate must be revised before the next
training attempt.

**Next work is dataset hardening, not DPO.** Inspect the RH-like failures, generate larger and
harder synthetic cases emphasizing row/value preservation, add a validation split, and rerun a
gentler SFT recipe. DPO should be considered only after SFT no longer regresses preservation or
increases RH-like behavior.

See [`results/collab_sft_v0.md`](results/collab_sft_v0.md) for full metrics and
[`docs/sft_v0_builder_notes.md`](docs/sft_v0_builder_notes.md) for interpretation and next steps.

Note: v1 hardening attempt did not promote — see [`results/collab_sft_v1.md`](results/collab_sft_v1.md).

---

### 5. SFT v1 — hardening attempt (gate: FAIL on stress)

**v1 was run. The adapter is not promoted.** v0 adapter is preserved unmodified.

#### Eval snapshot

| Metric | Base | SFT v1 | Delta |
|---|---|---|---|
| composite | 0.9569 | 0.9956 | +0.0387 |
| data_preservation | 0.9750 | 0.9875 | +0.0125 |
| unit_consistency | 0.8625 | 1.0000 | +0.1375 |
| RH-like cases | 2 | 1 | −1 |
| **stress data_preservation** | 0.2000 | **0.2500** | +0.0500 |

**Gate: FAIL on stress (4/5 PASS)** — stress `data_preservation_mean` 0.25 < 0.85 bar.

#### Promotion gate (v1)

Supersedes the v0 `+0.10` composite gate, which was mathematically unreachable from a
0.9569 base (composite capped at 1.0, maximum possible improvement ~0.043).

Conditions (all must pass):
1. `composite_mean >= base_composite - 0.005` (no regression)
2. `data_preservation >= base_data_preservation` (no regression)
3. `rh_like_count <= base_rh_like_count` (no increase)
4. At least one of `{unit_consistency, format_validity, completeness}` improves by `>= 0.02`
5. Preservation-stress `data_preservation_mean >= 0.85`

See `configs/sft_collab_eval_qwen25_3b.yaml` for the canonical gate definition.

#### Generate the stress split

```bash
python scripts/generate_tasks.py --task spreadsheet_clean_stress \
    --n 80 --seed 400 \
    --output data/generated/spreadsheet_train_stress_v1.jsonl
python scripts/generate_tasks.py --task spreadsheet_clean_stress \
    --n 40 --seed 300 \
    --output data/generated/spreadsheet_heldout_stress_v1.jsonl
```

#### Two-stage v1 eval

```bash
# 1. Stress eval first; emit per-case JSON
python eval/run_collab_model_eval.py \
    --adapter adapters/sft_collab_eval_qwen25_3b_v1/ \
    --eval-set data/generated/spreadsheet_heldout_stress_v1.jsonl \
    --emit-per-case results/collab_sft_v1_stress_per_case.json \
    --report-path results/collab_sft_v1_stress.md

# 2. Regular eval; chain stress per-case JSON for the 5th gate condition
python eval/run_collab_model_eval.py \
    --adapter adapters/sft_collab_eval_qwen25_3b_v1/ \
    --stress-results results/collab_sft_v1_stress_per_case.json \
    --report-path results/collab_sft_v1.md
```

Note: `results/collab_sft_v1_stress.md` will read as "Gate: FAIL — gate condition 5 not
evaluated" when run standalone — this is expected. That file documents the stress-set
numbers; the actual gate verdict lives in `collab_sft_v1.md`.

---

### 6. SFT v2 — data rebalance attempt (gate: FAIL with concern)

**v2 was run. The adapter is not promoted.** v0 and v1 adapters are preserved unmodified.

#### Eval snapshot

| Metric | Base | SFT v2 | Delta |
|---|---|---|---|
| composite | 0.9569 | 0.9912 | +0.0343 |
| data_preservation | 0.9750 | 0.9750 | +0.0000 |
| unit_consistency | 0.8625 | 1.0000 | +0.1375 |
| RH-like cases | 2 | 2 | +0 |
| **stress data_preservation** | 0.2000 | **0.2500** | +0.0500 |

**Gate: FAIL on stress with concern (4/5 PASS)** — stress `data_preservation_mean` 0.25 < 0.85 bar.
**Hypothesis (50% stress data lifts preservation) was refuted: identical to v1 despite 3× more stress
training cases.**

#### Headline diagnostic

The v2 training dataset doubled stress proportion from 25% to 50% (480 total = 240 regular +
240 stress), dropping the deletion-event ratio from 61.2% to 30.6%
(`docs/sft_v2_data_audit.md`). Conditions 1–4 all pass. Condition 5 fails at 0.25 — the same
score as v1, which had only 80 stress training cases. The stress data taught unit_consistency
cleanly (stress `unit_consistency` 0.70 → 1.00, identical to v1), but the row-preservation
instruction did not transfer at all. The flat trajectory across two experiments (0.25 at v1
with 80 stress cases; 0.25 at v2 with 240 stress cases) points to gradient competition or
signal asymmetry as the dominant failure mode, not data quantity. See
[`results/collab_sft_v2.md`](results/collab_sft_v2.md) for the full diagnostic and three v3
candidates (curriculum training, 75% rebalance, row-level reward shaping).

#### Reproducibility

- Config: `configs/sft_collab_eval_qwen25_3b_v2.yaml`
- Training data: `data/sft_collab_eval_full_v2/` (regenerate from seeds 100 + 500)
- Held-out splits: unchanged from v1 (`spreadsheet_heldout_v1.jsonl` seed=200, `spreadsheet_heldout_stress_v1.jsonl` seed=300)

---

### 7. SFT v3 — curriculum learning attempt (gate: FAIL — hypothesis refuted)

**v3 was run. The adapter is not promoted. v0, v1, and v2 adapters are preserved unmodified.**

Two-phase curriculum: 150 iters stress-only at lr=5e-5 (phase 1), then 150 iters mixed
at lr=2e-5 (phase 2, resumed from phase 1 final checkpoint, 40% LR to protect phase 1
preservation gains). Total 300 iters, same compute as v2.

#### Eval snapshot

| Metric | Base | SFT v3 | Delta |
|---|---|---|---|
| composite | 0.9569 | 0.9912 | +0.0343 |
| data_preservation | 0.9750 | 0.9750 | +0.0000 |
| unit_consistency | 0.8625 | 1.0000 | +0.1375 |
| RH-like cases | 2 | 2 | +0 |
| **stress data_preservation** | 0.2000 | **0.2500** | +0.0500 |

**Gate: FAIL — hypothesis refuted (4/5 PASS).** Stress `data_preservation_mean` 0.25 is
**identical to v1 and v2**. Three SFT iterations, three different interventions (recipe
gentleness in v1, data quantity in v2, curriculum ordering in v3), zero movement on the
target dimension.

#### Headline diagnostic

Phase 1 stress-only (no competing deletion gradients) drove train loss 0.883 → 0.267 on
stress data, with val loss tracking. Phase 2 resumed correctly from phase 1 weights (iter-1
val loss 0.388 vs fresh-start 1.109) and refined to 0.245 train / 0.262 val. Both phases
trained successfully. **Despite correct training, stress `data_preservation` remained at 0.25.**

This rules out gradient competition (H2) as the bottleneck — phase 1 had no competing
deletion gradients and still failed to push preservation above 0.25. It confirms signal
asymmetry (H3): the preservation signal is "don't generate something," and SFT learns
from positive demonstrations only. When the gold preserves rows, the model learns the
surface form but not the policy decision; co-located positive signals (unit conversion,
header preservation) transfer cleanly because they ARE positive demonstrations.

A side note from the v3 run: mlx-lm's `adapter_path` config field is save-only and does not
load existing weights for resume. Phase 2 needed the separate `resume_adapter_file` field
pointing at the phase-1 final safetensors. See `FAILURE_MODES.md` Mode 12.

#### What's next

The flat trajectory across v0/v1/v2/v3 stress `data_preservation` (0.25 ± 0.00 across three
SFT recipes and one curriculum experiment) is the harness's clearest experimental signal:
**standard SFT cannot teach the row-preservation policy through positive demonstrations**.
The gradient signal for "preserve this row" is too weak relative to the gradient signal for
"convert $M to $K" or "drop the annotation row," and the asymmetry persists regardless of
recipe gentleness, data quantity, or curriculum ordering.

The next experimental candidate is **DPO-on-preservation pairs** — a discriminative signal
of the form "this output preserves all rows; this other output drops some" that explicitly
trains the policy on the preservation decision rather than asking it to imitate a positive
demonstration. The collab-eval generator already produces stress cases (long tables, gold
preserves all rows); building preference pairs is one new generator + a DPO training step on
top of v3 (or directly on top of base, since the SFT-acquired schema is intact).

The full v4 prompt and design is queued for a separate session — not started here.

#### Reproducibility

| Field | Value |
|---|---|
| Phase 1 config | `configs/sft_collab_eval_qwen25_3b_v3_phase1.yaml` |
| Phase 2 config | `configs/sft_collab_eval_qwen25_3b_v3_phase2.yaml` |
| Phase 1 adapter | `adapters/sft_collab_eval_qwen25_3b_v3_phase1/` |
| Final adapter (v3) | `adapters/sft_collab_eval_qwen25_3b_v3/` |
| Phase 1 data | `data/sft_collab_eval_full_v3_phase1/` (regenerate from seed=500) |
| Phase 2 data | `data/sft_collab_eval_full_v2/` (regenerate from seeds 100 + 500) |
| Held-out splits | unchanged from v1 (seeds 200 regular, 300 stress) |
| Full report | [`results/collab_sft_v3.md`](results/collab_sft_v3.md) |

---

## Repo layout

```
collab-eval/
  collab_eval/
    base.py              # TaskSpec, Episode, TaskEnv (abstract)
    demo.py              # Demo runner (good + bad agent per task)
    env/tasks/
      doc_revision.py
      spreadsheet_clean.py
      citation_ground.py
    graders/
      deterministic.py   # Rule-based checks with Design note: comments
      llm_judge.py       # Optional Anthropic SDK grader (per-dimension)
      composite.py       # Weighted scoring + hard-fail caps
    generation/
      spreadsheet_generator.py  # Synthetic task generator (training cycle v0)
    policies/
      baselines.py       # Four heuristic policies for optimization loop
    inference/
      mlx_runner.py      # MLX-LM wrapper; degrades gracefully if unavailable
  tests/
    test_reward_hacking_cases.py  # Adversarial demonstrations (original 4)
    test_eval_extended.py         # Extended quality-range + RH probe cases (35)
    test_generation.py            # Generated task schema and coverage tests
    test_calibration.py           # Calibration record schema and sanity tests
    test_optimization_loop.py     # Policy outputs and overfit hacking tests
    test_split_validation.py      # Train/held-out overlap and gradeability checks
    test_sft_data.py              # SFT record schema and gold determinism tests
    test_baseline_runner.py       # Baseline runner graceful skip; train scaffold checks
  scripts/
    run_demo.py
    generate_tasks.py             # Generate synthetic task cases (with validation)
    run_judge_calibration.py      # Validate calibration set (offline + LLM)
    run_model_baseline.py         # Base-model inference baseline (MLX-LM)
    build_sft_data.py             # Build SFT records from generated cases (no model)
    train_collab_sft.py           # SFT training scaffold (--dry-run / --check-only)
    run_optimization_loop.py      # Four-policy optimization loop runner
  docs/
    rl_env_design.md              # Design note: reward decomposition, extensions
    generated_task_schema.md      # Schema for generated task JSONL records
    sft_data_schema.md            # Schema for SFT training records
  configs/
    sft_collab_eval_qwen25_3b.yaml  # MLX-LM LoRA config (Qwen2.5-3B)
  eval/
    run_collab_model_eval.py      # Base vs. SFT comparison; promotion gate check
  data/
    sample_docs/                  # Synthetic input documents (hardcoded tasks)
    generated/                    # Generated task JSONL (gitignored, reproducible)
    judge_calibration/            # Seed calibration set for LLM judge
    sft_collab_eval_sample.jsonl  # Committed SFT sample (20 records, seed 999)
  results/
    eval_results_v1.md            # Original 35-case grader results (preserved)
    training_cycle_v0_audit.md    # Phase 0 audit
    judge_calibration_v1.md       # Calibration run report
    optimization_loop_v0_raw.jsonl  # Per-case raw results
    optimization_loop_v0.md       # Four-policy comparison report
    training_cycle_v0.md          # End-to-end training cycle summary
    model_baseline_v0.md          # Base-model baseline (run status + metrics if run)
    collab_sft_v0.md              # SFT scaffold status (results if training has run)
```
