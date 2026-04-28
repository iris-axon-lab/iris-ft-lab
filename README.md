# iris-ft-lab

Fine-tuning lab for Trace Layer 2 — structured memory extraction and prospective intent modeling.

This is not a generic LLM fine-tuning sandbox. It is a focused personal lab for improving
Trace's core processing pipeline: transforming raw trace inputs into structured memory records,
detecting prospective commitments, and later matching intentions against observed outcomes.

---

## What this is, and why it exists

**Trace** is a local-first memory system built around a "witness, not coach" principle — it
surfaces structure, patterns, and prospective commitments from a person's raw notes and
reflections, without advice, pep talks, or prescriptive framing.

The weak link in Trace is **Layer 2 processing**: the model that turns raw trace text into
structured memory records. That model needs to:

- Classify memory tier correctly (`episodic`, `semantic`, `procedural`, `prospective`)
- Extract stated intent when it's genuinely present — and leave it null when it isn't
- Distinguish aspiration from commitment (the hardest boundary)
- Stay in witness mode: record what was said, not what should have been done

Off-the-shelf instruct models are reasonable but imprecise here, especially on the
aspiration-vs-commitment distinction and the no-coaching constraint. This lab is the
training and evaluation infrastructure for improving that.

---

## Results

> **Small gold eval / smoke eval** — 12 examples. Results are directional, not statistically robust.
> See [`eval/results.md`](eval/results.md) for the full report including confusion matrices,
> failure case analysis, and Trace-style integration examples.

| Metric | Baseline (no adapter) | Fine-Tuned (SFT v2) | + DPO v1 |
|--------|----------------------|---------------------|----------|
| Overall accuracy | 0/12 (0.0%) | 9/12 (75.0%) | **12/12 (100.0%)** |
| episodic | 0/2 (0%) | 2/2 (100%) | 2/2 (100%) |
| semantic | 0/3 (0%) | 3/3 (100%) | 3/3 (100%) |
| procedural | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| prospective | 0/6 (0%) | 3/6 (50%) | **6/6 (100%)** |
| aspiration-vs-commitment | 0/4 (0%) | 2/4 (50%) | 4/4 (100%) |
| parse errors | 0 (wrong schema) | 0 | 0 |

**Key finding (SFT):** The base model reliably outputs JSON but uses its own invented schema —
fields like `"memoryRecord"`, `"memory_records"`, `"extractedStatements"` — never the Trace
`memory_tier` field. A light SFT pass (100 examples, ~4 epochs) is sufficient to align the
model to the Trace schema and achieve 75% tier classification accuracy.

**Key finding (DPO):** The remaining 3 SFT errors were all hard prospective edge cases —
weak-signal commitments (`eval_008`), conditional commitments (`eval_009`), and mixed
episodic+prospective inputs (`eval_010`). A small DPO run (80 synthetic preference pairs,
150 iters at lr=5e-6, β=0.1) on top of the SFT-fused model corrected all three with no
regressions on the 9 cases SFT already got right. See [`eval/dpo_v1_results.md`](eval/dpo_v1_results.md)
for the full report.

**Model:** `mlx-community/Qwen2.5-3B-Instruct-4bit` |
**Adapter:** `outputs/sft_qwen25_3b_v2/` |
**Training data:** [`data/sft_train_100.jsonl`](data/sft_train_100.jsonl) (100 examples, balanced 25/tier) |
**Eval set:** [`data/eval_gold.jsonl`](data/eval_gold.jsonl) |
**Full notes:** [`eval/notes.md`](eval/notes.md)

**DPO adapter:** `outputs/dpo_qwen25_3b_v1/` |
**DPO config:** [`configs/dpo_trace_qwen25_3b.yaml`](configs/dpo_trace_qwen25_3b.yaml) |
**DPO data:** [`data/processed/dpo/`](data/processed/dpo/) (regenerate with `scripts/generate_synthetic_dpo.py`)

---

## Lessons & failure modes

Both pipelines hit real failure modes during development — mode collapse, a quantized-base
fusion defect, an unreachable promotion gate, training-distribution asymmetry, a missing
DPO backend, and several others. Each was diagnosed, fixed, and committed; the
consolidated catalog with symptoms, root causes, fixes, and cross-cutting patterns lives in
[`FAILURE_MODES.md`](FAILURE_MODES.md).

If you're iterating on either pipeline, **read it first.** Most of the cross-cutting
rules — "audit your gold output before training", "calibrate gates against base-model
evidence", "beware silent transformations on quantized weights", "tool-call timeouts are
a recipe parameter" — would have saved a session each time they were learned the hard way.

---

## Hardware

- MacBook Pro M4 / 64GB unified memory / 1TB storage
- Apple Silicon (Metal GPU backend)
- No external GPU required

**Why this hardware matters:**

64GB unified memory is unusually large for a laptop — it comfortably fits a 4-bit quantized
7B model (≈4GB) alongside activations, optimizer state, and LoRA adapters with room to spare.
You can run meaningful fine-tuning experiments locally without cloud compute.
DPO training peaks at ~19 GB on the float16 fused model (vs ~4 GB for SFT); smaller-RAM Macs
should expect to use the SFT path only.

---

## Why MLX-LM

[MLX](https://github.com/ml-explore/mlx) is Apple's native machine learning framework,
designed specifically for Apple Silicon's unified memory architecture.

- No PCIe transfer overhead — weights stay in unified memory across CPU and GPU
- MLX-LM wraps MLX for language model training and inference with minimal boilerplate
- LoRA training on a 3B model takes minutes per epoch on M4
- Everything runs locally — no data leaves the machine

CUDA-based frameworks (PyTorch + HuggingFace Trainer) work on Apple Silicon via CPU
but are substantially slower and miss the unified memory advantage. MLX is the right
tool for this hardware.

---

## Quick start

**For DPO only:** `pip install -U mlx-lm-lora` (third-party, separate from `mlx-lm`).
The DPO scaffold (`scripts/train_dpo.py`) checks for this and exits cleanly if missing.
See `data/dpo_design_notes.md` §5 for the rationale.

```bash
# 1. Clone and enter the repo
cd iris-ft-lab

# 2. Set up the environment
bash setup.sh
source .venv/bin/activate

# 3. Verify everything is working
make verify

# 4. Generate training data (no model required)
make generate-data          # writes data/sft_train_100.jsonl  (100 examples, 25/tier)

# 5. Prepare train/val splits
make prepare-data           # writes data/processed/sft/{train,valid}.jsonl

# 6. Run SFT training (model auto-downloads ~2GB on first use)
make sft                    # trains outputs/sft_qwen25_3b_v2/  (~4 min on M4)

# 7. Run baseline vs fine-tuned eval
make eval                   # writes eval/results.md + eval/results_raw.json
```

To download the model ahead of time:

```bash
python -c "from mlx_lm import load; load('mlx-community/Qwen2.5-3B-Instruct-4bit')"
```

To verify the published results without retraining:

```bash
make validate-results       # checks all artifacts exist and paths are consistent
```

---

## Eval Snapshot — collab-eval (document tasks)

> **Three SFT attempts and one DPO attempt; none promoted.** v0 mode-collapsed; v1
> hardened recipe but stress preservation flat at 0.25; v2 doubled stress data, still
> flat at 0.25; v3 ran two-phase curriculum, still flat at 0.25; DPO v0 (preference
> learning over preserve-vs-drop pairs) regressed to 0.20 — DPO learned the
> discrimination trivially but did not transfer that into generation-time row
> preservation, also costing −0.025 on unit_consistency. **Two instruments exhausted
> (SFT positive demonstrations and DPO preference learning).** The flat trajectory
> across four interventions points at preservation being a generation-time policy
> property that requires generation-time feedback. Next experimental candidate is
> **RL with grader as reward** (PPO/GRPO) — see
> [`collab-eval/results/collab_dpo_v0.md`](collab-eval/results/collab_dpo_v0.md) and
> [`collab-eval/docs/preservation_analysis.md`](collab-eval/docs/preservation_analysis.md)
> for the full diagnostic.

| Run | composite | data_preservation | unit_consistency | stress data_preservation | Verdict |
|---|---|---|---|---|---|
| Base | 0.9569 | 0.9750 | 0.8625 | 0.2000 | — |
| SFT v0 | 0.9110 | 0.7500 | 1.0000 | — | NOT PROMOTED (mode collapse) |
| SFT v1 | 0.9956 | 0.9875 | 1.0000 | 0.2500 | NOT PROMOTED (gate: FAIL on stress) |
| SFT v2 | 0.9912 | 0.9750 | 1.0000 | 0.2500 | NOT PROMOTED (with concern; hypothesis refuted) |
| SFT v3 | 0.9912 | 0.9750 | 1.0000 | 0.2500 | NOT PROMOTED (curriculum; hypothesis refuted) |
| DPO v0 | 0.9506 | 0.9750 | 0.8375 | 0.2000 | NOT PROMOTED (preference learning; hypothesis refuted) |

[Full reports](collab-eval/results/) · [Reward design rationale](collab-eval/docs/rl_env_design.md) · [Failure modes](FAILURE_MODES.md) · [Preservation analysis](collab-eval/docs/preservation_analysis.md)

---

## Repo layout

```
iris-ft-lab/
│
├── configs/
│   ├── sft_trace_qwen25_3b_v2.yaml # ★ Canonical SFT config (v2 — published results)
│   ├── sft_trace_qwen25_3b.yaml    #   Legacy v1 scaffold config (3-example, deprecated)
│   ├── dpo_trace_qwen25_3b.yaml    #   DPO scaffold config (Stage 3)
│   └── model_registry.yaml        #   All model paths live here — never in Python code
│
├── data/
│   ├── README.md                   # Schema documentation and data format reference
│   ├── sft_train_100.jsonl         # ★ Canonical training data (100 examples, 25/tier)
│   ├── eval_gold.jsonl             # ★ 12 gold eval examples (small eval / smoke eval)
│   ├── sample_sft.jsonl            #   3-example scaffold (legacy reference)
│   ├── sample_dpo.jsonl            #   3-example DPO scaffold
│   ├── raw_private/                #   Local workspace for real traces (gitignored)
│   └── processed/                  #   Train/val splits — regenerate with make prepare-data
│                                   #   (gitignored; .gitkeep preserves stub)
│
├── eval/                           # ★ Eval harness and results
│   ├── run_eval.py                 #   Baseline vs fine-tuned comparison eval
│   ├── results.md                  #   Published results report
│   ├── results_raw.json            #   Raw per-example inference results (JSON)
│   └── notes.md                   #   Training command, hyperparameters, loss log
│
├── scripts/
│   ├── setup_verify.py             # Verify Python, MLX, MLX-LM, Metal
│   ├── generate_synthetic_sft.py   # Generate sft_train_100.jsonl (no model needed)
│   ├── prepare_data.py             # Validate + split JSONL data
│   ├── train_sft.py                # SFT training wrapper (reads YAML only)
│   ├── train_dpo.py                # DPO scaffold (honest about current availability)
│   ├── eval_extraction.py          # Legacy extraction eval (used by eval-v1 target)
│   ├── eval_tracking.py            # Intention-reality gap characterization eval
│   └── run_trace_style_inference.py # Pipe-friendly inference entrypoint
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Tier distribution, intent frequency, example inspection
│   ├── 02_lora_walkthrough.ipynb   # LoRA intuition, rank/alpha/modules on Apple Silicon
│   └── 03_dpo_memory_policy.ipynb  # DPO rationale, chosen/rejected patterns, policy goals
│
├── tests/
│   ├── test_prepare_data.py        # Schema validation, split correctness, JSONL I/O
│   └── test_configs_load.py        # YAML structure, required keys, sane values
│
├── outputs/                        # Adapter weights after training (gitignored)
├── pyproject.toml
├── setup.sh
└── Makefile                        # make help for all targets
```

Items marked ★ are the canonical v2 artifacts that produced the published results.

---

## How to swap models

**Model paths live only in `configs/model_registry.yaml` and the YAML configs.**
No model ID appears in Python source.

To switch to the smoke-test model for faster iteration:

```bash
python scripts/train_sft.py \
    --config configs/sft_trace_qwen25_3b_v2.yaml \
    --model mlx-community/Qwen3-1.7B-4bit
```

To make it the default, edit `model.path` in `configs/sft_trace_qwen25_3b_v2.yaml`.

To upgrade to the 7B model after pipeline validation:

```yaml
# configs/sft_trace_qwen25_3b_v2.yaml
model:
  path: "mlx-community/Qwen2.5-7B-Instruct-4bit"
```

See `configs/model_registry.yaml` for all tested model paths and notes on when
to use each.

---

## Data schema overview

### Memory record (SFT target)

```json
{
  "content_summary": "Factual summary of what was recorded.",
  "memory_tier": "semantic | episodic | procedural | prospective",
  "emotional_valence": "positive | neutral | negative | mixed",
  "stated_intent": null,
  "topic_cluster": "domain / subdomain",
  "timestamp": "YYYY-MM-DD",
  "source": "daily_trace",
  "source_id": "trace_YYYYMMDD_NNN",
  "channel": "work | personal | ..."
}
```

### Intention-reality record (Stage 3 tracking)

```json
{
  "original_intent": "What was committed to.",
  "stated_when": "YYYY-MM-DD",
  "outcome_observed": "What actually happened.",
  "outcome_when": "YYYY-MM-DD",
  "gap_characterization": "fulfilled | abandoned | transformed | unresolved"
}
```

### The aspiration vs. commitment distinction

This is the critical boundary in the extraction pipeline:

| Signal | Tier |
|--------|------|
| "I'd love to get back into meditation someday." | `semantic` |
| "I keep thinking I should write more consistently." | `semantic` |
| "I should really reach out to David this week." | `prospective` |
| "I told Sam I'd send the draft by Thursday." | `prospective` |
| "I need to submit the proposal by Friday 5pm." | `prospective` |

The rule: **any plan, deadline, social accountability, or directional self-commitment
qualifies as prospective**. Pure aspiration without action-orientation stays semantic.

See `data/README.md` for the full schema reference and `eval_gold.jsonl` for worked
examples across all four tiers.

---

## Privacy and local-first notes

- **`data/raw_private/`** is the local workspace for real personal traces. It is
  excluded from git by `.gitignore`. Never commit it.
- **`data/processed/`** holds derived train/val splits. Regenerate with `make prepare-data`.
- All sample JSONL files in `data/` use entirely synthetic, fictional traces.
- Model weights download to your local cache (typically `~/.cache/huggingface/`).
  Nothing is uploaded.
- Adapter weights go to `outputs/` — also gitignored.

If you accidentally stage real traces: `git reset HEAD <file>` before committing.

---

## Known limitations

1. **DPO requires `mlx-lm-lora`.** The DPO backend is `mlx-lm-lora` (a separate third-party
   package, not the same as `mlx-lm`). Install with `pip install -U mlx-lm-lora`. Tested
   against v2.1.0. `train_dpo.py` checks availability at runtime and exits clearly if missing.

2. **`data/sft_train_100.jsonl` is synthetic.** The 100 training examples are template-based
   and do not cover the full range of real trace inputs. To improve quality further, generate
   examples from real traces in `data/raw_private/` and add them to the training set.

3. **Extraction eval depends on structured JSON output.** If the base model (without an
   adapter) doesn't reliably produce well-formed JSON, `eval_extraction.py` will show low
   accuracy due to parse failures, not actual tier errors. This clears up with even light
   SFT training.

4. **Model downloads can be large.** `Qwen2.5-3B-Instruct-4bit` is ~2GB. The 7B model
   is ~4GB. Plan disk space accordingly.

---

## 3-stage learning path

### Stage 1 — Foundation
*Environment, schema, data prep, smoke-test SFT*

- [ ] Run `bash setup.sh` and `make verify`
- [ ] Read `data/README.md` — understand the Trace schema before touching the model
- [ ] Run `make prepare-data` — validate sample data, inspect splits in `data/processed/`
- [ ] Open `notebooks/01_data_exploration.ipynb` — explore tier distribution and example structure
- [ ] Run a smoke-test SFT pass with Qwen3-1.7B:
      `python scripts/train_sft.py --config configs/sft_trace_qwen25_3b_v2.yaml --model mlx-community/Qwen3-1.7B-4bit --dry-run`
- [ ] Run inference on a raw trace:
      `echo "Finished debugging the race condition." | python scripts/run_trace_style_inference.py --config configs/sft_trace_qwen25_3b_v2.yaml`

### Stage 2 — Extraction quality
*Adapter training, eval loop, structured extraction*

- [ ] Run `make generate-data && make prepare-data` — build the 100-example training set
- [ ] Run full SFT on Qwen2.5-3B-Instruct: `make sft` — saves adapter to `outputs/sft_qwen25_3b_v2/`
- [ ] Run extraction eval: `make eval` — writes `eval/results.md` with baseline vs fine-tuned comparison
- [ ] Open `notebooks/02_lora_walkthrough.ipynb` — understand the LoRA settings
- [ ] Iterate on real data: add real traces from `data/raw_private/` to the training set for better coverage
- [ ] Upgrade to Qwen2.5-7B if 3B quality plateaus on aspiration-vs-commitment cases

### Stage 3 — Policy shaping
*DPO preference learning, prospective-memory policy, Trace-style integration*

DPO v1 is **PROMOTED** — 12/12 tier accuracy, 0 regressions. See [`eval/dpo_v1_results.md`](eval/dpo_v1_results.md).

- [x] Open `notebooks/03_dpo_memory_policy.ipynb` — understand the policy goal before training
- [x] Install DPO backend: `make install-dpo-backend` (`pip install -U mlx-lm-lora`)
- [x] Fuse SFT v2 adapter into base model (one-time): `make fuse-sft` → `outputs/qwen25_3b_sft_fused/` (~6 GB float16)
- [x] Generate preference pairs: `make generate-dpo-data` (80 train + 12 valid)
- [x] Run DPO from the fused-SFT model: `make dpo` → `outputs/dpo_qwen25_3b_v1/`
- [x] Eval DPO adapter: `make eval-dpo` — corrected all 3 prospective misses with 0 regressions
- [ ] Run `eval_tracking.py` on intention-reality pairs from real trace data
- [ ] Wire the trained adapter into Trace Layer 2 via `run_trace_style_inference.py`

---

## The full story

```
raw personal traces
       │
       ▼
 prepare_data.py        ← validates schema, creates train/val splits
       │
       ▼
   train_sft.py         ← LoRA SFT on Qwen2.5-3B; teaches Trace schema
       │
       ▼
 eval_extraction.py     ← tier accuracy, intent extraction, aspiration/commitment
       │
       ▼
   train_dpo.py         ← preference shaping; reinforces witness-not-coach
       │
       ▼
 eval_tracking.py       ← gap characterization: fulfilled/abandoned/transformed
       │
       ▼
run_trace_style_inference.py  ← pipe-friendly inference; JSON to stdout
       │
       ▼
   Trace Layer 2
```

---

## See also

- [trace](https://github.com/iris-axon-lab/trace) — a local-first longitudinal agent built on the same memory architecture principles.
