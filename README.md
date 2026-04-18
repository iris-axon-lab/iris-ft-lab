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

## Hardware

- MacBook Pro M4 / 64GB unified memory / 1TB storage
- Apple Silicon (Metal GPU backend)
- No external GPU required

**Why this hardware matters:**

64GB unified memory is unusually large for a laptop — it comfortably fits a 4-bit quantized
7B model (≈4GB) alongside activations, optimizer state, and LoRA adapters with room to spare.
You can run meaningful fine-tuning experiments locally without cloud compute.

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

```bash
# 1. Clone and enter the repo
cd iris-ft-lab

# 2. Set up the environment
bash setup.sh
source .venv/bin/activate

# 3. Verify everything is working
make verify

# 4. Validate and prepare the sample data
make prepare-data

# 5. Run the extraction eval (gold-distribution inspection, no model required)
python scripts/eval_extraction.py --eval-data data/eval_gold.jsonl --gold-only

# 6. When you have a model downloaded, run SFT
make sft

# 7. Eval the trained adapter
make eval
```

To download a model (MLX-LM handles this automatically on first use):

```bash
python -c "from mlx_lm import load; load('mlx-community/Qwen2.5-3B-Instruct-4bit')"
```

---

## Repo layout

```
iris-ft-lab/
│
├── configs/
│   ├── sft_trace_qwen25_3b.yaml    # SFT training config (LoRA, data, output paths)
│   ├── dpo_trace_qwen25_3b.yaml    # DPO scaffold config
│   └── model_registry.yaml        # All model paths live here — never in Python code
│
├── data/
│   ├── README.md                   # Schema documentation and data format reference
│   ├── sample_sft.jsonl            # 3 synthetic SFT examples (chat format)
│   ├── sample_dpo.jsonl            # 3 synthetic DPO preference pairs
│   ├── eval_gold.jsonl             # 12 synthetic eval examples with gold labels
│   ├── raw_private/                # Local workspace for real traces (gitignored)
│   └── processed/                  # Train/val splits from prepare_data.py (gitignored)
│
├── scripts/
│   ├── setup_verify.py             # Verify Python, MLX, MLX-LM, Metal
│   ├── prepare_data.py             # Validate + split JSONL data
│   ├── train_sft.py                # SFT training wrapper (reads YAML only)
│   ├── train_dpo.py                # DPO scaffold (honest about current availability)
│   ├── eval_extraction.py          # Extraction eval: tier, intent, aspiration/commitment
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
└── Makefile
```

---

## How to swap models

**Model paths live only in `configs/model_registry.yaml` and the YAML configs.**
No model ID appears in Python source.

To switch to the smoke-test model for faster iteration:

```bash
python scripts/train_sft.py \
    --config configs/sft_trace_qwen25_3b.yaml \
    --model mlx-community/Qwen3-1.7B-4bit
```

To make it the default, edit `model.path` in `configs/sft_trace_qwen25_3b.yaml`.

To upgrade to the 7B model after pipeline validation:

```yaml
# configs/sft_trace_qwen25_3b.yaml
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

1. **DPO is a scaffold.** MLX-LM's DPO API is evolving. `train_dpo.py` checks availability
   at runtime and exits honestly if it's not yet supported. Check the mlx-lm changelog
   before Stage 3 work.

2. **3 SFT examples are conceptual starters.** Real training requires at minimum ~50–200
   examples per tier to produce meaningful extraction quality. Generate from real traces
   in `data/raw_private/`.

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
      `python scripts/train_sft.py --config configs/sft_trace_qwen25_3b.yaml --model mlx-community/Qwen3-1.7B-4bit --dry-run`
- [ ] Run inference on a raw trace:
      `echo "Finished debugging the race condition." | python scripts/run_trace_style_inference.py --config configs/sft_trace_qwen25_3b.yaml`

### Stage 2 — Extraction quality
*Adapter training, eval loop, structured extraction*

- [ ] Generate 50–200 real SFT examples from your traces in `data/raw_private/`
- [ ] Run full SFT on Qwen2.5-3B-Instruct: `make sft`
- [ ] Run extraction eval: `make eval` — inspect tier accuracy and intent hit rate
- [ ] Open `notebooks/02_lora_walkthrough.ipynb` — understand the LoRA settings
- [ ] Iterate: adjust rank, target_modules, and epochs based on eval results
- [ ] Upgrade to Qwen2.5-7B if 3B quality plateaus on aspiration-vs-commitment cases

### Stage 3 — Policy shaping
*DPO scaffolding, prospective-memory policy, Trace-style integration*

- [ ] Open `notebooks/03_dpo_memory_policy.ipynb` — understand the policy goal before training
- [ ] Generate 50–500 DPO preference pairs (real traces, real preference judgments)
- [ ] Check DPO availability: `python scripts/train_dpo.py --check-only`
- [ ] Run DPO from the SFT checkpoint: `make dpo`
- [ ] Compare SFT vs DPO adapters on `eval_gold.jsonl` aspiration-vs-commitment cases
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
