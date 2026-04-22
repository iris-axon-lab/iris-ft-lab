# Training Notes — SFT v2

## Run summary

| Field | Value |
|-------|-------|
| Date | 2026-04-21 |
| Model | `mlx-community/Qwen2.5-3B-Instruct-4bit` |
| Framework | MLX-LM 0.31.2 |
| Hardware | MacBook Pro M4 / 64 GB unified memory |
| Adapter output | `outputs/sft_qwen25_3b_v2/` |

## Training command

```bash
source .venv/bin/activate

# Step 1 — generate training data
python scripts/generate_synthetic_sft.py \
    --output data/sft_train_100.jsonl \
    --shuffle --seed 42
# (then 35 more examples appended to reach 100 balanced examples)

# Step 2 — prepare train/val splits
python scripts/prepare_data.py \
    --sft data/sft_train_100.jsonl \
    --eval data/eval_gold.jsonl \
    --seed 42

# Step 3 — run SFT
python scripts/train_sft.py \
    --config configs/sft_trace_qwen25_3b_v2.yaml
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| LoRA num_layers | 16 (last 16 of 36 transformer blocks) |
| Learning rate | 1e-4 |
| Batch size | 4 |
| Total iters | 100 |
| Max seq length | 512 |
| Grad checkpoint | true |
| Random seed | 42 (data split) |

## Data

| Split | Count |
|-------|-------|
| Total | 100 |
| Train | 90 |
| Val | 10 |
| Tier distribution | 25 episodic / 25 semantic / 25 procedural / 25 prospective |

Training data file: `data/sft_train_100.jsonl`
Split seed: 42 (in `scripts/prepare_data.py`)

## Training loss log

| Iter | Train loss |
|------|-----------|
| 1    | (init) |
| 10   | 2.154 |
| 20   | 0.853 |
| 30   | 0.640 |
| 40   | 0.582 |
| 50   | 0.451 |
| 60   | 0.359 |
| 70   | 0.261 |
| 80   | 0.161 |
| 90   | 0.129 |
| 100  | 0.074 |

Validation loss: 4.146 (iter 1) → 1.167 (iter 100)
Trainable parameters: 6.652M / 3085.939M (0.216%)
Peak memory: 3.762 GB

## Eval command

```bash
python eval/run_eval.py \
    --eval-data data/eval_gold.jsonl \
    --config configs/sft_trace_qwen25_3b_v2.yaml \
    --ft-adapter outputs/sft_qwen25_3b_v2 \
    --output eval/results_raw.json \
    --write-md eval/results.md
```

## Key findings

- Baseline produces JSON with invented schemas (no `memory_tier` field) → 0/12 on tier accuracy
- Fine-tuned achieves 9/12 (75%) — all episodic, semantic, procedural correct
- Remaining 3 errors are all hard prospective edge cases: weak-signal commitment (eval_008),
  conditional commitment (eval_009), mixed episodic+prospective input (eval_010)
- Aspiration-vs-commitment accuracy: 0% → 50% (2/4 correct)
- No parse errors with fine-tuned adapter (model reliably outputs Trace schema JSON)
