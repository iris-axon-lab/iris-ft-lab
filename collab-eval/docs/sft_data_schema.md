# SFT Data Schema

## Overview

SFT records for spreadsheet-cleaning tasks follow a standard chat-messages format
compatible with MLX-LM fine-tuning. Each record contains a system prompt, a user
task description with the dirty CSV, and a deterministic gold assistant response.

## Gold Output Derivation

The gold assistant output is derived **deterministically** from the synthetic task
generator. The generator produces a dirty (messy) CSV from a known clean version.
`build_sft_data.py` recovers that clean version directly from `gold_or_reference_output`
in the generated case.

**No model is used to generate gold outputs.** No LLM, Anthropic API, Ollama, MLX,
or any other model is called during SFT data construction.

## Record Schema

```json
{
  "messages": [
    {
      "role": "system",
      "content": "<spreadsheet cleanup system prompt>"
    },
    {
      "role": "user",
      "content": "<task description and dirty CSV input>"
    },
    {
      "role": "assistant",
      "content": "<deterministic gold clean CSV output>"
    }
  ],
  "metadata": {
    "task_type": "spreadsheet_clean",
    "difficulty": "easy|medium|hard",
    "primary_dimension": "unit_consistency|data_preservation|format_validity|completeness",
    "generation_seed": 999,
    "case_id": "sc_gen_999_0000"
  }
}
```

## Field Descriptions

### messages

A 3-element array in `[system, user, assistant]` order.

| Role | Required | Content |
|------|----------|---------|
| system | yes | The spreadsheet cleanup task description and output format instructions |
| user | yes | Task directive plus the dirty CSV input |
| assistant | yes | The gold clean CSV; no explanation or commentary |

### metadata

| Field | Type | Description |
|-------|------|-------------|
| task_type | string | Always `"spreadsheet_clean"` in this dataset |
| difficulty | string | `"easy"`, `"medium"`, or `"hard"` — complexity of the noise introduced |
| primary_dimension | string | The grading dimension most challenged by this case's noise pattern |
| generation_seed | int | The seed passed to `generate_cases()` for this case |
| case_id | string | Unique case identifier; format `sc_gen_{seed}_{index:04d}` |

## Invariants

Every valid SFT record must satisfy:

1. `messages` has exactly 3 elements in `[system, user, assistant]` order.
2. `messages[2]["content"]` (assistant turn) is non-empty.
3. `messages[2]["content"]` is parseable as CSV (gold output is always valid CSV).
4. `metadata` contains all five required keys.
5. `metadata.primary_dimension` is one of the four valid dimensions.
6. `metadata.task_type == "spreadsheet_clean"`.
7. Gold output contains no private or proprietary strings.

## Files

| File | Records | Status | Notes |
|------|---------|--------|-------|
| `data/sft_collab_eval_sample.jsonl` | ≤ 20 | Committed | Seed 999; representative sample |
| `data/sft_collab_eval_full.jsonl` | ≤ 240 | Gitignored | Generated from training split |

## How to regenerate

```bash
# Generate training split first
python scripts/generate_tasks.py \
    --task spreadsheet_clean --n 240 --seed 100 \
    --output data/generated/spreadsheet_train_v1.jsonl

# Build full SFT data from training split
python scripts/build_sft_data.py \
    --input data/generated/spreadsheet_train_v1.jsonl \
    --output data/sft_collab_eval_full.jsonl

# Regenerate the committed sample (seed 999, n=20)
python -c "
import json, sys
sys.path.insert(0, '.')
from scripts.build_sft_data import build_sft_records_from_generator
records = build_sft_records_from_generator(n=20, seed=999)
with open('data/sft_collab_eval_sample.jsonl', 'w') as f:
    for r in records:
        f.write(json.dumps(r) + '\n')
"
```
