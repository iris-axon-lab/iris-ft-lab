# iris-ft-lab — Data

This directory holds training, preference, and evaluation data for Trace Layer 2
fine-tuning experiments. Only synthetic sample data is committed to version control.

---

## Directory layout

```
data/
  sample_sft.jsonl      # 3 synthetic SFT examples (tracked)
  sample_dpo.jsonl      # 3 synthetic DPO preference pairs (tracked)
  eval_gold.jsonl       # 12 synthetic eval examples with gold labels (tracked)
  raw_private/          # local workspace for real personal traces (gitignored)
  processed/            # train/val splits written by prepare_data.py (gitignored)
```

**Never commit files from `raw_private/` or `processed/`.**
Real personal traces live only in `raw_private/` and are excluded by `.gitignore`.
`processed/` holds derived artifacts (train/val splits); regenerate with `make prepare-data`.

---

## Trace memory record schema (SFT target)

Each SFT example teaches the model to transform a raw trace input into a
structured Trace-style record conforming to this schema:

```json
{
  "content_summary": "Concise factual summary of what was recorded.",
  "memory_tier": "semantic | episodic | procedural | prospective",
  "emotional_valence": "positive | neutral | negative | mixed",
  "stated_intent": null,
  "topic_cluster": "domain / subdomain",
  "timestamp": "YYYY-MM-DD",
  "source": "daily_trace | journal | meeting_note | ...",
  "source_id": "trace_YYYYMMDD_NNN",
  "channel": "work | personal | health | learning | ..."
}
```

### Memory tier definitions

| Tier | What it captures |
|------|-----------------|
| `episodic` | A specific past event or experience with temporal grounding |
| `semantic` | General self-knowledge, beliefs, patterns, or preferences |
| `procedural` | A learned technique, workflow, or repeatable skill |
| `prospective` | A forward-looking commitment, intention, or planned action |

### Aspiration vs. commitment — the critical distinction

This is the hardest classification boundary in the pipeline. The rule:

- **Aspiration → `semantic`** when there is no concrete plan, timeline, or
  accountability signal. ("I've been thinking I should write more.")
- **Commitment → `prospective`** when a plan, deadline, social accountability,
  or self-stated intention to act is present. ("I told Sam I'd send the draft
  by Thursday." / "I need to reach out to David this week.")

A weak hedge ("I should really...") is enough to qualify as `prospective` if
the person is clearly orienting toward action. Vague aspiration without any
directional signal stays `semantic`.

---

## Intention-reality tracking schema (Stage 3)

Used to evaluate whether prospective commitments were fulfilled over time:

```json
{
  "original_intent": "What was committed to.",
  "stated_when": "YYYY-MM-DD",
  "outcome_observed": "What actually happened.",
  "outcome_when": "YYYY-MM-DD",
  "gap_characterization": "fulfilled | abandoned | transformed | unresolved"
}
```

### Gap labels

| Label | Meaning |
|-------|---------|
| `fulfilled` | Intention was carried out as stated |
| `abandoned` | Intention was not acted on with no replacement |
| `transformed` | Intention evolved or was replaced by a different action |
| `unresolved` | Outcome not yet observed or ambiguous |

---

## Data formats

### SFT (`sample_sft.jsonl`, `processed/sft_train.jsonl`)

MLX-LM chat format — one JSON object per line:

```json
{
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "Raw trace input text."},
    {"role": "assistant", "content": "{...structured Trace record as JSON string...}"}
  ]
}
```

### DPO (`sample_dpo.jsonl`, `processed/dpo_train.jsonl`)

Prompt / chosen / rejected format — one JSON object per line:

```json
{
  "prompt":   "System prompt + raw trace input.",
  "chosen":   "Correct structured extraction (witness, not coach).",
  "rejected": "Incorrect extraction (over-flattened, over-coached, or mis-tiered)."
}
```

### Eval (`eval_gold.jsonl`)

Evaluation set with gold labels for structured comparison:

```json
{
  "id": "eval_001",
  "input": "Raw trace input text.",
  "gold": {
    "memory_tier": "episodic",
    "emotional_valence": "positive",
    "stated_intent": null,
    "topic_cluster": "engineering / debugging"
  },
  "eval_tags": ["tier_classification"],
  "note": "Optional clarification for tricky examples."
}
```

`eval_tags` values: `tier_classification`, `intent_extraction`,
`aspiration_vs_commitment`, `gap_characterization`

---

## Privacy rules

- `raw_private/` is the local workspace for real experiments. Never commit it.
- `processed/` holds derived train/val splits. Never commit it; regenerate as needed.
- Sample JSONL files in `data/` must use only synthetic, fictional traces.
- If you accidentally stage real traces, use `git reset HEAD <file>` before committing.
