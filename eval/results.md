# Eval Results — Baseline vs. Fine-Tuned (SFT v2)

> **Small gold eval / smoke eval** — 12 examples. Results are directional, not statistically robust.

## Summary Table

| Metric | Baseline (no adapter) | Fine-Tuned (SFT v2) |
|--------|----------------------|---------------------|
| Overall accuracy | 0/12 (0.0%) | 9/12 (75.0%) |
| episodic accuracy | 0/2 (0.0%) | 2/2 (100.0%) |
| semantic accuracy | 0/3 (0.0%) | 3/3 (100.0%) |
| procedural accuracy | 0/1 (0.0%) | 1/1 (100.0%) |
| prospective accuracy | 0/6 (0.0%) | 3/6 (50.0%) |
| aspiration-vs-commitment | 0/4 (0.0%) | 2/4 (50.0%) |
| parse errors | 0 (wrong schema — 12/12 partial parses) | 0 |

## Confusion Matrix — Baseline

| Gold \ Pred | episodic | semantic | procedural | prospective | parse_error |
|---|---|---|---|---|---|
| episodic | 0 | 0 | 0 | 0 | 2 |
| semantic | 0 | 0 | 0 | 0 | 3 |
| procedural | 0 | 0 | 0 | 0 | 1 |
| prospective | 0 | 0 | 0 | 0 | 6 |

## Confusion Matrix — Fine-Tuned

| Gold \ Pred | episodic | semantic | procedural | prospective | parse_error |
|---|---|---|---|---|---|
| episodic | 2 | 0 | 0 | 0 | 0 |
| semantic | 0 | 3 | 0 | 0 | 0 |
| procedural | 0 | 0 | 1 | 0 | 0 |
| prospective | 1 | 2 | 0 | 3 | 0 |

## Illustrative Examples

### Cases where fine-tuning improved the prediction

**eval_001** — gold: `episodic`
> Paired with Alex on the authentication refactor today. We got the token refresh logic working cleanly after two frustrating false starts. Really good session — Alex caught a subtle edge case I'd misse...
- Baseline predicted: `partial`
- Fine-tuned predicted: `episodic`  ✓

**eval_002** — gold: `episodic`
> Got pulled into an unplanned two-hour strategy meeting that derailed my entire planned afternoon. Left feeling frustrated and behind on sprint work. Nobody had an agenda.
- Baseline predicted: `partial`
- Fine-tuned predicted: `episodic`  ✓

**eval_003** — gold: `semantic`
> I've noticed over the past year that I do my clearest thinking in the first 90 minutes after waking, before I check messages or news. Afternoons are better for routine tasks and shallow work.
- Baseline predicted: `partial`
- Fine-tuned predicted: `semantic`  ✓

### Cases where both were correct

**eval_007** — gold: `semantic`
> I'd love to get back into regular meditation at some point. Used to do it more consistently a few years ago. No idea when I'd actually restart — life feels too packed right now.
- Baseline: schema mismatch (partial)  |  Fine-tuned: `semantic` ✓

**eval_004** — gold: `procedural`
> Finally figured out the right pattern for handling concurrent map access in Go — acquire a write lock only when mutating, use RLock for reads, and never hold the lock across a goroutine boundary.
- Baseline: schema mismatch (partial)  |  Fine-tuned: `procedural` ✓

### Cases where fine-tuning did not fix the error

These are the three hardest prospective examples — all involve weak signals or mixed tiers.

**eval_008** — gold: `prospective`, predicted: `semantic`
> I've been saying I'll reach out to David for weeks now. Every time I open my messages I think about it and then close them. I should really just do it this week.
- The phrase "I should really just do it this week" is a weak prospective signal. The model weighted the pattern-of-deferral framing and classified as semantic.

**eval_009** — gold: `prospective`, predicted: `semantic`
> If I can wrap the current sprint by Thursday, I'm going to take Friday afternoon completely off — no laptop, just read and walk. I need it.
- Conditional commitment. The model classified as semantic (no-action-yet framing) rather than prospective (forward-oriented specific plan).

**eval_010** — gold: `prospective`, predicted: `episodic`
> Shipped the v2.1 release today after three weeks of work. Felt good to get it across the line. I've promised the customer success team I'll have the migration guide written and sent to them by next Monday.
- Mixed input: episodic event (shipping) + prospective commitment (migration guide). The model picked up the dominant episodic framing and missed the open commitment.

## Interpretation

Overall tier accuracy improved by 75.0 percentage points (0.0% → 75.0%) on this 12-example smoke eval. The fine-tuned model gained on: episodic, semantic, procedural, prospective. Aspiration-vs-commitment accuracy improved (0% → 50%), which is the hardest boundary in the Trace schema. This subset is only a few examples; treat as indicative. With only 12 eval examples, individual example outcomes dominate the percentages. A larger held-out set (50+ examples) would distinguish real learning from variance.

## Trace-Style Integration Examples

Three realistic Trace inputs showing baseline vs. fine-tuned predictions.

> **Baseline note:** The baseline model (no adapter) outputs valid JSON but uses its own invented
> schema — fields like `"memoryRecord"`, `"memory"`, `"extractedStatements"` — never the Trace
> `memory_tier` field. All 12 baseline examples parsed as `partial` (JSON present, schema wrong).
> This is the pre-SFT state: the model knows JSON but has no knowledge of the Trace schema.

**Input:** _Told Marcus I'd have the retrospective write-up done by Thursday. Need to block time tomorrow morning._

- Baseline: schema mismatch — output `{"memory": [{"actor": "Marcus", "action": "receive", ...}]}` (no `memory_tier`)
- Fine-tuned: `prospective` ✓
- Gold tier: `prospective`

**Input:** _I'd love to get back into meditation at some point. Life feels too packed right now to actually start._

- Baseline: schema mismatch — output `{"memory": [{"event": "feeling", "details": "Life feels too packed", ...}]}` (no `memory_tier`)
- Fine-tuned: `semantic` ✓
- Gold tier: `semantic`

**Input:** _Finally cracked the right pattern for async error handling in the service layer — bubble up domain errors, swallow infrastructure ones._

- Baseline: schema mismatch — output `{"memory_records": [{"type": "service", "action": "pattern_cracked", ...}]}` (no `memory_tier`)
- Fine-tuned: `procedural` ✓
- Gold tier: `procedural`

## Reproducibility

- Config: `configs/sft_trace_qwen25_3b_v2.yaml`
- Adapter: `outputs/sft_qwen25_3b_v2`
- Eval set: `data/eval_gold.jsonl` (12 examples)
- Seed: 42 (train/val split in prepare_data.py)
- See `eval/notes.md` for full training command and hyperparameters.
