# DPO Design Notes — Phase 1 Audit

Generated: 2026-04-27. Based on `data/sample_dpo.jsonl`, `data/eval_gold.jsonl`, `eval/results.md`.

---

## 1. Per-axis audit of the 3 existing preference pairs

### Pair 1 — over_flatten axis

**Input:** "I've been meaning to reach out to Claire about the mentorship opportunity. She mentioned it twice and I keep saying I'll follow up. I really should just send the message this week."

**Chosen:** `memory_tier: prospective`, `stated_intent: "Reach out to Claire about the mentorship opportunity this week."`
**Rejected:** `memory_tier: semantic`, `stated_intent: null`, collapsed to "general interest in mentorship and values maintaining professional relationships"

**Axis classification:** over_flatten — a real prospective signal ("I really should just send the message this week") is present, but the rejected output ignores it and frames the entry as a stable pattern of networking interest.

**Plausibility check:** ✓ Plausibly wrong. This is exactly the failure mode of eval_008 (David reach-out) — the model weights the deferral pattern and assigns semantic rather than reading "this week" as a commitment signal.

**Quality check:** ✓ Chosen is clean witness output. Rejected is structurally valid JSON, uses the right fields, but makes a single wrong classification + drops stated_intent.

---

### Pair 2 — add_coaching axis

**Input:** "Missed another workout this week. Third week in a row now. Keep telling myself it's because of the project crunch, but I know I'm also just avoiding it."

**Chosen:** `memory_tier: episodic`, no advice, `stated_intent: null`
**Rejected:** `memory_tier: episodic` (same tier!), adds "should consider scheduling exercise sessions in advance and addressing the underlying avoidance behavior. Blocking calendar time may help", and fabricates `stated_intent: "Restart workout routine"` (the user stated no such intent)

**Axis classification:** add_coaching — the tier is correct in both; the difference is the rejected output inserts prescriptive framing and manufactures an intent the user never stated.

**Plausibility check:** ✓ Plausibly wrong. The advice ("Blocking calendar time may help") is exactly the kind of language a base or lightly-tuned model produces when it "helps" with an emotionally loaded entry.

**Quality check:** ✓ Chosen is clean. Rejected correctly differs only on coaching/fabricated-intent, with no confounding tier noise.

---

### Pair 3 — mis_tier_mixed axis

**Input:** "Told Sarah I'd review her pull request before noon today. Just saw her Slack message asking if I'm still on it."

**Chosen:** `memory_tier: prospective`, `stated_intent: "Review Sarah's pull request before noon today."`
**Rejected:** `memory_tier: episodic`, `stated_intent: null`, drops the commitment entirely, frames as "Sarah sent a Slack message today. There was a prior conversation about a pull request review."

**Axis classification:** mis_tier_mixed — the input contains both a concrete commitment (review PR before noon) and an episodic event (Sarah's Slack follow-up). The rejected output picks the surface episodic frame (the Slack message) and loses the open commitment.

**Plausibility check:** ✓ Plausibly wrong. This is the same failure mode as eval_010 (shipped release + migration guide commitment). The model is known to pick the dominant framing and drop the commitment — this is a realistic failure, not an absurd one.

**Quality check:** ✓ Clean single-axis difference: tier + stated_intent. No confounding changes to topic_cluster or emotional_valence.

---

## 2. Per-axis count and format check

| Axis | Count | Prompt format | Both parse as JSON | Rejected structurally valid |
|---|---|---|---|---|
| over_flatten | 1 | System + "\n\nInput: " prefix | ✓ | ✓ |
| add_coaching | 1 | System + "\n\nInput: " prefix | ✓ | ✓ |
| mis_tier_mixed | 1 | System + "\n\nInput: " prefix | ✓ | ✓ |

All three prompts use the same format: system prompt concatenated with `\n\nInput: <raw text>`. This matches the DPO format MLX-LM expects (a single prompt string, not a messages list).

**No pair has an implausibly wrong rejected output.** All three rejected outputs are valid JSON with all required Trace fields; the errors are policy-level (wrong tier, coaching language, dropped intent), not structural.

---

## 3. Target distribution for the new generator

Based on the three failure axes confirmed in eval_gold.jsonl (eval_008/009/010), the three existing sample pairs (one per axis), and the requirement to preserve the 9/12 SFT capabilities:

| Family | Axis | Target count | Proportion | Source / rationale |
|---|---|---|---|---|
| A | over_flatten | 25 | 27.8% | eval_008, sample pair 1 |
| B | add_coaching | 25 | 27.8% | sample pair 2 (voice axis — drives the coaching-frequency gate) |
| C | mis_tier_mixed | 25 | 27.8% | eval_010, sample pair 3 |
| D | conditional_commitment | 8 | 8.9% | eval_009 (within prompt's 5–10 range) |
| E | schema_drift / tier_swap distractors | 7 | 7.8% | preserves 9/12 cases SFT already nails |
| **Total** | | **90** | 100% | |

**Notes on the total count:**
- The prompt specifies "Total ~80 pairs" but per-family targets (25/25/25/5–10/5–10) sum to 85–95. We chose 90 to respect every per-family target. The validator's ±10% rule is per-family, not on the total.
- The generator's `--n` arg is honored as the *total* count; family proportions scale: e.g. `--n 80` yields A:22, B:22, C:22, D:7, E:7 (each within ±10% of its per-family target at this n).
- For the 12-pair validation set (`--n 12 --seed 99`), proportional split: A:3, B:3, C:3, D:2, E:1.

---

## 4. System prompt note

The sample DPO pairs use a slightly shorter system prompt than the SFT training data:
- **SFT:** "...Do not add advice, coaching, prescriptive framing, or interpretation. Witness and structure; do not suggest or evaluate."
- **DPO sample:** "...Do not add advice, coaching, or prescriptive framing."

For the new generator, use the full SFT system prompt (more complete) for consistency with the model's trained behavior. The DPO pairs should share the same system prompt the model was trained on, so the prompt+chosen matches what the model learned to produce.

---

## 5. Phase 1.3 blocker — RESOLVED via mlx-lm-lora v2.1.0

**Resolution date:** 2026-04-27

- **Backend switch:** Replaced `mlx-lm` DPO (unavailable in v0.31.3) with `mlx-lm-lora` v2.1.0 (Goekdeniz-Guelmez, PyPI). CLI: `mlx_lm_lora.train --train-mode dpo`. Data format `{prompt, chosen, rejected}` is unchanged — no Phase 2 data rework was needed.
- **Adapter-stacking choice (fused SFT as reference):** SFT v2 LoRA is fused into the base model first (`scripts/fuse_sft.py` → `outputs/qwen25_3b_sft_fused/`). This fused model is used as both `--model` and `--reference-model-path` for DPO. KL is anchored at SFT, preserving the schema-alignment SFT taught; the DPO LoRA is a clean separable artifact on top.
- **Artifacts:** `scripts/train_dpo.py` (updated dispatcher), `scripts/fuse_sft.py` (new fuse wrapper), `configs/dpo_trace_qwen25_3b.yaml` (updated config with fused-model paths, `learning_rate: 5e-6`, `iters: 150`).

### Background (historical)

**mlx-lm version:** 0.31.3
**DPO status at the time:** NOT AVAILABLE

Neither `mlx_lm.tuner.dpo_trainer.DPOTrainer` (ImportError) nor a `--dpo` flag on `mlx_lm.lora` was present. Upgrade to latest mlx-lm was attempted but 0.31.3 was already current. A manual DPO loop in MLX was explicitly excluded.

---

## 6. Phase 2.4 audit — post-fix sample review (2026-04-27)

Reviewed `data/sample_dpo_v0.jsonl` (6 pairs: 2× over_flatten, 2× add_coaching, 2× mis_tier_mixed) after fixing two template glitches in Family B.

**Pairs reviewed:**

| case_id | axis | result |
|---|---|---|
| dpo_over_000 | over_flatten | ✓ Chosen prospective+intent, rejected semantic+null. Policy axis clean. |
| dpo_over_001 | over_flatten | ✓ Chosen prospective+intent, rejected semantic+null. Policy axis clean. |
| dpo_add__022 | add_coaching | ✓ Tier same (episodic); rejected adds coaching + fabricated "Restart morning routine." intent. Fix confirmed: no duplicate word. |
| dpo_add__023 | add_coaching | ✓ Tier same (semantic); rejected adds coaching + fabricated "Address my pattern of under-charging for my work through deliberate practice." intent. Fix confirmed: grammatical noun form. |
| dpo_mis__044 | mis_tier_mixed | ✓ Chosen prospective+intent, rejected episodic+null. topic_cluster differs (expected on this axis). |
| dpo_mis__045 | mis_tier_mixed | ✓ Chosen prospective+intent, rejected episodic+null. Policy axis clean. |

**Verdict:** All 6 pairs pass. Rejected outputs are plausibly wrong. Chosen outputs have no accidental coaching. The two fixed add_coaching cases now differ only on the intended policy axis (coaching+fabricated intent vs faithful witness), with no grammatical artefacts introducing a spurious gradient.
