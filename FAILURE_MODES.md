# Failure modes — iris-ft-lab

Consolidated catalog of the ML and tooling failures hit while building the Trace SFT/DPO and collab-eval SFT/DPO pipelines. One paragraph per mode: symptom, root cause, fix, where the evidence lives. The failures here are real — every entry was a session blocker that was diagnosed, fixed, and committed, not a hypothetical.

This doc is mostly for the next person (or future you) walking into a similar bug. Read it before iterating on either pipeline. Start with the **"Three dimensions: Knowing, Doing, Deciding"** section below — it maps the 15 modes into three failure classes with different remedies, so you can orient quickly before reading individual entries.

---

## Three dimensions: Knowing, Doing, Deciding

The 15 modes below fall into three distinct failure classes. Recognizing the class before debugging is important: the remedies are structurally different.

**Knowing** — The model's learned representation is wrong or corrupted. The problem is in what the model ended up knowing after training, or in the data and infrastructure that shaped it. Fix: find the corruption source (data, recipe, or weight-level defect) and retrain from a clean state.

**Doing** — The execution layer failed. The problem is in the tooling, harness, or framework that surrounds training. The model itself is fine; the run was misconfigured, timed out, or used an API incorrectly. Fix: find the tooling constraint, fix the scaffold, re-run. No gradient information is lost.

**Deciding** — The policy layer failed. Either the trained model is making the wrong generation-time decision (inference-time policy failure), or the practitioner made the wrong instrument selection decision (meta-level policy failure). This class is the hardest: the training ran correctly, the tooling worked, but the choice of what to train or how to measure it was wrong. Fix: change the instrument or supervision signal — more of the same will not help.

---

### K — Knowing failures (Modes 1, 2, 3, 5, 7, 11)

| Sub-class | Description | Modes |
|---|---|---|
| K1 — Recipe / data quality | Hyperparameters, data imbalance, or template errors corrupt what the model learns | 1, 5, 7 |
| K2 — Infrastructure corrupts trained weights | The run completes cleanly; the artifact is silently wrong | 2, 3 |
| K3 — Signal ceiling | Data and recipe are correct, but the training signal cannot carry the required information | 11 |

**Canonical K2 example — Mode 2 (fusion defect):** `mlx_lm fuse` without `--dequantize` completed without error, produced a same-size output, and logged nothing anomalous. The fused weights were byte-equivalent to the original 4-bit base. The defect was invisible until the downstream eval scored 0/12 instead of expected 9/12. Lesson: verify the fused artifact size, not just the exit code.

**How eval tightened against K:** The Mode 2 experience added the hard-stop rule — baseline eval must score within ±1 of expected before any adapter eval runs. K3 (Mode 11) added the signal-ceiling check: if doubling data doesn't move the metric, stop iterating on data quantity and diagnose the gradient.

**Training-process impact:** K failures are often invisible at training time (the loss curve looks normal). The promotion gate is the first place they surface. This is why a well-calibrated gate matters more than clean training logs.

---

### D — Doing failures (Modes 9, 10, 12)

| Sub-class | Description | Modes |
|---|---|---|
| D1 — Harness / tooling constraints underestimated | The run infrastructure has limits that weren't accounted for | 9, 10 |
| D2 — Framework API misuse | A framework config field does something other than expected | 12 |

**Canonical D example — Mode 9 (Bash timeout):** The Bash tool has a 10-minute hard timeout; a 23-minute training run must be backgrounded and polled. The first collab-eval v2 training run was issued as a foreground command and silently killed mid-training. Lesson: any training run > 5 minutes must use `run_in_background` + until-loop polling.

**Pattern:** D failures are almost always fixable without retraining — they're about the scaffold, not the model. If the training run was killed, re-run it. If the config field did the wrong thing, fix the config.

---

### Dc — Deciding failures (Modes 4, 6, 8, 13, 14, 15)

| Sub-class | Description | Modes |
|---|---|---|
| Dc1 — Gate / promotion design | The success criterion was wrong before training started | 4 |
| Dc2 — Instrument selection | The chosen training method cannot, in principle, learn the target behavior | 6, 8, 13, 14 |
| Dc3 — Cumulative exhaustion | Multiple instruments have been tried; the trajectory itself is the signal | 15 |

**Canonical Dc3 example — Mode 15 (both instruments exhausted):** Three SFT configs (recipe, quantity, curriculum) and one DPO run all produced 0.20–0.25 stress `data_preservation`. Each individual run could have been attributed to a specific hyperparameter. The trajectory across all five runs cannot — it points at a structural property of the supervision signal, not a tunable parameter. Lesson: when two instrument families have been exhausted on the same target, the problem is in the instrument class, not the instance.

**How eval tightened against Dc:** The stress eval (gate condition 5) was added specifically to catch Dc2 failures — an adapter that looks fine on regular eval but fails on the target behavior. Without gate condition 5, SFT v1 (composite +0.039, unit_consistency +0.138) would have been promoted despite 0.25 stress preservation.

**The 6/3/6 split observation:** Knowing and Deciding have 6 modes each; Doing has only 3. This reflects where the real cost lies in an ML project: failures of representation and instrument selection compound across sessions; tooling failures are usually one-session blockers. The Deciding class is also harder to detect — a D failure usually announces itself (training crashed, timeout error), while a Dc failure only becomes visible when the eval results are in.

---

### Pre-flight checklist

Three questions to ask before starting a new training run:

- **K:** Is the artifact I'm training from clean? (Verify fused model size; check training data for template errors; confirm the recipe hasn't over-parameterized for dataset size.)
- **D:** Is the scaffold set up correctly? (Background command? Polling loop? Framework API used as documented?)
- **Dc:** Is this the right instrument for the target behavior? (What does the supervision signal reward? Is there a ceiling I'm about to hit? Has a previous run already hit it?)

---

## Index

| # | Mode | Pipeline | Where it bit |
|---|---|---|---|
| 1 | Mode collapse during SFT | collab-eval | v0 training |
| 2 | Fusion defect on quantized base | trace | DPO v0 training |
| 3 | Stale weights glob conflict | trace | DPO v1 startup |
| 4 | Unreachable promotion gate | collab-eval | v0 gate design |
| 5 | Training-distribution asymmetry | collab-eval | v0 + v1 data |
| 6 | DPO backend not in mlx-lm | trace | DPO Phase 1.3 |
| 7 | Template-substitution glitches | trace | DPO Phase 2 |
| 8 | Missing temp config in prompt | trace | DPO recovery Phase 4.1 |
| 9 | Bash 10-min timeout < training duration | tooling | collab-eval v2 training |
| 10 | Wall-time estimate ignored seq length | estimation | collab-eval v2 training |
| 11 | Quantity rebalance failed where signal/gradient was bottleneck | collab-eval | v2 |
| 12 | mlx-lm `adapter_path` is save-only, not load-on-resume | tooling | collab-eval v3 phase 2 |
| 13 | SFT instrument exhausted for absence-of-action signals | collab-eval | v1+v2+v3 trajectory |
| 14 | DPO discriminator collapse without generation-time policy change | collab-eval | DPO v0 |
| 15 | Both SFT and DPO instruments exhausted on absence-of-action signal — switch to RL | collab-eval | DPO v0 cumulative |

Below: cross-cutting patterns that fall out of these.

---

## 1. Mode collapse during SFT — collab-eval v0

**Symptom.** Adapter produced numerical convergence to repeated digits (`2444,1444`, `4444,4444`), year drift (`2024 → 2044 → 4444`), and literal token loops in some cases. Outputs *looked* like CSV (passing format_validity=1.0 and unit_consistency=1.0) but contained no real data. 88% of "missing" rows in RH-like cases were never produced — the model degenerated after the first few rows of generation.

**Root cause.** Hyperparameter excess on a small dataset: rank=8, alpha=20 (effective scale 2.5), zero dropout, 1000 iters over 240 cases ≈ 16 epochs. The adapter had too many degrees of freedom and no regularization, drove itself into a degenerate region.

**Fix.** Gentler recipe in v1: rank=4, scale=8.0 (= 2.0), dropout=0.05, iters=200, lr=5e-5 (was 1e-4), cosine schedule with warmup. Mode collapse went away.

**Where.** `collab-eval/docs/sft_v1_failure_audit.md` "Real failure mode: adapter mode collapse"; `collab-eval/configs/sft_collab_eval_qwen25_3b.yaml` (the v1 recipe).

---

## 2. Fusion defect on quantized base — trace DPO v0

**Symptom.** Fused-SFT baseline scored 0/12 instead of expected 9/12 — identical to the pre-SFT base behavior. The DPO adapter trained against this defective fused model produced clean training metrics (loss 0.693 → 0.001, val 0.002, margin 7.8) but was unusable.

**Root cause.** `mlx_lm fuse` was called without `--dequantize` on a 4-bit quantized base. MLX cannot represent a float LoRA delta inside a packed 4-bit integer tensor; the fuse command ran to completion but the merged weights were byte-equivalent to the original quantized base (file size delta of 194 bytes — within tar/safetensors header rounding, vs the 26 MB adapter that should have been merged).

**Fix.** Add `--dequantize` to the fuse command. The fused output is now ~6 GB float16 (vs the broken 1.6 GB 4-bit "fusion"). The §4.1 hard-stop rule (baseline ≠ 9/12 ± 1 → stop) caught this cleanly — without that floor, the eval would have run on a base-equivalent model, the DPO adapter would have looked broken for "DPO reasons," and we'd have iterated on hyperparameters chasing a phantom bug.

**Where.** `eval/dpo_v0_results.md` "Fusion Defect — Root Cause Analysis"; `scripts/fuse_sft.py` (the `--dequantize` flag).

---

## 3. Stale `model.safetensors` glob conflict — trace DPO v1 startup

**Symptom.** DPO training failed to start with a 506-parameter mismatch error.

**Root cause.** `mlx_lm`'s weight discovery globs `model*.safetensors`. After re-fusing with `--dequantize`, the directory contained both the old 4-bit `model.safetensors` (from the defective v0 fuse) and new float16 shards (`model-00001-of-XXX.safetensors`, etc.). All three loaded simultaneously, producing conflicting quantization keys.

**Fix.** Rename the stale 4-bit weight before training: `mv model.safetensors model.safetensors.defective_v0_bak`.

**Where.** `eval/dpo_v1_training_notes.md` "Pre-training fix" section. This is the kind of bug that's invisible without the right log line; capture it because it'll bite again the first time someone re-fuses without cleaning up.

---

## 4. Unreachable promotion gate — collab-eval v0

**Symptom.** Gate condition "composite improves by ≥ 0.10" was mathematically unreachable from a base composite of 0.9569 (max possible improvement = 0.0431). Even a perfect adapter would fail the gate. Discovered post-hoc when v0 was already trained.

**Root cause.** Gate threshold designed in the abstract without checking against base-model evidence. The "+0.10" felt like a strong improvement target; nobody verified the ceiling.

**Fix.** Replaced with a 5-condition no-regression-style gate (composite ≥ base − 0.005, dim no-regression, RH-like no-increase, one dim improves ≥ 0.02, stress data_preservation ≥ 0.85). The structural rule that fell out: **calibrate gates against base-model evidence before training.** If your base scores X, your gate's positive thresholds must be reachable from X without assuming a perfect ceiling.

**Where.** `SFT_ANALYSIS.md` §2.2(2); `collab-eval/results/collab_sft_v0.md` "Note on the +0.10 gate"; `collab-eval/configs/sft_collab_eval_qwen25_3b.yaml` (the revised gate).

---

## 5. Training-distribution asymmetry — collab-eval v0 + v1

**Symptom.** Even after the v1 mode-collapse fix, the adapter regressed on `data_preservation` (0.975 → 0.7500 in v0) and held flat (0.9750 → 0.9875) in v1, and the v1 stress-eval `data_preservation` came in at 0.25 vs the 0.85 gate.

**Root cause.** The training data only ever rewarded row deletion or Notes stripping. Audit (`docs/sft_v1_data_audit.md`) counted 147 deletion events across 240 v1 training cases and **0 preserve-and-convert demonstrations**. Whenever the input contained a noisy row (blank, duplicate, annotation, fabricated, `; orig $X.XXXM` in Notes), the gold deleted or stripped. The model learned the dominant signal: "drop suspicious content." Not surprisingly, it generalized that to ambiguous unit-normalization rows in the held-out eval.

**Fix.** Rebalance the training distribution by adding more preservation-stress cases (long tables where every row is real data and gold preserves all). v1 had 25% stress (80/320); v2 has 50% stress (240/480), dropping the deletion-event ratio from 61.2% to 30.6%. The v2 outcome will tell us whether 50% is enough.

**Lesson.** **Audit your gold output before training.** What does the gold consistently teach? In tasks with noise/cleanup tradeoffs, "gold strips X" and "gold preserves X" are different policies — and the gold's distribution of those decisions is what the model learns, regardless of what the prompt asks for.

**Where.** `collab-eval/docs/sft_v1_data_audit.md`, `sft_v1_failure_audit.md`, `sft_v2_data_plan.md`, `sft_v2_data_audit.md`. Audit script: `collab-eval/docs/audits/audit_gold_v1.py` (and v2 variant).

---

## 6. DPO backend not in mlx-lm — trace DPO Phase 1.3

**Symptom.** `python scripts/train_dpo.py --check-only` reported `DPO trainer not found in mlx-lm 0.31.3`. Neither `mlx_lm.tuner.dpo_trainer.DPOTrainer` (ImportError) nor a `--dpo` flag on `mlx_lm.lora` was present. Hard stop at Phase 1.3.

**Root cause.** Official `mlx-lm` (v0.31.3, current at the time) does not expose DPO training. The original DPO scaffold was written assuming MLX would land DPO support; it didn't.

**Fix.** Switch the DPO backend to **`mlx-lm-lora`** v2.1.0 (third-party, by Goekdeniz-Guelmez, on PyPI). CLI: `mlx_lm_lora.train --train-mode dpo`. Data format `{"prompt", "chosen", "rejected"}` matches what the existing trace generator already produces — no Phase 2 data rework was needed. Documented in `data/dpo_design_notes.md` §5.

**Lesson.** Don't assume framework-level features will land on your timeline. When a feature is "evolving" or "expected soon," a community fork that already has it is often the right unblock.

**Where.** `data/dpo_design_notes.md` §5; `scripts/train_dpo.py` (updated dispatcher); `configs/dpo_trace_qwen25_3b.yaml` (mlx-lm-lora-shaped config).

---

## 7. Template-substitution glitches in synthetic data — trace DPO Phase 2

**Symptom.** Two records out of 80 in the DPO preference set had ungrammatical `stated_intent` strings:
- `dpo_add__022`: `"Restart morning routine routine."` (duplicated word)
- `dpo_add__023`: `"Stop under-charge for my work through deliberate practice."` (ungrammatical compound)

Both were *plausibly wrong* in the policy sense (advice-fabricating rejected outputs) but the grammatical artifacts gave DPO a "prefer grammatical" gradient on those two cases that wasn't the policy axis we wanted to teach.

**Root cause.** Generator templates combined lexicon entries and format strings without checking for grammatical agreement. The lexicon entry "morning routine" plugged into `f"Restart {activity} routine."` yielded the duplicated word. The verb "under-charge" plugged into `f"Stop {behavior} for my work..."` yielded ungrammatical English.

**Fix.** Lexicon-level adjustment: change `"morning routine"` to `"morning"` and rework the second template to use a noun form. Both were 1–3 line changes; the rest of Phase 2's 78/80 records were already clean.

**Lesson.** When generating synthetic preference data, **scan the rejected outputs for surface artifacts** before training. Bad pairs aren't just "wrong policy" — grammar/length/punctuation differences become free training signals that the policy axis didn't intend.

**Where.** `scripts/generate_synthetic_dpo.py` (post-fix); commit `7bc6676`.

---

## 8. Missing temp config in prompt — trace DPO recovery Phase 4.1

**Symptom.** During the DPO v1 recovery run, `eval_extraction.py` failed with `FileNotFoundError: configs/eval_baseline_fused.yaml`. The user had executed Phase 1–3 manually, then ran the §4.1 baseline eval verbatim from the prompt and hit the missing file.

**Root cause.** The prompt described the temp config's contents in a fenced YAML block but didn't include an explicit "create this file" step. The previous Sonnet session that ran §3 may have created it implicitly; when the user re-ran §4.1 manually, the file didn't exist.

**Fix.** Use `configs/dpo_trace_qwen25_3b.yaml` directly. `eval_extraction.py` only reads `config["model"]["path"]`, and the DPO config already has the right value. No temp config is needed at all.

**Lesson.** **Prompts should not assume side effects from earlier prompt sections.** If a step depends on a file existing, either the prompt creates it explicitly or the prompt uses an existing file. "Create this YAML in your head and then run this command" is brittle; an executor reading the prompt linearly may skip the implicit step.

**Where.** `TRACE_DPO_V1_RECOVERY_PROMPT.md` §"Background" notes the immediate operator error; the recovery used the existing config directly.

---

## 9. Bash 10-min timeout < training duration — collab-eval v2 training

**Symptom.** During SFT v2 (300 iters on M-series Mac), the Bash command running training hit Claude Code's 10-minute Bash ceiling. On the first attempt the harness returned, Sonnet interpreted the return as "training completed" and started running eval on a partial adapter checkpoint. On the recovery attempt Sonnet tried `sleep 90 && tail -30 …` to wait between checks; the harness blocked the chained sleep with a specific guidance error.

**Root cause.** Two-layer mismatch: (a) Bash tool's max timeout is 10 minutes by harness design; (b) the training command's actual runtime exceeded it. The training process either continued running orphaned in the background, or was killed; either way the harness lost visibility.

**Fix (the pattern that works).** Two harness affordances combine cleanly:
1. Run training in the background: `Bash(..., run_in_background: true)`. Returns immediately; the process runs to completion in its own context, no tool-call timeout applies.
2. Poll for milestones with `until` loops: `until grep -q "Iter 100:" <logfile>; do sleep 10; done`, also run with `run_in_background: true`. The harness fires a notification when the until-loop exits, so the assistant wakes up at each milestone without burning context on idle waiting.

What does NOT work: `sleep N && <command>` chains in foreground. The harness explicitly blocks long leading sleeps with a guidance error pointing at exactly the pattern above. Don't fight it.

**Lesson.** **Tool-call timeouts are a recipe parameter.** Any training run that can take > 10 min must use `run_in_background`, and milestone checks must use until-loops, not chained sleeps. Document expected wall time in the prompt — v2 was claimed at "~10–15 min on M-series"; actual was ~75 min (see Mode 10).

**Where.** This conversation; collab-eval v2 training run logs; pattern is in Sonnet's recovery polling.

---

## 10. Wall-time estimate ignored seq length — collab-eval v2

**Symptom.** Phase 3-4 prompt estimated SFT v2 wall time at "~10–15 min on M-series Mac." Actual run was ~75 min — 5× the estimate. Per-iter time was ~15 sec/iter, vs v1's ~2.4 sec/iter on the same hardware and recipe.

**Root cause.** Wall-time estimate scaled linearly with iter count and dataset size, but ignored sequence length. v2's training data is 50% stress cases (long tables, 15–25 rows), pushing the average tokens-per-example near `max_seq_length: 2048`. v1's regular cases averaged 6–18 rows = 500–800 tokens — a quarter of the context. With everything else equal, per-iter compute scales roughly with seq length, and v2's avg seq length is ~3-4× v1's.

**Fix.** When estimating training wall time, scale by `iters × avg_tokens_per_example`, not just `iters`. Or measure: run a 10-iter smoke first, divide by 10, multiply by total iters and add 20% buffer. The 5-iter smoke run already in the v2 prompt would have surfaced this if the prompt had said "use the smoke run wall time × 60 to estimate the full run."

**Lesson.** **Per-iter time is dominated by per-example seq length.** When training data composition shifts toward longer examples (stress cases, longer documents, full-context examples), redo the wall-time math. Don't extrapolate from a previous run that used shorter examples.

**Where.** This conversation; v2 prompt's "~10–15 min" claim was wrong; actual ~75 min for 300 iters at ~1500–2000 tokens/example.

---

## 11. Quantity rebalance failed where signal/gradient structure was the bottleneck — collab-eval v2

**Symptom.** v1 had 25% stress training cases and stress `data_preservation` came in at
0.25 (vs the 0.85 gate). v2 tripled the stress count (80 → 240), reaching 50% stress
proportion, with the stated hypothesis that more preservation demonstrations would lift
the score. v2 stress `data_preservation` was identical to v1: 0.25. Zero movement.

**Root cause.** The v2 hypothesis ("quantity is the bottleneck") was wrong. The flat
trajectory across two experiments rules out quantity as the dominant cause. The model
learned the unit-conversion signal from stress data perfectly (stress unit_consistency
went 0.70 → 1.00 in both v1 and v2), so the stress demonstrations *are* being absorbed —
but only the gradient-friendly signals. Two more plausible hypotheses now: (H2) gradient
competition between stress's "preserve" and regular's "delete-on-noise" gradients within
mixed-batch steps; (H3) signal asymmetry — "don't drop a row" is a harder positive
demonstration than "convert X to Y" because the action space includes all possible
deletions and the gold doesn't show every preservation decision explicitly.

**Fix.** Don't iterate on quantity. v3 will test curriculum learning (separates the
gradient phases) as a cheap H2 test. If curriculum also produces no movement, H3 is
confirmed and the next instrument is DPO-on-preservation pairs, not more SFT data.

**Lesson.** **When a quantity rebalance produces zero movement on a target metric while
co-resident metrics improve, the bottleneck is structural, not quantitative.** Don't run
v3 with the same instrument as v2; switch instruments. The "zero movement" is the
informative signal — partial movement would have justified more rebalance, but flat
movement says "this knob doesn't control this dimension."

**Where.** `collab-eval/results/collab_sft_v2.md` "Hypothesis test" + "Likely next stage"
sections; `collab-eval/results/collab_sft_v1.md` for the v1 baseline this is compared
against.

---

## 12. `adapter_path` is save-only — to resume, use `resume_adapter_file` — collab-eval v3 phase 2

**Symptom.** v3 curriculum required phase 2 to resume from phase 1's final adapter
checkpoint (the whole point of the curriculum is for phase 2 to refine phase 1's
preservation gains, not start fresh). The phase 2 config set `adapter_path:
adapters/sft_collab_eval_qwen25_3b_v3/` with the phase-1 final adapter copied into that
directory; the expectation was that mlx-lm would load existing weights from there as
training-init. It didn't. Phase 2's iter-1 train loss was ~1.0 — same as a fresh start
on mixed data. Phase 1 weights were not loaded.

**Root cause.** mlx-lm's `adapter_path` config field is the *output* path for new
checkpoints; it does not control loading. To resume from existing weights, you must set
the separate `resume_adapter_file` field pointing at a specific safetensors file. The
two fields are independent: `adapter_path` says "write here", `resume_adapter_file` says
"start from here". Setting only `adapter_path` to a directory containing existing
checkpoints does not auto-load them.

**Fix.** Add `resume_adapter_file: <path-to-final-safetensors>` to the phase 2 config.
Verify the resume worked by checking phase 2's iter-1 val loss: should be substantially
below a fresh-start val loss for the same data (in v3, phase 2 iter-1 val was 0.388 with
resume, vs 1.109 for fresh start — clear confirmation).

**Lesson.** **The save vs load distinction in mlx-lm is silent until phase 2 starts from
random init.** A two-phase curriculum that fails to resume is effectively a single longer
phase on the second-phase data — different experiment, easy to mistake for "curriculum
didn't help" when actually the curriculum never ran. Always verify resume by inspecting
iter-1 train/val loss against the fresh-start baseline. The diagnostic in v3's phase 2
training notes is what caught this.

**Where.** `collab-eval/results/collab_sft_v3_phase2_train_notes.md` "Bug discovered
during run" section; `collab-eval/configs/sft_collab_eval_qwen25_3b_v3_phase2.yaml` (the
fixed config with `resume_adapter_file`).

---

## 13. SFT instrument exhausted for absence-of-action signals — collab-eval v1+v2+v3

**Symptom.** Across three SFT experiments — v1 (gentler recipe + 25% stress data), v2
(50% stress data, same recipe), v3 (curriculum: stress-only phase 1, then mixed phase 2
at lower LR) — stress `data_preservation` was 0.25 in every case. Three distinct
interventions touching three different parametric axes (recipe, quantity, ordering),
identical outcome. Co-located metrics on the same eval moved cleanly in every run
(stress `unit_consistency` 0.70 → 1.00 in all three).

**Root cause.** Preservation is an **absence-of-action policy**: "don't drop this row."
SFT learns from positive demonstrations (gold outputs); it has no mechanism to teach
"don't drop this row" except through what the gold does. When the gold contains a
preserved row, the model learns the surface form (this row appears in gold) but not the
policy decision (whether to drop on uncertainty). Positive signals like unit conversion
("convert this $M cell to $K") and header preservation ("output these column names")
transfer cleanly because they ARE positive demonstrations — the gold token sequence
literally exhibits the desired action. Preservation has no equivalent literal signal;
the desired action is the absence of a deletion the model would otherwise produce.

**Fix.** Not an SFT fix. The right instrument for absence-of-action policies is
preference learning (DPO/ORPO/KTO), where the discriminative signal "this output is
better than this other output" can directly express "preserve > drop" without requiring
the gold to demonstrate every preservation decision. The DPO loss compares two outputs
on the same input — exactly the comparison preservation requires.

**Lesson.** **A flat metric across multiple varied interventions = wrong instrument.**
When v1, v2, and v3 all produced 0.25 stress `data_preservation` despite varying recipe,
quantity, and ordering, the bottleneck is structural (instrument-fit), not parametric
(recipe). Three flat data points across three different interventions = SFT not the
right tool. Switch instruments before iterating further. This is the most efficient
possible negative result: each iteration gathered evidence that ruled out one hypothesis,
and the cumulative trajectory pointed cleanly at the next instrument.

**Where.** `collab-eval/results/collab_sft_v1.md`, `collab_sft_v2.md`, `collab_sft_v3.md`
(the three flat data points); `collab-eval/README.md` §4–§7 (the cumulative narrative).

---

## 14. DPO discriminator collapse without generation-time policy change — collab-eval DPO v0

**Symptom.** DPO v0 trained on 80 preference pairs (chosen = gold preserves all rows;
rejected = gold with rows dropped at random fraction [0.3, 0.7]). Training loss
collapsed to ~0.001 by iter 30 with train accuracy 1.0 from iter 30 onward; final
reward margin 7.7. By every training metric the run was a textbook success. But on
the held-out stress eval, `data_preservation` came in at 0.20 — actually below the v3
SFT baseline of 0.25, matching the no-adapter base model. DPO also regressed
`unit_consistency` (−0.025) and composite (−0.006) on regular eval.

**Root cause.** Two compounding effects:

1. **Discrimination ≠ generation.** DPO's loss compares two fixed outputs and updates
   the policy to assign higher log-probability to chosen than rejected. When the
   discrimination is trivial (CSV with N rows vs CSV with N−k rows differs in obvious
   token-count features), the model learns it as a classification task very fast. But
   classification over fixed outputs does not rewire the per-step generation decision
   "should I emit another row or stop?" — that decision happens during decoding, not
   during the comparison. The reward signal never reaches the layer where the
   generation choice is made.

2. **β=0.1 too low for crisp signals.** The DPO KL term penalizes divergence from the
   reference policy. With a crisp binary signal that's trivially learnable, β=0.1 lets
   the policy drift aggressively from the v3 SFT reference. The unit_consistency
   regression is the symptom: v3's `$M→$K` conversion knowledge was partially
   overwritten as the policy drifted toward maximizing the (already-saturated) chosen-
   vs-rejected log-ratio.

The reference model loaded correctly (iter-1 val loss = 0.693 = −log(0.5), ruling out
Mode 2). This is purely a policy-side failure: DPO learned the wrong target.

**Fix.** Not a DPO-tuning fix. The structural problem is that preference learning over
fixed outputs is the wrong instrument for a generation-time policy decision. Higher β
would slow the drift but not solve the underlying mismatch — the reward signal still
wouldn't reach the generation decision. The right instrument is RL with the grader as
reward at rollout time (Mode 15). For DPO experiments where this pattern is suspected,
diagnostic: if train loss collapses to <0.01 within the first 20% of iters AND a co-
located dimension regresses on held-out eval, the discriminator-not-generator pattern
is likely.

**Lesson.** **Preference learning over pre-computed pairs is for behaviors expressible
as token-sequence preferences.** When the target behavior is a generation-time
boundary decision (when to stop, how long to be), DPO's reward signal cannot reach the
right layer. Use RL with online rollouts instead. A DPO loss that collapses below
0.01 within the first 30 iters on a binary signal is a warning sign — not a sign of
fast convergence.

**Where.** `collab-eval/results/collab_dpo_v0.md`; `collab-eval/results/collab_dpo_v0_training_notes.md` "Loss curve notes" section; `collab-eval/docs/preservation_analysis.md` §4.

---

## 15. Both SFT and DPO instruments exhausted on absence-of-action signal — switch to RL — collab-eval DPO v0 cumulative

**Symptom.** Stress `data_preservation` across four interventions:

| Run | Instrument | Stress preservation |
|---|---|---|
| Base | (none) | 0.20 |
| SFT v1 | gentle recipe + 25% stress training | 0.25 |
| SFT v2 | + 50% stress training | 0.25 |
| SFT v3 | + curriculum (stress phase 1, mixed phase 2) | 0.25 |
| DPO v0 | preference pairs (preserve > drop) | 0.20 |

Five data points, two instrument families (SFT positive demonstrations × three
configurations; DPO preference learning × one configuration). The metric refuses to
move above 0.25. Co-located metrics on the same eval move cleanly (`unit_consistency`
0.70 → 1.00 on stress under all SFT runs).

**Root cause.** All four interventions share a structural flaw: they provide
supervision on **token sequences at training time**, not feedback on **generation
behavior at rollout time**. The grader's `row_count_preserved` is a generation-time
property — only knowable after the policy has decided when to stop emitting rows.
Token-level supervision (SFT) and pair-level supervision (DPO) both miss this. See
`collab-eval/docs/preservation_analysis.md` for the five compounding reasons in detail.

**Fix.** Switch to RL with the grader as reward (PPO or GRPO via mlx-lm-lora's
`--train-mode grpo`). Reward = composite_score with `data_preservation` weighted
heavily, plus an explicit RH-like penalty for row duplication (the harness already
flags `format_validity ≥ 0.9 AND data_preservation < 0.5` as RH-like; that condition
becomes a reward penalty term). KL-anchored at v3 SFT to preserve regular-eval gains.

**Lesson.** **When two distinct instrument families produce flat results across
multiple configurations of each, the bottleneck is the supervision signal's
relationship to the target behavior, not the instrument's tuning.** SFT and DPO both
provide pre-computed-output supervision; the target here is a generation-time policy.
Mismatch is structural — switching instrument families inside the "pre-computed
output" class (SFT → DPO) was the obvious move and we tried it; the next move has
to leave that class entirely. RL with rollout-time grader feedback is the next class.

**Where.** `collab-eval/results/collab_sft_v1.md`, `collab_sft_v2.md`, `collab_sft_v3.md`,
`collab_dpo_v0.md`; `collab-eval/docs/preservation_analysis.md` (definitive analysis);
`FAILURE_MODES.md` Mode 13 (the SFT-side observation that v0 confirmed and extended).

---

## Cross-cutting patterns

A few rules of thumb that fall out of the above modes. Worth applying as a pre-flight checklist before any new SFT/DPO run.

**Audit data before training.** What does the gold consistently teach? Run a small script that classifies what the gold *does* on each input pattern (deletes, preserves, converts, etc.). If the gold's policy distribution is one-sided in a way the eval doesn't fully test, your model will overfit that policy and the eval won't catch it. Modes 1 and 5 both stem from skipping this.

**Calibrate gates against base-model evidence.** Before declaring promotion thresholds, run baseline eval and compute "what's the maximum improvement reachable from here?" If your gate exceeds that, the gate is unreachable on its face. Mode 4.

**Beware silent transformations on quantized weights.** `mlx_lm fuse` on 4-bit base is a no-op. By analogy, watch for any operation that mixes precision tiers (4-bit + float16, fp8 activations + fp16 weights) and confirm the output via spot-check eval, not just file-size or no-error-thrown signals. Modes 2 and 3.

**A tool-call timeout is a recipe parameter.** If your training command can take longer than the harness's tool timeout, run it in the background and poll for milestones with `until` loops. Don't chain sleeps; the harness blocks them. Mode 9.

**Wall-time estimates must scale with seq length, not just iters.** When training data shifts toward longer examples (long tables, full-context documents), per-iter compute grows roughly linearly with avg tokens-per-example. Either compute `iters × avg_tokens` or measure via a smoke run; don't extrapolate from a previous run with shorter examples. Mode 10.

**Adapter-stacking strategy matters as much as hyperparameters.** Strategy 1 (fuse SFT, fresh LoRA on top, both `--model` and `--reference-model-path` = fused) vs Strategy 2 (resume LoRA on top of base, `--reference-model-path` = base) have completely different KL-anchoring behavior. Strategy 2 with `reference=base` actively undoes SFT's gains. Pick Strategy 1 unless you have specific evidence the other works in your tooling. See trace DPO `dpo_design_notes.md` §5 and Strategy 1/2 discussion.

**Synthetic preference data needs surface-level QA, not just policy-level QA.** Validators check JSON parse + schema; they don't check grammar, repeated words, or coincident formatting differences between chosen and rejected. Add a pass that scans rejected outputs for cosmetic artifacts. Mode 7.

**`adapter_path` in mlx-lm is save-only — to resume, set `resume_adapter_file` explicitly.**
The save vs load distinction is invisible until phase 2 starts from random init. Always
verify resume by inspecting iter-1 train/val loss against the fresh-start baseline. Mode 12.

**A flat metric across multiple varied interventions = wrong instrument.** When three
SFT iterations vary recipe, quantity, and ordering and produce identical results on a
target dimension, the bottleneck is structural not parametric. Three flat data points
is your signal to switch instruments (SFT → DPO/ORPO/PPO), not to iterate on the same
one with finer granularity. Mode 13.

**A DPO loss that collapses below 0.01 in the first 20% of iters is a warning sign,
not a sign of fast convergence.** When the chosen/rejected discrimination is trivial
(simple structural feature like row count), DPO learns it as classification without
transferring to generation behavior. Diagnose by checking a held-out eval immediately
after collapse — if a co-located dimension regressed, the discriminator-not-generator
pattern is likely. Mode 14.

**When two instrument families both produce flat results, the supervision-signal
class is wrong.** Don't iterate further within the same class — switch classes.
Pre-computed-output supervision (SFT positive demos, DPO preference pairs) and
rollout-time supervision (RL with reward) are different classes; if both pre-computed
attempts fail, the next move is rollout-time. Mode 15.

**Bucket A/B/C verdicts beat binary PROMOTED/NOT-PROMOTED.** Three-bucket classification ("PROMOTED" / "NOT PROMOTED with progress" / "NOT PROMOTED with concern") lets you document forward motion without shipping unsuitable adapters. This is the antidote to gate-creep — when v2 doesn't quite hit the gate but lifts the failing dimension by 0.4, you can record that as Bucket B and plan v3 with evidence, instead of being tempted to lower the gate. (Used in collab-eval v2 prompt; not yet a failure but a documented antipattern avoided.)

---

## Known gaps (intentional, not failures)

These are scoped-out work items the team decided not to do for legitimate reasons. Listed here so they're not mistaken for unfinished failures.

- **Voice eval for trace DPO.** Family B in the DPO training data targets coaching/advice behavior, but the Phase 4 eval was tier-only. Building a 10-case voice keyword eval (~1 hour) would close the loop. Skipped because trace already promoted at 12/12 on tier accuracy and there's no immediate downstream consumer.
- **Cross-project SFT/DPO infra refactor.** A shared `iris_ft_lab/training/` package was scoped in `SFT_ANALYSIS.md` §5 but deferred until both pipelines have shipped successful runs. As of v2 in flight, only trace has shipped (SFT + DPO); collab-eval v2 outcome will trigger this decision.
- **CI / test coverage for trace pipeline.** collab-eval has 206 tests; trace has none for the new DPO scaffolding (`fuse_sft.py`, `generate_synthetic_dpo.py`, `validate_dpo_data.py`). Acceptable for now because the artifacts are stable and the validators run inline at generation time.
- **ORPO / GRPO experimentation on collab-eval.** mlx-lm-lora supports both. If collab-eval v2 doesn't promote, ORPO (single-model, no reference) is a reasonable next instrument to try. Currently out of scope.
