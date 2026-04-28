# Why row preservation has been particularly hard to learn

The collab-eval harness has run four instruments against the row-preservation failure mode
on stress data (long tables, gold preserves all rows). All four produced 0.20–0.25 stress
`data_preservation` — essentially flat at base-model level:

| Run | Instrument | Stress `data_preservation` |
|---|---|---|
| Base | (none) | 0.20 |
| SFT v1 | gentle recipe + 25% stress training data | 0.25 |
| SFT v2 | same recipe + 50% stress training data | 0.25 |
| SFT v3 | curriculum (stress-only phase 1, mixed phase 2 at lower LR) | 0.25 |
| DPO v0 | preference pairs (chosen=preserve, rejected=drop) | 0.20 |

This document explains why this dimension has been resistant, in five compounding factors
ordered by explanatory weight.

## 1. Absence-of-action signal asymmetry

Positive demonstrations teach "convert `$M` to `$K`" because the gold token sequence
literally exhibits the action. They cannot teach "don't drop this row" because the gold
has nothing to point at — the desired action is the **absence** of a deletion the model
would otherwise produce. SFT learns from gold tokens; the policy decision "preserve vs
drop" is not embedded in any specific token, only in which tokens collectively appear.
v1, v2, v3 SFT all hit this wall regardless of recipe gentleness, data quantity, or
phase ordering.

This is `FAILURE_MODES.md` Mode 13 in its original form.

## 2. Token-economic disincentive at generation time

A 20-row CSV costs ~1000 tokens to produce. Past row 10–12, the model is deep into its
generation context. Termination has a structural incentive: every additional row is
sustained commitment without per-row positive feedback. The longer the table, the
stronger the "wrap up" pressure.

This is why stress eval scores low for **both base and SFT models** on long tables but
not on short ones — the cliff isn't a learned behavior, it's a generation-time
attentional pattern that the model carries over from its instruction-tuned prior.
SFT can't shift this because the gold doesn't model the cost of producing each row;
it only shows the final correct output.

## 3. Reward dilution — single missing row is a tiny loss component

A 20-row gold output has ~1000 supervision tokens. Missing one row costs ~50 tokens of
token-level cross-entropy. Divided across a batch, it's a small fraction of the total
loss. The gradient signal on "you should have generated this row" is correspondingly
small relative to "you should have converted this `$M` cell."

The optimizer climbs the strongest gradient first. Unit-conversion errors are
concentrated (a single token gets wrong); preservation errors are diluted (many tokens
of "missing row" get a small per-token signal). This is why `unit_consistency` reliably
hits 1.0 across SFT runs while `data_preservation` doesn't move.

## 4. DPO learned discrimination, not generation (Mode 14)

DPO v0's training loss collapsed to 0.001 by iter 30 with train accuracy 1.0 — the
model trivially learned "20-row CSV > 14-row CSV" as a discrimination task between
fixed outputs. But that's classification over **already-existing** sequences. At
inference time, the model has to decide **when to stop generating** token by token.

DPO sharpens log-ratios on fixed outputs; it does not rewire generation-time
decision-making. β=0.1 provided insufficient KL anchoring against such a crisp binary
signal — the policy drifted aggressively from v3 SFT (which explains the
unit_consistency regression as collateral damage during drift).

This is the new lesson from DPO v0: **preference learning over fixed outputs is not
the right instrument for a policy decision that is structurally about generation
boundaries.** Even when DPO learns the discrimination perfectly, the gradient never
reaches the per-step generation policy in a way that changes when the model stops.

## 5. Curriculum ruled out gradient competition (H2) as the bottleneck

v3 phase 1 (stress-only, 150 iters at lr=5e-5) gave preservation a clean gradient
run with no competing deletion signals. The model still didn't push above 0.25. This
ruled out the gradient-competition hypothesis: it's not "the deletion signal from
regular cases is overwriting the preservation signal from stress cases." The stress
signal itself, even pure, doesn't carry the right information through SFT positive
demonstrations.

This rules-out matters because it tells us "tune SFT harder" is not the path. The
bottleneck is in the supervision signal's fundamental nature, not in how it competes
with other signals.

## What's a viable next step

The four instruments tried (SFT recipe, SFT data quantity, SFT curriculum, DPO
preferences) all share a property: **they provide supervision on token sequences at
training time, not feedback on generation behavior at rollout time.** The grader's
`row_count_preserved` is a *generation-time* property — only knowable after the policy
has decided when to stop.

**Reinforcement learning with the grader as reward (v5)** is the only standard
instrument that:

- Rewards the *generation-time decision* directly, not via log-ratios over fixed outputs.
- Provides per-rollout feedback that accumulates across the long table (vs SFT's diluted
  per-token signal).
- Can be combined with the existing harness's RH-detection to penalize reward hacking
  (row duplication to game `row_count_preserved`).

Concretely for v5:

- **Algorithm:** PPO or GRPO over stress-case rollouts. mlx-lm-lora supports
  `--train-mode grpo`.
- **Reward:** composite_score with `data_preservation` weighted heavily, plus an explicit
  RH-like penalty term for duplicated/fabricated rows (since the harness's RH-detection
  already flags `format_validity ≥ 0.9 AND data_preservation < 0.5` post-hoc).
- **Reference policy:** v3 SFT (KL-anchored to preserve regular-eval gains, same as the
  DPO setup).
- **Risk:** rollout cost is 10–50× per training step vs DPO. Reward hacking via row
  duplication is a real risk and needs the explicit penalty.

The v5 prompt (RL with grader) is intentionally left out of scope for this session. The analysis above is the motivation document; the design will be drafted in a dedicated session.

## What this trajectory tells us about the harness

The flat trajectory is the harness doing its job. The promotion gate (specifically the
0.85 stress threshold) has correctly held the line through four interventions, none of
which produced a deployable adapter. Without that gate, v1's 0.05-lift composite would
have promoted an adapter that drops 75% of rows on long tables — exactly the failure
mode the harness was designed to detect. The fact that no instrument has cleared the
bar yet is information, not failure.
