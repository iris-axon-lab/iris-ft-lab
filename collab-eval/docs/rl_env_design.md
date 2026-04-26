# RL Environment Design Note: Document Tasks as a Reward Design Problem

> Tasks, documents, and failure cases in this artifact are fully synthetic and derived
> from public evaluation patterns. No proprietary data or internal workflows are referenced.

---

## 1. Why document tasks are harder to reward than code tasks

Code tasks have a natural, cheap reward signal: does the code pass the tests? The ground
truth is executable and deterministic. A grader that runs the test suite is not making a
judgment call; it is reading a binary outcome.

Document tasks have no equivalent. "Is this revision better?" is not a question with a
recoverable ground truth. The space of valid outputs is large, the quality differences
are often subtle, and different raters will disagree on what counts as an improvement.
This creates three problems for RL reward design:

**1. No binary oracle.** You cannot write a function that reliably returns True for good
revisions and False for bad ones. Any such function will be wrong in a non-trivial fraction
of cases, and agents will find the failure modes.

**2. Multi-dimensional quality.** A document can be clearer but less faithful. It can
follow instructions while being worse overall. These trade-offs cannot be collapsed into
a single scalar without losing information that matters for policy design.

**3. Lazy proxy metrics dominate.** Metrics that correlate with quality in expectation
(fluency, word count compliance, format validity) are often much cheaper to satisfy than
the underlying quality criteria they proxy for. Agents that optimize proxies without
satisfying the underlying criteria are the canonical reward hacking failure mode.

This harness is designed around these three problems. It does not solve them, but it
makes them explicit and structures the reward signal to resist the most common failure modes.

---

## 2. Design principles

### 2.1 Reward decomposition

Rather than a single holistic score, each task uses 3–4 named dimensions. This serves
two purposes:

- **Interpretability.** A composite score drop can be attributed to a specific dimension,
  making it possible to diagnose agent failure modes rather than just observe them.
- **Anti-gaming.** A multi-dimensional rubric is harder to satisfy with a single lazy
  strategy. An agent that pads text to hit a word-count limit will score well on
  `instruction_following` but poorly on `over_editing`. An agent that drops CSV rows to
  avoid messy data will score well on `format_validity` but poorly on `data_preservation`.

### 2.2 Deterministic checks where possible

Every dimension that has a recoverable ground truth uses a deterministic check. CSV
parseability, row counts, word counts, citation marker presence, and passive voice ratios
are all checkable without a model. These checks are:

- Cheap (no API call required).
- Auditable (the logic is transparent and inspectable).
- Stable (they do not change between runs).

The heuristics are imperfect. The passive voice regex misses complex constructions.
The CSV row count check does not verify values, only counts. Citation presence checks
markers, not accuracy. Each heuristic has a `# Design note:` comment in the source
explaining the known failure mode and why the heuristic is used anyway. The principle
is: cheap and transparent beats clever and opaque.

### 2.3 LLM judges only where needed

For dimensions that genuinely require judgment — faithfulness, citation accuracy,
hallucination detection — the harness uses an LLM judge. Three design choices reduce
the risk that the judge is itself gamed:

**Per-dimension scoring.** The judge evaluates one dimension at a time with a specific
rubric, not holistically. This prevents the judge from trading off across dimensions in
a single inference pass, and produces structured output that maps directly back to the
reward decomposition.

**Short rubrics.** Each rubric is one paragraph, describing the scoring scale explicitly.
Long, vague rubrics produce inconsistent scores and are easier to optimize against.

**Optional.** The harness runs fully in deterministic-only mode without an API key.
Placeholder scores of 0.5 signal "unassessed" rather than "passing". This makes the
gap between deterministic and full evaluation explicit rather than hiding it.

### 2.4 Hard-fail flags

Some failure modes are severe enough that partial credit should not apply. An output
that is not parseable CSV has failed completely at the spreadsheet task, regardless of
how well it scores on other dimensions. An output longer than 360 words has violated
a hard constraint.

When a hard-fail flag is triggered, the composite score is capped at a low threshold
(0.2–0.3). This prevents an agent from scoring well overall by excelling at cheap
dimensions while catastrophically failing an important one.

The cap is set low enough that a hard-failing output cannot be confused with a marginal
pass, but not at zero — partial credit for what the agent did correctly aids debugging.

### 2.5 Adversarial cases as first-class artifacts

`tests/test_reward_hacking_cases.py` contains four executable adversarial demonstrations.
These are not structural smoke tests; they are explicit embodiments of known reward
hacking failure modes, each with a comment explaining why a naive grader would fail.
They run offline with pytest and serve as regression tests: if a grader change causes
an adversarial case to pass when it should fail, the test catches it.

---

## 3. Reward hacking failure modes this harness targets

| Failure mode | Naive grader behavior | Harness mitigation |
|---|---|---|
| Verbosity padding | Passes if output is "within limit" | Soft word-count decay + hard-fail at 120% of limit |
| Silent row dropping | Passes format-validity check | `row_count_preserved` against expected row count |
| Citation hallucination | Passes if citation markers are present | `citation_accurate` + `hallucination_flag` (LLM) |
| Constraint gaming (active voice + new claims) | Passes voice check | `faithfulness` held at 0.5 until LLM assesses it |
| Holistic judge gaming | Agent learns judge's preferences | Per-dimension scoring with isolated rubrics |
| Format-only compliance | Passes CSV parse check | `data_preservation` weighted at 0.35 |

The harness does not fully prevent reward hacking — no harness does. It makes specific
failure modes explicit and raises the cost of the most common exploits.

---

## 4. Why these three task types

**Document revision** is the canonical open-ended document task. It combines structural
editing (where quality is judged holistically) with constraint satisfaction (where quality
is partially deterministic). The tension between faithfulness and instruction-following
is a real challenge for reward design and cannot be collapsed into a single score.

**Spreadsheet cleanup** is a format-normalization task with fully recoverable ground
truth in some dimensions (CSV validity, unit consistency, row count) and genuine judgment
in others (what counts as a "data value" worth preserving). It illustrates that even
tasks that look fully deterministic have a judgment gap when the input is messy.

**Citation-grounded editing** is chosen because it has a hard-to-detect failure mode:
citation hallucination. An agent can produce a document that looks well-cited but whose
citations are not actually traceable to the source material. This failure mode requires
an LLM judge and is expensive to catch with rules alone. It represents the class of
tasks where model-based grading is genuinely necessary.

Together, the three tasks cover the main reward design trade-offs: fully deterministic,
partially deterministic, and requiring model judgment. They are small enough to inspect
manually and realistic enough to be interesting.

---

## 5. Reward decomposition table

| Dimension | Task types | Grader type | Gameable by | Mitigation |
|---|---|---|---|---|
| `instruction_following` | doc_revision | Deterministic | Contractions, unusual active constructions | Soft decay, not binary cliff |
| `faithfulness` | doc_revision | LLM | N/A (LLM judge required) | Per-dimension rubric |
| `over_editing` | doc_revision | Deterministic | Truncating input quote | Word-count ratio check |
| `quality_delta` | doc_revision | LLM | N/A (LLM judge required) | Per-dimension rubric |
| `data_preservation` | spreadsheet_clean | Deterministic | Row duplication | Expected count + value spot-checks |
| `format_validity` | spreadsheet_clean | Deterministic | Valid CSV with wrong schema | Combined with other dimensions |
| `unit_consistency` | spreadsheet_clean | Deterministic | Hiding units in Notes text | Regex + per-cell check in full mode |
| `completeness` | spreadsheet_clean | Deterministic | Column renaming | Header set comparison |
| `citation_present` | citation_ground | Deterministic | Empty brackets | `citation_accurate` (LLM) required |
| `citation_accurate` | citation_ground | LLM | Fabricating plausible quotes | Rubric requires source traceability |
| `hallucination_flag` | citation_ground | LLM | Subtle claim strengthening | Explicit rubric: "stronger than source supports" |
| `argument_preservation` | citation_ground | Deterministic + LLM | Rephrasing anchor phrases | Anchor phrase check + LLM holistic |

---

## 6. Extension paths

This harness is intentionally small. It does not implement a training loop. These are
the natural extension directions, roughly in order of difficulty.

### 6.1 Synthetic task generation

**Status as of training cycle v0:** Implemented for spreadsheet_clean. Stub for
doc_revision and citation_ground.

The spreadsheet generator (`collab_eval/generation/spreadsheet_generator.py`) produces
parameterized messy inputs with seeded noise: blank rows, mixed $K/$M units, annotation
rows, extra columns, renamed columns. Each generated case includes `expected_metadata`
sufficient for fully offline deterministic grading. The generator is deterministic:
same seed → same output.

For doc_revision and citation_ground, generation requires a template library or a
generator model to produce varied source documents with seeded structural and factual
problems. These are stubs in this cycle.

The key constraint is that generated tasks must have seeded, auditable ground truth.
Tasks generated by asking a model to "write a document with problems" are hard to
validate — you cannot be sure the model seeded the problems it claimed to seed.

### 6.2 Judge calibration (seed calibration harness)

**Status as of training cycle v0:** Seed calibration harness implemented. Human
calibration not implemented.

`data/judge_calibration/seed_calibration_v1.jsonl` contains 24 synthetic calibration
examples covering all four LLM-required dimensions (faithfulness, quality_delta,
citation_accurate, hallucination_flag) with expected scores and bands. The
`run_judge_calibration.py` script validates schema offline and optionally runs the LLM
judge to compute per-dimension agreement.

**Gate:** dimensions with < 0.80 agreement are excluded from training-signal claims.
Dimensions with < 0.70 are hard-excluded and marked uncalibrated.

This is "seed calibration" — the artifact creator's judgment, not human rater agreement.
A production setup would route a sample of episodes to human raters to establish
inter-rater reliability before using judge scores as training signal. That step is not
implemented here.

### 6.3 What should remain SFT-first

For document tasks, RL makes sense for fine-grained policy shaping after a base level
of task competence is established. An agent that cannot produce coherent prose should
not be RL-trained on document revision — the reward signal is too sparse and the action
space too large. SFT on demonstration data should establish basic competence first.

A reasonable heuristic: use SFT until the agent reliably produces outputs in the correct
format and avoids catastrophic failures (hard-fail flags). Once hard-fail rates drop below
~10%, the reward signal is dense enough to make RL productive. Attempting RL before
this threshold is reached tends to produce degenerate policies that exploit grader
weaknesses rather than improving actual task performance.

### 6.4 Open problems this harness does not solve

- **Reward model stability over training.** The LLM judge's scores may drift as the
  policy changes if the judge was not trained adversarially against the policy.
- **Task diversity.** Three task types with one example each is not a training
  distribution. Generalization requires much greater task variety.
- **Long-horizon tasks.** All three tasks assume a single-turn completion. Multi-turn
  document editing — where the agent revises, receives feedback, and revises again — is
  a harder environment to reward correctly and is not addressed here.
- **Inter-rater reliability.** The LLM judge's rubrics have not been calibrated against
  human raters. Calibration is essential before using the harness for RL training.
  This is especially hard for document quality because rater disagreement is asymmetric:
  raters converge reliably on bad outputs but diverge significantly on what "good" looks
  like, which means a calibration set requires oversampling the high-quality range.
- **Grader adversarial robustness.** The deterministic heuristics are cheap to game
  by a sufficiently capable agent that specifically targets them. The harness is designed
  to be harder to game than a single holistic score, not impossible to game.

---

## 7. Training cycle v0 — findings from optimization pressure

Training cycle v0 ran four heuristic policies on 80 generated spreadsheet_clean cases
and measured the grader's behavior under optimization pressure. Key findings:

### 7.1 Grader weaknesses confirmed under pressure

**Row count check is the most exploitable weakness.** `row_count_preserved` counts data
rows but does not check row identity. The overfit_policy drops the second half of each
time series, duplicates the first half to reach the expected count, and scores
`data_preservation = 1.0` on 13 probe cases. An agent that discovers this strategy would
appear to preserve data while silently truncating it. Fix: add a per-row identity or
uniqueness check.

**Unit regex misses embedded $M in Notes free-text.** `unit_normalized` uses the pattern
`\$M|\bM\b`. This pattern does not match `$1.800M` because `\$M` requires `$` immediately
before `M` (digits intervene) and `\bM\b` requires a word boundary before `M` (but digit
`0` precedes `M`, no boundary). An agent that converts data cells to integer $K but stores
the original $M value in Notes as `orig $1.800M` scores `unit_consistency = 1.0` while
the $M data is still present in the output. Fix: cell-by-cell numeric parser.

### 7.2 What template selection reveals

The reward_aware_policy (inference-time template search over 4 candidates using the
deterministic grader) achieves mean composite 1.0 vs. naive_policy's 0.93. The full
improvement comes from unit normalization: template B normalizes $M → $K while naive
passes $M through unchanged. This demonstrates that access to the grader's signal at
inference time provides a meaningful advantage — and that a model which "discovers"
this signal access pattern would achieve the same improvement.

### 7.3 The floor is already high

naive_policy achieves mean composite 0.93 with essentially no effort (blank-row removal
only). This means the harness's signal-to-noise ratio is low for most cases: only 7%
of the composite score is at risk from naive behavior. The implication: a training loop
on this harness would spend most of its signal budget on marginal improvements in the
remaining 7%, while missing the deeper failure modes (duplication, unit hiding) that
score 1.0 on the current grader.

### 7.4 Preconditions for a real RL loop

From training cycle v0, the following must be true before this can be called an RL
environment:

1. **A trainable policy exists.** Currently there is no model — only fixed heuristic functions.
2. **Row-identity check is added** to close the most exploitable weakness.
3. **Hard-fail rate is below ~10%.** naive_policy has 0% hard-fails on spreadsheet_clean,
   which is a good sign, but the other task types (doc_revision, citation_ground) need
   evaluation with a model before this claim can be made across the board.
4. **LLM judge calibration gate passes** for at least the dimensions intended for training
   signal. Currently all LLM dimensions are unassessed.

See `results/training_cycle_v0.md` for the full summary and Go/No-go assessment.
