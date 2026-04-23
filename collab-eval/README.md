# collab-eval

A task environment and grader harness for open-ended document manipulation tasks,
built to make reward design explicit. The core problem: document tasks have no binary
oracle, so naive reward functions get gamed. This harness addresses that with decomposed
rewards, deterministic checks where the ground truth is recoverable, LLM judges only
where judgment is genuinely required, and hard-fail caps that prevent catastrophic
outputs from hiding behind good scores on cheap dimensions.

All tasks, documents, and failure cases are fully synthetic.

---

## What this demonstrates

Each task is chosen because it has a specific, non-obvious grader failure mode:

- **Document revision:** An agent can produce fluent, active-voice prose that satisfies
  every deterministic check while silently introducing unsupported claims. Word count and
  passive-voice ratio cannot catch this — only a faithfulness judge can. The demo shows a
  padded agent that exceeds the word limit and gets hard-capped, and explains why the
  faithfulness dimension stays at 0.5 until an LLM assesses it.

- **Spreadsheet cleanup:** An agent can produce perfectly valid, correctly-headed CSV by
  quietly dropping the difficult rows. Format-validity alone gives this a perfect score.
  `data_preservation` catches it by checking row count against a known expected value.

- **Citation-grounded editing:** An agent can insert `[Source B]` next to a claim that
  Source B does not actually support — citation hallucination. The citation-present check
  (deterministic) rewards the form; `citation_accurate` and `hallucination_flag` (LLM)
  are required to catch the substance. The demo shows a fully uncited output triggering
  a hard-fail cap, making the gap between "looks cited" and "is cited" concrete.

The adversarial cases in `tests/test_reward_hacking_cases.py` make each of these
failure modes executable and runnable as regression tests.

---

## What this is

`collab-eval` provides three runnable task environments, deterministic graders for each,
an optional LLM-as-judge layer, and a composite grader with hard-fail caps. It is an
evaluation and environment harness — no training loop.

The design note at [`docs/rl_env_design.md`](docs/rl_env_design.md) covers reward
decomposition strategy, failure modes, and extension paths to a fuller RL setup.

---

## Setup

```bash
cd collab-eval
pip install -r requirements.txt
```

Python 3.11+ required.

---

## Run the demo

```bash
# Deterministic-only mode (no API key required):
python scripts/run_demo.py

# With LLM grading (Anthropic API key required):
ANTHROPIC_API_KEY=sk-... python scripts/run_demo.py

# Optional: use a different model for the LLM judge:
COLLAB_EVAL_JUDGE_MODEL=claude-sonnet-4-6 python scripts/run_demo.py
```

Each task section prints two episodes: a good agent output and a bad one that games
a naive grader, so the harness's purpose is visible in the output itself.

---

## Run the tests

```bash
# From the collab-eval/ directory:
pytest tests/ -v

# Or from the repo root:
pytest collab-eval/tests/ -v
```

Tests pass offline without Anthropic credentials. The adversarial cases are executable
demonstrations of reward hacking failure modes, not just smoke tests.

---

## Reward decomposition

Each task uses 3–4 named dimensions. The table below shows which are deterministic,
which require an LLM judge, and what the main exploit is for each.

**Document revision**

| Dimension | Grader | Gameable by |
|---|---|---|
| `instruction_following` | deterministic | compressing to word limit with contractions |
| `faithfulness` | LLM | adding unsupported claims in active voice |
| `over_editing` | deterministic | shortening the quoted input text |
| `quality_delta` | LLM | surface improvements that miss the real problems |

**Spreadsheet cleanup**

| Dimension | Grader | Gameable by |
|---|---|---|
| `data_preservation` | deterministic | duplicating rows to inflate count |
| `format_validity` | deterministic | valid CSV with wrong schema |
| `unit_consistency` | deterministic | hiding units in free-text Notes field |
| `completeness` | deterministic | renaming columns |

**Citation-grounded editing**

| Dimension | Grader | Gameable by |
|---|---|---|
| `citation_present` | deterministic | inserting empty brackets `[]` |
| `citation_accurate` | LLM | fabricating plausible-sounding quotes |
| `hallucination_flag` | LLM | citing a source that only loosely supports the claim |
| `argument_preservation` | deterministic | rephrasing anchor phrases |

Hard-fail caps: any output that is not parseable CSV, exceeds 120% of the word limit,
or contains zero citation markers is capped at ≤ 0.3 regardless of other dimension scores.

---

## Repo layout

```
collab-eval/
  collab_eval/
    base.py              # TaskSpec, Episode, TaskEnv (abstract)
    demo.py              # Demo runner (good + bad agent per task)
    env/tasks/
      doc_revision.py
      spreadsheet_clean.py
      citation_ground.py
    graders/
      deterministic.py   # Rule-based checks with Design note: comments
      llm_judge.py       # Optional Anthropic SDK grader (per-dimension)
      composite.py       # Weighted scoring + hard-fail caps
  tests/
    test_reward_hacking_cases.py  # Adversarial demonstrations
  scripts/
    run_demo.py
  docs/
    rl_env_design.md     # Design note: reward decomposition, failure modes, extensions
  data/sample_docs/      # Synthetic input documents
```
