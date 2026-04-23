# collab-eval

A task environment and grader harness for open-ended document manipulation tasks,
built to make reward design explicit. The core problem: document tasks have no binary
oracle, so naive reward functions get gamed. This harness addresses that with decomposed
rewards, deterministic checks where the ground truth is recoverable, LLM judges only
where judgment is genuinely required, and hard-fail caps that prevent catastrophic
outputs from hiding behind good scores on cheap dimensions.

All tasks, documents, and failure cases are fully synthetic.

---

## What this is

`collab-eval` provides three runnable task environments (document revision, spreadsheet
cleanup, citation-grounded editing), deterministic graders for each, an optional
LLM-as-judge layer, and a composite grader with hard-fail caps. It is an evaluation
and environment harness — no training loop.

The adversarial test cases in `tests/test_reward_hacking_cases.py` are executable
demonstrations of specific reward hacking failure modes, not just smoke tests.

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

---

## Run the tests

```bash
# From the collab-eval/ directory:
pytest tests/ -v

# Or from the repo root:
pytest collab-eval/tests/ -v
```

Tests pass offline without Anthropic credentials. The adversarial cases in
`tests/test_reward_hacking_cases.py` are executable demonstrations of reward
hacking failure modes, not just structural smoke tests.

---

## Task types

| Task | Input | Key difficulty |
|---|---|---|
| `doc_revision` | Project update doc with seeded problems | Faithfulness vs. instruction-following trade-off |
| `spreadsheet_clean` | Messy CSV with unit inconsistencies and blank rows | Silent row dropping is undetectable by format checks |
| `citation_ground` | Research brief needing inline citations | Hallucination trap: one source only loosely supports its claim |

---

## Reward design approach

Each task decomposes reward into 3–4 named dimensions. Deterministic checks (word count,
CSV validity, citation marker presence) are used wherever the ground truth is recoverable.
LLM judges are used only for dimensions that genuinely require judgment (faithfulness,
citation accuracy, hallucination detection). Hard-fail caps prevent catastrophically bad
outputs from scoring above a low threshold regardless of partial dimension scores.

See [`docs/rl_env_design.md`](docs/rl_env_design.md) for the full design rationale.

---

## Repo layout

```
collab-eval/
  collab_eval/
    base.py              # TaskSpec, Episode, TaskEnv (abstract)
    demo.py              # Demo runner
    env/tasks/
      doc_revision.py    # Document revision task
      spreadsheet_clean.py
      citation_ground.py
    graders/
      deterministic.py   # Rule-based checks
      llm_judge.py       # Optional Anthropic SDK grader
      composite.py       # Weighted scoring + hard-fail caps
  tests/
    test_reward_hacking_cases.py  # Adversarial demonstrations
  scripts/
    run_demo.py
  docs/
    rl_env_design.md     # Design note (reward decomposition, hacking cases, extensions)
  data/sample_docs/      # Synthetic input documents
```
