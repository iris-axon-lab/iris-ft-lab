## Judge Calibration v1 — Seed Calibration Report

**Cases:** 24
**Dimensions covered:** faithfulness, quality_delta, citation_accurate, hallucination_flag
**LLM judge ran:** no (ANTHROPIC_API_KEY not set)

---

### Schema validation

All records pass schema validation.

### Deterministic sanity checks

All sanity checks pass (band/score consistency OK).

### LLM judge agreement

LLM judge not run (no API key). All LLM-required dimensions are **unassessed**.

To run with LLM judge:

```bash
ANTHROPIC_API_KEY=sk-... python scripts/run_judge_calibration.py
```

### Gate status (offline)

| Dimension | Status |
|-----------|--------|
| citation_accurate | unassessed (no API key) |
| faithfulness | unassessed (no API key) |
| hallucination_flag | unassessed (no API key) |
| quality_delta | unassessed (no API key) |

### Calibration set coverage

| Dimension | Cases |
|-----------|-------|
| citation_accurate | 6 |
| faithfulness | 6 |
| hallucination_flag | 6 |
| quality_delta | 6 |

### Note on calibration terminology

This is **seed calibration** — the expected scores reflect the artifact creator's best judgment of what a well-calibrated LLM judge should produce for these synthetic examples. This is NOT human-calibrated data. The expected scores serve as a sanity check (do obvious low-quality outputs score low?) rather than a gold standard.

For any dimension where the LLM judge agreement is below the gate threshold (80%), that dimension's scores should be excluded from training signal claims. See `docs/rl_env_design.md` for calibration rationale.
