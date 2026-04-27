# docs/audits/

One-shot audit scripts from SFT v1 Phase 1 analysis. Committed for
reproducibility; **not maintained**. If the generator or eval schema changes
these scripts will likely break.

## Scripts

### audit_gold_v1.py — Phase 1.2 data audit

Diffs input vs gold for all 240 cases in `data/generated/spreadsheet_train_v1.jsonl`.
Produces `docs/sft_v1_data_audit.md`.

```bash
# From collab-eval/ root:
python docs/audits/audit_gold_v1.py
```

### audit_rh_v1.py — Phase 1.4 RH-like failure audit

Characterizes the 20 RH-like cases from `results/collab_sft_v0_eval_raw.jsonl`.
Produces `docs/sft_v1_failure_audit.md`.

```bash
# From collab-eval/ root:
python docs/audits/audit_rh_v1.py
```

### audit_gold_v2.py — Phase 2.4 data audit

Diffs input vs gold for v2 data. Runs two sections by default: (1) v2 regular
only (240 cases, seed=100) for a direct v1↔v2 comparison; (2) full v2 train
set (480 cases, regular + stress) for the actual train-distribution rollup.
Produces `docs/sft_v2_data_audit.md`.

```bash
# From collab-eval/ root (runs both sections, writes sft_v2_data_audit.md):
python docs/audits/audit_gold_v2.py

# Single-source invocation (optional):
python docs/audits/audit_gold_v2.py \
    --input data/generated/spreadsheet_train_v2.jsonl \
    --input data/generated/spreadsheet_train_stress_v2.jsonl \
    --label full_v2
```

Key result: deletion ratio drops from 61.2% (v1: 147/240) to 30.6% (v2: 147/480),
a 30.6 pp reduction — above the 15 pp threshold for Phase 3 to proceed.

## Warning

The committed markdown outputs (`docs/sft_v1_data_audit.md`,
`docs/sft_v1_failure_audit.md`) are the canonical artifacts. Re-running the
scripts would overwrite them; `audit_rh_v1.py` in particular will lose the
manually-added narrative sections in `sft_v1_failure_audit.md` — restore
from git history if needed.

`docs/sft_v2_data_audit.md` is fully generated (no manual edits); it is safe
to re-run `audit_gold_v2.py` to reproduce it.
