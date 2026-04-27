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

## Warning

The committed markdown outputs (`docs/sft_v1_data_audit.md`,
`docs/sft_v1_failure_audit.md`) are the canonical artifacts. Re-running the
scripts would overwrite them; `audit_rh_v1.py` in particular will lose the
manually-added narrative sections in `sft_v1_failure_audit.md` — restore
from git history if needed.
