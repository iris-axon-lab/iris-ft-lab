# SFT v1 — Phase 1.2 data audit

Source: `data/generated/spreadsheet_train_v1.jsonl` (n=240, seed=100)

Method: parsed each input + gold CSV, flagged where gold removes a row or strips a noise-bearing fragment present in the input.


## Whole-dataset rollup

| Behavior | Count |
|---|---|
| total cases | 240 |
| input had duplicate rows | 41 → gold drops them in 41 |
| input had `(...)` annotation row | 71 → gold drops in 71 |
| input had `---` separator | 8 → gold drops in 8 |
| input had `extra fabricated entry` | 11 → gold drops in 11 |
| input had `; orig $X.XXXM` in Notes | 16 → gold strips in 16 |

## Per-dimension rollup

| dim | n | drop_dup | drop_annot | drop_sep | drop_fab | strip_orig_$M |
|---|---|---|---|---|---|---|
| completeness | 60 | 0 | 0 | 0 | 0 | 0 |
| data_preservation | 60 | 41 | 11 | 0 | 11 | 0 |
| format_validity | 60 | 0 | 60 | 8 | 0 | 0 |
| unit_consistency | 60 | 0 | 0 | 0 | 0 | 16 |

## Sampled cases (5 per primary_dimension)

### completeness

| case_id | difficulty | rows in→gold (exp) | findings |
|---|---|---|---|
| `sc_gen_100_0001` | hard | in=16 gold=16 expected=16 | — |
| `sc_gen_100_0007` | easy | in=9 gold=9 expected=9 | — |
| `sc_gen_100_0017` | medium | in=10 gold=10 expected=10 | — |
| `sc_gen_100_0021` | medium | in=8 gold=8 expected=8 | — |
| `sc_gen_100_0030` | hard | in=12 gold=12 expected=12 | — |

### data_preservation

| case_id | difficulty | rows in→gold (exp) | findings |
|---|---|---|---|
| `sc_gen_100_0011` | hard | in=18 gold=14 expected=14 | DROP_DUP(2),DROP_ANNOT,DROP_FABRICATED |
| `sc_gen_100_0013` | medium | in=15 gold=14 expected=14 | DROP_DUP(1) |
| `sc_gen_100_0015` | medium | in=13 gold=12 expected=12 | DROP_DUP(1) |
| `sc_gen_100_0020` | medium | in=13 gold=12 expected=12 | DROP_DUP(1) |
| `sc_gen_100_0023` | easy | in=7 gold=7 expected=7 | — |

### format_validity

| case_id | difficulty | rows in→gold (exp) | findings |
|---|---|---|---|
| `sc_gen_100_0000` | hard | in=14 gold=12 expected=12 | DROP_ANNOT,DROP_SEP |
| `sc_gen_100_0002` | medium | in=13 gold=12 expected=12 | DROP_ANNOT |
| `sc_gen_100_0004` | easy | in=9 gold=8 expected=8 | DROP_ANNOT |
| `sc_gen_100_0006` | medium | in=15 gold=14 expected=14 | DROP_ANNOT |
| `sc_gen_100_0009` | medium | in=9 gold=8 expected=8 | DROP_ANNOT |

### unit_consistency

| case_id | difficulty | rows in→gold (exp) | findings |
|---|---|---|---|
| `sc_gen_100_0003` | medium | in=10 gold=10 expected=10 | — |
| `sc_gen_100_0005` | hard | in=11 gold=11 expected=11 | STRIP_NOTES_ORIG_$M |
| `sc_gen_100_0008` | hard | in=13 gold=13 expected=13 | STRIP_NOTES_ORIG_$M |
| `sc_gen_100_0016` | hard | in=15 gold=15 expected=15 | STRIP_NOTES_ORIG_$M |
| `sc_gen_100_0022` | hard | in=11 gold=11 expected=11 | STRIP_NOTES_ORIG_$M |

## Hypothesis test: 'delete suspicious content' vs 'preserve and convert'

Total **gold-acts-by-deleting** events across the 240-case train set: **147** (sum of column 1 in the rollup above).

Of the 240 cases, every case from `_data_preservation_noise` and `_format_validity_noise` paths produces at least one of: dup-drop, annotation-drop, separator-drop, fabricated-drop. Every `_unit_consistency_noise` hard case produces a Notes-stripping signal (`; orig $X.XXXM` in input, absent in gold).

There are **0** cases in the train set where the gold instructs preserve-and-convert on a row that *looks* droppable (e.g., a row whose Revenue is `$1.800M`-tagged where the gold keeps the row and rewrites the value as `1800`). The closest signal is the non-hidden `X.XXX M` cells in `_unit_consistency_noise` easy/medium — gold converts those to `$K` integers, but the row was never structurally tempting to delete.

Verdict: the training distribution **only ever rewards row deletion or Notes stripping** as the response to noise rows. There is no preserve-under-uncertainty signal. This confirms the v0 failure mode described in `SFT_ANALYSIS.md` §2.2(4) and motivates §2.2 of the v1 preservation-stress generator.
