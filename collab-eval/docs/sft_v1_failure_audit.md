# SFT v1 — Phase 1.4 RH-like failure audit

Definition: cases from `results/collab_sft_v0_eval_raw.jsonl` where `format_validity >= 0.9 AND data_preservation < 0.5` (matches `eval/run_collab_model_eval.py:91-95`).

Total RH-like cases under v0 SFT: **20**.


## Per-case breakdown

| case_id | dim | diff | rows_in→sft (gold) | base_pres | sft_pres | missing_quarters → category |
|---|---|---|---|---|---|---|
| `sc_gen_200_0001` | unit_consistency | medium | 12→6 (12) | 1.00 | 0.00 | Q1 2023, Q2 2023, Q3 2023, Q4 2023, Q2 2024, Q3 2024, Q4 2024 → has_$M_notation, has_$M_notation, genuine_data, genuine_data, genuine_data, genuine_data, genuine_data |
| `sc_gen_200_0003` | data_preservation | medium | 14→10 (13) | 1.00 | 0.00 | Q1 2023, Q2 2023, Q3 2023, Q4 2023 → duplicate_in_input, genuine_data, genuine_data, genuine_data, genuine_data |
| `sc_gen_200_0007` | unit_consistency | medium | 13→11 (13) | 1.00 | 0.00 | Q1 2023, Q2 2023, Q3 2023, Q4 2023 → has_$M_notation, genuine_data, genuine_data, has_$M_notation |
| `sc_gen_200_0008` | format_validity | hard | 12→10 (12) | 1.00 | 0.00 | Q1 2023, Q2 2023, Q3 2023, Q4 2023 → genuine_data, genuine_data, genuine_data, genuine_data |
| `sc_gen_200_0015` | format_validity | hard | 18→14 (18) | 1.00 | 0.00 | Q3 2022, Q4 2022, Q1 2023, Q2 2023, Q3 2023, Q4 2023, Q1 2024, Q2 2024 → genuine_data, genuine_data, genuine_data, genuine_data, genuine_data, genuine_data, genuine_data, genuine_data |
| `sc_gen_200_0018` | format_validity | easy | 10→6 (10) | 1.00 | 0.00 | Q1 2023, Q2 2023, Q3 2023, Q4 2023, Q2 2024 → genuine_data, genuine_data, genuine_data, genuine_data, genuine_data |
| `sc_gen_200_0021` | format_validity | easy | 8→4 (8) | 1.00 | 0.00 | Q1 2023, Q2 2023, Q3 2023, Q4 2023, Q2 2024, Q3 2024, Q4 2024 → genuine_data, genuine_data, genuine_data, genuine_data, genuine_data, genuine_data, genuine_data |
| `sc_gen_200_0023` | unit_consistency | medium | 8→6 (8) | 1.00 | 0.00 | Q3 2021, Q4 2021 → genuine_data, genuine_data |
| `sc_gen_200_0024` | completeness | medium | 10→6 (10) | 1.00 | 0.00 | Q1 2023, Q2 2023, Q3 2023, Q4 2023, Q2 2024 → genuine_data, genuine_data, genuine_data, genuine_data, genuine_data |
| `sc_gen_200_0029` | unit_consistency | hard | 18→10 (18) | 0.00 | 0.00 | Q1 2023, Q2 2023, Q3 2023, Q4 2023, Q2 2024, Q3 2024, Q4 2024, Q1 2025, Q2 2025 → has_orig_$M_in_notes, has_$M_notation, has_orig_$M_in_notes, genuine_data, genuine_data, has_$M_notation, genuine_data, genuine_data, genuine_data |
| `sc_gen_200_0035` | completeness | medium | 9→4 (9) | 1.00 | 0.00 | Q1 2022, Q2 2022, Q3 2022, Q4 2022, Q1 2023 → genuine_data, genuine_data, genuine_data, genuine_data, genuine_data |
| `sc_gen_200_0039` | format_validity | medium | 8→4 (8) | 1.00 | 0.00 |   Q1 2023 ,   Q2 2023 ,   Q3 2023 , Q4 2023,   Q2 2024 , Q4 2024 → genuine_data, genuine_data, genuine_data, genuine_data, genuine_data, genuine_data |
| `sc_gen_200_0043` | format_validity | hard | 14→10 (14) | 1.00 | 0.00 | Q1 2023, Q2 2023, Q3 2023, Q4 2023, Q3 2024, Q4 2024, Q1 2025, Q2 2025 → genuine_data, genuine_data, genuine_data, genuine_data, genuine_data, genuine_data, genuine_data, genuine_data |
| `sc_gen_200_0053` | format_validity | medium | 10→6 (10) | 1.00 | 0.00 |   Q4 2022 ,   Q1 2023 , Q2 2023, Q3 2023, Q4 2023,   Q1 2024 ,   Q2 2024  → genuine_data, genuine_data, genuine_data, genuine_data, genuine_data, genuine_data, genuine_data |
| `sc_gen_200_0058` | format_validity | hard | 11→10 (11) | 1.00 | 0.00 | Q1 2023, Q2 2023, Q3 2023 → genuine_data, genuine_data, genuine_data |
| `sc_gen_200_0062` | format_validity | medium | 8→4 (8) | 1.00 | 0.00 | Q1 2023,   Q2 2023 , Q3 2023, Q4 2023, Q4 2024 → genuine_data, genuine_data, genuine_data, genuine_data, genuine_data |
| `sc_gen_200_0065` | unit_consistency | medium | 10→6 (10) | 1.00 | 0.00 | Q1 2023, Q2 2023, Q3 2023, Q4 2023, Q2 2024 → has_$M_notation, genuine_data, genuine_data, genuine_data, genuine_data |
| `sc_gen_200_0066` | unit_consistency | hard | 10→6 (10) | 1.00 | 0.00 | Q1 2023, Q2 2023, Q3 2023, Q4 2023 → genuine_data, has_orig_$M_in_notes, genuine_data, has_$M_notation |
| `sc_gen_200_0071` | data_preservation | medium | 9→7 (8) | 1.00 | 0.00 | Q1 2023, Q2 2023, Q3 2023, Q4 2023 → duplicate_in_input, genuine_data, genuine_data, genuine_data, genuine_data |
| `sc_gen_200_0077` | data_preservation | medium | 15→9 (14) | 1.00 | 0.00 | Q1 2023, Q2 2023, Q3 2023, Q4 2023, Q2 2024, Q3 2024, Q4 2024, Q1 2025 → genuine_data, genuine_data, genuine_data, genuine_data, genuine_data, genuine_data, genuine_data, genuine_data, genuine_data |

## Aggregate classification

Each row dropped by SFT but present in input is categorized by why it *might* have looked droppable to the model.

| Category | Count |
|---|---|
| has_$M_notation | 8 |
| has_orig_$M_in_notes | 3 |
| extra_fabricated | 0 |
| duplicate_in_input | 2 |
| genuine_data | 100 |

## By primary_dimension

| dim | count of RH-like |
|---|---|
| completeness | 2 |
| data_preservation | 3 |
| format_validity | 9 |
| unit_consistency | 6 |

## Detailed per-case rows

### `sc_gen_200_0001` (unit_consistency / medium)

- input rows: 12, sft output rows: 6, gold rows: 12
- known_failure_modes (generator-seeded): ['mixed_units', 'mixed_unit_suffixes']
- missing input rows in sft output:
    - `Q1 2023,2.328 M,1.683 M,31,Jan 2023 product launch` → **has_$M_notation**
    - `Q2 2023,2.614 M,1.704 M,33,Apr 2023 market expansion` → **has_$M_notation**
    - `Q3 2023,2785,1755,33,Jul 2023 office lease renewed` → **genuine_data**
    - `Q4 2023,3082$K,1963,33,Oct 2023 office lease renewed` → **genuine_data**
    - `Q2 2024,3555,2396,37,Apr 2024 product launch` → **genuine_data**
    - `Q3 2024,3755,2382,39,Jul 2024 market expansion` → **genuine_data**
    - `Q4 2024,3934,2505,42,Oct 2024 new tooling deployed` → **genuine_data**

### `sc_gen_200_0003` (data_preservation / medium)

- input rows: 14, sft output rows: 10, gold rows: 13
- known_failure_modes (generator-seeded): ['blank_rows', 'duplicated_rows']
- missing input rows in sft output:
    - `Q1 2023,1769,1037,22,Jan 2023 market expansion` → **genuine_data**
    - `Q2 2023,1852,1176,23,Apr 2023 Q2 targets met` → **genuine_data**
    - `Q3 2023,1984,1302,24,Jul 2023 market expansion` → **genuine_data**
    - `Q4 2023,2104,1225,24,Oct 2023 office lease renewed` → **genuine_data**

### `sc_gen_200_0007` (unit_consistency / medium)

- input rows: 13, sft output rows: 11, gold rows: 13
- known_failure_modes (generator-seeded): ['mixed_units']
- missing input rows in sft output:
    - `Q1 2023,2.006 M,1.494 M,44,Jan 2023 budget review completed` → **has_$M_notation**
    - `Q2 2023,2042,1338,44,Apr 2023 series funding closed` → **genuine_data**
    - `Q3 2023,2157,1423,46,Jul 2023 hiring freeze lifted` → **genuine_data**
    - `Q4 2023,2.238 M,1.536 M,48,Oct 2023 compliance audit passed` → **has_$M_notation**

### `sc_gen_200_0008` (format_validity / hard)

- input rows: 12, sft output rows: 10, gold rows: 12
- known_failure_modes (generator-seeded): ['annotation_row', 'malformed_row_width', 'mid_table_separator']
- missing input rows in sft output:
    - `Q1 2023,2452,1792,26` → **genuine_data**
    - `Q2 2023,2589,1997,26,Apr 2023 partnership signed` → **genuine_data**
    - `Q3 2023,2831,2124,26,Jul 2023 product launch` → **genuine_data**
    - `Q4 2023,2946,2303,29,Oct 2023 Q4 targets met` → **genuine_data**

### `sc_gen_200_0015` (format_validity / hard)

- input rows: 18, sft output rows: 14, gold rows: 18
- known_failure_modes (generator-seeded): ['annotation_row', 'malformed_row_width', 'mid_table_separator']
- missing input rows in sft output:
    - `Q3 2022,3402,2852,44,Jul 2022 headcount increase` → **genuine_data**
    - `Q4 2022,3597,2845,47,Oct 2022 hiring freeze lifted` → **genuine_data**
    - `Q1 2023,3868,2985,47,Jan 2023 Q1 targets met` → **genuine_data**
    - `Q2 2023,4091,3354,50,Apr 2023 baseline quarter` → **genuine_data**
    - `Q3 2023,4110,3094,53,Jul 2023 headcount increase` → **genuine_data**
    - `Q4 2023,4215,3348,54,Oct 2023 partnership signed` → **genuine_data**
    - `Q1 2024,4223,3327,55,Jan 2024 headcount increase` → **genuine_data**
    - `Q2 2024,4441,3674,58,Apr 2024 new tooling deployed` → **genuine_data**

### `sc_gen_200_0018` (format_validity / easy)

- input rows: 10, sft output rows: 6, gold rows: 10
- known_failure_modes (generator-seeded): ['annotation_row']
- missing input rows in sft output:
    - `Q1 2023,2187,1440,19,Jan 2023 headcount increase` → **genuine_data**
    - `Q2 2023,2256,1619,21,Apr 2023 headcount increase` → **genuine_data**
    - `Q3 2023,2217,1482,24,Jul 2023 budget review completed` → **genuine_data**
    - `Q4 2023,2429,1712,27,Oct 2023 baseline quarter` → **genuine_data**
    - `Q2 2024,2846,1834,30,Apr 2024 Q2 targets met` → **genuine_data**

### `sc_gen_200_0021` (format_validity / easy)

- input rows: 8, sft output rows: 4, gold rows: 8
- known_failure_modes (generator-seeded): ['annotation_row']
- missing input rows in sft output:
    - `Q1 2023,1170,850,29,Jan 2023 compliance audit passed` → **genuine_data**
    - `Q2 2023,1360,951,30,Apr 2023 office lease renewed` → **genuine_data**
    - `Q3 2023,1515,1075,31,Jul 2023 partnership signed` → **genuine_data**
    - `Q4 2023,1647,1092,31,Oct 2023 hiring freeze lifted` → **genuine_data**
    - `Q2 2024,2125,1466,33,Apr 2024 product launch` → **genuine_data**
    - `Q3 2024,2194,1533,36,Jul 2024 hiring freeze lifted` → **genuine_data**
    - `Q4 2024,2351,1495,36,Oct 2024 compliance audit passed` → **genuine_data**

### `sc_gen_200_0023` (unit_consistency / medium)

- input rows: 8, sft output rows: 6, gold rows: 8
- known_failure_modes (generator-seeded): ['mixed_units']
- missing input rows in sft output:
    - `Q3 2021,3214,2111,47,Jul 2021 hiring freeze lifted` → **genuine_data**
    - `Q4 2021,3265,2007,49,Oct 2021 baseline quarter` → **genuine_data**

### `sc_gen_200_0024` (completeness / medium)

- input rows: 10, sft output rows: 6, gold rows: 10
- known_failure_modes (generator-seeded): ['extra_column', 'extra_column_region']
- missing input rows in sft output:
    - `Q1 2023,1973,1273,35,Jan 2023 Q1 targets met,AMER` → **genuine_data**
    - `Q2 2023,2136,1276,37,Apr 2023 headcount increase,AMER` → **genuine_data**
    - `Q3 2023,2106,1348,38,Jul 2023 headcount increase,APAC` → **genuine_data**
    - `Q4 2023,2187,1312,39,Oct 2023 partnership signed,APAC` → **genuine_data**
    - `Q2 2024,2362,1590,42,Apr 2024 new tooling deployed,APAC` → **genuine_data**

### `sc_gen_200_0029` (unit_consistency / hard)

- input rows: 18, sft output rows: 10, gold rows: 18
- known_failure_modes (generator-seeded): ['mixed_units', 'units_hidden_in_notes', 'mixed_unit_suffixes']
- missing input rows in sft output:
    - `Q1 2023,3460,2500,30,Jan 2023 baseline quarter; orig $3.460M` → **has_orig_$M_in_notes**
    - `Q2 2023,3.421 M,2.686 M,33,Apr 2023 new tooling deployed` → **has_$M_notation**
    - `Q3 2023,3628,2613,34,Jul 2023 headcount increase; orig $3.628M` → **has_orig_$M_in_notes**
    - `Q4 2023,3679,2778,37,Oct 2023 hiring freeze lifted` → **genuine_data**
    - `Q2 2024,3878,2771,42,Apr 2024 new tooling deployed` → **genuine_data**
    - `Q3 2024,4.153 M,3.185 M,44,Jul 2024 baseline quarter` → **has_$M_notation**
    - `Q4 2024,4186$K,3263,44,Oct 2024 series funding closed` → **genuine_data**
    - `Q1 2025,4385,3263,45,Jan 2025 hiring freeze lifted` → **genuine_data**
    - `Q2 2025,4512,3302,45,Apr 2025 office lease renewed` → **genuine_data**

### `sc_gen_200_0035` (completeness / medium)

- input rows: 9, sft output rows: 4, gold rows: 9
- known_failure_modes (generator-seeded): ['extra_column', 'extra_column_region']
- missing input rows in sft output:
    - `Q1 2022,2892,2367,28,Jan 2022 budget review completed,EMEA` → **genuine_data**
    - `Q2 2022,2883,2290,28,Apr 2022 baseline quarter,EMEA` → **genuine_data**
    - `Q3 2022,3130,2621,29,Jul 2022 office lease renewed,APAC` → **genuine_data**
    - `Q4 2022,3279,2650,32,Oct 2022 series funding closed,AMER` → **genuine_data**
    - `Q1 2023,3336,2491,35,Jan 2023 headcount increase,EMEA` → **genuine_data**

### `sc_gen_200_0039` (format_validity / medium)

- input rows: 8, sft output rows: 4, gold rows: 8
- known_failure_modes (generator-seeded): ['annotation_row', 'extra_whitespace']
- missing input rows in sft output:
    - `  Q1 2023 , 2220 , 1798 , 30 , Jan 2023 new tooling deployed` → **genuine_data**
    - `  Q2 2023 , 2208 , 1834 , 33 , Apr 2023 market expansion` → **genuine_data**
    - `  Q3 2023 , 2493 , 1955 , 33 , Jul 2023 product launch` → **genuine_data**
    - `Q4 2023,2483,1824,36,Oct 2023 series funding closed` → **genuine_data**
    - `  Q2 2024 , 2559 , 1909 , 38 , Apr 2024 Q2 targets met` → **genuine_data**
    - `Q4 2024,2693,2033,39,Oct 2024 compliance audit passed` → **genuine_data**

### `sc_gen_200_0043` (format_validity / hard)

- input rows: 14, sft output rows: 10, gold rows: 14
- known_failure_modes (generator-seeded): ['annotation_row', 'malformed_row_width', 'mid_table_separator']
- missing input rows in sft output:
    - `Q1 2023,3466,1952,38,Jan 2023 product launch` → **genuine_data**
    - `Q2 2023,3495,1933,41,Apr 2023 Q2 targets met` → **genuine_data**
    - `Q3 2023,3712,2090,42,Jul 2023 baseline quarter` → **genuine_data**
    - `Q4 2023,3823,2120,44,Oct 2023 series funding closed,extra_field` → **genuine_data**
    - `Q3 2024,4127,2439,49,Jul 2024 new tooling deployed,extra_field` → **genuine_data**
    - `Q4 2024,4128,2240,50,Oct 2024 budget review completed` → **genuine_data**
    - `Q1 2025,4149,2171,53,Jan 2025 office lease renewed` → **genuine_data**
    - `Q2 2025,4105,2218,54,Apr 2025 headcount increase` → **genuine_data**

### `sc_gen_200_0053` (format_validity / medium)

- input rows: 10, sft output rows: 6, gold rows: 10
- known_failure_modes (generator-seeded): ['annotation_row', 'extra_whitespace']
- missing input rows in sft output:
    - `  Q4 2022 , 1212 , 818 , 25 , Oct 2022 series funding closed` → **genuine_data**
    - `  Q1 2023 , 1501 , 1117 , 26 , Jan 2023 series funding closed` → **genuine_data**
    - `Q2 2023,1601,1198,29,Apr 2023 partnership signed` → **genuine_data**
    - `Q3 2023,1648,1145,31,Jul 2023 Q3 targets met` → **genuine_data**
    - `Q4 2023,1863,1271,33,Oct 2023 office lease renewed` → **genuine_data**
    - `  Q1 2024 , 1904 , 1435 , 34 , Jan 2024 partnership signed` → **genuine_data**
    - `  Q2 2024 , 2201 , 1522 , 34 , Apr 2024 headcount increase` → **genuine_data**

### `sc_gen_200_0058` (format_validity / hard)

- input rows: 11, sft output rows: 10, gold rows: 11
- known_failure_modes (generator-seeded): ['annotation_row', 'malformed_row_width', 'mid_table_separator']
- missing input rows in sft output:
    - `Q1 2023,3180,2187,43,Jan 2023 Q1 targets met` → **genuine_data**
    - `Q2 2023,3265,2250,43,Apr 2023 series funding closed` → **genuine_data**
    - `Q3 2023,3378,2317,44,Jul 2023 partnership signed` → **genuine_data**

### `sc_gen_200_0062` (format_validity / medium)

- input rows: 8, sft output rows: 4, gold rows: 8
- known_failure_modes (generator-seeded): ['annotation_row', 'extra_whitespace']
- missing input rows in sft output:
    - `Q1 2023,987,719,13,Jan 2023 Q1 targets met` → **genuine_data**
    - `  Q2 2023 , 986 , 694 , 13 , Apr 2023 baseline quarter` → **genuine_data**
    - `Q3 2023,1256,877,14,Jul 2023 Q3 targets met` → **genuine_data**
    - `Q4 2023,1313,901,16,Oct 2023 new tooling deployed` → **genuine_data**
    - `Q4 2024,2163,1483,22,Oct 2024 new tooling deployed` → **genuine_data**

### `sc_gen_200_0065` (unit_consistency / medium)

- input rows: 10, sft output rows: 6, gold rows: 10
- known_failure_modes (generator-seeded): ['mixed_units', 'mixed_unit_suffixes']
- missing input rows in sft output:
    - `Q1 2023,1.532 M,1.020 M,23,Jan 2023 headcount increase` → **has_$M_notation**
    - `Q2 2023,1558,983,24,Apr 2023 market expansion` → **genuine_data**
    - `Q3 2023,1806$K,1226,25,Jul 2023 compliance audit passed` → **genuine_data**
    - `Q4 2023,2023,1297,27,Oct 2023 new tooling deployed` → **genuine_data**
    - `Q2 2024,2422,1490,29,Apr 2024 Q2 targets met` → **genuine_data**

### `sc_gen_200_0066` (unit_consistency / hard)

- input rows: 10, sft output rows: 6, gold rows: 10
- known_failure_modes (generator-seeded): ['mixed_units', 'units_hidden_in_notes']
- missing input rows in sft output:
    - `Q1 2023,1831,1300,20,Jan 2023 baseline quarter` → **genuine_data**
    - `Q2 2023,2059,1443,22,Apr 2023 partnership signed; orig $2.059M` → **has_orig_$M_in_notes**
    - `Q3 2023,2124,1445,22,Jul 2023 product launch` → **genuine_data**
    - `Q4 2023,2.373 M,1.770 M,25,Oct 2023 series funding closed` → **has_$M_notation**

### `sc_gen_200_0071` (data_preservation / medium)

- input rows: 9, sft output rows: 7, gold rows: 8
- known_failure_modes (generator-seeded): ['blank_rows', 'duplicated_rows']
- missing input rows in sft output:
    - `Q1 2023,2396,1421,33,Jan 2023 baseline quarter` → **genuine_data**
    - `Q2 2023,2536,1432,35,Apr 2023 partnership signed` → **genuine_data**
    - `Q3 2023,2545,1342,37,Jul 2023 Q3 targets met` → **genuine_data**
    - `Q4 2023,2824,1652,40,Oct 2023 partnership signed` → **genuine_data**

### `sc_gen_200_0077` (data_preservation / medium)

- input rows: 15, sft output rows: 9, gold rows: 14
- known_failure_modes (generator-seeded): ['blank_rows', 'duplicated_rows']
- missing input rows in sft output:
    - `Q1 2023,2563,1520,41,Jan 2023 office lease renewed` → **genuine_data**
    - `Q2 2023,2607,1353,42,Apr 2023 office lease renewed` → **genuine_data**
    - `Q3 2023,2661,1385,43,Jul 2023 market expansion` → **genuine_data**
    - `Q4 2023,2818,1680,43,Oct 2023 market expansion` → **genuine_data**
    - `Q2 2024,3059,1624,44,Apr 2024 office lease renewed` → **genuine_data**
    - `Q3 2024,3312,1855,47,Jul 2024 product launch` → **genuine_data**
    - `Q4 2024,3469,1946,48,Oct 2024 baseline quarter` → **genuine_data**
    - `Q1 2025,3686,2032,49,Jan 2025 partnership signed` → **genuine_data**

## Headline

Across the 20 RH-like cases, of the 113 rows dropped by the SFT adapter:
- **10%** had `$M` notation (raw or hidden in Notes) — could plausibly be "trade-row-for-units"
- **2%** were duplicates or fabricated entries the gold would also drop
- **88%** were genuine input data rows the model failed to preserve

The 88% number is too high to be explained by "the model traded suspicious rows
for unit cleanliness." Inspection of the actual SFT outputs reveals a different
failure mode: **the adapter is mode-collapsed, not selectively row-dropping.**

## Real failure mode: adapter mode collapse

Reading the raw SFT outputs (see `results/collab_sft_v0_eval_raw.jsonl`) shows
that the v0 adapter produces:

1. **Numerical mode collapse.** Revenue/OpEx values across all preserved rows
   converge to a small set of repeated digits — typically `2444,1444` or
   `4444,4444` — regardless of the input values. Example
   (`sc_gen_200_0015`): every output row has `Revenue=4444 OpEx=4444 Headcount=44`
   for the back half of the table.
2. **Year drift.** Quarter labels lose fidelity past the first few rows. `2024`
   becomes `2044`; later still, `4444`. Example (`sc_gen_200_0021`,
   8-row input): output is `Q1 2024 / Q2 2044 / Q3 2044 / Q4 2044` — the model
   produces correctly-shaped CSV but with corrupted years.
3. **Token loops.** At least one case (`sc_gen_200_0035`) emits a literal
   repeating sequence: `Oct 2021 hiring freeze lifted,AP-2444,1444,24,Oct 2021
   hiring freeze lifted,AP-2444,1444,...` until max_tokens.
4. **Notes-cell hallucination.** Note month abbreviations drift (`Apr` → `Aap`,
   `APr`); event labels are reassigned to wrong quarters.

The rows my Phase 1.4 categorization flagged as "genuine_data" were not
**chosen** for deletion by the model — they were **never produced** because
the adapter is in a degenerate generation mode after a few rows. The output is
shaped enough like CSV to score `format_validity = 1.0`, `unit_consistency = 1.0`
(no `$M` patterns appear in mode-collapsed digits), and `completeness = 1.0`
(headers are correct), which is exactly the RH-like signature the harness was
designed to flag.

## Implications for v1

This refines, but does not invalidate, `SFT_ANALYSIS.md` §2.2:

- **Hyperparameter excess is the dominant cause.** Rank 8, alpha 20 (scale=2.5),
  zero dropout, and 1000 iters over 240 examples (≈16 epochs) drove the adapter
  into a degenerate region. The v1 recipe (rank=4, scale=8, dropout=0.05,
  iters=200, lr 5e-5) directly addresses this — and would address it even if
  the dataset were already balanced.
- **Dataset hardening is still needed but is the secondary fix.** Even a
  perfectly-recipied SFT on the current data would only ever demonstrate
  "delete suspicious rows" (per Phase 1.2). A model that learns this signal
  cleanly without mode-collapsing would still be biased toward deletion in
  ambiguous cases — just less catastrophically.
- **The +0.10 promotion gate is unreachable** regardless of recipe; this is
  recorded in `SFT_ANALYSIS.md` §2.2(2) and `collab_sft_v0.md`.
- **A "preservation-stress" eval split is still warranted.** Even after fixing
  mode collapse, we want to detect any residual deletion-bias that survives
  v1's gentler recipe.
