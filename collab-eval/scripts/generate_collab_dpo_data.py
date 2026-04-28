#!/usr/bin/env python3
"""
Generate DPO preference pairs for collab-eval row-preservation policy.

Mirrors scripts/generate_synthetic_dpo.py structurally for future refactor ease.

Single family: preserve_vs_drop
  Chosen:   gold output (preserves all rows; canonical clean CSV)
  Rejected: gold with a random subset of rows dropped (matches SFT failure mode)

Drop pattern distribution:
  50%: random subset (uniform sampling without replacement)
  30%: contiguous tail (drop last N rows)
  20%: contiguous middle band (drop rows i:j around the table center)

Drop fraction: uniform random in [0.3, 0.7] per case.

Usage:
  python collab-eval/scripts/generate_collab_dpo_data.py \\
      --n 80 --seed 600 --output collab-eval/data/processed/dpo/train.jsonl
  python collab-eval/scripts/generate_collab_dpo_data.py \\
      --n 12 --seed 601 --output collab-eval/data/processed/dpo/valid.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
from pathlib import Path

# ── Ensure collab_eval package is importable ──────────────────────────────────

_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))

from collab_eval.generation.spreadsheet_generator import generate_preservation_stress_cases

# ── Constants ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a data cleaning assistant. Given a messy CSV spreadsheet, produce a "
    "clean version that:\n"
    "- Preserves all original data rows (do not drop rows with real data)\n"
    "- Outputs valid CSV\n"
    "- Normalizes all Revenue and OpEx values to $K (no $M notation; "
    "convert $M values by multiplying by 1000)\n"
    "- Preserves all required columns: Quarter, Revenue, OpEx, Headcount, Notes\n\n"
    "Output ONLY the clean CSV. No explanation or commentary."
)

# Drop pattern weights: (pattern_name, weight)
DROP_PATTERNS = [
    ("random", 50),
    ("tail", 30),
    ("middle", 20),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def case_id_hash(seed: int, case_id: str, idx: int) -> str:
    h = hashlib.sha1(f"{seed}|{case_id}|{idx}".encode()).hexdigest()[:8]
    return f"dpo_pres_{idx:03d}_{h}"


def build_user_content(dirty_csv: str) -> str:
    return (
        "Clean the following spreadsheet CSV so it can be loaded into a data pipeline.\n\n"
        f"{dirty_csv}"
    )


def parse_csv_rows(csv_text: str) -> tuple[list[str], list[list[str]]]:
    """Return (header_row, data_rows) from CSV text. Header is first row."""
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def rows_to_csv(header: list[str], data_rows: list[list[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(data_rows)
    return buf.getvalue()


def drop_rows(
    rng: random.Random,
    data_rows: list[list[str]],
    drop_fraction: float,
    pattern: str,
) -> tuple[list[list[str]], int]:
    """Return (remaining_rows, n_dropped)."""
    n = len(data_rows)
    n_drop = max(1, round(n * drop_fraction))
    n_drop = min(n_drop, n - 1)  # always keep at least 1 row

    if pattern == "random":
        drop_indices = set(rng.sample(range(n), n_drop))
        remaining = [r for i, r in enumerate(data_rows) if i not in drop_indices]
    elif pattern == "tail":
        remaining = data_rows[: n - n_drop]
    elif pattern == "middle":
        center = n // 2
        start = max(0, center - n_drop // 2)
        end = min(n, start + n_drop)
        remaining = data_rows[:start] + data_rows[end:]
    else:
        raise ValueError(f"Unknown drop pattern: {pattern}")

    return remaining, n_drop


def pick_pattern(rng: random.Random) -> str:
    names = [p for p, _ in DROP_PATTERNS]
    weights = [w for _, w in DROP_PATTERNS]
    return rng.choices(names, weights=weights, k=1)[0]


# ── Record generation ─────────────────────────────────────────────────────────

def generate_pair(
    rng: random.Random,
    case,
    idx: int,
    seed: int,
) -> dict | None:
    """Build one DPO pair from a stress case. Returns None if CSV is malformed."""
    import dataclasses
    case_dict = dataclasses.asdict(case) if dataclasses.is_dataclass(case) else case

    gold_csv: str = case_dict["gold_or_reference_output"]
    dirty_csv: str = case_dict["input"]
    case_id: str = case_dict["case_id"]

    header, data_rows = parse_csv_rows(gold_csv)
    if not header or not data_rows:
        return None

    drop_fraction = rng.uniform(0.3, 0.7)
    pattern = pick_pattern(rng)
    remaining_rows, n_dropped = drop_rows(rng, data_rows, drop_fraction, pattern)

    if len(remaining_rows) >= len(data_rows):
        return None  # drop logic failed

    rejected_csv = rows_to_csv(header, remaining_rows)
    prompt = f"{SYSTEM_PROMPT}\n\n{build_user_content(dirty_csv)}"

    return {
        "prompt": prompt,
        "chosen": gold_csv,
        "rejected": rejected_csv,
        "metadata": {
            "case_id": case_id,
            "axis": "preserve_vs_drop",
            "drop_pattern": pattern,
            "drop_fraction": round(drop_fraction, 4),
            "rows_dropped": n_dropped,
            "rows_total": len(data_rows),
            "generation_seed": seed,
        },
    }


# ── Generation ────────────────────────────────────────────────────────────────

def generate(n: int, seed: int) -> list[dict]:
    master_rng = random.Random(seed)

    # Generate enough stress cases to fill n pairs (with some margin for failures)
    n_cases = max(n, n + 10)
    cases = generate_preservation_stress_cases(n=n_cases, seed=seed)

    records: list[dict] = []
    seen_prompts: set[str] = set()

    for i, case in enumerate(cases):
        if len(records) >= n:
            break
        instance_rng = random.Random(master_rng.randint(0, 2**31 - 1))
        pair = generate_pair(instance_rng, case, i, seed)
        if pair is None:
            continue
        if pair["prompt"] in seen_prompts:
            continue
        seen_prompts.add(pair["prompt"])
        records.append(pair)

    return records[:n]


# ── Validation ────────────────────────────────────────────────────────────────

def validate_record(record: dict) -> list[str]:
    errors: list[str] = []
    for field in ("prompt", "chosen", "rejected", "metadata"):
        if field not in record:
            errors.append(f"missing top-level field: {field}")
    if errors:
        return errors

    for field in ("chosen", "rejected"):
        try:
            list(csv.reader(io.StringIO(record[field])))
        except csv.Error as e:
            errors.append(f"{field} does not parse as CSV: {e}")

    try:
        chosen_header, chosen_rows = parse_csv_rows(record["chosen"])
        rejected_header, rejected_rows = parse_csv_rows(record["rejected"])
        if chosen_header != rejected_header:
            errors.append("chosen and rejected have different header rows")
        if len(chosen_rows) <= len(rejected_rows):
            errors.append(
                f"chosen row count ({len(chosen_rows)}) must be > rejected row count ({len(rejected_rows)})"
            )
        if record["chosen"].strip() == record["rejected"].strip():
            errors.append("chosen and rejected are identical")
    except Exception as e:
        errors.append(f"CSV comparison error: {e}")

    md = record.get("metadata", {})
    if not isinstance(md, dict):
        errors.append("metadata is not a dict")
    else:
        if md.get("axis") != "preserve_vs_drop":
            errors.append(f"metadata.axis must be 'preserve_vs_drop', got {md.get('axis')!r}")
        df = md.get("drop_fraction")
        if df is None or not (0.3 <= df <= 0.7):
            errors.append(f"metadata.drop_fraction out of [0.3, 0.7]: {df}")

    return errors


# ── CLI ───────────────────────────────────────────────────────────────────────

def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs for collab-eval row-preservation policy."
    )
    parser.add_argument("--n", type=int, default=80, help="Number of pairs to generate.")
    parser.add_argument("--seed", type=int, default=600, help="RNG seed.")
    parser.add_argument(
        "--output",
        default="data/processed/dpo/train.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    if args.n < 1:
        print("ERROR: --n must be at least 1.", file=sys.stderr)
        sys.exit(2)

    print(f"Generating {args.n} DPO pairs (seed={args.seed}, axis=preserve_vs_drop)")
    records = generate(args.n, args.seed)

    if len(records) < args.n:
        print(
            f"WARNING: only generated {len(records)} pairs (requested {args.n}); "
            f"stress generator may have insufficient cases.",
            file=sys.stderr,
        )

    out = Path(args.output)
    write_jsonl(records, out)
    print(f"Wrote {len(records)} records to {out}")

    # Inline validation
    print("\nRunning inline validation...")
    errs = 0
    for i, rec in enumerate(records):
        rec_errs = validate_record(rec)
        if rec_errs:
            errs += 1
            print(f"  [{i}] ({rec['metadata'].get('case_id', '?')})")
            for e in rec_errs:
                print(f"      - {e}")
    if errs == 0:
        print("  OK: all records validate.")
    else:
        print(f"  FAIL: {errs} records failed validation.")
        sys.exit(1)

    # Pattern distribution summary
    from collections import Counter
    pattern_counts = Counter(r["metadata"]["drop_pattern"] for r in records)
    print("\nDrop pattern distribution:")
    for pat, count in sorted(pattern_counts.items()):
        print(f"  {pat:<10} {count}")
    print()


if __name__ == "__main__":
    main()
