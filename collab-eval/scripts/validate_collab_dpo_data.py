#!/usr/bin/env python3
"""
Standalone validator for collab-eval DPO preference data.

Mirrors scripts/validate_dpo_data.py structurally. Adapted for the
preserve_vs_drop axis: chosen/rejected are CSV text, not JSON records.

Validates each record:
  - Top-level fields: prompt, chosen, rejected, metadata
  - prompt is a non-empty string
  - chosen and rejected parse as valid CSV
  - chosen and rejected share the same header row
  - chosen row count > rejected row count (preservation axis)
  - chosen and rejected differ beyond whitespace
  - metadata.axis == "preserve_vs_drop"
  - metadata.drop_fraction in [0.3, 0.7]
  - No two records share the same prompt (deduplicated by hash)

Exit codes:
  0 — all records validate
  1 — at least one record has errors
  2 — file not found / unreadable

Usage:
  python collab-eval/scripts/validate_collab_dpo_data.py data/processed/dpo/train.jsonl
  python collab-eval/scripts/validate_collab_dpo_data.py data/processed/dpo/valid.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path


def parse_csv(text: str) -> tuple[bool, str, list[str], list[list[str]]]:
    """Return (ok, error_msg, header, data_rows)."""
    try:
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
    except csv.Error as e:
        return False, str(e), [], []
    if not rows:
        return False, "empty CSV", [], []
    return True, "", rows[0], rows[1:]


def validate_record(record: dict) -> list[str]:
    errors: list[str] = []

    for field in ("prompt", "chosen", "rejected", "metadata"):
        if field not in record:
            errors.append(f"missing top-level field: {field}")
    if errors:
        return errors

    if not isinstance(record["prompt"], str) or not record["prompt"].strip():
        errors.append("prompt is empty or not a string")

    chosen_ok, chosen_err, chosen_header, chosen_rows = parse_csv(record["chosen"])
    if not chosen_ok:
        errors.append(f"chosen does not parse as CSV: {chosen_err}")

    rejected_ok, rejected_err, rejected_header, rejected_rows = parse_csv(record["rejected"])
    if not rejected_ok:
        errors.append(f"rejected does not parse as CSV: {rejected_err}")

    if chosen_ok and rejected_ok:
        if chosen_header != rejected_header:
            errors.append(
                f"chosen and rejected have different headers: {chosen_header!r} vs {rejected_header!r}"
            )
        if len(chosen_rows) <= len(rejected_rows):
            errors.append(
                f"chosen row count ({len(chosen_rows)}) must be > rejected row count "
                f"({len(rejected_rows)}) for preserve_vs_drop axis"
            )
        if record["chosen"].strip() == record["rejected"].strip():
            errors.append("chosen and rejected are identical (differ only on whitespace or not at all)")

    md = record.get("metadata", {})
    if not isinstance(md, dict):
        errors.append("metadata is not an object")
    else:
        if md.get("axis") != "preserve_vs_drop":
            errors.append(
                f"metadata.axis must be 'preserve_vs_drop', got {md.get('axis')!r}"
            )
        df = md.get("drop_fraction")
        if df is None or not isinstance(df, (int, float)) or not (0.3 <= float(df) <= 0.7):
            errors.append(f"metadata.drop_fraction out of [0.3, 0.7]: {df!r}")

    return errors


def prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate collab-eval DPO preference JSONL data."
    )
    parser.add_argument("path", help="Path to JSONL file to validate.")
    args = parser.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"ERROR: file not found: {p}", file=sys.stderr)
        return 2

    total = 0
    errors_count = 0
    pattern_counts: dict[str, int] = {}
    seen_prompt_hashes: dict[str, int] = {}

    print(f"Validating {p}")
    with p.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [line {lineno}] JSONL parse error: {e}")
                errors_count += 1
                continue

            errs = validate_record(record)
            if errs:
                errors_count += 1
                for e in errs:
                    print(f"  [line {lineno}] {e}")
                continue

            ph = prompt_hash(record["prompt"])
            if ph in seen_prompt_hashes:
                print(
                    f"  [line {lineno}] duplicate prompt (first seen at line {seen_prompt_hashes[ph]})"
                )
                errors_count += 1
            else:
                seen_prompt_hashes[ph] = lineno

            md = record["metadata"]
            pat = md.get("drop_pattern", "unknown")
            pattern_counts[pat] = pattern_counts.get(pat, 0) + 1

    print()
    print(f"Records:             {total}")
    print(f"Records with errors: {errors_count}")
    print()
    print("Drop pattern distribution:")
    for pat, count in sorted(pattern_counts.items()):
        print(f"  {pat:<10} {count}")
    print()

    if errors_count == 0:
        print("OK: all records validate.")
        return 0
    print(f"FAIL: {errors_count} record(s) had errors.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
