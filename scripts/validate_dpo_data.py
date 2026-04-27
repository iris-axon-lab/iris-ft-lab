#!/usr/bin/env python3
"""
Standalone validator for synthetic DPO preference data.

Mirrors the inline validator inside generate_synthetic_dpo.py so the same
checks can be run on any JSONL file that claims to follow the Trace DPO
format. Useful as a Phase 2.3 smoke check before training.

Validates each record:
  - Top-level fields: prompt, chosen, rejected, metadata
  - chosen and rejected parse as JSON objects
  - Both contain all required Trace fields (memory_tier, content_summary,
    stated_intent, emotional_valence, topic_cluster, timestamp, source,
    source_id, channel)
  - memory_tier values are in VALID_TIERS
  - emotional_valence values are in VALID_VALENCES
  - chosen and rejected differ on at least one of:
    memory_tier, stated_intent, content_summary
  - metadata contains axis, target_tier, generation_seed, case_id

Usage:
  python scripts/validate_dpo_data.py data/processed/dpo/train.jsonl
  python scripts/validate_dpo_data.py data/processed/dpo/valid.jsonl

Exit codes:
  0 — all records validate
  1 — at least one record has errors
  2 — file not found / unreadable
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED_FIELDS = (
    "memory_tier",
    "content_summary",
    "stated_intent",
    "emotional_valence",
    "topic_cluster",
    "timestamp",
    "source",
    "source_id",
    "channel",
)

VALID_TIERS = {"episodic", "semantic", "procedural", "prospective"}
VALID_VALENCES = {"positive", "neutral", "negative", "mixed"}
KNOWN_AXES = {
    "over_flatten",
    "add_coaching",
    "mis_tier_mixed",
    "conditional_commitment",
    "schema_drift",
}


def parse_record_field(record: dict, field: str) -> tuple[bool, str]:
    raw = record.get(field, "")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f"{field} is not valid JSON: {e}"
    if not isinstance(parsed, dict):
        return False, f"{field} did not parse to a JSON object"
    missing = [f for f in REQUIRED_FIELDS if f not in parsed]
    if missing:
        return False, f"{field} missing required fields: {missing}"
    if parsed["memory_tier"] not in VALID_TIERS:
        return False, f"{field} has invalid memory_tier: {parsed['memory_tier']}"
    if parsed["emotional_valence"] not in VALID_VALENCES:
        return False, f"{field} has invalid emotional_valence: {parsed['emotional_valence']}"
    return True, ""


def validate_record(record: dict) -> list[str]:
    errors: list[str] = []
    for required_top_level in ("prompt", "chosen", "rejected", "metadata"):
        if required_top_level not in record:
            errors.append(f"missing top-level field: {required_top_level}")
    if errors:
        return errors

    if not isinstance(record["prompt"], str) or not record["prompt"].strip():
        errors.append("prompt is empty or not a string")

    chosen_ok, chosen_msg = parse_record_field(record, "chosen")
    if not chosen_ok:
        errors.append(chosen_msg)
    rejected_ok, rejected_msg = parse_record_field(record, "rejected")
    if not rejected_ok:
        errors.append(rejected_msg)

    if chosen_ok and rejected_ok:
        chosen = json.loads(record["chosen"])
        rejected = json.loads(record["rejected"])
        axis = record["metadata"].get("axis") if isinstance(record["metadata"], dict) else None
        if axis in KNOWN_AXES:
            differs_on = (
                chosen["memory_tier"] != rejected["memory_tier"]
                or chosen["stated_intent"] != rejected["stated_intent"]
                or chosen["content_summary"] != rejected["content_summary"]
            )
            if not differs_on:
                errors.append(
                    f"chosen and rejected do not differ on memory_tier, stated_intent, "
                    f"or content_summary (axis: {axis})"
                )

    md = record.get("metadata", {})
    if not isinstance(md, dict):
        errors.append("metadata is not an object")
    else:
        for k in ("axis", "target_tier", "generation_seed", "case_id"):
            if k not in md:
                errors.append(f"metadata missing field: {k}")
        if "axis" in md and md["axis"] not in KNOWN_AXES:
            errors.append(f"metadata.axis has unknown value: {md['axis']}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate synthetic DPO preference JSONL data."
    )
    parser.add_argument("path", help="Path to JSONL file to validate.")
    parser.add_argument("--strict-axes", action="store_true",
                        help="Require metadata.axis to be one of the known axes "
                             "(off by default — unknown axes only warn).")
    args = parser.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"ERROR: file not found: {p}", file=sys.stderr)
        return 2

    total = 0
    errors_count = 0
    axis_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    duplicate_prompts: dict[str, int] = {}

    print(f"Validating {p}")
    with p.open() as f:
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

            md = record["metadata"]
            axis_counts[md["axis"]] += 1
            chosen = json.loads(record["chosen"])
            tier_counts[chosen["memory_tier"]] += 1
            duplicate_prompts[record["prompt"]] = duplicate_prompts.get(record["prompt"], 0) + 1

    dupes = {p: c for p, c in duplicate_prompts.items() if c > 1}
    if dupes:
        print(f"  WARNING: {len(dupes)} duplicate prompts found "
              f"(max repeats: {max(dupes.values())})")

    print()
    print(f"Records:          {total}")
    print(f"Records with errors: {errors_count}")
    print()
    print("Per-axis counts:")
    for axis, count in sorted(axis_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {axis:<28} {count}")
    print()
    print("Chosen.memory_tier counts:")
    for tier, count in sorted(tier_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {tier:<28} {count}")
    print()

    if errors_count == 0:
        print("OK: all records validate.")
        return 0
    print(f"FAIL: {errors_count} record(s) had errors.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
