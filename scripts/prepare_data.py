#!/usr/bin/env python3
"""
Validate and prepare Trace FT Lab data files.

What this script does:
  1. Validates JSONL files against expected schemas (SFT, DPO, eval).
  2. Splits SFT and DPO data deterministically into train/val (90/10 by default).
  3. Writes split artifacts to data/processed/.

Usage:
  python scripts/prepare_data.py --help
  python scripts/prepare_data.py \
      --sft data/sample_sft.jsonl \
      --dpo data/sample_dpo.jsonl \
      --eval data/eval_gold.jsonl

  # Custom split ratio and seed:
  python scripts/prepare_data.py --sft data/sample_sft.jsonl --val-ratio 0.15 --seed 99
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any


# ── Schema definitions ────────────────────────────────────────────────────────

# Required top-level keys for each format.
SFT_REQUIRED_KEYS = {"messages"}
DPO_REQUIRED_KEYS = {"prompt", "chosen", "rejected"}
EVAL_REQUIRED_KEYS = {"id", "input", "gold"}

# Required keys inside the gold record of an eval example.
EVAL_GOLD_REQUIRED_KEYS = {"memory_tier", "emotional_valence", "topic_cluster"}

VALID_MEMORY_TIERS = {"semantic", "episodic", "procedural", "prospective"}
VALID_VALENCES = {"positive", "neutral", "negative", "mixed"}


# ── Validation ────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load a JSONL file; return list of parsed objects."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} — invalid JSON: {e}") from e
    return records


def validate_sft_record(record: dict, idx: int, path: str) -> None:
    missing = SFT_REQUIRED_KEYS - record.keys()
    if missing:
        raise ValueError(f"{path}[{idx}] missing keys: {missing}")

    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(
            f"{path}[{idx}] 'messages' must be a list with at least 2 items."
        )
    for msg in messages:
        if "role" not in msg or "content" not in msg:
            raise ValueError(
                f"{path}[{idx}] each message must have 'role' and 'content'."
            )


def validate_dpo_record(record: dict, idx: int, path: str) -> None:
    missing = DPO_REQUIRED_KEYS - record.keys()
    if missing:
        raise ValueError(f"{path}[{idx}] missing keys: {missing}")
    for key in DPO_REQUIRED_KEYS:
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f"{path}[{idx}] '{key}' must be a non-empty string.")


def validate_eval_record(record: dict, idx: int, path: str) -> None:
    missing = EVAL_REQUIRED_KEYS - record.keys()
    if missing:
        raise ValueError(f"{path}[{idx}] missing keys: {missing}")

    gold = record.get("gold", {})
    missing_gold = EVAL_GOLD_REQUIRED_KEYS - gold.keys()
    if missing_gold:
        raise ValueError(f"{path}[{idx}] gold missing keys: {missing_gold}")

    tier = gold.get("memory_tier")
    if tier not in VALID_MEMORY_TIERS:
        raise ValueError(
            f"{path}[{idx}] invalid memory_tier '{tier}'. "
            f"Valid: {VALID_MEMORY_TIERS}"
        )

    valence = gold.get("emotional_valence")
    if valence not in VALID_VALENCES:
        raise ValueError(
            f"{path}[{idx}] invalid emotional_valence '{valence}'. "
            f"Valid: {VALID_VALENCES}"
        )


def validate_file(path: str, format: str) -> list[dict]:
    """Load and validate a JSONL file. Returns records if valid."""
    records = load_jsonl(path)
    if not records:
        raise ValueError(f"{path} is empty.")

    validators = {
        "sft": validate_sft_record,
        "dpo": validate_dpo_record,
        "eval": validate_eval_record,
    }
    validate_fn = validators[format]

    for idx, record in enumerate(records):
        validate_fn(record, idx, path)

    return records


# ── Split ─────────────────────────────────────────────────────────────────────

def split_records(
    records: list[dict], val_ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """
    Deterministic 90/10 (or custom ratio) train/val split.

    Shuffles with a fixed seed, then slices. Preserves reproducibility
    across runs as long as the source file doesn't change.
    """
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)

    n_val = max(1, math.floor(len(shuffled) * val_ratio))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


# ── Output ────────────────────────────────────────────────────────────────────

def write_jsonl(records: list[dict], path: str) -> None:
    """Write list of dicts to a JSONL file, creating parent dirs if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):>4} records → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and split Trace FT Lab JSONL data into train/val artifacts "
            "in data/processed/."
        )
    )
    parser.add_argument("--sft", help="Path to SFT JSONL file.")
    parser.add_argument("--dpo", help="Path to DPO JSONL file.")
    parser.add_argument("--eval", help="Path to eval gold JSONL file.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Fraction of data to use for validation (default: 0.10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to write split artifacts (default: data/processed/).",
    )
    args = parser.parse_args()

    if not any([args.sft, args.dpo, args.eval]):
        parser.print_help()
        print("\nError: provide at least one of --sft, --dpo, --eval.")
        sys.exit(1)

    print(f"\niris-ft-lab data preparation (seed={args.seed}, val_ratio={args.val_ratio})")
    print("─" * 60)

    errors = []

    if args.sft:
        print(f"\nValidating SFT: {args.sft}")
        try:
            records = validate_file(args.sft, "sft")
            print(f"  {len(records)} records valid.")
            train, val = split_records(records, args.val_ratio, args.seed)
            # mlx-lm requires files named exactly train.jsonl / valid.jsonl
            write_jsonl(train, os.path.join(args.output_dir, "sft", "train.jsonl"))
            write_jsonl(val, os.path.join(args.output_dir, "sft", "valid.jsonl"))
        except (ValueError, FileNotFoundError) as e:
            errors.append(str(e))
            print(f"  ERROR: {e}")

    if args.dpo:
        print(f"\nValidating DPO: {args.dpo}")
        try:
            records = validate_file(args.dpo, "dpo")
            print(f"  {len(records)} records valid.")
            train, val = split_records(records, args.val_ratio, args.seed)
            write_jsonl(train, os.path.join(args.output_dir, "dpo", "train.jsonl"))
            write_jsonl(val, os.path.join(args.output_dir, "dpo", "valid.jsonl"))
        except (ValueError, FileNotFoundError) as e:
            errors.append(str(e))
            print(f"  ERROR: {e}")

    if args.eval:
        print(f"\nValidating eval: {args.eval}")
        try:
            records = validate_file(args.eval, "eval")
            print(f"  {len(records)} records valid.")
            # Eval set is not split — copy as-is to processed/
            write_jsonl(records, os.path.join(args.output_dir, "eval_gold.jsonl"))
        except (ValueError, FileNotFoundError) as e:
            errors.append(str(e))
            print(f"  ERROR: {e}")

    print("\n" + "─" * 60)
    if errors:
        print(f"Completed with {len(errors)} error(s). Fix above and re-run.\n")
        sys.exit(1)
    else:
        print("All files validated and written to data/processed/.\n")


if __name__ == "__main__":
    main()
