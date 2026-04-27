"""
Build SFT training data for spreadsheet-cleaning tasks.

Gold output derivation rule:
  The gold output is derived DETERMINISTICALLY from the synthetic task generator.
  The generator created the dirty input from a known clean/reference version.
  This script recovers that known clean output from the `gold_or_reference_output`
  field already stored in each generated case.

  NO model is called. No LLM, no Ollama, no MLX, no API.

SFT record schema:
  {
    "messages": [
      {"role": "system", "content": "<spreadsheet cleanup system prompt>"},
      {"role": "user", "content": "<task input / dirty spreadsheet spec>"},
      {"role": "assistant", "content": "<deterministic gold clean output>"}
    ],
    "metadata": {
      "task_type": "spreadsheet_clean",
      "difficulty": "easy|medium|hard",
      "primary_dimension": "unit_consistency|data_preservation|format_validity|completeness",
      "generation_seed": <int>,
      "case_id": "<str>"
    }
  }

Inputs:
  By default this script reads cases from BOTH the regular generator
  (data/generated/spreadsheet_train_v1.jsonl, seed=100) AND the
  preservation-stress generator (data/generated/spreadsheet_train_stress_v1.jsonl,
  seed=400). Pass --inputs to override.

Train/valid split (added in v1):
  After concatenating all inputs, cases are sorted by case_id. Every 10th
  case in the sorted order goes to valid.jsonl; the remainder goes to
  train.jsonl. This is a deterministic 90/10 split with no per-run
  randomness — re-running this script produces byte-identical files. Stress
  cases (case_id prefix `sc_stress_`) interleave with regular cases (prefix
  `sc_gen_`) in the sort, so both train and valid contain a mix.

CLI:
  --inputs   one or more JSONL files of generated cases (comma-separated).
             default: spreadsheet_train_v1.jsonl,spreadsheet_train_stress_v1.jsonl
  --output-dir  directory to write train.jsonl and valid.jsonl
                default: data/sft_collab_eval_full
  --limit    max records to emit total (0 = no limit)
  --sample   if set, write <= 20 records to data/sft_collab_eval_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))

from collab_eval.generation.spreadsheet_generator import load_jsonl

_DEFAULT_INPUTS = [
    str(_ROOT / "data" / "generated" / "spreadsheet_train_v1.jsonl"),
    str(_ROOT / "data" / "generated" / "spreadsheet_train_stress_v1.jsonl"),
]
# MLX-LM expects a directory with train.jsonl and (optionally) valid.jsonl.
_DEFAULT_OUTPUT_DIR = str(_ROOT / "data" / "sft_collab_eval_full")
_SAMPLE_OUTPUT = _ROOT / "data" / "sft_collab_eval_sample.jsonl"
_SAMPLE_SIZE = 20
_VALID_FRACTION_DENOM = 10  # 1 in N goes to valid; 90/10 split when N=10

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


def _build_user_content(case: dict) -> str:
    return (
        "Clean the following spreadsheet CSV so it can be loaded into a data pipeline.\n\n"
        f"{case['input']}"
    )


def case_to_sft_record(case: dict) -> dict:
    """
    Convert a generated case dict to an SFT record.

    The gold assistant output is taken directly from `gold_or_reference_output`,
    which was produced deterministically by the generator — no model involved.
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_content(case)},
            {"role": "assistant", "content": case["gold_or_reference_output"]},
        ],
        "metadata": {
            "task_type": "spreadsheet_clean",
            "difficulty": case["difficulty"],
            "primary_dimension": case["primary_dimension"],
            "generation_seed": case["generation_seed"],
            "case_id": case["case_id"],
        },
    }


def build_sft_records_from_generator(n: int = 20, seed: int = 999) -> list[dict]:
    """
    Generate n cases on-the-fly and return SFT records.
    Used to build the committed sample file without needing a pre-generated JSONL.
    """
    from collab_eval.generation.spreadsheet_generator import generate_cases
    cases = generate_cases(n=n, seed=seed)
    return [case_to_sft_record(vars(c) if not isinstance(c, dict) else c) for c in cases]


def _dataclass_to_dict(case) -> dict:
    """Convert GeneratedCase dataclass to dict."""
    import dataclasses
    if dataclasses.is_dataclass(case):
        return dataclasses.asdict(case)
    return case


def _split_train_valid(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Deterministic 90/10 split by case_id.

    Sort all records by metadata.case_id, then take every Nth (N=10) into the
    valid set; the rest go to train. Re-running with the same inputs produces
    byte-identical splits.
    """
    sorted_records = sorted(records, key=lambda r: r["metadata"]["case_id"])
    train: list[dict] = []
    valid: list[dict] = []
    for i, rec in enumerate(sorted_records):
        if i % _VALID_FRACTION_DENOM == 0:
            valid.append(rec)
        else:
            train.append(rec)
    return train, valid


def run(input_paths: list[str], output_dir: str, limit: int, write_sample: bool) -> None:
    all_cases: list[dict] = []
    for p in input_paths:
        cases = load_jsonl(p)
        if not cases:
            print(f"No cases found in {p}")
            print(
                "Regenerate inputs first:\n"
                "  python scripts/generate_tasks.py --task spreadsheet_clean "
                "--n 240 --seed 100 --output data/generated/spreadsheet_train_v1.jsonl\n"
                "  python scripts/generate_tasks.py --task spreadsheet_clean_stress "
                "--n 80 --seed 400 --output "
                "data/generated/spreadsheet_train_stress_v1.jsonl"
            )
            sys.exit(1)
        all_cases.extend(cases)
        print(f"Loaded {len(cases)} cases from {p}")

    if limit > 0:
        all_cases = all_cases[:limit]

    records = [case_to_sft_record(c) for c in all_cases]

    train_records, valid_records = _split_train_valid(records)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with valid_path.open("w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train_records)} records → {train_path}")
    print(f"Wrote {len(valid_records)} records → {valid_path}")

    if write_sample:
        sample = records[:_SAMPLE_SIZE]
        _SAMPLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with _SAMPLE_OUTPUT.open("w", encoding="utf-8") as f:
            for r in sample:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(sample)} sample records → {_SAMPLE_OUTPUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT training data from generated cases.")
    parser.add_argument(
        "--inputs",
        default=",".join(_DEFAULT_INPUTS),
        help=(
            "Comma-separated JSONL inputs (default: regular train + stress train)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory for train.jsonl + valid.jsonl (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max records to emit total (default: 0 = no limit)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help=f"Also write the first {_SAMPLE_SIZE} records to {_SAMPLE_OUTPUT}.",
    )
    args = parser.parse_args()
    input_paths = [p.strip() for p in args.inputs.split(",") if p.strip()]
    run(
        input_paths=input_paths,
        output_dir=args.output_dir,
        limit=args.limit,
        write_sample=args.sample,
    )


if __name__ == "__main__":
    main()
