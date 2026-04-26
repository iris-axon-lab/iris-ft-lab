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

CLI:
  --input    JSONL file of generated cases (from generate_tasks.py)
             default: data/generated/spreadsheet_train_v1.jsonl
  --output   JSONL file to write SFT records to
             default: data/sft_collab_eval_full.jsonl
  --limit    max records to emit (0 = no limit)
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

_DEFAULT_INPUT = str(_ROOT / "data" / "generated" / "spreadsheet_train_v1.jsonl")
# MLX-LM expects a directory with train.jsonl, not a flat .jsonl file.
_DEFAULT_OUTPUT = str(_ROOT / "data" / "sft_collab_eval_full" / "train.jsonl")
_SAMPLE_OUTPUT = _ROOT / "data" / "sft_collab_eval_sample.jsonl"
_SAMPLE_SIZE = 20

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


def run(input_path: str, output_path: str, limit: int, write_sample: bool) -> None:
    cases = load_jsonl(input_path)
    if not cases:
        print(f"No cases found in {input_path}")
        print("Generate first: python scripts/generate_tasks.py --task spreadsheet_clean --n 240 --seed 100 --output data/generated/spreadsheet_train_v1.jsonl")
        print("Then: python scripts/build_sft_data.py  (writes to data/sft_collab_eval_full/train.jsonl)")
        sys.exit(1)

    if limit > 0:
        cases = cases[:limit]

    records = [case_to_sft_record(c) for c in cases]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} SFT records → {out}")

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
        "--input",
        default=_DEFAULT_INPUT,
        help=f"Input JSONL of generated cases (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help=f"Output JSONL for SFT records (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max records to emit (default: 0 = no limit)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help=f"Also write the first {_SAMPLE_SIZE} records to {_SAMPLE_OUTPUT}.",
    )
    args = parser.parse_args()
    run(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        write_sample=args.sample,
    )


if __name__ == "__main__":
    main()
