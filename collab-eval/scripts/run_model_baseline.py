"""
Run a base-model inference baseline on held-out spreadsheet-cleaning tasks.

Scores outputs with the existing deterministic composite grader.
Does NOT use an LLM judge. Does NOT add a new grader.

Default model: mlx-community/Qwen2.5-3B-Instruct-4bit

CLI:
  --tasks   path to held-out JSONL
            default: data/generated/spreadsheet_heldout_v1.jsonl
  --model   model identifier
            default: mlx-community/Qwen2.5-3B-Instruct-4bit
  --limit   int, default 20 for smoke test
  --output  path for raw JSONL output
            default: results/model_baseline_v0_raw.jsonl
  --dry-run print config and exit; do not run inference

Behavior when model is unavailable:
  Prints "Model unavailable: <reason>. Skipping inference. No raw results written."
  Exits 0. Does not write raw JSONL.
  May write results/model_baseline_v0.md with "Baseline not yet run. Model unavailable: <reason>."

Baseline decision gate:
  Proceed to actual SFT training only if:
  - composite mean >= 0.30
  - hard-fail rate <= 40%

  This is a provisional engineering gate, not a statistical claim.
  If either condition fails: document the result; do NOT train; the SFT scaffold
  may still be committed as infrastructure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))

from collab_eval.graders import deterministic as det
from collab_eval.graders.composite import weighted_score, apply_hard_fail_cap
from collab_eval.inference.mlx_runner import DEFAULT_MODEL, check_mlx_available, load_model
from collab_eval.generation.spreadsheet_generator import load_jsonl

_DEFAULT_TASKS = str(_ROOT / "data" / "generated" / "spreadsheet_heldout_v1.jsonl")
_DEFAULT_OUTPUT = str(_ROOT / "results" / "model_baseline_v0_raw.jsonl")
_BASELINE_MD = _ROOT / "results" / "model_baseline_v0.md"

_WEIGHTS: dict[str, float] = {
    "data_preservation": 0.35,
    "format_validity": 0.25,
    "unit_consistency": 0.25,
    "completeness": 0.15,
}

_SYSTEM_PROMPT = (
    "You are a data cleaning assistant. Given a messy CSV spreadsheet, produce a "
    "clean version that:\n"
    "- Preserves all original data rows (do not drop rows with real data)\n"
    "- Outputs valid CSV\n"
    "- Normalizes all Revenue and OpEx values to $K (no $M notation)\n"
    "- Preserves all required columns: Quarter, Revenue, OpEx, Headcount, Notes\n\n"
    "Output ONLY the clean CSV. No explanation or commentary."
)


def _build_prompt(case: dict) -> str:
    user_content = (
        "Clean the following spreadsheet CSV so it can be loaded into a data pipeline.\n\n"
        f"{case['input']}"
    )
    return f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


def grade_case(case: dict, output: str) -> tuple[dict[str, float], list[str], float]:
    """
    Grade a model output against a generated case's expected_metadata.

    Uses only the existing deterministic grader. Does not call any LLM.
    Returns (scores, flags, composite).
    """
    meta = case["expected_metadata"]
    expected_rows = meta["expected_row_count"]
    required_cols = set(meta["required_columns"])
    forbidden = meta["unit_normalization"]["forbidden_pattern"]

    parseable = det.csv_parseable(output)
    flags: list[str] = []

    if not parseable:
        flags.append("csv_not_parseable")
        scores = {
            "data_preservation": 0.0,
            "format_validity": 0.0,
            "unit_consistency": 0.0,
            "completeness": 0.0,
        }
        composite = apply_hard_fail_cap(0.0, cap=0.2)
        return scores, flags, composite

    preservation = 1.0 if det.row_count_preserved(output, expected_rows) else 0.0
    unit = 1.0 if det.unit_normalized(output, forbidden) else 0.0
    completeness = det.headers_preserved(output, required_cols)

    scores = {
        "data_preservation": preservation,
        "format_validity": 1.0,
        "unit_consistency": unit,
        "completeness": round(completeness, 3),
    }
    composite = weighted_score(scores, _WEIGHTS)
    if flags:
        composite = apply_hard_fail_cap(composite, cap=0.2)
    return scores, flags, round(composite, 3)


def _write_unavailable_md(reason: str) -> None:
    _BASELINE_MD.parent.mkdir(parents=True, exist_ok=True)
    _BASELINE_MD.write_text(
        f"Baseline not yet run. Model unavailable: {reason}.\n",
        encoding="utf-8",
    )


def _write_baseline_md(results: list[dict]) -> None:
    composites = [r["composite"] for r in results]
    hard_fails = [r for r in results if r["flags"]]

    dim_scores: dict[str, list[float]] = {
        "data_preservation": [],
        "format_validity": [],
        "unit_consistency": [],
        "completeness": [],
    }
    for r in results:
        for dim in dim_scores:
            dim_scores[dim].append(r["scores"].get(dim, 0.0))

    parseability_rate = sum(1 for r in results if "csv_not_parseable" not in r["flags"]) / len(results)
    mean_composite = mean(composites)
    hard_fail_rate = len(hard_fails) / len(results)
    dim_means = {dim: round(mean(vals), 4) for dim, vals in dim_scores.items()}

    gate_composite = mean_composite >= 0.30
    gate_hard_fail = hard_fail_rate <= 0.40
    gate_passed = gate_composite and gate_hard_fail
    gate_verdict = "PASS — proceed to SFT training." if gate_passed else "FAIL — do not train; SFT scaffold may still be committed."

    # Representative failure cases (up to 5)
    failures = [r for r in results if r["composite"] < 0.5][:5]

    lines = [
        "## Model Baseline v0",
        "",
        "**Status:** Baseline run complete.",
        f"**Model:** {results[0].get('model', DEFAULT_MODEL)}",
        f"**Cases evaluated:** {len(results)}",
        "",
        "### Aggregate metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Parseability rate | {parseability_rate:.1%} |",
        f"| Mean composite score | {mean_composite:.4f} |",
        f"| Hard-fail rate | {hard_fail_rate:.1%} |",
        "",
        "### Per-dimension means",
        "",
        "| Dimension | Mean |",
        "|-----------|------|",
        f"| data_preservation | {dim_means['data_preservation']:.4f} |",
        f"| unit_consistency | {dim_means['unit_consistency']:.4f} |",
        f"| format_validity | {dim_means['format_validity']:.4f} |",
        f"| completeness | {dim_means['completeness']:.4f} |",
        "",
        "### Baseline decision gate",
        "",
        "Proceed to actual SFT training only if composite mean >= 0.30 AND hard-fail rate <= 40%.",
        "",
        f"| Gate condition | Value | Status |",
        f"|----------------|-------|--------|",
        f"| composite mean >= 0.30 | {mean_composite:.4f} | {'PASS' if gate_composite else 'FAIL'} |",
        f"| hard-fail rate <= 40% | {hard_fail_rate:.1%} | {'PASS' if gate_hard_fail else 'FAIL'} |",
        f"| Overall | | **{gate_verdict}** |",
        "",
        "### Notes",
        "",
        "This is a base-model baseline, not a fine-tuned result. "
        "No gradient updates have been applied to the model.",
    ]

    if failures:
        lines += ["", "### Representative failure cases", ""]
        for r in failures:
            lines += [
                f"**Case:** {r['case_id']}  ",
                f"Composite: {r['composite']} | Flags: {r['flags']}  ",
                f"Scores: {r['scores']}  ",
                "",
            ]

    # Reward-hacking-like behavior check
    rh_candidates = [
        r for r in results
        if r["scores"].get("format_validity", 0) >= 0.9
        and r["scores"].get("data_preservation", 1) < 0.5
    ]
    if rh_candidates:
        lines += [
            "### Reward-hacking-like behavior",
            "",
            f"{len(rh_candidates)} cases scored high on format_validity but low on "
            "data_preservation — consistent with row-dropping behavior.",
            "",
        ]
    else:
        lines += [
            "### Reward-hacking-like behavior",
            "",
            "No obvious reward-hacking patterns observed in this run.",
            "",
        ]

    _BASELINE_MD.parent.mkdir(parents=True, exist_ok=True)
    _BASELINE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote baseline report → {_BASELINE_MD}")


def run(
    tasks_path: str,
    model_id: str,
    limit: int,
    output_path: str,
    dry_run: bool,
) -> None:
    if dry_run:
        print("=== dry-run: config only, no inference ===")
        print(f"  tasks:  {tasks_path}")
        print(f"  model:  {model_id}")
        print(f"  limit:  {limit}")
        print(f"  output: {output_path}")
        return

    # Check model availability before loading tasks
    avail, reason = check_mlx_available()
    if not avail:
        msg = f"Model unavailable: {reason}. Skipping inference. No raw results written."
        print(msg)
        _write_unavailable_md(reason)
        return

    # Load tasks
    tasks_file = Path(tasks_path)
    if not tasks_file.exists():
        print(f"Tasks file not found: {tasks_path}")
        print("Run: python scripts/generate_tasks.py --task spreadsheet_clean --n 80 --seed 200 --output data/generated/spreadsheet_heldout_v1.jsonl")
        sys.exit(1)

    cases = load_jsonl(tasks_path)
    if limit > 0:
        cases = cases[:limit]
    print(f"Loaded {len(cases)} cases from {tasks_path}")

    # Attempt model load
    try:
        model, tokenizer = load_model(model_id)
    except RuntimeError as exc:
        reason = str(exc)
        msg = f"Model unavailable: {reason}. Skipping inference. No raw results written."
        print(msg)
        _write_unavailable_md(reason)
        return

    # Run inference and score
    from collab_eval.inference.mlx_runner import generate as mlx_generate

    results: list[dict] = []
    for i, case in enumerate(cases):
        prompt = _build_prompt(case)
        try:
            output = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=512)
        except Exception as exc:
            print(f"  [{i+1}/{len(cases)}] inference error for {case['case_id']}: {exc}")
            output = ""

        scores, flags, composite = grade_case(case, output)
        results.append({
            "case_id": case["case_id"],
            "model": model_id,
            "output": output,
            "scores": scores,
            "flags": flags,
            "composite": composite,
        })
        print(f"  [{i+1}/{len(cases)}] {case['case_id']}: composite={composite:.3f} flags={flags}")

    # Write raw JSONL
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} raw results → {out_path}")

    # Write report
    _write_baseline_md(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run base-model inference baseline on held-out tasks."
    )
    parser.add_argument(
        "--tasks",
        default=_DEFAULT_TASKS,
        help=f"Held-out JSONL path (default: {_DEFAULT_TASKS})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max cases to evaluate (default: 20 for smoke test; 0 = no limit)",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help=f"Raw JSONL output path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit; do not run inference.",
    )
    args = parser.parse_args()
    run(
        tasks_path=args.tasks,
        model_id=args.model,
        limit=args.limit,
        output_path=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
