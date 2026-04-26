"""
Compare base model vs. SFT adapter on held-out spreadsheet-cleaning tasks.

Uses ONLY the held-out task split. Scores with the existing deterministic
composite grader. Does not add a new grader. Does not use an LLM judge.

Promotion gate (also in README and results/collab_sft_v0.md):
  Promote SFT adapter only if ALL are true:
  - composite improves by >= 0.10 over base
  - hard-fail rate does not increase vs. base
  - format_validity does not regress vs. base
  - no obvious increase in reward-hacking behavior

CLI:
  --model-base   base model identifier
                 default: mlx-community/Qwen2.5-3B-Instruct-4bit
  --adapter      path to SFT adapter (optional; if omitted, base-only eval)
  --tasks        path to held-out JSONL
                 default: data/generated/spreadsheet_heldout_v1.jsonl
  --output       path to write comparison JSONL
                 default: results/collab_sft_v0_eval_raw.jsonl
  --limit        max cases (default: 0 = no limit)
  --dry-run      print config and exit

Metrics reported:
  - composite
  - data_preservation
  - format_validity
  - unit_consistency
  - completeness
  - hard-fail rate
  - parseability rate
  - reward-hacking-like behavior (format_validity high, data_preservation low)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))

from collab_eval.generation.spreadsheet_generator import load_jsonl
from collab_eval.inference.mlx_runner import DEFAULT_MODEL, check_mlx_available, load_model
from scripts.run_model_baseline import grade_case, _build_prompt, _write_unavailable_md

_DEFAULT_TASKS = str(_ROOT / "data" / "generated" / "spreadsheet_heldout_v1.jsonl")
_DEFAULT_OUTPUT = str(_ROOT / "results" / "collab_sft_v0_eval_raw.jsonl")
_SFT_MD = _ROOT / "results" / "collab_sft_v0.md"

_COMPOSITE_IMPROVEMENT_THRESHOLD = 0.10
_HARD_FAIL_MAX_INCREASE = 0.0  # must not increase
_FORMAT_VALIDITY_MIN = 0.0     # must not regress


def _run_inference(model, tokenizer, cases: list[dict]) -> list[dict]:
    from collab_eval.inference.mlx_runner import generate as mlx_generate
    results = []
    for i, case in enumerate(cases):
        prompt = _build_prompt(case)
        try:
            output = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=512)
        except Exception as exc:
            print(f"  [{i+1}/{len(cases)}] error: {exc}")
            output = ""
        scores, flags, composite = grade_case(case, output)
        results.append({
            "case_id": case["case_id"],
            "output": output,
            "scores": scores,
            "flags": flags,
            "composite": composite,
        })
        print(f"  [{i+1}/{len(cases)}] {case['case_id']}: composite={composite:.3f}")
    return results


def _aggregate(results: list[dict]) -> dict:
    composites = [r["composite"] for r in results]
    dims = ["data_preservation", "format_validity", "unit_consistency", "completeness"]
    dim_means = {}
    for dim in dims:
        vals = [r["scores"].get(dim, 0.0) for r in results]
        dim_means[dim] = round(mean(vals), 4)
    hard_fail_rate = sum(1 for r in results if r["flags"]) / len(results)
    parseability = sum(1 for r in results if "csv_not_parseable" not in r["flags"]) / len(results)
    rh_count = sum(
        1 for r in results
        if r["scores"].get("format_validity", 0) >= 0.9
        and r["scores"].get("data_preservation", 1) < 0.5
    )
    return {
        "composite_mean": round(mean(composites), 4),
        "hard_fail_rate": round(hard_fail_rate, 4),
        "parseability_rate": round(parseability, 4),
        "dim_means": dim_means,
        "reward_hacking_count": rh_count,
        "n": len(results),
    }


def _check_promotion_gate(base_agg: dict, sft_agg: dict) -> tuple[bool, list[str]]:
    issues: list[str] = []

    composite_delta = sft_agg["composite_mean"] - base_agg["composite_mean"]
    if composite_delta < _COMPOSITE_IMPROVEMENT_THRESHOLD:
        issues.append(
            f"composite improvement {composite_delta:+.4f} < required {_COMPOSITE_IMPROVEMENT_THRESHOLD:+.2f}"
        )

    hf_delta = sft_agg["hard_fail_rate"] - base_agg["hard_fail_rate"]
    if hf_delta > _HARD_FAIL_MAX_INCREASE:
        issues.append(
            f"hard-fail rate increased by {hf_delta:+.4f}"
        )

    fv_delta = (
        sft_agg["dim_means"]["format_validity"] - base_agg["dim_means"]["format_validity"]
    )
    if fv_delta < 0:
        issues.append(
            f"format_validity regressed by {fv_delta:.4f}"
        )

    rh_delta = sft_agg["reward_hacking_count"] - base_agg["reward_hacking_count"]
    if rh_delta > 0:
        issues.append(
            f"reward-hacking-like cases increased by {rh_delta}"
        )

    return len(issues) == 0, issues


def _write_sft_md_no_results() -> None:
    _SFT_MD.parent.mkdir(parents=True, exist_ok=True)
    _SFT_MD.write_text(
        "SFT scaffold ready. Training has not been run. No results to report.\n",
        encoding="utf-8",
    )


def _write_sft_md_with_results(
    base_agg: dict,
    sft_agg: dict | None,
    gate_passed: bool | None,
    gate_issues: list[str],
    model_base: str,
    adapter: str | None,
) -> None:
    lines = [
        "## SFT Eval v0",
        "",
        f"**Base model:** {model_base}",
        f"**Adapter:** {adapter or 'none (base-only eval)'}",
        f"**Cases evaluated:** {base_agg['n']}",
        "",
        "### Base model results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mean composite | {base_agg['composite_mean']:.4f} |",
        f"| Hard-fail rate | {base_agg['hard_fail_rate']:.1%} |",
        f"| Parseability rate | {base_agg['parseability_rate']:.1%} |",
        f"| data_preservation | {base_agg['dim_means']['data_preservation']:.4f} |",
        f"| format_validity | {base_agg['dim_means']['format_validity']:.4f} |",
        f"| unit_consistency | {base_agg['dim_means']['unit_consistency']:.4f} |",
        f"| completeness | {base_agg['dim_means']['completeness']:.4f} |",
        f"| RH-like cases | {base_agg['reward_hacking_count']} |",
    ]

    if sft_agg is not None:
        composite_delta = sft_agg["composite_mean"] - base_agg["composite_mean"]
        lines += [
            "",
            "### SFT adapter results",
            "",
            "| Metric | Base | SFT | Delta |",
            "|--------|------|-----|-------|",
            f"| Mean composite | {base_agg['composite_mean']:.4f} | {sft_agg['composite_mean']:.4f} | {composite_delta:+.4f} |",
            f"| Hard-fail rate | {base_agg['hard_fail_rate']:.1%} | {sft_agg['hard_fail_rate']:.1%} | {sft_agg['hard_fail_rate'] - base_agg['hard_fail_rate']:+.1%} |",
            f"| data_preservation | {base_agg['dim_means']['data_preservation']:.4f} | {sft_agg['dim_means']['data_preservation']:.4f} | {sft_agg['dim_means']['data_preservation'] - base_agg['dim_means']['data_preservation']:+.4f} |",
            f"| format_validity | {base_agg['dim_means']['format_validity']:.4f} | {sft_agg['dim_means']['format_validity']:.4f} | {sft_agg['dim_means']['format_validity'] - base_agg['dim_means']['format_validity']:+.4f} |",
            f"| unit_consistency | {base_agg['dim_means']['unit_consistency']:.4f} | {sft_agg['dim_means']['unit_consistency']:.4f} | {sft_agg['dim_means']['unit_consistency'] - base_agg['dim_means']['unit_consistency']:+.4f} |",
            f"| completeness | {base_agg['dim_means']['completeness']:.4f} | {sft_agg['dim_means']['completeness']:.4f} | {sft_agg['dim_means']['completeness'] - base_agg['dim_means']['completeness']:+.4f} |",
            f"| RH-like cases | {base_agg['reward_hacking_count']} | {sft_agg['reward_hacking_count']} | {sft_agg['reward_hacking_count'] - base_agg['reward_hacking_count']:+d} |",
            "",
            "### Promotion gate",
            "",
            "Promote adapter only if ALL are true:",
            "- composite improves by >= 0.10 over base",
            "- hard-fail rate does not increase",
            "- format_validity does not regress",
            "- no obvious increase in reward-hacking behavior",
            "",
        ]
        if gate_passed:
            lines += [
                "**Gate: PASS** — adapter is experimental; small-N caveat applies.",
                "N = {}, which is insufficient for statistical claims.".format(base_agg['n']),
            ]
        else:
            lines += [
                "**Gate: FAIL** — adapter is not promoted.",
                "",
                "Reasons:",
            ]
            for issue in gate_issues:
                lines.append(f"- {issue}")
    else:
        lines += [
            "",
            "No SFT adapter evaluated in this run.",
        ]

    _SFT_MD.parent.mkdir(parents=True, exist_ok=True)
    _SFT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote SFT eval report → {_SFT_MD}")


def run(
    model_base: str,
    adapter: str | None,
    tasks_path: str,
    output_path: str,
    limit: int,
    dry_run: bool,
) -> None:
    if dry_run:
        print("=== dry-run: config only ===")
        print(f"  model-base: {model_base}")
        print(f"  adapter:    {adapter or 'none'}")
        print(f"  tasks:      {tasks_path}")
        print(f"  output:     {output_path}")
        print(f"  limit:      {limit}")
        return

    avail, reason = check_mlx_available()
    if not avail:
        print(f"Model unavailable: {reason}. Skipping inference.")
        _write_sft_md_no_results()
        return

    tasks_file = Path(tasks_path)
    if not tasks_file.exists():
        print(f"Tasks file not found: {tasks_path}")
        sys.exit(1)

    cases = load_jsonl(tasks_path)
    if limit > 0:
        cases = cases[:limit]
    print(f"Loaded {len(cases)} cases")

    try:
        print(f"Loading base model: {model_base}")
        base_model, base_tokenizer = load_model(model_base)
    except RuntimeError as exc:
        print(f"Model unavailable: {exc}. Skipping inference.")
        _write_sft_md_no_results()
        return

    print("Running base model inference...")
    base_results = _run_inference(base_model, base_tokenizer, cases)
    base_agg = _aggregate(base_results)

    sft_results = None
    sft_agg = None
    gate_passed = None
    gate_issues: list[str] = []

    if adapter:
        try:
            print(f"Loading adapter: {adapter}")
            from mlx_lm import load as mlx_load
            sft_model, sft_tokenizer = mlx_load(model_base, adapter_path=adapter)
            print("Running SFT adapter inference...")
            sft_results = _run_inference(sft_model, sft_tokenizer, cases)
            sft_agg = _aggregate(sft_results)
            gate_passed, gate_issues = _check_promotion_gate(base_agg, sft_agg)
        except Exception as exc:
            print(f"Failed to load adapter {adapter!r}: {exc}")

    # Write raw output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in base_results:
            f.write(json.dumps({"split": "base", **r}, ensure_ascii=False) + "\n")
        if sft_results:
            for r in sft_results:
                f.write(json.dumps({"split": "sft", **r}, ensure_ascii=False) + "\n")
    print(f"Wrote raw results → {out_path}")

    _write_sft_md_with_results(base_agg, sft_agg, gate_passed, gate_issues, model_base, adapter)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare base model vs. SFT adapter on held-out tasks."
    )
    parser.add_argument(
        "--model-base",
        default=DEFAULT_MODEL,
        help=f"Base model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to SFT adapter (optional)",
    )
    parser.add_argument(
        "--tasks",
        default=_DEFAULT_TASKS,
        help=f"Held-out JSONL path (default: {_DEFAULT_TASKS})",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help=f"Raw JSONL output path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max cases (default: 0 = no limit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit.",
    )
    args = parser.parse_args()
    run(
        model_base=args.model_base,
        adapter=args.adapter,
        tasks_path=args.tasks,
        output_path=args.output,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
