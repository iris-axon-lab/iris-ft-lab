"""
Compare base model vs. SFT adapter on held-out spreadsheet-cleaning tasks.

Uses ONLY the held-out task split. Scores with the existing deterministic
composite grader. Does not add a new grader. Does not use an LLM judge.

Promotion gate (revised in v1 — see configs/sft_collab_eval_qwen25_3b.yaml
and SFT_ANALYSIS.md §2 for context):
  Promote SFT adapter if ALL are true:
  - composite_mean       >= base_composite - 0.005      (no regression)
  - data_preservation    >= base_data_preservation       (no regression)
  - rh_like_count        <= base_rh_like_count           (no increase)
  - at least one of {unit_consistency, format_validity, completeness}
    improves by >= 0.02                                  (forward progress)
  - preservation-stress data_preservation_mean >= 0.85   (new bar)

The 5th condition requires running this script a second time against the
preservation-stress eval set and passing the per-case JSON via
--stress-results, which extracts data_preservation_mean and applies the gate.
If --stress-results is not provided, the 5th condition is reported as
"not evaluated" and the gate is conservatively marked FAIL.

CLI:
  --model-base       base model identifier
                     default: mlx-community/Qwen2.5-3B-Instruct-4bit
  --adapter          path to SFT adapter (optional; if omitted, base-only eval)
  --eval-set/--tasks path to held-out JSONL
                     default: data/generated/spreadsheet_heldout_v1.jsonl
  --output           path to write raw comparison JSONL (per-case)
                     default: results/collab_sft_v1_eval_raw.jsonl
  --emit-per-case    optional path to write per-case results as a single JSON
                     file (alongside --output JSONL). Used to pipe stress
                     results into a subsequent --stress-results call.
  --stress-results   optional path to a per-case JSON from a prior run on
                     the preservation-stress eval set; enables the 5th gate.
  --report-path      optional path for the markdown report
                     default: results/collab_sft_v1.md
  --limit            max cases (default: 0 = no limit)
  --dry-run          print config and exit

This script never writes to results/collab_sft_v0.md or
results/collab_sft_v0_eval_raw.jsonl — those are immutable v0 records.

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
_DEFAULT_OUTPUT = str(_ROOT / "results" / "collab_sft_v1_eval_raw.jsonl")
_DEFAULT_REPORT = str(_ROOT / "results" / "collab_sft_v1.md")

# v1 promotion-gate thresholds (see module docstring).
_COMPOSITE_REGRESSION_TOLERANCE = 0.005   # composite_delta >= -0.005
_DIMENSION_IMPROVEMENT_MIN = 0.02         # one of {unit, fv, completeness} >= +0.02
# Why 0.85? The stress split is the gate's *promotion bar*, not the
# adapter's expected next-step score. Base model scores 0.20 on stress; v1
# reached 0.25. We require 0.85 because below that, the adapter is still
# materially worse than gold (which always preserves all rows by construction
# → data_preservation=1.00). A 0.50 bar would let an adapter promote while
# still dropping half the rows on long tables — the failure mode this gate
# exists to prevent. Revise only with explicit justification in collab_sft_v2.md.
_STRESS_DATA_PRESERVATION_MIN = 0.85      # stress eval data_preservation_mean >= 0.85
_IMPROVEMENT_DIMENSIONS = ("unit_consistency", "format_validity", "completeness")


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


def _check_promotion_gate(
    base_agg: dict,
    sft_agg: dict,
    stress_data_preservation: float | None = None,
) -> tuple[bool, list[str], list[str]]:
    """v1 5-condition gate. Returns (passed, issues, condition_lines).

    `stress_data_preservation` is the data_preservation_mean from running the
    same adapter on the preservation-stress eval set. If None, condition #5
    is reported as "not evaluated" and the gate fails conservatively.
    """
    issues: list[str] = []
    conditions: list[str] = []

    # 1. composite no regression
    composite_delta = sft_agg["composite_mean"] - base_agg["composite_mean"]
    cond1_pass = composite_delta >= -_COMPOSITE_REGRESSION_TOLERANCE
    conditions.append(
        f"composite_mean delta {composite_delta:+.4f} "
        f">= -{_COMPOSITE_REGRESSION_TOLERANCE:.3f}: "
        f"{'PASS' if cond1_pass else 'FAIL'}"
    )
    if not cond1_pass:
        issues.append(
            f"composite regressed: delta {composite_delta:+.4f} below "
            f"tolerance -{_COMPOSITE_REGRESSION_TOLERANCE:.3f}"
        )

    # 2. data_preservation no regression
    dp_delta = (
        sft_agg["dim_means"]["data_preservation"]
        - base_agg["dim_means"]["data_preservation"]
    )
    cond2_pass = dp_delta >= 0.0
    conditions.append(
        f"data_preservation delta {dp_delta:+.4f} >= 0: "
        f"{'PASS' if cond2_pass else 'FAIL'}"
    )
    if not cond2_pass:
        issues.append(f"data_preservation regressed by {dp_delta:.4f}")

    # 3. rh_like_count no increase
    rh_delta = sft_agg["reward_hacking_count"] - base_agg["reward_hacking_count"]
    cond3_pass = rh_delta <= 0
    conditions.append(
        f"rh_like_count delta {rh_delta:+d} <= 0: "
        f"{'PASS' if cond3_pass else 'FAIL'}"
    )
    if not cond3_pass:
        issues.append(f"reward-hacking-like cases increased by {rh_delta}")

    # 4. at least one of {unit_consistency, format_validity, completeness}
    #    improves by >= 0.02
    best_dim = None
    best_delta = -1.0
    for dim in _IMPROVEMENT_DIMENSIONS:
        delta = sft_agg["dim_means"][dim] - base_agg["dim_means"][dim]
        if delta > best_delta:
            best_delta = delta
            best_dim = dim
    cond4_pass = best_delta >= _DIMENSION_IMPROVEMENT_MIN
    conditions.append(
        f"best of {{{', '.join(_IMPROVEMENT_DIMENSIONS)}}} "
        f"delta {best_delta:+.4f} (dim={best_dim}) "
        f">= +{_DIMENSION_IMPROVEMENT_MIN:.2f}: "
        f"{'PASS' if cond4_pass else 'FAIL'}"
    )
    if not cond4_pass:
        issues.append(
            f"no improvement dimension reached +{_DIMENSION_IMPROVEMENT_MIN:.2f} "
            f"(best: {best_dim}={best_delta:+.4f})"
        )

    # 5. preservation-stress eval data_preservation_mean >= 0.85
    if stress_data_preservation is None:
        cond5_pass = False
        conditions.append(
            f"stress data_preservation_mean: NOT EVALUATED "
            f"(pass --stress-results to enable)"
        )
        issues.append(
            "preservation-stress eval not evaluated (gate condition 5 missing)"
        )
    else:
        cond5_pass = stress_data_preservation >= _STRESS_DATA_PRESERVATION_MIN
        conditions.append(
            f"stress data_preservation_mean {stress_data_preservation:.4f} "
            f">= {_STRESS_DATA_PRESERVATION_MIN:.2f}: "
            f"{'PASS' if cond5_pass else 'FAIL'}"
        )
        if not cond5_pass:
            issues.append(
                f"preservation-stress data_preservation_mean "
                f"{stress_data_preservation:.4f} < "
                f"{_STRESS_DATA_PRESERVATION_MIN:.2f}"
            )

    return (
        cond1_pass and cond2_pass and cond3_pass and cond4_pass and cond5_pass,
        issues,
        conditions,
    )


def _load_stress_per_case(path: str) -> float:
    """Load a per-case JSON file (from a prior --emit-per-case run) and
    return data_preservation_mean across cases.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cases = data["cases"] if isinstance(data, dict) and "cases" in data else data
    vals = [c["scores"]["data_preservation"] for c in cases]
    return mean(vals) if vals else 0.0


def _write_sft_md_no_results(report_path: str) -> None:
    p = Path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "SFT scaffold ready. Training has not been run. No results to report.\n",
        encoding="utf-8",
    )


def _write_sft_md_with_results(
    base_agg: dict,
    sft_agg: dict | None,
    gate_passed: bool | None,
    gate_issues: list[str],
    gate_conditions: list[str],
    model_base: str,
    adapter: str | None,
    report_path: str,
    stress_data_preservation: float | None = None,
) -> None:
    lines = [
        "## SFT Eval v1",
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
            "### Promotion gate (v1)",
            "",
            "Promote adapter only if ALL are true:",
            f"- composite_mean does not regress by more than {_COMPOSITE_REGRESSION_TOLERANCE:.3f}",
            "- data_preservation does not regress",
            "- rh_like_count does not increase",
            f"- best of {{{', '.join(_IMPROVEMENT_DIMENSIONS)}}} improves by >= {_DIMENSION_IMPROVEMENT_MIN:.2f}",
            f"- preservation-stress data_preservation_mean >= {_STRESS_DATA_PRESERVATION_MIN:.2f}",
            "",
            "Per-condition results:",
            "",
        ]
        for cond in gate_conditions:
            lines.append(f"- {cond}")
        lines.append("")
        if stress_data_preservation is not None:
            lines.append(
                f"Stress eval data_preservation_mean: {stress_data_preservation:.4f}"
            )
            lines.append("")
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

    p = Path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote SFT eval report → {p}")


def run(
    model_base: str,
    adapter: str | None,
    tasks_path: str,
    output_path: str,
    report_path: str,
    limit: int,
    dry_run: bool,
    emit_per_case: str | None = None,
    stress_results: str | None = None,
) -> None:
    if dry_run:
        print("=== dry-run: config only ===")
        print(f"  model-base:      {model_base}")
        print(f"  adapter:         {adapter or 'none'}")
        print(f"  tasks:           {tasks_path}")
        print(f"  output:          {output_path}")
        print(f"  report-path:     {report_path}")
        print(f"  emit-per-case:   {emit_per_case or 'none'}")
        print(f"  stress-results:  {stress_results or 'none'}")
        print(f"  limit:           {limit}")
        return

    avail, reason = check_mlx_available()
    if not avail:
        print(f"Model unavailable: {reason}. Skipping inference.")
        _write_sft_md_no_results(report_path)
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
        _write_sft_md_no_results(report_path)
        return

    print("Running base model inference...")
    base_results = _run_inference(base_model, base_tokenizer, cases)
    base_agg = _aggregate(base_results)

    sft_results = None
    sft_agg = None
    gate_passed = None
    gate_issues: list[str] = []
    gate_conditions: list[str] = []
    stress_dp: float | None = None

    if adapter:
        try:
            print(f"Loading adapter: {adapter}")
            from mlx_lm import load as mlx_load
            sft_model, sft_tokenizer = mlx_load(model_base, adapter_path=adapter)
            print("Running SFT adapter inference...")
            sft_results = _run_inference(sft_model, sft_tokenizer, cases)
            sft_agg = _aggregate(sft_results)
            if stress_results:
                stress_dp = _load_stress_per_case(stress_results)
                print(f"Loaded stress eval data_preservation_mean: {stress_dp:.4f}")
            gate_passed, gate_issues, gate_conditions = _check_promotion_gate(
                base_agg, sft_agg, stress_data_preservation=stress_dp
            )
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

    if emit_per_case:
        per_case_path = Path(emit_per_case)
        per_case_path.parent.mkdir(parents=True, exist_ok=True)
        per_case_results = sft_results if sft_results is not None else base_results
        per_case_payload = {
            "model_base": model_base,
            "adapter": adapter,
            "tasks_path": tasks_path,
            "split": "sft" if sft_results is not None else "base",
            "cases": per_case_results,
        }
        per_case_path.write_text(
            json.dumps(per_case_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote per-case JSON → {per_case_path}")

    _write_sft_md_with_results(
        base_agg=base_agg,
        sft_agg=sft_agg,
        gate_passed=gate_passed,
        gate_issues=gate_issues,
        gate_conditions=gate_conditions,
        model_base=model_base,
        adapter=adapter,
        report_path=report_path,
        stress_data_preservation=stress_dp,
    )


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
        "--eval-set",
        dest="tasks",
        default=_DEFAULT_TASKS,
        help=f"Held-out JSONL path (default: {_DEFAULT_TASKS})",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help=f"Raw JSONL output path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--report-path",
        default=_DEFAULT_REPORT,
        help=f"Markdown report path (default: {_DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--emit-per-case",
        default=None,
        help=(
            "Optional path to write per-case results as a single JSON file "
            "(used to feed --stress-results in a follow-up call)."
        ),
    )
    parser.add_argument(
        "--stress-results",
        default=None,
        help=(
            "Optional path to a per-case JSON file from a prior preservation-stress "
            "run; enables the 5th gate condition."
        ),
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
        report_path=args.report_path,
        limit=args.limit,
        dry_run=args.dry_run,
        emit_per_case=args.emit_per_case,
        stress_results=args.stress_results,
    )


if __name__ == "__main__":
    main()
