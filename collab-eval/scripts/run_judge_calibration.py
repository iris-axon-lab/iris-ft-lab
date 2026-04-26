"""
Judge / rubric calibration runner.

Validates the seed calibration set at data/judge_calibration/seed_calibration_v1.jsonl
and optionally runs the LLM judge if ANTHROPIC_API_KEY is set.

Always runs offline:
  - Schema validation of all calibration records
  - Deterministic sanity checks (band/score consistency)

Runs if ANTHROPIC_API_KEY is set:
  - LLM judge scoring per dimension
  - Per-dimension expected-band agreement
  - Disagreement and borderline case report

Gate:
  - Agreement >= 0.80 per dimension → dimension marked calibrated
  - Agreement 0.70-0.79 → warn, exclude from training signal claims
  - Agreement < 0.70 → mark uncalibrated, hard exclude

Writes results/judge_calibration_v1.md.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))

DATA_PATH = _ROOT / "data" / "judge_calibration" / "seed_calibration_v1.jsonl"
RESULTS_PATH = _ROOT / "results" / "judge_calibration_v1.md"

REQUIRED_FIELDS = [
    "case_id", "task_type", "dimension", "input", "output",
    "rubric", "expected_score", "expected_score_band", "rationale", "known_failure_mode",
]
VALID_TASK_TYPES = {"doc_revision", "spreadsheet_clean", "citation_ground"}
VALID_DIMENSIONS = {"faithfulness", "quality_delta", "citation_accurate", "hallucination_flag"}
VALID_BANDS = {"low", "medium", "high"}

AGREEMENT_GATE = 0.80
WARN_THRESHOLD = 0.70


def load_cases() -> list[dict]:
    cases: list[dict] = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"  ERROR: Failed to parse line {i}: {exc}", file=sys.stderr)
    return cases


def validate_schema(cases: list[dict]) -> list[str]:
    errors: list[str] = []
    for case in cases:
        cid = case.get("case_id", f"<line {cases.index(case)+1}>")
        for field in REQUIRED_FIELDS:
            if field not in case:
                errors.append(f"{cid}: missing field '{field}'")
        if case.get("task_type") not in VALID_TASK_TYPES:
            errors.append(f"{cid}: invalid task_type '{case.get('task_type')}'")
        if case.get("dimension") not in VALID_DIMENSIONS:
            errors.append(f"{cid}: invalid dimension '{case.get('dimension')}'")
        if case.get("expected_score_band") not in VALID_BANDS:
            errors.append(f"{cid}: invalid expected_score_band '{case.get('expected_score_band')}'")
        try:
            s = float(case.get("expected_score", -1))
            if not (0.0 <= s <= 1.0):
                errors.append(f"{cid}: expected_score {s} out of [0.0, 1.0]")
        except (TypeError, ValueError):
            errors.append(f"{cid}: expected_score is not a float")
    return errors


def sanity_checks(cases: list[dict]) -> list[str]:
    """Deterministic checks: band/score consistency."""
    issues: list[str] = []
    for case in cases:
        cid = case.get("case_id", "?")
        band = case.get("expected_score_band")
        try:
            score = float(case.get("expected_score", 0.5))
        except (TypeError, ValueError):
            continue
        if band == "low" and score > 0.45:
            issues.append(f"{cid}: band=low but expected_score={score:.2f} > 0.45")
        elif band == "high" and score < 0.55:
            issues.append(f"{cid}: band=high but expected_score={score:.2f} < 0.55")
        elif band == "medium" and (score < 0.3 or score > 0.75):
            issues.append(f"{cid}: band=medium but expected_score={score:.2f} outside [0.3, 0.75]")
    return issues


def _score_to_band(score: float) -> str:
    if score < 0.4:
        return "low"
    elif score < 0.7:
        return "medium"
    return "high"


def run_llm_calibration(cases: list[dict]) -> dict[str, list[dict]]:
    """
    Run LLM judge on all calibration cases.
    Returns per-dimension results list.
    Each result: {case_id, expected_band, actual_score, actual_band, match, reasoning}
    """
    from collab_eval.graders.llm_judge import score_dimension, LLMJudgeUnavailableError

    results: dict[str, list[dict]] = {}
    total = len(cases)
    for i, case in enumerate(cases, 1):
        dim = case["dimension"]
        cid = case["case_id"]
        print(f"  Scoring {cid} ({dim}) [{i}/{total}]...")
        try:
            result = score_dimension(
                dimension=dim,
                spec_intent=f"Score the '{dim}' dimension of this agent output.",
                input_doc=case["input"],
                agent_output=case["output"],
                source_docs=None,
            )
            actual_score = result.score
            actual_band = _score_to_band(actual_score)
            expected_band = case["expected_score_band"]
            results.setdefault(dim, []).append({
                "case_id": cid,
                "expected_band": expected_band,
                "actual_score": actual_score,
                "actual_band": actual_band,
                "match": actual_band == expected_band,
                "reasoning": result.reasoning,
            })
        except Exception as exc:
            print(f"  WARNING: LLM judge failed for {cid}: {exc}", file=sys.stderr)
            results.setdefault(dim, []).append({
                "case_id": cid,
                "expected_band": case["expected_score_band"],
                "actual_score": None,
                "actual_band": None,
                "match": False,
                "reasoning": f"error: {exc}",
            })
    return results


def compute_agreement(llm_results: dict[str, list[dict]]) -> dict[str, float]:
    agreement: dict[str, float] = {}
    for dim, results in llm_results.items():
        valid = [r for r in results if r["actual_band"] is not None]
        if not valid:
            agreement[dim] = 0.0
        else:
            matches = sum(1 for r in valid if r["match"])
            agreement[dim] = round(matches / len(valid), 3)
    return agreement


def dimension_gate_status(agreement: dict[str, float]) -> dict[str, str]:
    status: dict[str, str] = {}
    for dim, rate in agreement.items():
        if rate >= AGREEMENT_GATE:
            status[dim] = "calibrated"
        elif rate >= WARN_THRESHOLD:
            status[dim] = "warn_exclude_from_training_signal"
        else:
            status[dim] = "uncalibrated_hard_exclude"
    return status


def write_report(
    cases: list[dict],
    schema_errors: list[str],
    sanity_issues: list[str],
    llm_results: dict[str, list[dict]] | None,
    agreement: dict[str, float] | None,
    gate_status: dict[str, str] | None,
    llm_ran: bool,
) -> None:
    lines: list[str] = []
    lines.append("## Judge Calibration v1 — Seed Calibration Report\n")
    lines.append(f"**Cases:** {len(cases)}")
    lines.append(f"**Dimensions covered:** faithfulness, quality_delta, citation_accurate, hallucination_flag")
    lines.append(f"**LLM judge ran:** {'yes' if llm_ran else 'no (ANTHROPIC_API_KEY not set)'}")
    lines.append("")

    lines.append("---\n")
    lines.append("### Schema validation\n")
    if schema_errors:
        lines.append(f"**{len(schema_errors)} error(s) found:**\n")
        for e in schema_errors:
            lines.append(f"- {e}")
    else:
        lines.append("All records pass schema validation.")
    lines.append("")

    lines.append("### Deterministic sanity checks\n")
    if sanity_issues:
        lines.append(f"**{len(sanity_issues)} issue(s) found:**\n")
        for issue in sanity_issues:
            lines.append(f"- {issue}")
    else:
        lines.append("All sanity checks pass (band/score consistency OK).")
    lines.append("")

    if not llm_ran:
        lines.append("### LLM judge agreement\n")
        lines.append("LLM judge not run (no API key). All LLM-required dimensions are **unassessed**.\n")
        lines.append("To run with LLM judge:\n")
        lines.append("```bash")
        lines.append("ANTHROPIC_API_KEY=sk-... python scripts/run_judge_calibration.py")
        lines.append("```\n")
        lines.append("### Gate status (offline)\n")
        lines.append("| Dimension | Status |")
        lines.append("|-----------|--------|")
        for dim in sorted(VALID_DIMENSIONS):
            lines.append(f"| {dim} | unassessed (no API key) |")
        lines.append("")
    else:
        lines.append("### LLM judge agreement\n")
        lines.append("| Dimension | Cases | Agreement | Gate |")
        lines.append("|-----------|-------|-----------|------|")
        for dim in sorted(agreement or {}):
            rate = (agreement or {})[dim]
            status = (gate_status or {}).get(dim, "unknown")
            n = len((llm_results or {}).get(dim, []))
            gate_str = "✓ calibrated" if status == "calibrated" else ("⚠ warn" if "warn" in status else "✗ uncalibrated")
            lines.append(f"| {dim} | {n} | {rate:.3f} | {gate_str} |")
        lines.append("")

        if llm_results:
            lines.append("### Disagreements and borderline cases\n")
            for dim, results in sorted((llm_results or {}).items()):
                mismatches = [r for r in results if not r["match"] and r["actual_band"] is not None]
                if mismatches:
                    lines.append(f"**{dim}** — {len(mismatches)} disagreement(s):\n")
                    for r in mismatches:
                        lines.append(
                            f"- `{r['case_id']}`: expected={r['expected_band']}, "
                            f"actual={r['actual_band']} (score={r['actual_score']:.2f}). "
                            f"Reasoning: {r['reasoning']}"
                        )
                    lines.append("")

    lines.append("### Calibration set coverage\n")
    dim_counts: dict[str, int] = {}
    for case in cases:
        d = case.get("dimension", "?")
        dim_counts[d] = dim_counts.get(d, 0) + 1
    lines.append("| Dimension | Cases |")
    lines.append("|-----------|-------|")
    for d, c in sorted(dim_counts.items()):
        lines.append(f"| {d} | {c} |")
    lines.append("")

    lines.append("### Note on calibration terminology\n")
    lines.append(
        "This is **seed calibration** — the expected scores reflect the artifact creator's "
        "best judgment of what a well-calibrated LLM judge should produce for these synthetic "
        "examples. This is NOT human-calibrated data. The expected scores serve as a sanity "
        "check (do obvious low-quality outputs score low?) rather than a gold standard.\n"
    )
    lines.append(
        "For any dimension where the LLM judge agreement is below the gate threshold "
        f"({AGREEMENT_GATE:.0%}), that dimension's scores should be excluded from training "
        "signal claims. See `docs/rl_env_design.md` for calibration rationale."
    )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")


def main() -> None:
    print("=== Judge Calibration Runner ===\n")

    if not DATA_PATH.exists():
        print(f"ERROR: Calibration data not found at {DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading calibration cases from {DATA_PATH}...")
    cases = load_cases()
    print(f"  Loaded {len(cases)} cases.\n")

    print("Running schema validation...")
    schema_errors = validate_schema(cases)
    if schema_errors:
        for e in schema_errors:
            print(f"  ERROR: {e}")
        print(f"\n{len(schema_errors)} schema error(s). Stopping.", file=sys.stderr)
        sys.exit(1)
    print("  OK — all records valid.\n")

    print("Running deterministic sanity checks...")
    sanity_issues = sanity_checks(cases)
    if sanity_issues:
        for issue in sanity_issues:
            print(f"  WARN: {issue}")
    else:
        print("  OK — all sanity checks pass.\n")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    llm_ran = False
    llm_results = None
    agreement = None
    gate_status = None

    if api_key:
        print("ANTHROPIC_API_KEY found — running LLM judge calibration...")
        llm_results = run_llm_calibration(cases)
        agreement = compute_agreement(llm_results)
        gate_status = dimension_gate_status(agreement)

        print("\nAgreement results:")
        for dim, rate in sorted(agreement.items()):
            status = gate_status[dim]
            print(f"  {dim}: {rate:.1%} → {status}")
        llm_ran = True
    else:
        print("ANTHROPIC_API_KEY not set — skipping LLM judge.")
        print("Offline validation only. LLM dimensions marked 'unassessed'.\n")

    write_report(
        cases=cases,
        schema_errors=schema_errors,
        sanity_issues=sanity_issues,
        llm_results=llm_results,
        agreement=agreement,
        gate_status=gate_status,
        llm_ran=llm_ran,
    )

    if schema_errors or sanity_issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
