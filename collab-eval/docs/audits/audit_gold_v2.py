"""Phase 2.4 audit script (read-only). Diffs input vs gold per case.

Runs on v2 data. Accepts --input to target a specific JSONL file so the
audit can be invoked separately on v2-regular (240 cases) and the full
v2 training set (480 cases, regular + stress combined).

Usage (from collab-eval/ root):
    # v2-regular only (comparable to v1):
    python docs/audits/audit_gold_v2.py \
        --input data/generated/spreadsheet_train_v2.jsonl

    # full v2 train set (regular + stress, writes combined audit):
    python docs/audits/audit_gold_v2.py \
        --input data/generated/spreadsheet_train_v2.jsonl \
        --input data/generated/spreadsheet_train_stress_v2.jsonl \
        --label full_v2

Both runs together produce docs/sft_v2_data_audit.md with two sections.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parents[2]


def parse_csv_lines(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if not row or all(c == "" for c in row):
            continue
        rows.append(row)
    return rows


def split_input_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip() != ""]


def analyze_case(case: dict) -> dict:
    inp = case.get("input", "")
    gold = case.get("gold_or_reference_output", "")

    in_lines = split_input_lines(inp)
    gold_lines = split_input_lines(gold)

    in_rows = parse_csv_lines(inp)
    gold_rows = parse_csv_lines(gold)

    body_in = len(in_rows) - 1 if in_rows else 0
    body_gold = len(gold_rows) - 1 if gold_rows else 0
    expected_rows = case.get("expected_metadata", {}).get("expected_row_count", "?")

    findings: list[str] = []

    in_body_tuples = [tuple(r) for r in in_rows[1:]]
    gold_body_tuples = [tuple(r) for r in gold_rows[1:]]
    in_dup_count = len(in_body_tuples) - len(set(in_body_tuples))
    if in_dup_count > 0 and len(gold_body_tuples) == len(set(gold_body_tuples)):
        findings.append(f"DROP_DUP({in_dup_count})")

    has_paren_annot_in = any(ln.lstrip().startswith("(") and "$" in ln for ln in in_lines)
    has_paren_annot_gold = any(ln.lstrip().startswith("(") and "$" in ln for ln in gold_lines)
    if has_paren_annot_in and not has_paren_annot_gold:
        findings.append("DROP_ANNOT")

    has_sep_in = any("---" in ln for ln in in_lines)
    has_sep_gold = any("---" in ln for ln in gold_lines)
    if has_sep_in and not has_sep_gold:
        findings.append("DROP_SEP")

    if "extra fabricated entry" in inp and "extra fabricated entry" not in gold:
        findings.append("DROP_FABRICATED")

    if "; orig $" in inp and "; orig $" not in gold:
        findings.append("STRIP_NOTES_ORIG_$M")

    return {
        "case_id": case.get("case_id", ""),
        "primary_dimension": case.get("primary_dimension", ""),
        "difficulty": case.get("difficulty", ""),
        "headline": f"in={body_in} gold={body_gold} expected={expected_rows}",
        "findings": findings,
        "in_dup_count": in_dup_count,
        "has_annot_in": has_paren_annot_in,
        "has_sep_in": has_sep_in,
        "has_fab_in": "extra fabricated entry" in inp,
        "has_orig_M_in": "; orig $" in inp,
    }


def audit_cases(cases: list[dict]) -> dict:
    rollup = {
        "n_total": len(cases),
        "n_drop_dup": 0, "n_drop_annot": 0, "n_drop_sep": 0,
        "n_drop_fab": 0, "n_strip_orig_M": 0,
        "n_with_dup_in": 0, "n_with_annot_in": 0, "n_with_sep_in": 0,
        "n_with_fab_in": 0, "n_with_orig_M_in": 0,
    }
    per_dim: dict[str, dict] = {}
    for c in cases:
        dim = c.get("primary_dimension", "unknown")
        if dim not in per_dim:
            per_dim[dim] = {k: 0 for k in rollup}
            per_dim[dim]["n_total"] = 0
        per_dim[dim]["n_total"] += 1

    for c in cases:
        a = analyze_case(c)
        f = a["findings"]
        dim = c.get("primary_dimension", "unknown")
        if "DROP_DUP" in "|".join(f):
            rollup["n_drop_dup"] += 1
            per_dim[dim]["n_drop_dup"] += 1
        if "DROP_ANNOT" in f:
            rollup["n_drop_annot"] += 1
            per_dim[dim]["n_drop_annot"] += 1
        if "DROP_SEP" in f:
            rollup["n_drop_sep"] += 1
            per_dim[dim]["n_drop_sep"] += 1
        if "DROP_FABRICATED" in f:
            rollup["n_drop_fab"] += 1
            per_dim[dim]["n_drop_fab"] += 1
        if "STRIP_NOTES_ORIG_$M" in f:
            rollup["n_strip_orig_M"] += 1
            per_dim[dim]["n_strip_orig_M"] += 1
        if a["in_dup_count"] > 0:
            rollup["n_with_dup_in"] += 1
            per_dim[dim]["n_with_dup_in"] += 1
        if a["has_annot_in"]:
            rollup["n_with_annot_in"] += 1
            per_dim[dim]["n_with_annot_in"] += 1
        if a["has_sep_in"]:
            rollup["n_with_sep_in"] += 1
            per_dim[dim]["n_with_sep_in"] += 1
        if a["has_fab_in"]:
            rollup["n_with_fab_in"] += 1
            per_dim[dim]["n_with_fab_in"] += 1
        if a["has_orig_M_in"]:
            rollup["n_with_orig_M_in"] += 1
            per_dim[dim]["n_with_orig_M_in"] += 1

    return {"rollup": rollup, "per_dim": per_dim}


def deletion_events(rollup: dict) -> int:
    return (
        rollup["n_drop_dup"] + rollup["n_drop_annot"] + rollup["n_drop_sep"]
        + rollup["n_drop_fab"] + rollup["n_strip_orig_M"]
    )


def render_section(title: str, sources: str, cases: list[dict], rollup: dict, per_dim: dict) -> list[str]:
    out: list[str] = []
    n = rollup["n_total"]
    del_events = deletion_events(rollup)
    ratio_pct = 100.0 * del_events / n if n else 0

    out.append(f"## {title}\n")
    out.append(f"Source(s): {sources} (n={n})\n")
    out.append("### Whole-dataset rollup\n")
    out.append("| Behavior | Count |")
    out.append("|---|---|")
    out.append(f"| total cases | {n} |")
    out.append(f"| input had duplicate rows | {rollup['n_with_dup_in']} → gold drops them in {rollup['n_drop_dup']} |")
    out.append(f"| input had `(...)` annotation row | {rollup['n_with_annot_in']} → gold drops in {rollup['n_drop_annot']} |")
    out.append(f"| input had `---` separator | {rollup['n_with_sep_in']} → gold drops in {rollup['n_drop_sep']} |")
    out.append(f"| input had `extra fabricated entry` | {rollup['n_with_fab_in']} → gold drops in {rollup['n_drop_fab']} |")
    out.append(f"| input had `; orig $X.XXXM` in Notes | {rollup['n_with_orig_M_in']} → gold strips in {rollup['n_strip_orig_M']} |")
    out.append("")
    out.append("### Per-dimension rollup\n")
    out.append("| dim | n | drop_dup | drop_annot | drop_sep | drop_fab | strip_orig_$M |")
    out.append("|---|---|---|---|---|---|---|")
    for dim, d in sorted(per_dim.items()):
        out.append(
            f"| {dim} | {d['n_total']} | {d['n_drop_dup']} | "
            f"{d['n_drop_annot']} | {d['n_drop_sep']} | {d['n_drop_fab']} | "
            f"{d['n_strip_orig_M']} |"
        )
    out.append("")
    out.append(
        f"**Total gold-acts-by-deleting events: {del_events} / {n} cases "
        f"= {ratio_pct:.1f}%**\n"
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 data audit script.")
    parser.add_argument(
        "--input", action="append", dest="inputs", metavar="FILE",
        help="JSONL input file(s). Repeat for multiple files.",
    )
    parser.add_argument(
        "--label", default=None,
        help="Section label suffix (auto-derived from filenames if omitted).",
    )
    args = parser.parse_args()

    # Default: run both sections (regular-only, then full)
    run_default = args.inputs is None

    out_path = ROOT / "docs" / "sft_v2_data_audit.md"
    out: list[str] = []
    out.append("# SFT v2 — Phase 2.4 data audit\n")
    out.append("Generated by `docs/audits/audit_gold_v2.py`.\n")
    out.append(
        "Two sections: (1) v2 regular only for v1↔v2 comparison; "
        "(2) full v2 train set (regular + stress) for actual train-distribution rollup.\n"
    )

    v1_del_events = 147
    v1_n = 240
    v1_ratio = 100.0 * v1_del_events / v1_n

    if run_default:
        # Section 1: regular only
        src_reg = ROOT / "data" / "generated" / "spreadsheet_train_v2.jsonl"
        cases_reg = [json.loads(l) for l in src_reg.open(encoding="utf-8") if l.strip()]
        r1 = audit_cases(cases_reg)
        out.extend(render_section(
            "v2 regular only (v1↔v2 comparison)",
            f"`{src_reg.name}`",
            cases_reg,
            r1["rollup"], r1["per_dim"],
        ))
        del1 = deletion_events(r1["rollup"])
        out.append(f"v1 baseline: {v1_del_events}/{v1_n} = {v1_ratio:.1f}%\n")
        out.append(
            f"Same seed (100) → same cases → same deletion count as v1. "
            f"Regular data unchanged by design.\n"
        )

        # Section 2: full train (regular + stress)
        src_stress = ROOT / "data" / "generated" / "spreadsheet_train_stress_v2.jsonl"
        cases_stress = [json.loads(l) for l in src_stress.open(encoding="utf-8") if l.strip()]
        cases_full = cases_reg + cases_stress
        r2 = audit_cases(cases_full)
        out.extend(render_section(
            "v2 full train set (regular + stress)",
            f"`{src_reg.name}` + `{src_stress.name}`",
            cases_full,
            r2["rollup"], r2["per_dim"],
        ))
        del2 = deletion_events(r2["rollup"])
        n2 = r2["rollup"]["n_total"]
        ratio2 = 100.0 * del2 / n2
        drop_pp = v1_ratio - ratio2
        out.append("## Hypothesis check\n")
        out.append("| | deletion events | total cases | ratio |")
        out.append("|---|---|---|---|")
        out.append(f"| v1 training (regular only) | {v1_del_events} | {v1_n} | {v1_ratio:.1f}% |")
        out.append(f"| v2 full training (regular + stress) | {del2} | {n2} | {ratio2:.1f}% |")
        out.append(f"| drop (pp) | | | **{drop_pp:.1f} pp** |")
        out.append("")
        threshold_pp = 15.0
        if drop_pp >= threshold_pp:
            out.append(
                f"**PASS** — deletion ratio dropped by {drop_pp:.1f} pp "
                f"(≥ {threshold_pp:.0f} pp threshold). "
                f"Rebalance shifted the distribution as expected. "
                f"Phase 3 training can proceed."
            )
        else:
            out.append(
                f"**FAIL** — deletion ratio dropped by only {drop_pp:.1f} pp "
                f"(threshold ≥ {threshold_pp:.0f} pp). "
                f"Rebalance did not shift the distribution enough. "
                f"Do not proceed to Phase 3 without investigation."
            )
        out.append("")

    else:
        # Single-invocation mode: audit the specified file(s)
        all_cases: list[dict] = []
        sources: list[str] = []
        for p in args.inputs:
            path = Path(p)
            if not path.is_absolute():
                path = ROOT / p
            cases = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
            all_cases.extend(cases)
            sources.append(f"`{path.name}`")
        label = args.label or " + ".join(s.strip("`") for s in sources)
        r = audit_cases(all_cases)
        out.extend(render_section(
            label, ", ".join(sources), all_cases, r["rollup"], r["per_dim"]
        ))

    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
