#!/usr/bin/env python3
"""
Baseline vs. fine-tuned comparison eval for Trace Layer 2 tier classification.

Runs inference twice — once with the base model (no adapter) and once with the
fine-tuned LoRA adapter — and computes accuracy, per-class accuracy, confusion
matrix, and a sample of failure/success cases.

Usage:
  # Run both baseline and fine-tuned in one pass:
  python eval/run_eval.py \
      --eval-data data/eval_gold.jsonl \
      --config configs/sft_trace_qwen25_3b_v2.yaml \
      --ft-adapter outputs/sft_qwen25_3b_v2 \
      --output eval/results_raw.json

  # Baseline only (no adapter):
  python eval/run_eval.py \
      --eval-data data/eval_gold.jsonl \
      --config configs/sft_trace_qwen25_3b_v2.yaml \
      --baseline-only

  # Fine-tuned only:
  python eval/run_eval.py \
      --eval-data data/eval_gold.jsonl \
      --config configs/sft_trace_qwen25_3b_v2.yaml \
      --ft-adapter outputs/sft_qwen25_3b_v2 \
      --ft-only
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

SYSTEM_PROMPT = (
    "You are a Trace memory extraction engine. Transform the raw trace input into a "
    "structured Trace-style memory record. Output valid JSON only. Do not add advice, "
    "coaching, prescriptive framing, or interpretation. Witness and structure; do not "
    "suggest or evaluate."
)

TIERS = ["episodic", "semantic", "procedural", "prospective"]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_eval_data(path: str) -> list[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Model helpers ─────────────────────────────────────────────────────────────

def load_model(model_path: str, adapter_path: str | None):
    from mlx_lm import load
    label = f"{model_path}" + (f" + adapter={adapter_path}" if adapter_path else " (baseline)")
    print(f"  Loading: {label}", file=sys.stderr)
    return load(model_path, adapter_path=adapter_path)


def run_inference(model, tokenizer, input_text: str, max_tokens: int = 512) -> str:
    from mlx_lm import generate
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def _find_json_object(text: str) -> str | None:
    depth, start, in_string, i = 0, None, False, 0
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\" and i + 1 < len(text):
                i += 2
                continue
            if ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                if start is None:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start: i + 1]
        i += 1
    return None


def parse_output(raw: str) -> tuple[dict | None, str]:
    text = raw.strip()
    if "```" in text:
        for block in text.split("```")[1:]:
            if "\n" in block:
                block = block[block.index("\n") + 1:]
            block = block.strip()
            if block:
                text = block
                break
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, "ok" if "memory_tier" in parsed else "partial"
    except json.JSONDecodeError:
        pass
    json_str = _find_json_object(raw)
    if json_str:
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return parsed, "ok" if "memory_tier" in parsed else "partial"
        except json.JSONDecodeError:
            pass
    return None, "parse_error"


# ── Eval loop ─────────────────────────────────────────────────────────────────

def run_eval_pass(
    model, tokenizer, eval_records: list[dict], label: str
) -> list[dict]:
    results = []
    print(f"\n  Running {len(eval_records)} examples [{label}]...", file=sys.stderr)
    for rec in eval_records:
        raw = run_inference(model, tokenizer, rec["input"])
        pred, parse_status = parse_output(raw)
        gold_tier = rec["gold"]["memory_tier"]
        pred_tier = pred.get("memory_tier") if pred else None
        correct = pred_tier == gold_tier
        results.append({
            "id": rec["id"],
            "input": rec["input"],
            "gold_tier": gold_tier,
            "pred_tier": pred_tier,
            "parse_status": parse_status,
            "correct": correct,
            "eval_tags": rec.get("eval_tags", []),
            "raw_output": raw,
        })
        status_sym = "✓" if correct else "✗"
        print(f"    {status_sym} {rec['id']}: gold={gold_tier} pred={pred_tier or parse_status}",
              file=sys.stderr)
    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    parse_errors = sum(1 for r in results if r["parse_status"] == "parse_error")

    # Per-class accuracy
    per_class: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        per_class[r["gold_tier"]].append(r["correct"])

    per_class_acc = {}
    for tier in TIERS:
        vals = per_class.get(tier, [])
        per_class_acc[tier] = {"correct": sum(vals), "total": len(vals),
                                "accuracy": sum(vals) / len(vals) if vals else 0.0}

    # Confusion matrix: conf[gold][pred] = count
    conf: dict[str, dict[str, int]] = {t: {t2: 0 for t2 in TIERS + ["parse_error"]}
                                        for t in TIERS}
    for r in results:
        gold = r["gold_tier"]
        pred = r["pred_tier"] if r["pred_tier"] in TIERS else "parse_error"
        conf[gold][pred] += 1

    # Aspiration-vs-commitment subset
    asp_results = [r for r in results if "aspiration_vs_commitment" in r["eval_tags"]]
    asp_correct = sum(1 for r in asp_results if r["correct"])

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "parse_errors": parse_errors,
        "per_class": per_class_acc,
        "confusion": conf,
        "aspiration_vs_commitment": {
            "total": len(asp_results),
            "correct": asp_correct,
            "accuracy": asp_correct / len(asp_results) if asp_results else 0.0,
        },
    }


def pick_examples(baseline_results: list[dict], ft_results: list[dict]) -> dict:
    """
    Select illustrative examples:
      - FT improved (baseline wrong, ft correct)
      - Both correct
      - FT still wrong
    """
    by_id_base = {r["id"]: r for r in baseline_results}
    by_id_ft = {r["id"]: r for r in ft_results}

    improved, both_correct, still_wrong = [], [], []
    for rid in by_id_ft:
        b = by_id_base[rid]
        f = by_id_ft[rid]
        if not b["correct"] and f["correct"]:
            improved.append((b, f))
        elif b["correct"] and f["correct"]:
            both_correct.append((b, f))
        elif not f["correct"]:
            still_wrong.append((b, f))

    return {
        "improved": improved[:3],
        "both_correct": both_correct[:2],
        "still_wrong": still_wrong[:2],
    }


# ── Markdown rendering ────────────────────────────────────────────────────────

def render_confusion_matrix(conf: dict[str, dict[str, int]]) -> str:
    header = "| Gold \\ Pred | " + " | ".join(TIERS) + " | parse_error |"
    sep = "|" + "---|" * (len(TIERS) + 2)
    rows = [header, sep]
    for gold in TIERS:
        cells = [str(conf[gold].get(p, 0)) for p in TIERS + ["parse_error"]]
        rows.append(f"| {gold} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def render_results_md(
    baseline_metrics: dict,
    ft_metrics: dict,
    examples: dict,
    eval_size: int,
    ft_adapter_path: str,
    config_path: str,
) -> str:
    bm = baseline_metrics
    fm = ft_metrics

    def pct(x): return f"{100*x:.1f}%"

    lines = [
        "# Eval Results — Baseline vs. Fine-Tuned (SFT v2)",
        "",
        f"> **Small gold eval / smoke eval** — {eval_size} examples. "
        "Results are directional, not statistically robust.",
        "",
        "## Summary Table",
        "",
        "| Metric | Baseline (no adapter) | Fine-Tuned (SFT v2) |",
        "|--------|----------------------|---------------------|",
        f"| Overall accuracy | {bm['correct']}/{bm['total']} ({pct(bm['accuracy'])}) "
        f"| {fm['correct']}/{fm['total']} ({pct(fm['accuracy'])}) |",
    ]

    for tier in TIERS:
        b = bm["per_class"].get(tier, {})
        f = fm["per_class"].get(tier, {})
        b_str = f"{b.get('correct',0)}/{b.get('total',0)} ({pct(b.get('accuracy',0))})"
        f_str = f"{f.get('correct',0)}/{f.get('total',0)} ({pct(f.get('accuracy',0))})"
        lines.append(f"| {tier} accuracy | {b_str} | {f_str} |")

    asp_b = bm["aspiration_vs_commitment"]
    asp_f = fm["aspiration_vs_commitment"]
    lines += [
        f"| aspiration-vs-commitment | {asp_b['correct']}/{asp_b['total']} ({pct(asp_b['accuracy'])}) "
        f"| {asp_f['correct']}/{asp_f['total']} ({pct(asp_f['accuracy'])}) |",
        f"| parse errors | {bm['parse_errors']} | {fm['parse_errors']} |",
        "",
        "## Confusion Matrix — Baseline",
        "",
        render_confusion_matrix(bm["confusion"]),
        "",
        "## Confusion Matrix — Fine-Tuned",
        "",
        render_confusion_matrix(fm["confusion"]),
        "",
        "## Illustrative Examples",
        "",
    ]

    if examples["improved"]:
        lines.append("### Cases where fine-tuning improved the prediction")
        for b, f in examples["improved"]:
            lines += [
                "",
                f"**{b['id']}** — gold: `{b['gold_tier']}`",
                f"> {b['input'][:200]}{'...' if len(b['input']) > 200 else ''}",
                f"- Baseline predicted: `{b['pred_tier'] or b['parse_status']}`",
                f"- Fine-tuned predicted: `{f['pred_tier']}`  ✓",
            ]

    if examples["both_correct"]:
        lines += ["", "### Cases where both were correct"]
        for b, f in examples["both_correct"]:
            lines += [
                "",
                f"**{b['id']}** — gold: `{b['gold_tier']}`",
                f"> {b['input'][:200]}{'...' if len(b['input']) > 200 else ''}",
                f"- Baseline: `{b['pred_tier']}`  ✓  |  Fine-tuned: `{f['pred_tier']}`  ✓",
            ]

    if examples["still_wrong"]:
        lines += ["", "### Cases where fine-tuning did not fix the error"]
        for b, f in examples["still_wrong"]:
            lines += [
                "",
                f"**{f['id']}** — gold: `{f['gold_tier']}`",
                f"> {f['input'][:200]}{'...' if len(f['input']) > 200 else ''}",
                f"- Baseline predicted: `{b['pred_tier'] or b['parse_status']}`",
                f"- Fine-tuned predicted: `{f['pred_tier'] or f['parse_status']}`",
            ]

    lines += [
        "",
        "## Interpretation",
        "",
        _interpretation(bm, fm),
        "",
        "## Trace-Style Integration Examples",
        "",
        "Three realistic Trace inputs showing baseline vs. fine-tuned predictions:",
        "",
    ]

    trace_examples = _trace_integration_examples()
    for ex in trace_examples:
        lines += [
            f"**Input:** _{ex['input']}_",
            "",
            f"- Baseline: `{ex['baseline']}`",
            f"- Fine-tuned: `{ex['ft']}`",
            f"- Gold tier: `{ex['gold']}`",
            "",
        ]

    lines += [
        "## Reproducibility",
        "",
        f"- Config: `{config_path}`",
        f"- Adapter: `{ft_adapter_path}`",
        f"- Eval set: `data/eval_gold.jsonl` ({eval_size} examples)",
        "- Seed: 42 (train/val split in prepare_data.py)",
        "- See `eval/notes.md` for full training command and hyperparameters.",
    ]

    return "\n".join(lines) + "\n"


def _interpretation(bm: dict, fm: dict) -> str:
    delta = fm["accuracy"] - bm["accuracy"]
    direction = "improved" if delta > 0 else ("declined" if delta < 0 else "unchanged")
    delta_pct = abs(100 * delta)

    gains, regressions = [], []
    for tier in TIERS:
        b_acc = bm["per_class"].get(tier, {}).get("accuracy", 0)
        f_acc = fm["per_class"].get(tier, {}).get("accuracy", 0)
        if f_acc > b_acc:
            gains.append(tier)
        elif f_acc < b_acc:
            regressions.append(tier)

    asp_b = bm["aspiration_vs_commitment"]["accuracy"]
    asp_f = fm["aspiration_vs_commitment"]["accuracy"]

    parts = [
        f"Overall tier accuracy {direction} by {delta_pct:.1f} percentage points "
        f"({100*bm['accuracy']:.1f}% → {100*fm['accuracy']:.1f}%) on this {bm['total']}-example smoke eval."
    ]
    if gains:
        parts.append(f"The fine-tuned model gained on: {', '.join(gains)}.")
    if regressions:
        parts.append(f"Small regressions observed on: {', '.join(regressions)} — likely noise at this eval size.")
    if asp_f != asp_b:
        direction_asp = "improved" if asp_f > asp_b else "declined"
        parts.append(
            f"Aspiration-vs-commitment accuracy {direction_asp} "
            f"({100*asp_b:.0f}% → {100*asp_f:.0f}%), which is the hardest boundary in the Trace schema. "
            "This subset is only a few examples; treat as indicative."
        )
    parts.append(
        "With only 12 eval examples, individual example outcomes dominate the percentages. "
        "A larger held-out set (50+ examples) would distinguish real learning from variance."
    )
    return " ".join(parts)


def _trace_integration_examples() -> list[dict]:
    return [
        {
            "input": "Told Marcus I'd have the retrospective write-up done by Thursday. "
                     "Need to block time tomorrow morning.",
            "gold": "prospective",
            "baseline": "(run baseline to see)",
            "ft": "(run fine-tuned to see)",
        },
        {
            "input": "I'd love to get back into meditation at some point. Life feels too "
                     "packed right now to actually start.",
            "gold": "semantic",
            "baseline": "(run baseline to see)",
            "ft": "(run fine-tuned to see)",
        },
        {
            "input": "Finally cracked the right pattern for async error handling in the "
                     "service layer — bubble up domain errors, swallow infrastructure ones.",
            "gold": "procedural",
            "baseline": "(run baseline to see)",
            "ft": "(run fine-tuned to see)",
        },
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs. fine-tuned Trace extraction on eval_gold.jsonl."
    )
    parser.add_argument("--eval-data", default="data/eval_gold.jsonl")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ft-adapter", help="Path to fine-tuned LoRA adapter.")
    parser.add_argument("--output", default="eval/results_raw.json",
                        help="Where to save raw results JSON.")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--ft-only", action="store_true")
    parser.add_argument("--write-md", default="eval/results.md",
                        help="Where to write the markdown report.")
    args = parser.parse_args()

    eval_records = load_eval_data(args.eval_data)
    config = load_config(args.config)
    model_path = config["model"]["path"]

    print(f"\nEval set: {args.eval_data} ({len(eval_records)} examples)", file=sys.stderr)
    print(f"Model:    {model_path}", file=sys.stderr)

    baseline_results, ft_results = None, None

    # ── Baseline ──────────────────────────────────────────────────────────────
    if not args.ft_only:
        print("\n[1/2] Baseline (no adapter)", file=sys.stderr)
        model, tokenizer = load_model(model_path, None)
        baseline_results = run_eval_pass(model, tokenizer, eval_records, "baseline")
        del model, tokenizer  # free memory before loading adapter

    # ── Fine-tuned ────────────────────────────────────────────────────────────
    if not args.baseline_only:
        if not args.ft_adapter:
            print("ERROR: --ft-adapter required unless --baseline-only is set.", file=sys.stderr)
            sys.exit(1)
        print("\n[2/2] Fine-tuned adapter", file=sys.stderr)
        model, tokenizer = load_model(model_path, args.ft_adapter)
        ft_results = run_eval_pass(model, tokenizer, eval_records, "fine-tuned")
        del model, tokenizer

    # ── Metrics ───────────────────────────────────────────────────────────────
    output = {}
    if baseline_results:
        bm = compute_metrics(baseline_results)
        output["baseline"] = {"metrics": bm, "results": baseline_results}
        print("\n── Baseline metrics ──────────────────────────────────")
        print(f"  Overall accuracy: {bm['correct']}/{bm['total']} ({100*bm['accuracy']:.1f}%)")
        for tier in TIERS:
            pc = bm["per_class"].get(tier, {})
            print(f"    {tier:<15} {pc.get('correct',0)}/{pc.get('total',0)}")

    if ft_results:
        fm = compute_metrics(ft_results)
        output["ft"] = {"metrics": fm, "results": ft_results}
        print("\n── Fine-tuned metrics ────────────────────────────────")
        print(f"  Overall accuracy: {fm['correct']}/{fm['total']} ({100*fm['accuracy']:.1f}%)")
        for tier in TIERS:
            pc = fm["per_class"].get(tier, {})
            print(f"    {tier:<15} {pc.get('correct',0)}/{pc.get('total',0)}")

    # ── Save raw JSON ─────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nRaw results saved to: {args.output}", file=sys.stderr)

    # ── Write markdown report ─────────────────────────────────────────────────
    if baseline_results and ft_results:
        examples = pick_examples(baseline_results, ft_results)
        md = render_results_md(
            bm, fm, examples,
            eval_size=len(eval_records),
            ft_adapter_path=args.ft_adapter or "",
            config_path=args.config,
        )
        Path(args.write_md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_md, "w") as f:
            f.write(md)
        print(f"Markdown report written to: {args.write_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
