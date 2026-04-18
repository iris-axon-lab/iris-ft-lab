#!/usr/bin/env python3
"""
Evaluate structured extraction quality against eval_gold.jsonl.

Metrics reported:
  - Tier classification accuracy (overall and per-tier)
  - Intent extraction hit rate
  - Aspiration vs. commitment accuracy

For each eval example, the script runs the fine-tuned model on the raw input,
parses the JSON output, and compares against the gold labels.

Usage:
  python scripts/eval_extraction.py \
      --eval-data data/eval_gold.jsonl \
      --config configs/sft_trace_qwen25_3b.yaml

  # Skip model inference (inspect gold distribution only):
  python scripts/eval_extraction.py \
      --eval-data data/eval_gold.jsonl \
      --gold-only

  # Use a trained adapter:
  python scripts/eval_extraction.py \
      --eval-data data/eval_gold.jsonl \
      --config configs/sft_trace_qwen25_3b.yaml \
      --adapter outputs/sft_qwen25_3b
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

SYSTEM_PROMPT = (
    "You are a Trace memory extraction engine. Transform the raw trace input into a "
    "structured Trace-style memory record. Output valid JSON only. Do not add advice, "
    "coaching, prescriptive framing, or interpretation. Witness and structure; do not "
    "suggest or evaluate."
)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_eval_data(path: str) -> list[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Model inference ───────────────────────────────────────────────────────────

def load_model(config: dict, adapter_path: str | None):
    """Load model and tokenizer via mlx_lm."""
    from mlx_lm import load

    model_path = config["model"]["path"]
    print(f"  Loading model: {model_path}")
    if adapter_path:
        print(f"  Adapter:       {adapter_path}")

    model, tokenizer = load(model_path, adapter_path=adapter_path)
    return model, tokenizer


def run_inference(model, tokenizer, input_text: str, max_tokens: int = 512) -> str:
    """Run a single inference pass and return the model's raw text output."""
    from mlx_lm import generate

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
    ]

    # Apply the model's chat template.
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return response


def _find_json_object(text: str) -> str | None:
    """
    Find the first complete {...} JSON object in text using brace counting.

    Handles JSON embedded in prose, after markdown fences, or mixed with
    explanatory text. Respects string literals so braces inside "..." are
    not counted as structural.
    """
    depth = 0
    start = None
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\" and i + 1 < len(text):
                i += 2  # skip escaped character
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
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        return text[start : i + 1]
        i += 1
    return None


def parse_model_output(raw: str) -> tuple[dict | None, str]:
    """
    Attempt to parse the model's raw output as a Trace JSON record.

    Returns (parsed_dict_or_None, parse_status) where parse_status is one of:
      "ok"          — parsed and has expected keys
      "partial"     — parsed but missing 'memory_tier' (wrong schema / wrong keys)
      "parse_error" — could not extract valid JSON at all

    Strategy:
      1. Strip markdown code fences (```json ... ``` or ``` ... ```)
      2. Try direct json.loads on the cleaned text
      3. Fall back to brace-scanning to find a JSON object anywhere in the output
         (handles models that prepend prose before the JSON)
    """
    text = raw.strip()

    # Strip markdown code fence if present.
    if "```" in text:
        # Grab the content between the first opening and last closing fence.
        inside = text.split("```")
        for block in inside[1:]:
            # Drop optional language tag (e.g. "json\n")
            if "\n" in block:
                block = block[block.index("\n") + 1 :]
            block = block.strip()
            if block:
                text = block
                break

    # Try direct parse first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            status = "ok" if "memory_tier" in parsed else "partial"
            return parsed, status
    except json.JSONDecodeError:
        pass

    # Fall back: scan for first JSON object anywhere in the raw output.
    json_str = _find_json_object(raw)
    if json_str:
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                status = "ok" if "memory_tier" in parsed else "partial"
                return parsed, status
        except json.JSONDecodeError:
            pass

    return None, "parse_error"


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_tier(pred: dict | None, gold: dict) -> bool:
    if pred is None:
        return False
    return pred.get("memory_tier") == gold.get("memory_tier")


def compare_intent(pred: dict | None, gold: dict) -> bool | None:
    """
    Returns True/False when both have a stated_intent to compare,
    or None when the example doesn't test intent extraction.
    """
    gold_intent = gold.get("stated_intent")
    if gold_intent is None:
        # Gold says no intent — check model doesn't fabricate one.
        if pred is not None and pred.get("stated_intent") is not None:
            return False  # false positive
        return None  # not an intent test case
    # Gold has an intent — check model extracted something non-null.
    if pred is None:
        return False
    return pred.get("stated_intent") is not None


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(results: list[dict]) -> None:
    total = len(results)
    tier_correct = sum(1 for r in results if r["tier_correct"])
    parse_errors = sum(1 for r in results if r["parse_status"] == "parse_error")
    partial_parses = sum(1 for r in results if r["parse_status"] == "partial")
    intent_tests = [r for r in results if r["intent_result"] is not None]
    intent_correct = sum(1 for r in intent_tests if r["intent_result"])
    aspiration_tests = [
        r for r in results if "aspiration_vs_commitment" in r["eval_tags"]
    ]
    aspiration_correct = sum(1 for r in aspiration_tests if r["tier_correct"])

    print("\n── Extraction Eval Results " + "─" * 34)
    print(f"  Examples evaluated:          {total}")
    print(f"  Parse errors (no JSON):      {parse_errors}")
    print(f"  Partial parses (wrong keys): {partial_parses}")
    print()
    print(f"  Tier classification accuracy: {tier_correct}/{total} "
          f"({100*tier_correct/total:.1f}%)")

    # Per-tier breakdown
    per_tier: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        per_tier[r["gold_tier"]].append(r["tier_correct"])
    for tier in ["episodic", "semantic", "procedural", "prospective"]:
        vals = per_tier.get(tier, [])
        if vals:
            acc = sum(vals) / len(vals)
            print(f"    {tier:<15} {sum(vals)}/{len(vals)} ({100*acc:.0f}%)")

    print()
    if intent_tests:
        print(f"  Intent extraction hit rate:  {intent_correct}/{len(intent_tests)} "
              f"({100*intent_correct/len(intent_tests):.1f}%)")
    if aspiration_tests:
        print(f"  Aspiration vs commitment:    {aspiration_correct}/{len(aspiration_tests)} "
              f"({100*aspiration_correct/len(aspiration_tests):.1f}%)")

    print()
    failures = [r for r in results if not r["tier_correct"]]
    if failures:
        print(f"  Tier misclassifications ({len(failures)}):")
        for r in failures:
            display = r["pred_tier"] if r["pred_tier"] else r["parse_status"]
            print(f"    [{r['id']}] gold={r['gold_tier']} pred={display}")
    else:
        print("  No tier misclassifications.")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Trace extraction quality against gold eval set."
    )
    parser.add_argument(
        "--eval-data",
        default="data/eval_gold.jsonl",
        help="Path to eval_gold.jsonl (default: data/eval_gold.jsonl).",
    )
    parser.add_argument(
        "--config",
        help="Path to YAML config (used to find the model path).",
    )
    parser.add_argument(
        "--adapter",
        help="Path to trained LoRA adapter directory (optional).",
    )
    parser.add_argument(
        "--gold-only",
        action="store_true",
        help="Print gold label distribution only; skip model inference.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print raw model output for each example. Essential for debugging parse failures.",
    )
    args = parser.parse_args()

    if not Path(args.eval_data).exists():
        print(f"ERROR: eval data not found: {args.eval_data}")
        sys.exit(1)

    eval_records = load_eval_data(args.eval_data)
    print(f"\niris-ft-lab extraction eval — {len(eval_records)} examples")

    if args.gold_only:
        # Just show distribution.
        from collections import Counter
        tiers = Counter(r["gold"]["memory_tier"] for r in eval_records)
        print("\nGold tier distribution:")
        for tier, count in sorted(tiers.items()):
            print(f"  {tier:<15} {count}")
        print()
        sys.exit(0)

    if not args.config:
        print("ERROR: --config is required unless --gold-only is set.")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load model.
    print()
    try:
        model, tokenizer = load_model(config, args.adapter)
    except ImportError:
        print("ERROR: mlx_lm not available. Run: pip install mlx-lm")
        sys.exit(1)

    # Run eval loop.
    results = []
    print(f"\nRunning inference on {len(eval_records)} examples...\n")

    for record in eval_records:
        raw_output = run_inference(model, tokenizer, record["input"])
        pred, parse_status = parse_model_output(raw_output)
        gold = record["gold"]

        tier_correct = compare_tier(pred, gold)
        intent_result = compare_intent(pred, gold)
        pred_tier = pred.get("memory_tier") if pred else None

        results.append({
            "id": record["id"],
            "gold_tier": gold["memory_tier"],
            "pred_tier": pred_tier,
            "parse_status": parse_status,
            "tier_correct": tier_correct,
            "intent_result": intent_result,
            "eval_tags": record.get("eval_tags", []),
        })

        status = "OK" if tier_correct else "MISS"
        display_pred = pred_tier if pred_tier else parse_status
        print(f"  [{status}] {record['id']}: gold={gold['memory_tier']} pred={display_pred}")
        if args.verbose:
            print(f"         raw: {raw_output[:120].replace(chr(10), ' ')}{'...' if len(raw_output) > 120 else ''}")
            if parse_status == "partial":
                print(f"         parsed keys: {list(pred.keys()) if pred else '—'}")

    print_results(results)


if __name__ == "__main__":
    main()
