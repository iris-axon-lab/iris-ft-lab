#!/usr/bin/env python3
"""
Evaluate intention-reality tracking outputs.

This script tests the model's ability to characterize the gap between a stated
prospective intent and an observed outcome. The task is the second schema in
the Trace architecture — the intention-reality tracker.

Gap labels:
  fulfilled   — The intention was carried out as stated.
  abandoned   — Not acted on, with no replacement.
  transformed — Evolved into a different but related action.
  unresolved  — Outcome not yet observed or ambiguous.

The model should witness and characterize the gap without prescribing what the
person "should have" done differently.

Usage:
  python scripts/eval_tracking.py
  python scripts/eval_tracking.py --config configs/sft_trace_qwen25_3b.yaml \
      --adapter outputs/sft_qwen25_3b
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

# ── Synthetic tracking eval set ───────────────────────────────────────────────
# These examples are defined inline rather than in a separate file,
# keeping this script self-contained for educational inspection.

TRACKING_EVAL = [
    {
        "id": "track_001",
        "original_intent": "Submit grant proposal draft to committee by Friday 5pm",
        "stated_when": "2025-10-14",
        "outcome_observed": "Proposal submitted on Saturday morning, one day after the deadline, with committee approval to proceed",
        "outcome_when": "2025-10-16",
        "gold_gap": "transformed",
        "note": "Deadline was missed but the outcome was still achieved in a modified form.",
    },
    {
        "id": "track_002",
        "original_intent": "Reach out to David about the job referral this week",
        "stated_when": "2025-10-07",
        "outcome_observed": "No message sent. Three weeks later, the role was filled.",
        "outcome_when": "2025-10-28",
        "gold_gap": "abandoned",
        "note": "Explicit non-action over a defined window.",
    },
    {
        "id": "track_003",
        "original_intent": "Block Friday afternoon for reading and rest after the sprint",
        "stated_when": "2025-10-15",
        "outcome_observed": "Took Friday afternoon off as planned. Read for two hours.",
        "outcome_when": "2025-10-17",
        "gold_gap": "fulfilled",
        "note": "Clear fulfillment matching the stated intent.",
    },
    {
        "id": "track_004",
        "original_intent": "Review Sarah's pull request before noon",
        "stated_when": "2025-10-19",
        "outcome_observed": "Review started at 2pm and completed by 3pm; Sarah acknowledged it was helpful.",
        "outcome_when": "2025-10-19",
        "gold_gap": "transformed",
        "note": "Deadline missed but intent fulfilled later the same day.",
    },
    {
        "id": "track_005",
        "original_intent": "Restrict Slack to 10am and 3pm daily starting tomorrow",
        "stated_when": "2025-10-20",
        "outcome_observed": "No follow-up trace entries mention Slack habits over the next two weeks.",
        "outcome_when": "2025-11-03",
        "gold_gap": "unresolved",
        "note": "Absence of evidence — gap cannot be characterized without more signal.",
    },
]

TRACKING_PROMPT_TEMPLATE = """\
You are a Trace intention-reality tracker. Given the original intent and the observed outcome below, characterize the gap using exactly one of these labels: fulfilled, abandoned, transformed, unresolved.

Output valid JSON only, using this schema:
{{
  "original_intent": "...",
  "stated_when": "...",
  "outcome_observed": "...",
  "outcome_when": "...",
  "gap_characterization": "fulfilled | abandoned | transformed | unresolved"
}}

Do not add advice, interpretation, or prescriptive framing.

Original intent: {original_intent}
Stated when: {stated_when}
Outcome observed: {outcome_observed}
Outcome when: {outcome_when}"""

VALID_GAP_LABELS = {"fulfilled", "abandoned", "transformed", "unresolved"}


# ── Inference ─────────────────────────────────────────────────────────────────

def run_tracking_inference(model, tokenizer, example: dict) -> str:
    from mlx_lm import generate

    prompt_text = TRACKING_PROMPT_TEMPLATE.format(
        original_intent=example["original_intent"],
        stated_when=example["stated_when"],
        outcome_observed=example["outcome_observed"],
        outcome_when=example["outcome_when"],
    )
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)


def parse_gap_label(raw: str) -> str | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    try:
        parsed = json.loads(text)
        label = parsed.get("gap_characterization", "").lower()
        return label if label in VALID_GAP_LABELS else None
    except json.JSONDecodeError:
        # Try extracting label directly if the model produced partial output.
        for label in VALID_GAP_LABELS:
            if label in text.lower():
                return label
        return None


# ── Gold-only mode ────────────────────────────────────────────────────────────

def print_gold_distribution() -> None:
    from collections import Counter
    labels = Counter(e["gold_gap"] for e in TRACKING_EVAL)
    print("\nGold gap label distribution:")
    for label, count in sorted(labels.items()):
        print(f"  {label:<15} {count}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate intention-reality tracking (gap characterization)."
    )
    parser.add_argument(
        "--config",
        help="Path to YAML config. Required unless --gold-only.",
    )
    parser.add_argument(
        "--adapter",
        help="Path to trained LoRA adapter (optional).",
    )
    parser.add_argument(
        "--gold-only",
        action="store_true",
        help="Show gold distribution only; skip model inference.",
    )
    args = parser.parse_args()

    print(f"\niris-ft-lab tracking eval — {len(TRACKING_EVAL)} examples")

    if args.gold_only:
        print_gold_distribution()
        sys.exit(0)

    if not args.config:
        print("ERROR: --config is required. Use --gold-only to skip inference.")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    try:
        from mlx_lm import load
        model_path = config["model"]["path"]
        print(f"  Loading model: {model_path}")
        model, tokenizer = load(model_path, adapter_path=args.adapter)
    except ImportError:
        print("ERROR: mlx_lm not available. Run: pip install mlx-lm")
        sys.exit(1)

    correct = 0
    results = []

    print(f"\nRunning tracking inference...\n")
    for example in TRACKING_EVAL:
        raw = run_tracking_inference(model, tokenizer, example)
        pred_label = parse_gap_label(raw)
        gold_label = example["gold_gap"]
        match = pred_label == gold_label
        if match:
            correct += 1

        results.append({"id": example["id"], "gold": gold_label, "pred": pred_label, "correct": match})
        status = "OK" if match else "MISS"
        print(f"  [{status}] {example['id']}: gold={gold_label} pred={pred_label or 'parse_error'}")

    total = len(results)
    print(f"\n── Tracking Eval Results {'─' * 36}")
    print(f"  Gap characterization accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    misses = [r for r in results if not r["correct"]]
    if misses:
        print(f"\n  Misclassifications ({len(misses)}):")
        for r in misses:
            print(f"    [{r['id']}] gold={r['gold']} pred={r['pred'] or 'parse_error'}")
    print()


if __name__ == "__main__":
    main()
