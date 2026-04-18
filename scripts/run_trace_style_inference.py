#!/usr/bin/env python3
"""
Trace-style inference entrypoint.

Takes a raw trace text input, runs the model, and outputs a structured
Trace-style memory record as JSON to stdout. Designed to be pipe-friendly.

Usage:
  # From argument:
  python scripts/run_trace_style_inference.py \
      --config configs/sft_trace_qwen25_3b.yaml \
      --input "Finished the sprint retrospective and sent notes to the team."

  # From stdin (pipe-friendly):
  echo "Finished the sprint retrospective." | \
      python scripts/run_trace_style_inference.py \
          --config configs/sft_trace_qwen25_3b.yaml

  # With a trained adapter:
  python scripts/run_trace_style_inference.py \
      --config configs/sft_trace_qwen25_3b.yaml \
      --adapter outputs/sft_qwen25_3b \
      --input "Need to send the report by Friday."

  # Pretty-print with jq:
  echo "Worked on the auth refactor." | \
      python scripts/run_trace_style_inference.py \
          --config configs/sft_trace_qwen25_3b.yaml | jq .
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

SYSTEM_PROMPT = (
    "You are a Trace memory extraction engine. Transform the raw trace input into a "
    "structured Trace-style memory record. Output valid JSON only. Do not add advice, "
    "coaching, prescriptive framing, or interpretation. Witness and structure; do not "
    "suggest or evaluate."
)


def load_config(config_path: str) -> dict:
    if not Path(config_path).exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(config: dict, adapter_path: str | None):
    from mlx_lm import load

    model_path = config["model"]["path"]
    print(f"Loading model: {model_path}", file=sys.stderr)
    if adapter_path:
        print(f"Adapter:       {adapter_path}", file=sys.stderr)
    return load(model_path, adapter_path=adapter_path)


def run_inference(model, tokenizer, input_text: str, max_tokens: int) -> str:
    from mlx_lm import generate

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def parse_to_json(raw: str) -> dict:
    """
    Parse model output as JSON. Strip markdown fences if present.
    Returns the parsed dict, or a structured error record if parsing fails.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Return a well-structured error record rather than crashing.
        # This makes the output still pipeable — downstream tools can
        # detect the error key.
        return {
            "error": "parse_failed",
            "raw_output": raw,
            "note": (
                "Model output could not be parsed as JSON. "
                "Consider rerunning with a trained adapter or checking the prompt."
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Trace-style extraction inference. "
            "Outputs a structured JSON record to stdout."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. configs/sft_trace_qwen25_3b.yaml).",
    )
    parser.add_argument(
        "--input",
        help=(
            "Raw trace text to process. If omitted, reads from stdin. "
            "Enables pipe usage: echo '...' | python run_trace_style_inference.py ..."
        ),
    )
    parser.add_argument(
        "--adapter",
        help="Path to trained LoRA adapter directory (optional).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens to generate (default: 512).",
    )
    args = parser.parse_args()

    # Read input from argument or stdin.
    if args.input:
        input_text = args.input.strip()
    else:
        if sys.stdin.isatty():
            print(
                "No --input provided and stdin is a terminal.\n"
                "Usage: echo 'your trace text' | python scripts/run_trace_style_inference.py ...",
                file=sys.stderr,
            )
            sys.exit(1)
        input_text = sys.stdin.read().strip()

    if not input_text:
        print("ERROR: empty input.", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)

    try:
        model, tokenizer = load_model(config, args.adapter)
    except ImportError:
        print("ERROR: mlx_lm not installed. Run: pip install mlx-lm", file=sys.stderr)
        sys.exit(1)

    raw_output = run_inference(model, tokenizer, input_text, args.max_tokens)
    record = parse_to_json(raw_output)

    # Print JSON to stdout — one record per run, pipe-friendly.
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
