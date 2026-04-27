"""
SFT training scaffold for collab_eval spreadsheet-cleaning tasks.

This script does NOT train by default. Use --dry-run, --check-only, or --train.

Promotion gate (revised in v1; canonical definition lives in
configs/sft_collab_eval_qwen25_3b.yaml and eval/run_collab_model_eval.py):
  Promote SFT adapter only if ALL are true:
  - composite_mean does not regress by more than 0.005
  - data_preservation does not regress
  - rh_like_count does not increase
  - one of {unit_consistency, format_validity, completeness} improves by >= 0.02
  - preservation-stress data_preservation_mean >= 0.85

The v0 gate (+0.10 composite improvement) was retired because the base model
already scored composite 0.9569, leaving max possible improvement ~0.043.

CLI:
  --config     YAML config path
               default: configs/sft_collab_eval_qwen25_3b.yaml
  --dry-run    Print the exact mlx_lm.lora training command; do not train.
  --check-only Verify config, data path, and mlx_lm availability; exit 0.
  --train      Run training via subprocess (explicit; disabled by default).

Behavior when dependencies are missing:
  Prints a "not available" message and exits 0.
  Does not raise an exception.

Do NOT run training in tests or CI.
Do NOT commit adapter outputs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))

_DEFAULT_CONFIG = str(_ROOT / "configs" / "sft_collab_eval_qwen25_3b.yaml")


def _check_dependencies() -> tuple[bool, list[str]]:
    """Check that mlx_lm CLI is available and PyYAML is importable."""
    issues: list[str] = []
    try:
        import yaml  # noqa: F401
    except ImportError:
        issues.append("PyYAML not installed (pip install pyyaml)")
    # Prefer importability check; CLI probe is a secondary fallback
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        try:
            result = subprocess.run(
                ["mlx_lm.lora", "--help"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                issues.append("mlx_lm not installed (pip install mlx-lm)")
        except (FileNotFoundError, OSError):
            issues.append("mlx_lm not installed (pip install mlx-lm)")
    return len(issues) == 0, issues


def _load_config(config_path: str) -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dry_run_command(config_path: str) -> str:
    return f"mlx_lm.lora --config {config_path}"


def run_check_only(config_path: str) -> None:
    print("=== check-only: verifying dependencies and config ===")

    all_ok, issues = _check_dependencies()
    if issues:
        for issue in issues:
            print(f"  NOT AVAILABLE: {issue}")
        print("Training dependencies not available. Exit 0 (infrastructure check only).")

    config_file = Path(config_path)
    if not config_file.exists():
        print(f"  Config not found: {config_path}")
        return

    print(f"  Config: {config_path} [OK]")
    config = _load_config(config_path)

    # a. Check train: true
    if not config.get("train"):
        print("  WARN: config 'train' is not true — training will not run")
    else:
        print("  train: true [OK]")

    # b. Check data is not WikiSQL
    data_val = config.get("data", "<not set>")
    if data_val == "mlx-community/WikiSQL":
        print("  ERROR: config 'data' is still set to mlx-community/WikiSQL — fix required")
    elif not data_val or data_val == "<not set>":
        print("  WARN: config 'data' not set")
    else:
        print(f"  data: {data_val}")

    # c. Check data is a local path
    if data_val and data_val != "mlx-community/WikiSQL" and data_val != "<not set>":
        if data_val.startswith("mlx-community/") or data_val.startswith("huggingface/"):
            print(f"  WARN: config 'data' looks like a HuggingFace dataset, not a local path")
        else:
            print("  data is a local path [OK]")

    # d. Check local train.jsonl exists
    train_jsonl = _ROOT / data_val / "train.jsonl" if data_val and not data_val.startswith("/") else Path(data_val) / "train.jsonl"
    if train_jsonl.exists():
        print(f"  {train_jsonl}: exists [OK]")
    else:
        print(f"  MISSING: {train_jsonl}")
        print("  To create it, run:")
        print("    python scripts/generate_tasks.py --task spreadsheet_clean --n 240 --seed 100 \\")
        print("        --output data/generated/spreadsheet_train_v1.jsonl")
        print("    python scripts/build_sft_data.py")
        print(f"    (writes to {train_jsonl})")

    # e. Check adapter_path is local and gitignored
    adapter_path = config.get("adapter_path", "<not set>")
    print(f"  adapter_path: {adapter_path}")
    gitignore = _ROOT / ".gitignore"
    if gitignore.exists():
        gitignore_content = gitignore.read_text()
        if "adapters/" in gitignore_content:
            print("  adapter_path parent gitignored [OK]")
        else:
            print("  WARN: adapters/ not found in .gitignore — add it to avoid committing weights")

    # f. Report mlx_lm availability
    if not issues:
        print("  mlx_lm: available [OK]")
        print("  Note: model will be downloaded at training time if not cached.")
    else:
        print("  Note: install mlx-lm before running training.")

    print()
    print("Baseline gate requirement:")
    print("  Run scripts/run_model_baseline.py and verify:")
    print("  - composite mean >= 0.30")
    print("  - hard-fail rate <= 40%")
    print("  before proceeding to actual training.")
    print()
    print("Promotion gate requirement (post-training, v1):")
    print("  Promote adapter only if ALL are true:")
    print("  - composite_mean does not regress by more than 0.005")
    print("  - data_preservation does not regress")
    print("  - rh_like_count does not increase")
    print("  - one of {unit_consistency, format_validity, completeness} improves by >= 0.02")
    print("  - preservation-stress data_preservation_mean >= 0.85")


def run_dry_run(config_path: str) -> None:
    print("=== dry-run: printing MLX-LM training command ===")

    all_ok, issues = _check_dependencies()
    if issues:
        for issue in issues:
            print(f"  NOT AVAILABLE: {issue}")
        print("Cannot confirm full environment. Command preview (may not be runnable):")

    cmd = _dry_run_command(config_path)
    print()
    print("Training command (not executed):")
    print(cmd)
    print()
    print("Note: training data must exist at the path configured in the YAML.")
    print("Generate with:")
    print("  python scripts/generate_tasks.py --task spreadsheet_clean --n 240 --seed 100 \\")
    print("      --output data/generated/spreadsheet_train_v1.jsonl")
    print("  python scripts/build_sft_data.py")
    print("  (writes to data/sft_collab_eval_full/train.jsonl)")


def run_train(config_path: str) -> None:
    print("=== train: running MLX-LM LoRA training ===")

    all_ok, issues = _check_dependencies()
    if not all_ok:
        for issue in issues:
            print(f"  NOT AVAILABLE: {issue}")
        print("Cannot train. Exit 1.")
        sys.exit(1)

    config = _load_config(config_path)
    data_val = config.get("data", "")
    if data_val:
        train_jsonl = _ROOT / data_val / "train.jsonl" if not data_val.startswith("/") else Path(data_val) / "train.jsonl"
        if not train_jsonl.exists():
            print(f"ERROR: local train.jsonl not found: {train_jsonl}")
            print("Generate first:")
            print("  python scripts/generate_tasks.py --task spreadsheet_clean --n 240 --seed 100 \\")
            print("      --output data/generated/spreadsheet_train_v1.jsonl")
            print("  python scripts/build_sft_data.py")
            sys.exit(1)

    adapter_path = config.get("adapter_path", "adapters/sft_collab_eval_qwen25_3b/")
    adapter_dir = _ROOT / adapter_path if not adapter_path.startswith("/") else Path(adapter_path)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    cmd = _dry_run_command(config_path)
    print(f"Running: {cmd}")
    result = subprocess.run(cmd.split(), cwd=str(_ROOT))
    sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SFT training scaffold. Does not train by default."
    )
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help=f"YAML config path (default: {_DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print exact MLX-LM training command; do not train.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify config, data, and mlx_lm availability; print status; exit 0.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training via subprocess. Must be explicit. Verifies train.jsonl exists first.",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.check_only and not args.train:
        print("No action specified.")
        print("Use --dry-run to see the training command, --check-only to verify setup,")
        print("or --train to run training (requires local train.jsonl and mlx-lm).")
        print("Training is not run by default.")
        parser.print_help()
        return

    if args.check_only:
        run_check_only(args.config)
    elif args.dry_run:
        run_dry_run(args.config)
    elif args.train:
        run_train(args.config)


if __name__ == "__main__":
    main()
