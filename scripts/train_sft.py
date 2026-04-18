#!/usr/bin/env python3
"""
SFT training wrapper for Trace Layer 2 extraction.

Thin wrapper around MLX-LM LoRA training. All model and training parameters
are loaded from a YAML config — nothing is hardcoded in this file.

Usage:
  python scripts/train_sft.py --config configs/sft_trace_qwen25_3b.yaml
  python scripts/train_sft.py --config configs/sft_trace_qwen25_3b.yaml --dry-run
  python scripts/train_sft.py --config configs/sft_trace_qwen25_3b.yaml \
      --model mlx-community/Qwen3-1.7B-4bit   # override model at CLI
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_mlx_lm_command(
    config: dict, model_override: str | None = None
) -> tuple[list[str], str]:
    """
    Build the mlx_lm lora training command and a temporary mlx-lm config YAML.

    Returns (cmd, tmp_config_path). The caller is responsible for deleting
    tmp_config_path after the subprocess completes.

    Why a temp config YAML?
    LoRA parameters (rank, alpha, dropout, target modules) are not exposed as
    CLI flags in current mlx-lm — they must be provided via a config file
    passed with the -c flag. This function translates our YAML config into
    the format mlx-lm expects.

    mlx-lm invocation (current API):
      python -m mlx_lm lora -c <config.yaml>

    Note: `python -m mlx_lm.lora` is deprecated; use `python -m mlx_lm lora`.
    Note: mlx-lm uses --iters (total steps), not --epochs.
    """
    model_path = model_override or config["model"]["path"]
    train_cfg = config["training"]
    lora_cfg = config["lora"]
    data_cfg = config["data"]
    out_cfg = config["output"]

    # Build the mlx-lm config dict.
    #
    # Key design decisions:
    # - num_layers must be set explicitly; mlx-lm defaults to 0 (no LoRA) when absent.
    #   It controls how many transformer blocks from the end of the model receive LoRA.
    # - lora_parameters carries rank/alpha/dropout only. We do NOT pass a 'keys' field
    #   for target module names — mlx-lm's internal module-name matching is version-specific
    #   and passing bare names like 'q_proj' can silently yield 0 matches on some builds.
    #   Omitting 'keys' lets mlx-lm use its built-in defaults for the loaded model.
    mlx_config = {
        "model": model_path,
        "train": True,
        "data": str(Path(data_cfg["train"]).parent),
        "batch_size": train_cfg["batch_size"],
        "learning_rate": train_cfg["learning_rate"],
        "iters": train_cfg.get("iters", 600),
        "max_seq_length": train_cfg["max_seq_length"],
        "adapter_path": out_cfg["adapter_path"],
        "save_every": out_cfg["save_every"],
        "steps_per_report": out_cfg["steps_per_report"],
        # num_layers: number of transformer blocks from the end to apply LoRA to.
        # Must be explicit — absent means 0 (no LoRA) in current mlx-lm.
        "num_layers": lora_cfg.get("num_layers", 16),
        "lora_parameters": {
            "rank": lora_cfg["rank"],
            "alpha": float(lora_cfg["alpha"]),
            "dropout": lora_cfg["dropout"],
            "scale": float(lora_cfg["alpha"]) / lora_cfg["rank"],
            # Do NOT include 'keys' here — target module selection is handled
            # internally by mlx-lm based on the model architecture.
        },
    }

    if train_cfg.get("grad_checkpoint"):
        mlx_config["grad_checkpoint"] = True

    # Write to a named temp file. mlx_lm lora requires a file path, not stdin.
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_mlx_lm_config.yaml", delete=False
    )
    yaml.dump(mlx_config, tmp, default_flow_style=False)
    tmp.close()

    cmd = [sys.executable, "-m", "mlx_lm", "lora", "-c", tmp.name]
    return cmd, tmp.name


def check_data_files(config: dict) -> bool:
    """Verify that the required data files exist before starting training."""
    ok = True
    for key in ("train", "valid"):
        path = config["data"].get(key)
        if path and not Path(path).exists():
            print(
                f"  ERROR: {key} data file not found: {path}\n"
                f"         Run: make prepare-data"
            )
            ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "SFT training wrapper for Trace Layer 2. "
            "Reads all parameters from YAML config."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to SFT YAML config (e.g. configs/sft_trace_qwen25_3b.yaml).",
    )
    parser.add_argument(
        "--model",
        help=(
            "Override the model path from config. Useful for smoke-testing "
            "with a smaller model without editing the YAML."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the training command without executing it.",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    if not Path(args.config).exists():
        print(f"ERROR: config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    print(f"\niris-ft-lab SFT training")
    print(f"  Config:  {args.config}")
    print(f"  Model:   {args.model or config['model']['path']}")
    print(f"  Adapter: {config['output']['adapter_path']}")
    print()

    # ── Validate data ─────────────────────────────────────────────────────────
    if not check_data_files(config):
        print("\nAborting: required data files are missing.")
        sys.exit(1)

    # ── Build command ─────────────────────────────────────────────────────────
    cmd, tmp_config = build_mlx_lm_command(config, model_override=args.model)

    if args.dry_run:
        print("Dry run — mlx-lm config that would be written:")
        with open(tmp_config) as f:
            print("  " + f.read().replace("\n", "\n  "))
        print("Command:")
        print("  " + " \\\n    ".join(cmd))
        print()
        os.unlink(tmp_config)
        sys.exit(0)

    # ── Run ───────────────────────────────────────────────────────────────────
    print("Launching MLX-LM LoRA training...")
    print("  " + " ".join(cmd))
    print()

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\nTraining complete. Adapter saved to: {config['output']['adapter_path']}")
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}.")
        print("Check the output above for MLX-LM error details.")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(
            "\nERROR: mlx_lm module not found.\n"
            "       Ensure your virtualenv is active and mlx-lm is installed:\n"
            "         source .venv/bin/activate\n"
            "         pip install mlx-lm"
        )
        sys.exit(1)
    finally:
        # Clean up the temporary config file.
        if os.path.exists(tmp_config):
            os.unlink(tmp_config)


if __name__ == "__main__":
    main()
