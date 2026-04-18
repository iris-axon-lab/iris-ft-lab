#!/usr/bin/env python3
"""
DPO training scaffold for Trace prospective-memory policy shaping.

STATUS: Scaffold / stub.

MLX-native DPO support is evolving. This script:
  1. Loads the DPO config from YAML.
  2. Checks whether the installed mlx-lm version supports DPO training.
  3. If DPO is available, constructs and launches the training command.
  4. If DPO is not yet available, exits honestly with clear TODO guidance.

This file is structured for easy extension — see the TODO sections below.

Usage:
  python scripts/train_dpo.py --config configs/dpo_trace_qwen25_3b.yaml
  python scripts/train_dpo.py --config configs/dpo_trace_qwen25_3b.yaml --dry-run
  python scripts/train_dpo.py --check-only   # just test DPO availability
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


# ── DPO availability check ────────────────────────────────────────────────────

def check_dpo_available() -> tuple[bool, str]:
    """
    Probe the installed mlx-lm version for DPO support.

    Returns (available: bool, message: str).
    """
    try:
        import mlx_lm

        version = getattr(mlx_lm, "__version__", "unknown")

        # TODO: Update this check once the mlx-lm DPO API stabilizes.
        # As of mlx-lm 0.19.x, DPO training is available via the --dpo flag
        # on mlx_lm.lora. Check the changelog for your installed version.
        try:
            # Attempt to import DPO-specific internals to verify availability.
            from mlx_lm.tuner.dpo_trainer import DPOTrainer  # noqa: F401
            return True, f"DPO trainer found (mlx-lm {version})"
        except ImportError:
            return (
                False,
                f"DPO trainer not found in mlx-lm {version}. "
                f"Upgrade: pip install --upgrade mlx-lm"
            )

    except ImportError:
        return False, "mlx_lm not installed. Run: pip install mlx-lm"


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Command builder ───────────────────────────────────────────────────────────

def build_dpo_command(config: dict, model_override: str | None = None) -> list[str]:
    """
    Build the mlx_lm.lora DPO training command from config.

    TODO: Verify the exact DPO flags for your installed mlx-lm version.
    The --dpo flag and --ref-model flag names may change between releases.
    Check: python -m mlx_lm.lora --help
    """
    model_path = model_override or config["model"]["path"]
    train_cfg = config["training"]
    lora_cfg = config["lora"]
    dpo_cfg = config["dpo"]
    out_cfg = config["output"]
    data_cfg = config["data"]

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model_path,
        "--train",
        "--dpo",                                         # enable DPO mode
        "--data", str(Path(data_cfg["train"]).parent),
        "--batch-size", str(train_cfg["batch_size"]),
        "--learning-rate", str(train_cfg["learning_rate"]),
        "--lora-rank", str(lora_cfg["rank"]),
        "--lora-alpha", str(lora_cfg["alpha"]),
        "--max-seq-length", str(train_cfg["max_seq_length"]),
        "--adapter-path", out_cfg["adapter_path"],
        "--save-every", str(out_cfg["save_every"]),
        "--steps-per-report", str(out_cfg["steps_per_report"]),
        "--dpo-beta", str(dpo_cfg["beta"]),
    ]

    # Reference model for KL divergence baseline.
    ref_model = dpo_cfg.get("reference_model")
    if ref_model:
        cmd += ["--ref-model", ref_model]

    return cmd


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DPO training scaffold for Trace prospective-memory policy shaping. "
            "Checks MLX-LM DPO availability before attempting to train."
        )
    )
    parser.add_argument(
        "--config",
        help="Path to DPO YAML config (e.g. configs/dpo_trace_qwen25_3b.yaml).",
    )
    parser.add_argument(
        "--model",
        help="Override the model path from config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the training command without executing it.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check DPO availability; do not train.",
    )
    args = parser.parse_args()

    # ── Check DPO availability ─────────────────────────────────────────────
    available, msg = check_dpo_available()
    print(f"\niris-ft-lab DPO training scaffold")
    print(f"  DPO status: {'AVAILABLE' if available else 'NOT AVAILABLE'} — {msg}")

    if args.check_only:
        sys.exit(0 if available else 1)

    if not available:
        print(
            "\n── TODO: DPO not yet available in your mlx-lm installation ──────────\n"
            "\n"
            "  Next steps:\n"
            "  1. Upgrade mlx-lm:   pip install --upgrade mlx-lm\n"
            "  2. Check DPO docs:   python -m mlx_lm.lora --help | grep dpo\n"
            "  3. If DPO is available under a different flag, update\n"
            "     build_dpo_command() in this file.\n"
            "  4. See notebook 03_dpo_memory_policy.ipynb for the policy\n"
            "     design rationale and what the training objective targets.\n"
            "\n"
            "  When DPO becomes available:\n"
            "  - Run: make prepare-data  (ensures data/processed/dpo_*.jsonl exists)\n"
            "  - Run: make dpo\n"
            "\n"
            "  Data format expected: {\"prompt\": \"...\", \"chosen\": \"...\", \"rejected\": \"...\"}\n"
            "  See data/sample_dpo.jsonl for examples.\n"
        )
        sys.exit(1)

    # ── Load config ────────────────────────────────────────────────────────
    if not args.config:
        print("\nERROR: --config is required when not using --check-only.")
        sys.exit(1)

    if not Path(args.config).exists():
        print(f"ERROR: config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    print(f"\n  Config:  {args.config}")
    print(f"  Model:   {args.model or config['model']['path']}")
    print(f"  Beta:    {config['dpo']['beta']}")
    print(f"  Adapter: {config['output']['adapter_path']}")

    # ── Build command ──────────────────────────────────────────────────────
    cmd = build_dpo_command(config, model_override=args.model)

    if args.dry_run:
        print("\nDry run — command that would be executed:")
        print("  " + " \\\n    ".join(cmd))
        print()
        sys.exit(0)

    # ── Run ────────────────────────────────────────────────────────────────
    print("\nLaunching MLX-LM DPO training...")
    print("  " + " \\\n    ".join(cmd))
    print()

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\nDPO training complete. Adapter saved to: {config['output']['adapter_path']}")
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"\nDPO training failed with exit code {e.returncode}.")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
