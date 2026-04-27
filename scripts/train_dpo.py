#!/usr/bin/env python3
"""
DPO training scaffold for Trace Layer 2 prospective-memory policy shaping.

Backend: mlx-lm-lora (mlx_lm_lora.train --train-mode dpo).
Install: pip install -U mlx-lm-lora
Tested against mlx-lm-lora v2.1.0.

Adapter-stacking strategy: SFT v2 adapter is fused into the base model first
(see scripts/fuse_sft.py -- requires --dequantize for 4-bit base models).
The fused model is used as both --model and --reference-model-path,
anchoring KL at SFT behavior.

Usage:
  python scripts/train_dpo.py --config configs/dpo_trace_qwen25_3b.yaml
  python scripts/train_dpo.py --config configs/dpo_trace_qwen25_3b.yaml --dry-run
  python scripts/train_dpo.py --check-only   # just test DPO availability
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import yaml


# ── DPO availability check ────────────────────────────────────────────────────

def check_dpo_available() -> tuple[bool, str]:
    """Probe for mlx_lm_lora installation."""
    try:
        import mlx_lm_lora
        ver = getattr(mlx_lm_lora, "__version__", "unknown")
        return True, f"mlx_lm_lora v{ver} available"
    except ImportError:
        return False, "mlx_lm_lora not installed. Run: pip install mlx-lm-lora"


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Command builder ───────────────────────────────────────────────────────────

def build_dpo_command(
    config: dict, model_override: str | None = None
) -> tuple[list[str], str]:
    """
    Build the mlx_lm_lora DPO training command from config.

    Returns (cmd, tmp_lora_config_path).
    Caller is responsible for deleting tmp_lora_config_path after use.

    mlx-lm-lora v2.1.0: no --lora-parameters CLI flag; LoRA params are
    passed via a generated temp YAML config (-c flag) instead.
    """
    model_path = model_override or config["model"]["path"]
    ref_path = config["reference_model"]["path"]
    train_cfg = config["training"]
    lora_cfg = config["lora"]
    dpo_cfg = config["dpo"]
    out_cfg = config["output"]
    data_cfg = config["data"]

    # Write lora_parameters (rank/dropout/scale) to a temp YAML because
    # mlx-lm-lora v2.1.0 has no CLI flag for them.
    lora_params = {
        "rank": lora_cfg["rank"],
        "dropout": lora_cfg["dropout"],
        "scale": float(lora_cfg["alpha"]) / lora_cfg["rank"],
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="dpo_lora_params_"
    )
    yaml.dump({"lora_parameters": lora_params}, tmp)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    cmd = [
        sys.executable, "-m", "mlx_lm_lora", "train",
        "--model", model_path,
        "--train",
        "--train-mode", "dpo",
        "--data", data_cfg["dir"],
        "--reference-model-path", ref_path,
        "--beta", str(dpo_cfg["beta"]),
        "--dpo-cpo-loss-type", dpo_cfg.get("loss_type", "sigmoid"),
        "--learning-rate", str(train_cfg["learning_rate"]),
        "--batch-size", str(train_cfg["batch_size"]),
        "--iters", str(train_cfg["iters"]),
        "--max-seq-length", str(train_cfg["max_seq_length"]),
        "--num-layers", str(lora_cfg["num_layers"]),
        "-c", tmp_path,
        "--adapter-path", out_cfg["adapter_path"],
        "--save-every", str(out_cfg["save_every"]),
        "--steps-per-report", str(out_cfg["steps_per_report"]),
    ]

    return cmd, tmp_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DPO training scaffold for Trace Layer 2 prospective-memory policy shaping. "
            "Uses mlx_lm_lora.train --train-mode dpo."
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
            "\nERROR: mlx_lm_lora is not installed.\n"
            "  Run: pip install mlx-lm-lora\n"
            "  Then: python scripts/fuse_sft.py  # fuse SFT v2 before DPO\n"
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
    print(f"\n  Config:   {args.config}")
    print(f"  Model:    {args.model or config['model']['path']}")
    print(f"  Ref:      {config['reference_model']['path']}")
    print(f"  Beta:     {config['dpo']['beta']}")
    print(f"  Adapter:  {config['output']['adapter_path']}")

    # ── Build command ──────────────────────────────────────────────────────
    cmd, tmp_lora_config = build_dpo_command(config, model_override=args.model)

    try:
        if args.dry_run:
            print("\nDry run — command that would be executed:")
            print("  " + " \\\n    ".join(cmd))
            import mlx_lm_lora as _lib
            lora_cfg = config["lora"]
            lora_params = {
                "rank": lora_cfg["rank"],
                "dropout": lora_cfg["dropout"],
                "scale": float(lora_cfg["alpha"]) / lora_cfg["rank"],
            }
            import json
            print(f"\n  LoRA params (passed via -c temp YAML, mlx-lm-lora v2.1.0 has no --lora-parameters flag):")
            print(f"    {json.dumps(lora_params)}")
            print()
            sys.exit(0)

        # ── Run ────────────────────────────────────────────────────────────
        import subprocess
        print("\nLaunching DPO training via mlx_lm_lora.train --train-mode dpo ...")
        print("  " + " \\\n    ".join(cmd))
        print()

        result = subprocess.run(cmd, check=True)
        print(f"\nDPO training complete. Adapter saved to: {config['output']['adapter_path']}")
        sys.exit(result.returncode)

    except SystemExit:
        raise
    except Exception as e:
        print(f"\nDPO training failed: {e}")
        sys.exit(1)
    finally:
        if tmp_lora_config and os.path.exists(tmp_lora_config):
            os.unlink(tmp_lora_config)


if __name__ == "__main__":
    main()
