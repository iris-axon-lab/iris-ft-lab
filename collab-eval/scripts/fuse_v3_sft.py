#!/usr/bin/env python3
"""
Fuse the SFT v3 LoRA adapter into the base model before DPO.

Strategy: fused model is used as both --model and --reference-model-path
for DPO, anchoring KL at v3 SFT behavior with a clean separable DPO LoRA on top.

Usage:
  python collab-eval/scripts/fuse_v3_sft.py           # skip if output already exists
  python collab-eval/scripts/fuse_v3_sft.py --force   # overwrite existing fused model
"""

import argparse
import subprocess
import sys
from pathlib import Path

SFT_ADAPTER = "adapters/sft_collab_eval_qwen25_3b_v3"
BASE_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
FUSED_OUT = "adapters/qwen25_3b_collab_v3_fused"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse SFT v3 adapter into base model.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing fused model.")
    args = parser.parse_args()

    if not Path(SFT_ADAPTER).exists():
        print(f"ERROR: SFT v3 adapter not found at {SFT_ADAPTER!r}. Cannot retrain — stopping.")
        sys.exit(1)

    if Path(FUSED_OUT).exists() and not args.force:
        print(f"Fused model already exists at {FUSED_OUT!r}. Skipping (use --force to overwrite).")
        sys.exit(0)

    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", BASE_MODEL,
        "--adapter-path", SFT_ADAPTER,
        "--save-path", FUSED_OUT,
        "--dequantize",   # required: merging LoRA deltas into 4-bit quant without this is a no-op
    ]
    print("Fusing SFT v3 adapter into base model ...")
    print("  " + " \\\n    ".join(cmd))
    result = subprocess.run(cmd, check=True)
    print(f"\nFused model saved to {FUSED_OUT!r}")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
