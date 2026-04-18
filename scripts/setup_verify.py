#!/usr/bin/env python3
"""
Verify the iris-ft-lab Python environment.

Checks:
  - Python version (3.10+ required)
  - mlx import and Metal/Apple Silicon availability
  - mlx_lm import
  - pyyaml import

Usage:
  python scripts/setup_verify.py
  python scripts/setup_verify.py --quiet   # exit-code only, minimal output
"""

import sys
import argparse


def check_python(quiet: bool) -> bool:
    version = sys.version_info
    ok = version >= (3, 10)
    if not quiet:
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] Python {version.major}.{version.minor}.{version.micro}")
    if not ok:
        print(
            f"  ERROR: Python 3.10+ required. "
            f"Current: {version.major}.{version.minor}. "
            f"Install via pyenv or conda."
        )
    return ok


def check_mlx(quiet: bool) -> bool:
    try:
        import mlx.core as mx

        # Check whether the Metal GPU backend is available.
        # mx.default_device() returns the active device.
        device = mx.default_device()
        device_str = str(device)
        metal_available = "gpu" in device_str.lower() or "metal" in device_str.lower()

        if not quiet:
            print(f"  [OK ] mlx imported — default device: {device_str}")
            if metal_available:
                print("  [OK ] Metal GPU backend active (Apple Silicon confirmed)")
            else:
                # CPU fallback still works; just flag it.
                print(
                    "  [WARN] Metal GPU not detected — running on CPU. "
                    "Training will be slow. Ensure you're on Apple Silicon."
                )
        return True

    except ImportError:
        print(
            "  [FAIL] mlx not found.\n"
            "         Install: pip install mlx\n"
            "         Requires Apple Silicon (M1/M2/M3/M4)."
        )
        return False
    except Exception as e:
        print(f"  [FAIL] mlx imported but error during device check: {e}")
        return False


def check_mlx_lm(quiet: bool) -> bool:
    try:
        import mlx_lm  # noqa: F401

        # Verify the key entry points are accessible.
        from mlx_lm import load, generate  # noqa: F401

        if not quiet:
            print("  [OK ] mlx_lm imported — load() and generate() available")
        return True

    except ImportError:
        print(
            "  [FAIL] mlx_lm not found.\n"
            "         Install: pip install mlx-lm"
        )
        return False
    except Exception as e:
        print(f"  [FAIL] mlx_lm import error: {e}")
        return False


def check_pyyaml(quiet: bool) -> bool:
    try:
        import yaml  # noqa: F401

        if not quiet:
            print("  [OK ] pyyaml imported")
        return True
    except ImportError:
        print("  [FAIL] pyyaml not found.\n         Install: pip install pyyaml")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify iris-ft-lab Python environment and MLX availability."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output; use exit code to detect failure.",
    )
    args = parser.parse_args()

    if not args.quiet:
        print("\niris-ft-lab environment check")
        print("─" * 40)

    results = [
        check_python(args.quiet),
        check_mlx(args.quiet),
        check_mlx_lm(args.quiet),
        check_pyyaml(args.quiet),
    ]

    all_ok = all(results)

    if not args.quiet:
        print("─" * 40)
        if all_ok:
            print("All checks passed. Ready to run iris-ft-lab.\n")
        else:
            failed = sum(1 for r in results if not r)
            print(f"{failed} check(s) failed. Fix the issues above, then re-run.\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
