#!/usr/bin/env bash
# iris-ft-lab environment setup
# Requires Python 3.10+ on an Apple Silicon Mac.
# Usage: bash setup.sh

set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

echo "==> Checking Python version"
$PYTHON --version

echo "==> Creating virtual environment in ${VENV_DIR}/"
$PYTHON -m venv "$VENV_DIR"

echo "==> Activating virtual environment"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip --quiet

echo "==> Installing core dependencies (mlx, mlx-lm, pyyaml)"
pip install mlx mlx-lm pyyaml

echo "==> Installing dev dependencies (pytest, notebook, matplotlib, pandas)"
pip install pytest pytest-cov notebook matplotlib pandas

echo ""
echo "Setup complete."
echo ""
echo "  Activate:  source .venv/bin/activate"
echo "  Verify:    python scripts/setup_verify.py"
echo "  Quick run: make verify"
