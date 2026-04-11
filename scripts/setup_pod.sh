#!/usr/bin/env bash
# Bootstraps a fresh RunPod container for the 3BP basin project.
# Idempotent: safe to re-run after pod respawn.
set -euo pipefail

WORKDIR=/workspace/3bp
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== creating venv ==="
if [[ ! -x venv/bin/python ]]; then
    python3 -m venv venv
fi

echo "=== upgrading pip ==="
venv/bin/pip install --quiet --upgrade pip wheel setuptools

echo "=== installing jax[cuda12] and project deps ==="
venv/bin/pip install --no-cache-dir \
    "jax[cuda12]" \
    diffrax \
    equinox \
    optax \
    matplotlib \
    pyarrow \
    pandas \
    pytest \
    numpy \
    scipy \
    tqdm

echo
echo "=== installed versions ==="
venv/bin/python --version
venv/bin/pip list 2>/dev/null | grep -iE "^(jax|jaxlib|diffrax|equinox|optax|numpy|scipy|matplotlib|pyarrow) " || true

echo
echo "=== bootstrap complete ==="
echo "activate with: source /workspace/3bp/venv/bin/activate"
