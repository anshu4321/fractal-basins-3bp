#!/usr/bin/env bash
# Bootstraps a fresh RunPod container for the 3BP basin project.
# Idempotent: safe to re-run after pod respawn.
set -euo pipefail

WORKDIR=/workspace/3bp
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== upgrading runpodctl to 2.1.9 ==="
# The pod's pre-installed runpodctl is 1.14.x; its `send` output uses an older
# format that the local pod-helper recv regex does not parse. Replacing it with
# 2.1.9 keeps both ends of pod recv on the same protocol.
RPCTL_VERSION="${RPCTL_VERSION:-v2.1.9}"
if [[ "$(runpodctl -v 2>/dev/null | awk '{print $2}' | cut -d- -f1 || true)" != "${RPCTL_VERSION#v}" ]]; then
    arch="$(uname -m)"
    case "$arch" in
        x86_64|amd64) RPCTL_ARCH="amd64" ;;
        aarch64|arm64) RPCTL_ARCH="arm64" ;;
        *) echo "  unknown arch $arch, leaving runpodctl alone"; RPCTL_ARCH="" ;;
    esac
    if [[ -n "$RPCTL_ARCH" ]]; then
        url="https://github.com/runpod/runpodctl/releases/download/${RPCTL_VERSION}/runpodctl-linux-${RPCTL_ARCH}"
        target=/usr/local/bin/runpodctl
        curl -sL -o "$target" "$url" && chmod +x "$target"
        # Also drop a stable copy under /workspace so it survives pod respawns
        # (the bootstrap re-symlinks /usr/local/bin/runpodctl from this on restart).
        mkdir -p /workspace/3bp/bin
        cp "$target" /workspace/3bp/bin/runpodctl
        echo "  installed: $(runpodctl -v 2>/dev/null || echo failed)"
    fi
fi

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
