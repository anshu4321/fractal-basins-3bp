#!/usr/bin/env bash
# Convenience: run all Phase 2 chunk 1 outputs in one shot on the pod.
#
#   cd /workspace/3bp
#   bash scripts/run_phase2_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# 1. Re-generate the figure-eight energy plot with the Dirac style
echo "=== [1/3] make_energy_plot ==="
venv/bin/python scripts/make_energy_plot.py

# 2. Aesthetic figure-eight animation (mp4 + gif + still)
echo "=== [2/3] animate_figure_eight ==="
venv/bin/python scripts/animate_figure_eight.py

# 3. Phase 2 smoke test: batch integrator + basin preview + dashboard
echo "=== [3/3] phase2_smoke ==="
venv/bin/python scripts/phase2_smoke.py

echo
echo "=== all Phase 2 outputs generated in figures/ ==="
ls -la figures/
