#!/bin/bash
# Master overnight experiment chain.
# Waits for current velocity_space_search.py to finish, then runs:
# 1. extract_basin_map.py (full 500x500 map, ~47 min)
# 2. run_baselines_velocity.py (Group 4, 4 methods x 3 seeds, ~8-12 hr)
# 3. run_stability_velocity.py (Group 3 monodromy, ~30 min)
# 4. run_catalog_match.py (Group 2 catalog match, ~10 min)
#
# All output goes to their respective .log files in experiments/orbit_discovery/

set -e  # stop on error

cd ~/3bp/experiments/orbit_discovery
echo "========================================"
echo "MASTER OVERNIGHT QUEUE STARTED: $(date)"
echo "========================================"

# 1. Wait for current velocity_space_search.py to finish
echo ""
echo "[Stage 0] Waiting for current velocity_space_search.py to finish..."
while pgrep -f "python3.*velocity_space_search.py" > /dev/null; do
    sleep 30
done
echo "[Stage 0] Current run finished: $(date)"

# 2. Extract full 500x500 basin map
echo ""
echo "[Stage 1] Extracting full basin map: $(date)"
PYTHONUNBUFFERED=1 python3 -u extract_basin_map.py > extract_basin.log 2>&1
echo "[Stage 1] Basin map done: $(date)"

# 3. Catalog comparison (quick, can run even before baselines)
echo ""
echo "[Stage 2] Catalog comparison: $(date)"
PYTHONUNBUFFERED=1 python3 -u run_catalog_match.py > catalog_match.log 2>&1
echo "[Stage 2] Catalog comparison done: $(date)"

# 4. Stability analysis on verified orbits so far
echo ""
echo "[Stage 3] Stability analysis: $(date)"
PYTHONUNBUFFERED=1 python3 -u run_stability_velocity.py > stability.log 2>&1 || echo "Stability failed, continuing"
echo "[Stage 3] Stability done: $(date)"

# 5. Group 4 baselines (THE BIG ONE)
echo ""
echo "[Stage 4] Group 4 baselines (3 seeds x 4 methods): $(date)"
PYTHONUNBUFFERED=1 python3 -u run_baselines_velocity.py > baselines.log 2>&1
echo "[Stage 4] Baselines done: $(date)"

# 6. Re-run stability with ALL verified orbits (from baselines too)
echo ""
echo "[Stage 5] Final stability analysis: $(date)"
PYTHONUNBUFFERED=1 python3 -u run_stability_velocity.py > stability_final.log 2>&1
echo "[Stage 5] Final stability done: $(date)"

# 7. Re-run catalog match with all verified orbits
echo ""
echo "[Stage 6] Final catalog match: $(date)"
PYTHONUNBUFFERED=1 python3 -u run_catalog_match.py > catalog_final.log 2>&1
echo "[Stage 6] Final catalog match done: $(date)"

echo ""
echo "========================================"
echo "MASTER QUEUE COMPLETE: $(date)"
echo "========================================"
