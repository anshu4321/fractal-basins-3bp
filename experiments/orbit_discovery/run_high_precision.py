"""Group 2 Test 2.4: High-precision verification of candidate novel orbits.

Two-stage verification:
1. Start from double-precision LM-refined ICs (v1, v2, T)
2. Run Newton refinement at mpmath precision using Taylor integrator
3. Report closure residual at the tightened precision

If the orbit is truly periodic, Newton at high precision should drive
the residual below 1e-20 (novelty threshold per PROMPT Test 2.4).
If it's a numerical artifact, the residual will plateau at some floor.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mpmath as mp
from mega3bp.taylor_hp import integrate, closure_residual


def F_residual_hp(v1, v2, T, n_steps, order, dps):
    """12-component residual state(T) - state(0) at mpmath precision."""
    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        q0 = [[mp.mpf(-1), mp.mpf(0)],
              [mp.mpf(1), mp.mpf(0)],
              [mp.mpf(0), mp.mpf(0)]]
        v1m = mp.mpf(v1) if isinstance(v1, mp.mpf) else mp.mpf(str(v1))
        v2m = mp.mpf(v2) if isinstance(v2, mp.mpf) else mp.mpf(str(v2))
        Tm = mp.mpf(T) if isinstance(T, mp.mpf) else mp.mpf(str(T))
        p0 = [[v1m, v2m], [v1m, v2m], [-2*v1m, -2*v2m]]

        qf, pf = integrate(q0, p0, Tm, n_steps=n_steps, order=order, dps=dps)

        F = []
        for i in range(3):
            for d in range(2):
                F.append(qf[i][d] - q0[i][d])
        for i in range(3):
            for d in range(2):
                F.append(pf[i][d] - p0[i][d])
        norm = mp.sqrt(sum(x * x for x in F))
        return F, norm
    finally:
        mp.mp.dps = old_dps


def newton_refine_hp(v1_0, v2_0, T0, n_steps=400, order=20, dps=30,
                     max_iter=8, eps=1e-15):
    """Newton-Raphson refinement at mpmath precision.

    Computes Jacobian by finite differences (12 function evaluations per iter).
    Not as efficient as forward-mode autodiff, but straightforward to implement.
    """
    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        v1 = mp.mpf(str(v1_0))
        v2 = mp.mpf(str(v2_0))
        T = mp.mpf(str(T0))

        best_residual = mp.mpf("inf")
        best_params = (v1, v2, T)

        print(f"  Newton-HP at {dps} dps, order {order}, {n_steps} steps")
        for it in range(max_iter):
            F, normF = F_residual_hp(v1, v2, T, n_steps, order, dps)
            print(f"    iter {it}: ||F|| = {mp.nstr(normF, 10)}")
            if normF < best_residual:
                best_residual = normF
                best_params = (v1, v2, T)
            if normF < mp.mpf(f"1e-{dps - 5}"):
                print(f"    converged")
                break

            # Jacobian via central differences: J[:, k] = (F(p + h e_k) - F(p - h e_k)) / (2h)
            h_fd = mp.mpf(eps)
            params = [v1, v2, T]
            J = []  # 12x3
            for k in range(3):
                params_p = params.copy()
                params_m = params.copy()
                params_p[k] = params_p[k] + h_fd
                params_m[k] = params_m[k] - h_fd
                Fp, _ = F_residual_hp(params_p[0], params_p[1], params_p[2], n_steps, order, dps)
                Fm, _ = F_residual_hp(params_m[0], params_m[1], params_m[2], n_steps, order, dps)
                col = [(Fp[i] - Fm[i]) / (2 * h_fd) for i in range(12)]
                J.append(col)
            # J[k][i] = dF_i/dp_k. We want 12x3 matrix with rows indexed by i.
            J_mat = mp.matrix(12, 3)
            for i in range(12):
                for k in range(3):
                    J_mat[i, k] = J[k][i]
            F_vec = mp.matrix(F)

            # Solve (J^T J) dp = J^T F (Gauss-Newton)
            JtJ = J_mat.T * J_mat
            JtF = J_mat.T * F_vec
            damping = mp.matrix([[mp.mpf("1e-8") if i == j else mp.mpf(0)
                                   for j in range(3)] for i in range(3)])
            dp = mp.lu_solve(JtJ + damping, JtF)

            v1 = v1 - dp[0]
            v2 = v2 - dp[1]
            T = T - dp[2]

        v1_b, v2_b, T_b = best_params
        return {
            "v1": str(v1_b),
            "v2": str(v2_b),
            "T": str(T_b),
            "residual": mp.nstr(best_residual, 20),
            "residual_float": float(best_residual),
            "converged_below_1e_20": bool(best_residual < mp.mpf("1e-20")),
            "converged_below_1e_15": bool(best_residual < mp.mpf("1e-15")),
        }
    finally:
        mp.mp.dps = old_dps


def collect_novel_candidates():
    """Gather all CANDIDATE NOVEL orbits from catalog match results."""
    match_path = Path(__file__).parent / "catalog_match_results.json"
    if not match_path.exists():
        return []
    with open(match_path) as f:
        d = json.load(f)
    return [m for m in d.get("matches", []) if m["classification"] == "CANDIDATE NOVEL"]


def main():
    OUT_DIR = Path(__file__).parent
    print("=" * 60)
    print("GROUP 2 TEST 2.4: HIGH-PRECISION VERIFICATION")
    print("=" * 60)

    candidates = collect_novel_candidates()
    if not candidates:
        print("No candidate novel orbits found.")
        return 1

    # Deduplicate by approximate (|v1|, |v2|, T)
    seen = set()
    unique = []
    for c in candidates:
        key = (round(abs(c["v1"]), 4), round(abs(c["v2"]), 4), round(c["T"], 2))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    print(f"Unique candidate novel orbits to verify: {len(unique)}")

    results = []
    for i, cand in enumerate(unique):
        print(f"\n--- Orbit {i+1}/{len(unique)}: "
              f"v1={cand['v1']:.6f}, v2={cand['v2']:.6f}, T={cand['T']:.4f} ---")
        t0 = time.perf_counter()
        result = newton_refine_hp(cand["v1"], cand["v2"], cand["T"],
                                  n_steps=400, order=20, dps=30, max_iter=6)
        result["original"] = {"v1": cand["v1"], "v2": cand["v2"], "T": cand["T"]}
        result["nearest_catalog"] = cand.get("name", "unknown")
        result["wall_time_s"] = time.perf_counter() - t0
        print(f"  Final ||F|| = {result['residual']} ({result['wall_time_s']:.0f}s)")
        results.append(result)

    # Save
    out = OUT_DIR / "high_precision_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")

    n_verified_20 = sum(1 for r in results if r["converged_below_1e_20"])
    n_verified_15 = sum(1 for r in results if r["converged_below_1e_15"])
    print(f"\n=== SUMMARY ===")
    print(f"Verified below 1e-20: {n_verified_20}/{len(unique)}")
    print(f"Verified below 1e-15: {n_verified_15}/{len(unique)}")
    print(f"Interpretation:")
    print(f"  If all fail 1e-20: our orbits are 'near-periodic' but might be numerical")
    print(f"  If all pass 1e-15: genuinely periodic (novel claim holds)")
    print(f"  If all pass 1e-20: rigorously verified novel orbits")


if __name__ == "__main__":
    sys.exit(main())
