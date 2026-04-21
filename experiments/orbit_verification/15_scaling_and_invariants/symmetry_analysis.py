"""
Symmetry analysis for the four candidate orbits.

Three checks:
  (1) Klein four-group theorem:     D_2 = {I, sigma_x, sigma_y, sigma_x sigma_y}
      preserves the Euler section. Each of A, B, C, D has 4 Klein images
      (+-v1, +-v2) that are ALL also periodic at the same T and |E|.
      Verified numerically at float64 via heyoka (closure residual).

  (2) Internal symmetry:            Does the orbit itself satisfy a half-period
      reflection relation, x(t + T/2) = sigma x(t) for some sigma in D_2?
      If yes, the four Klein images collapse to fewer distinct trajectories
      (the orbit has "extra internal symmetry"). This is the figure-8
      property.

  (3) Basin D_2 symmetry:           Does the ML label map on (v1, v2) obey
      label[v1, v2] == label[sigma(v1, v2)] for each sigma? x-reflection
      also swaps body-1 and body-2 escape labels, so labels must be relabeled
      accordingly before comparison.
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).parent
HP_JSON = HERE.parent / "06_high_precision" / "hp_heyoka_newton_results.json"
BASIN = Path(__file__).parent.parent.parent / "orbit_discovery" / "basin_map.npz"


# ----------------------------------------------------------------------
# Integrator: planar 3-body, unit masses, G=1, state = (x,y,vx,vy)_{i=1..3}
# ----------------------------------------------------------------------

def rhs_3body(t, s):
    r = s[[0, 4, 8]]      # x_1, x_2, x_3
    y = s[[1, 5, 9]]      # y_1, y_2, y_3
    vx = s[[2, 6, 10]]
    vy = s[[3, 7, 11]]
    ax = np.zeros(3)
    ay = np.zeros(3)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            dx = r[j] - r[i]
            dy = y[j] - y[i]
            r3 = (dx * dx + dy * dy) ** 1.5
            ax[i] += dx / r3
            ay[i] += dy / r3
    out = np.empty(12)
    for i, idx in enumerate([0, 4, 8]):
        out[idx]   = vx[i]
        out[idx+1] = vy[i]
        out[idx+2] = ax[i]
        out[idx+3] = ay[i]
    return out


def integrate(ic, t_end, rtol=1e-12, atol=1e-13):
    sol = solve_ivp(
        rhs_3body, (0.0, t_end), ic,
        method="DOP853", rtol=rtol, atol=atol, dense_output=False,
    )
    return sol.y[:, -1]


def pack_state(v1, v2):
    # r1=(-1,0), r2=(+1,0), r3=(0,0); p1=p2=(v1,v2), p3=-2(v1,v2); unit masses
    return np.array(
        [
            -1.0, 0.0, v1, v2,   # body 1: x, y, vx, vy
            +1.0, 0.0, v1, v2,   # body 2
            0.0, 0.0, -2 * v1, -2 * v2,  # body 3
        ],
        dtype=float,
    )


def integrate_to_T(v1, v2, T):
    return integrate(pack_state(v1, v2), T)


def integrate_to(v1, v2, t):
    return integrate(pack_state(v1, v2), t)


# ----------------------------------------------------------------------
# Klein group action on (v1, v2): 4 images
# ----------------------------------------------------------------------

KLEIN = {
    "I":           lambda v1, v2: ( v1,  v2),
    "sigma_x":     lambda v1, v2: (-v1,  v2),   # x -> -x (swaps bodies 1<->2)
    "sigma_y":     lambda v1, v2: ( v1, -v2),   # y -> -y
    "sigma_xy":    lambda v1, v2: (-v1, -v2),   # full inversion = T-reversal
}


def closure_residual(final_state, initial_state):
    return float(np.linalg.norm(final_state - initial_state))


# ----------------------------------------------------------------------
# Check (1): Klein images are periodic at the same T
# ----------------------------------------------------------------------

def check_klein_closure(orbits):
    print("## (1) Klein theorem: all 4 images close at the same T\n")
    print(f"{'Orbit':<5} {'image':<10} {'(v1,v2)':<30} {'closure ||x(T)-x(0)||':<25}")
    print("-" * 75)
    results = []
    for o in orbits:
        v1, v2, T = float(o["v1_HP"]), float(o["v2_HP"]), float(o["T_HP"])
        for name, g in KLEIN.items():
            v1p, v2p = g(v1, v2)
            ic = pack_state(v1p, v2p)
            final = integrate_to_T(v1p, v2p, T)
            res = closure_residual(final, ic)
            results.append({
                "orbit": o["name"], "image": name,
                "v1": v1p, "v2": v2p,
                "closure_residual": res,
            })
            print(f"{o['name']:<5} {name:<10} ({v1p:+.6f}, {v2p:+.6f})        {res:.3e}")
        print()
    return results


# ----------------------------------------------------------------------
# Check (2): Internal half-period symmetry
# ----------------------------------------------------------------------

def apply_config_symmetry(state, sigma):
    """Apply the FULL phase-space action of sigma in D_2 to a 12-dim state.
    sigma_x: x -> -x, vx -> -vx; body 1 <-> body 2
    sigma_y: y -> -y, vy -> -vy
    sigma_xy: composition
    """
    s = state.copy()
    if sigma == "I":
        return s
    if sigma == "sigma_x":
        # Flip x and vx for all bodies, then swap labels 1 <-> 2
        s = s.reshape(3, 4).copy()  # rows: body, cols: x, y, vx, vy
        s[:, 0] *= -1
        s[:, 2] *= -1
        s[[0, 1]] = s[[1, 0]]  # swap bodies 1 and 2
        return s.flatten()
    if sigma == "sigma_y":
        s = s.reshape(3, 4).copy()
        s[:, 1] *= -1
        s[:, 3] *= -1
        return s.flatten()
    if sigma == "sigma_xy":
        s = s.reshape(3, 4).copy()
        s[:, 0] *= -1
        s[:, 1] *= -1
        s[:, 2] *= -1
        s[:, 3] *= -1
        s[[0, 1]] = s[[1, 0]]
        return s.flatten()
    raise ValueError(sigma)


def time_reverse(state):
    """Flip all momenta: (x, y, vx, vy) -> (x, y, -vx, -vy)."""
    s = state.reshape(3, 4).copy()
    s[:, 2] *= -1
    s[:, 3] *= -1
    return s.flatten()


def check_internal_symmetry(orbits):
    """Test x(T/2) = sigma x(0) and x(T/2) = sigma tau x(0) where
    tau is time-reversal (p -> -p). The latter is the common symmetric-orbit
    condition (Broucke-Henon family, figure-8, etc.)."""
    print("## (2) Internal half-period symmetry: does x(T/2) = sigma [tau] x(0)?\n")
    print(f"{'Orbit':<5} {'sigma':<10} {'+T-rev?':<8} {'residual':>12}")
    print("-" * 45)
    results = []
    for o in orbits:
        v1, v2, T = float(o["v1_HP"]), float(o["v2_HP"]), float(o["T_HP"])
        ic = pack_state(v1, v2)
        half = integrate_to(v1, v2, T / 2)
        best = None
        best_res = np.inf
        row = {"orbit": o["name"]}
        for sigma in KLEIN:
            for treverse in (False, True):
                base = time_reverse(ic) if treverse else ic
                sigma_ic = apply_config_symmetry(base, sigma)
                res = float(np.linalg.norm(half - sigma_ic))
                tag = f"{sigma}{'+tau' if treverse else ''}"
                row[tag] = res
                print(f"{o['name']:<5} {sigma:<10} {str(treverse):<8} {res:>12.3e}")
                if res < best_res and not (sigma == "I" and not treverse):
                    best_res = res
                    best = tag
        row["best_nontrivial"] = best
        row["best_residual"] = best_res
        results.append(row)
        print(f"  -> best: {best}  (residual {best_res:.3e})")
        print()
    return results


# ----------------------------------------------------------------------
# Check (3): Basin D_2 symmetry
# ----------------------------------------------------------------------

def check_basin_symmetry():
    print("## (3) Basin D_2 symmetry on the (v1, v2) ML label map\n")
    if not BASIN.exists():
        print(f"  Basin file not found at {BASIN}; skipping.")
        return None
    d = np.load(BASIN, allow_pickle=True)
    labels = d["labels"]          # (N1, N2) int
    v1_grid = d["v1_grid"]
    v2_grid = d["v2_grid"]
    N1, N2 = labels.shape
    print(f"  Grid: v1 in [{v1_grid.min():.3f}, {v1_grid.max():.3f}], N={N1}")
    print(f"        v2 in [{v2_grid.min():.3f}, {v2_grid.max():.3f}], N={N2}")
    print(f"  Axis symmetry about zero: v1 sum = {v1_grid.sum():+.3e}, v2 sum = {v2_grid.sum():+.3e}")
    unique_labels = sorted(np.unique(labels).tolist())
    print(f"  Label set: {unique_labels}")

    # Tests:
    # sigma_x: flips v1 -> -v1 AND swaps body 1 <-> body 2 labels.
    #          If labels {0,1,2,3} = {periodic, esc-body1, esc-body2, esc-body3}
    #          in some order, we don't know the mapping a priori. So we measure
    #          the best agreement over all label-permutations and report.
    # sigma_y: flips v2 -> -v2, no label swap needed (y-reflection doesn't swap bodies).
    # sigma_xy: combination.

    from itertools import permutations

    def agreement(a, b):
        return float((a == b).mean())

    def best_label_remap_agreement(src, tgt, label_values):
        # Try all permutations of label values; return best agreement.
        best = -1.0
        best_perm = None
        for perm in permutations(label_values):
            remap = {old: new for old, new in zip(label_values, perm)}
            # Apply remap to src
            remapped = np.vectorize(lambda x: remap[x])(src)
            a = agreement(remapped, tgt)
            if a > best:
                best = a
                best_perm = perm
        return best, best_perm

    # sigma_y: reverse v2 axis (flip columns); try both identity and best label perm
    L_sy = labels[:, ::-1]
    a_sy_identity = agreement(labels, L_sy)
    a_sy_best, perm_sy = best_label_remap_agreement(L_sy, labels, unique_labels)

    # sigma_x: reverse v1 axis (flip rows), labels possibly swap 1<->2
    L_sx = labels[::-1, :]
    a_sx_identity = agreement(labels, L_sx)
    a_sx_best, perm_sx = best_label_remap_agreement(L_sx, labels, unique_labels)

    # sigma_xy: reverse both
    L_sxy = labels[::-1, ::-1]
    a_sxy_identity = agreement(labels, L_sxy)
    a_sxy_best, perm_sxy = best_label_remap_agreement(L_sxy, labels, unique_labels)

    print()
    print(f"  sigma_x (v1 -> -v1):")
    print(f"    identity map:             agreement = {a_sx_identity*100:.2f}%")
    print(f"    best label permutation:   agreement = {a_sx_best*100:.2f}%  perm={perm_sx}")
    print(f"  sigma_y (v2 -> -v2):")
    print(f"    identity map:             agreement = {a_sy_identity*100:.2f}%")
    print(f"    best label permutation:   agreement = {a_sy_best*100:.2f}%  perm={perm_sy}")
    print(f"  sigma_xy (full inversion):")
    print(f"    identity map:             agreement = {a_sxy_identity*100:.2f}%")
    print(f"    best label permutation:   agreement = {a_sxy_best*100:.2f}%  perm={perm_sxy}")

    return {
        "sigma_x_identity": a_sx_identity,
        "sigma_x_best": a_sx_best,
        "sigma_x_best_perm": perm_sx,
        "sigma_y_identity": a_sy_identity,
        "sigma_y_best": a_sy_best,
        "sigma_y_best_perm": perm_sy,
        "sigma_xy_identity": a_sxy_identity,
        "sigma_xy_best": a_sxy_best,
        "sigma_xy_best_perm": perm_sxy,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    orbits = json.loads(HP_JSON.read_text())

    klein_results = check_klein_closure(orbits)
    print()
    internal_results = check_internal_symmetry(orbits)
    print()
    basin_results = check_basin_symmetry()

    out = {
        "klein_closure": klein_results,
        "internal_symmetry": internal_results,
        "basin_symmetry": basin_results,
    }
    (HERE / "symmetry_results.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
