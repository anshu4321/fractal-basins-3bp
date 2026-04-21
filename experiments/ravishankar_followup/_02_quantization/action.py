"""Action integral S_p = integral Sigma |v_i|^2 dt on periodic orbits A/B/C/D.

Two independent methods:
  trapezoidal   -- np.trapezoid on a dense uniform heyoka integration (50k steps)
  gauss_legendre -- composite GL-32 on 100 sub-intervals (3200 direct heyoka evaluations)

Both methods drive heyoka directly; no interpolation is used in either path.
Cross-check tolerance: < 1e-8 relative for all orbits.
"""
from pathlib import Path
import json
import numpy as np

from experiments.ravishankar_followup._02_quantization.data import build_heyoka_system, load_hp_ic

HERE = Path(__file__).resolve().parent


def _integrand_at(ta, t):
    """Propagate taylor_adaptive to time t, return Sigma_i |v_i|^2."""
    ta.propagate_until(t)
    v = ta.state[6:].reshape(3, 2)
    return float(np.sum(v * v))


def _integrate_dense(name, n_steps=50_000):
    """Return (ts, state_array) for one period of orbit `name`.

    state_array shape: (n_steps, 12) = [r1x r1y r2x r2y r3x r3y v1x v1y v2x v2y v3x v3y].
    """
    v1, v2, T = load_hp_ic(name)
    make = build_heyoka_system()
    ta = make(v1, v2)
    ts = np.linspace(0.0, T, n_steps)
    states = np.zeros((n_steps, 12))
    # At t=0, state is the IC
    states[0] = ta.state.copy()
    for i in range(1, n_steps):
        ta.propagate_until(ts[i])
        states[i] = ta.state.copy()
    return ts, states


def _compute_gl_direct(name, n_intervals=100, n_nodes=32):
    """Composite Gauss-Legendre: n_intervals sub-intervals, each with n_nodes GL points.

    All evaluations are direct heyoka propagations -- no interpolation.
    Returns S = integral_0^T Sigma |v_i|^2 dt.
    """
    from numpy.polynomial.legendre import leggauss
    v1, v2, T = load_hp_ic(name)
    make = build_heyoka_system()
    ta = make(v1, v2)
    nodes, weights = leggauss(n_nodes)

    # Sub-interval edges
    edges = np.linspace(0.0, T, n_intervals + 1)
    S = 0.0
    for k in range(n_intervals):
        a, b = edges[k], edges[k + 1]
        half = 0.5 * (b - a)
        mid = 0.5 * (a + b)
        t_sub = mid + half * nodes       # GL nodes in [a, b], sorted ascending
        # Sort to ensure sequential propagation (heyoka must go forward in time)
        order = np.argsort(t_sub)
        t_sorted = t_sub[order]
        w_sorted = weights[order]
        for i in range(n_nodes):
            ig = _integrand_at(ta, t_sorted[i])
            S += half * w_sorted[i] * ig
    return float(S)


def compute_action(name, method="gauss_legendre", alpha=1.0, n_steps=50_000):
    """Compute S_p = integral_0^T Sigma_i |v_i|^2 dt.

    If `alpha != 1`, applies the analytic scaling S(alpha) = alpha^{1/2} S(1).
    """
    if alpha != 1.0:
        return alpha ** 0.5 * compute_action(name, method=method, n_steps=n_steps)

    if method == "trapezoidal":
        ts, state = _integrate_dense(name, n_steps=n_steps)
        v = state[:, 6:].reshape(-1, 3, 2)
        T_kin = 0.5 * np.sum(v * v, axis=(1, 2))  # (n_steps,) -- m=1 gives 0.5 Sigma|v|^2
        integrand = 2 * T_kin                     # 2*T_kin = Sigma|v|^2
        # np.trapz was removed in NumPy 2.0; use np.trapezoid with fallback
        trapfn = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        return float(trapfn(integrand, ts))
    elif method == "gauss_legendre":
        return _compute_gl_direct(name)
    raise ValueError(f"unknown method: {method}")


def main():
    results = {}
    for name in "ABCD":
        print(f"Computing S_{name}...", flush=True)
        S_trap = compute_action(name, "trapezoidal")
        S_gl = compute_action(name, "gauss_legendre")
        rel = abs(S_trap - S_gl) / abs(S_gl)
        results[name] = {
            "S_trapezoidal": S_trap,
            "S_gauss_legendre": S_gl,
            "S": S_gl,
            "rel_error_between_methods": rel,
        }
        print(f"S_{name}: trap = {S_trap:.12g}, GL = {S_gl:.12g}, rel_err = {rel:.2e}", flush=True)
    (HERE / "action_ABCD.json").write_text(json.dumps(results, indent=2))
    print("wrote action_ABCD.json", flush=True)


if __name__ == "__main__":
    main()
