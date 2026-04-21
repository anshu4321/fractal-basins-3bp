"""Gauss-Newton refinement with per-candidate wall-time bound."""
import json
import signal
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

from .section import section_ic
from .eom import accelerations

HERE = Path(__file__).resolve().parent

T_GUESSES = [6.0, 7.75, 10.0]  # figure-8 full period ~7.75, plus generous bracket
PER_CANDIDATE_TIMEOUT = 60  # seconds


class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def handler(signum, frame):
        raise TimeoutError("timed out")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def _flatten(r, v): return np.concatenate([r.ravel(), v.ravel()])


def _ode(t, y):
    r = y[:6].reshape(3, 2); v = y[6:].reshape(3, 2)
    return np.concatenate([v.ravel(), accelerations(r).ravel()])


def integrate_from_u(u, T):
    r0, v0 = section_ic(tuple(u))
    y0 = _flatten(r0, v0)
    sol = solve_ivp(_ode, (0, T), y0, method="DOP853", rtol=1e-9, atol=1e-11,
                    max_step=0.2)
    return y0, sol.y[:, -1]


def newton_refine(u0, T0, max_iter=30, tol=1e-6):
    u = np.asarray(u0, dtype=float).copy()
    T = float(T0)
    for it in range(max_iter):
        y0, yT = integrate_from_u(u, T)
        residual = yT - y0
        res_norm = float(np.linalg.norm(residual))
        if res_norm < tol:
            return {"u": u.tolist(), "T": T, "residual": res_norm,
                    "iterations": it, "converged": True}
        eps_u, eps_T = 1e-6, 1e-4
        J = np.zeros((12, 3))
        for k, (du, dT) in enumerate([
            (np.array([eps_u, 0.0]), 0.0),
            (np.array([0.0, eps_u]), 0.0),
            (np.array([0.0, 0.0]), eps_T),
        ]):
            u_p = u + du; T_p = T + dT
            y0_p, yT_p = integrate_from_u(u_p, T_p)
            res_p = yT_p - y0_p
            J[:, k] = (res_p - residual) / (eps_u if k < 2 else eps_T)
        try:
            delta, *_ = np.linalg.lstsq(J, -residual, rcond=None)
        except np.linalg.LinAlgError:
            break
        max_step = max(abs(delta).max(), 1.0)
        if max_step > 0.5:
            delta = delta * (0.5 / max_step)
        u = u + delta[:2]
        T = T + delta[2]
        if T < 0.5 or T > 50:
            break
    return {"u": u.tolist(), "T": T, "residual": res_norm,
            "iterations": max_iter, "converged": res_norm < tol}


def run_all():
    cands = json.loads((HERE / "candidates_ICs.json").read_text())
    refined = []
    for i, c in enumerate(cands):
        u0 = (c["u_x"], c["u_y"])
        best = None
        try:
            with time_limit(PER_CANDIDATE_TIMEOUT):
                for T0 in T_GUESSES:
                    res = newton_refine(u0, T0)
                    if best is None or res["residual"] < best["residual"]:
                        best = res
                        if res["converged"]:
                            break
        except TimeoutError:
            best = best or {"u": list(u0), "T": float("nan"),
                            "residual": float("inf"),
                            "iterations": -1, "converged": False,
                            "timeout": True}
        status = "OK " if best.get("converged") else "FAIL"
        print(f"  #{i+1:2d} {status} u0=({c['u_x']:.3f},{c['u_y']:.3f}) -> "
              f"u=({best['u'][0]:.4f},{best['u'][1]:.4f}), "
              f"T={best['T']:.4f}, res={best['residual']:.2e}")
        if best.get("converged"):
            entry = {**c, **best}
            refined.append(entry)
    (HERE / "refined_candidates.json").write_text(json.dumps(refined, indent=2))
    print(f"\n{len(refined)}/{len(cands)} converged at residual < 1e-6")
    return refined


if __name__ == "__main__":
    run_all()
