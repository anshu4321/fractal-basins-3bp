"""High-precision Taylor-series integrator for the planar 3-body problem.

Uses mpmath for arbitrary precision arithmetic. Implements the recurrence
for 1/r^3 as a power series in time, which is the trickiest part.

For each time step h:
  1. Represent q(t), p(t) as truncated Taylor series around the current time
  2. Compute all pairwise separations r_ij(t) and their norms squared d_ij(t)
  3. Compute 1/d_ij^(3/2) via power-series recurrence
  4. Compute force F_i(t) = sum_j r_ji(t) / |r_ji(t)|^3 as power series
  5. Integrate: q_i^[k+1] = p_i^[k]/(k+1), p_i^[k+1] = F_i^[k]/(k+1)
  6. Evaluate at t=h: q(h) = sum q^[k] h^k

Reference: Jorba & Zou 2005 "A Software Package for the Numerical
Integration of ODEs by Means of High-Order Taylor Methods."
"""
from __future__ import annotations

import mpmath as mp


def _taylor_multiply(A, B, N):
    """Taylor series product C = A * B up to order N.

    A, B, C are lists of length N+1.
    C[n] = sum_{m=0}^n A[m] * B[n-m]
    """
    C = [mp.mpf(0)] * (N + 1)
    for n in range(N + 1):
        s = mp.mpf(0)
        for m in range(n + 1):
            s += A[m] * B[n - m]
        C[n] = s
    return C


def _taylor_pow_alpha(G, alpha, N):
    """Taylor series of G^alpha, given G as list of coefficients up to order N.

    Recurrence derived from g * f' = alpha * g' * f:
      f_{n+1} = [alpha * sum_{m=0}^n (m+1) g_{m+1} f_{n-m}
                - sum_{m=1}^n g_m * (n-m+1) * f_{n-m+1}] / [(n+1) * g_0]
    """
    F = [mp.mpf(0)] * (N + 1)
    F[0] = G[0] ** alpha
    for n in range(N):
        if n + 1 > N:
            break
        term1 = mp.mpf(0)
        for m in range(n + 1):
            if m + 1 <= N:
                term1 += alpha * (m + 1) * G[m + 1] * F[n - m]
        term2 = mp.mpf(0)
        for m in range(1, n + 1):
            if n - m + 1 <= N:
                term2 += G[m] * (n - m + 1) * F[n - m + 1]
        F[n + 1] = (term1 - term2) / ((n + 1) * G[0])
    return F


def _eval_taylor(F, h, N):
    """Evaluate Taylor series sum F[k] * h^k for k=0..N, Horner-style."""
    acc = F[N]
    for k in range(N - 1, -1, -1):
        acc = acc * h + F[k]
    return acc


def taylor_step(q, p, h, order=24):
    """One Taylor step of size h for the 3-body problem at mpmath precision.

    Args:
        q: list of 3 bodies, each [x, y] as mpmath numbers
        p: same shape
        h: step size (mpmath number)
        order: Taylor truncation order (N)

    Returns:
        (q_new, p_new) at time t + h
    """
    N = order
    zero = mp.mpf(0)

    # Initialize Taylor coefficients: q[i][d][k], p[i][d][k]
    # i = body (0..2), d = dim (0..1), k = Taylor order (0..N)
    q_tay = [[[zero] * (N + 1) for _ in range(2)] for _ in range(3)]
    p_tay = [[[zero] * (N + 1) for _ in range(2)] for _ in range(3)]
    for i in range(3):
        for d in range(2):
            q_tay[i][d][0] = mp.mpf(q[i][d])
            p_tay[i][d][0] = mp.mpf(p[i][d])

    # Compute Taylor coefficients order by order
    for k in range(N):
        # Compute r_ji[d] = q_j - q_i as Taylor series (for each pair, dim)
        # Only 3 unordered pairs: (0,1), (0,2), (1,2)
        # We'll compute r_{i,j} for i<j
        r = {}
        for i in range(3):
            for j in range(i + 1, 3):
                r[(i, j)] = [
                    [q_tay[j][d][kk] - q_tay[i][d][kk] for kk in range(k + 1)]
                    for d in range(2)
                ]

        # Compute d_ij = r_ij[0]^2 + r_ij[1]^2 as Taylor series
        d_pair = {}
        for (i, j), rij in r.items():
            rx = rij[0]
            ry = rij[1]
            rx2 = _taylor_multiply(rx, rx, k)
            ry2 = _taylor_multiply(ry, ry, k)
            d_pair[(i, j)] = [rx2[kk] + ry2[kk] for kk in range(k + 1)]

        # Compute u_ij = d_ij^(-3/2) as Taylor series
        u_pair = {}
        for (i, j), dij in d_pair.items():
            u_pair[(i, j)] = _taylor_pow_alpha(dij, mp.mpf("-1.5"), k)

        # Compute F_i = sum over pairs of signed (q_j - q_i) * u_ij
        # For pair (i, j) with i < j:
        #   contribution to body i: +r_ij * u_ij (pointing from i to j)
        #   contribution to body j: -r_ij * u_ij
        F_tay = [[[zero] * (k + 1) for _ in range(2)] for _ in range(3)]
        for (i, j), rij in r.items():
            uij = u_pair[(i, j)]
            for d in range(2):
                contrib = _taylor_multiply(rij[d], uij, k)
                for kk in range(k + 1):
                    F_tay[i][d][kk] += contrib[kk]   # body i: +r_ij * u
                    F_tay[j][d][kk] -= contrib[kk]   # body j: -r_ij * u

        # Advance Taylor coefficients: q^[k+1] = p^[k]/(k+1), p^[k+1] = F^[k]/(k+1)
        kp1 = k + 1
        for i in range(3):
            for d in range(2):
                q_tay[i][d][k + 1] = p_tay[i][d][k] / kp1
                p_tay[i][d][k + 1] = F_tay[i][d][k] / kp1

    # Evaluate at t = h
    q_new = [[_eval_taylor(q_tay[i][d], h, N) for d in range(2)] for i in range(3)]
    p_new = [[_eval_taylor(p_tay[i][d], h, N) for d in range(2)] for i in range(3)]
    return q_new, p_new


def integrate(q0, p0, T, n_steps=2000, order=24, dps=50):
    """Integrate 3BP from (q0, p0) for time T using fixed-step Taylor.

    Args:
        q0, p0: initial conditions (Python lists / numpy arrays, converted to mpmath)
        T: total time
        n_steps: number of steps
        order: Taylor order per step
        dps: decimal places of precision

    Returns:
        (qf, pf) at time T, as lists of mpmath numbers
    """
    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        q = [[mp.mpf(q0[i][d]) for d in range(2)] for i in range(3)]
        p = [[mp.mpf(p0[i][d]) for d in range(2)] for i in range(3)]
        h = mp.mpf(T) / n_steps

        for step in range(n_steps):
            q, p = taylor_step(q, p, h, order=order)

        return q, p
    finally:
        mp.mp.dps = old_dps


def closure_residual(q0, p0, qf, pf):
    """L2 norm of state(T) - state(0) at mpmath precision.

    Returns an mpmath number.
    """
    s = mp.mpf(0)
    for i in range(3):
        for d in range(2):
            dq = mp.mpf(qf[i][d]) - mp.mpf(q0[i][d])
            dp = mp.mpf(pf[i][d]) - mp.mpf(p0[i][d])
            s += dq * dq + dp * dp
    return mp.sqrt(s)


def verify_orbit_hp(v1, v2, T, n_steps=2000, order=24, dps=50):
    """Verify a velocity-space orbit at high precision.

    Starts from collinear configuration ((-1, 0), (1, 0), (0, 0)) with
    velocities (v1, v2), (v1, v2), (-2v1, -2v2).

    Returns dict with closure residual at the requested precision.
    """
    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        q0 = [[mp.mpf(-1), mp.mpf(0)],
              [mp.mpf(1), mp.mpf(0)],
              [mp.mpf(0), mp.mpf(0)]]
        v1m = mp.mpf(str(v1))
        v2m = mp.mpf(str(v2))
        p0 = [[v1m, v2m],
              [v1m, v2m],
              [-2 * v1m, -2 * v2m]]
        Tm = mp.mpf(str(T))

        qf, pf = integrate(q0, p0, Tm, n_steps=n_steps, order=order, dps=dps)

        residual = closure_residual(q0, p0, qf, pf)

        return {
            "v1": str(v1),
            "v2": str(v2),
            "T": str(T),
            "n_steps": n_steps,
            "order": order,
            "dps": dps,
            "residual": mp.nstr(residual, 20),
            "residual_float": float(residual),
            "qf": [[mp.nstr(qf[i][d], 16) for d in range(2)] for i in range(3)],
            "pf": [[mp.nstr(pf[i][d], 16) for d in range(2)] for i in range(3)],
        }
    finally:
        mp.mp.dps = old_dps
