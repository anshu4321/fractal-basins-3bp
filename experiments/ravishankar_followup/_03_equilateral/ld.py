"""Compute the label-disagreement (LD) field on a 512x512 fine grid.

LD is the Shannon entropy of the classifier softmax at each (u_x, u_y) point.
High-LD regions mark the fractal basin boundary - the search prior for new
periodic orbits.
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FINE_N = 512
U_LO, U_HI = -2.0, 2.0


def _compute_energy_grid(u_x, u_y):
    """Compute E for each (u_x, u_y) point using the section IC."""
    from .section import section_ic
    from .eom import energy
    E = np.zeros_like(u_x)
    for i in range(u_x.shape[0]):
        for j in range(u_x.shape[1]):
            r, v = section_ic((u_x[i, j], u_y[i, j]))
            E[i, j] = energy(r, v)
    return E


def compute_ld_field():
    # Fine grid
    us = np.linspace(U_LO, U_HI, FINE_N)
    U_x, U_y = np.meshgrid(us, us, indexing="ij")

    # E field: compute analytically (cheap - just position + potential)
    import time
    t0 = time.time()
    E = _compute_energy_grid(U_x, U_y).astype(np.float32)
    print(f"E grid computed in {time.time() - t0:.2f}s")

    # Feature vector (matches classifier.py featurize)
    X = np.stack([
        U_x, U_y, E, np.sqrt(U_x * U_x + U_y * U_y),
        U_x * U_y, U_x * U_x, U_y * U_y,
    ], axis=-1).astype(np.float32).reshape(-1, 7)

    # Normalize using classifier's stored mu/sd
    import pickle
    meta = pickle.load(open(HERE / "classifier.pkl", "rb"))
    mu = np.asarray(meta["feature_mu"], dtype=np.float32)
    sd = np.asarray(meta["feature_sd"], dtype=np.float32)
    X_norm = (X - mu) / np.maximum(sd, 1e-8)

    # Load classifier
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from mega3bp.ml import BasinMLP, fourier_features

    arch = meta["arch"]
    n_freqs = int(arch.get("fourier_n_freqs", arch.get("n_freqs", 8)))
    template = BasinMLP(
        in_dim=arch["in_dim"],
        hidden=arch["hidden"],
        n_layers=arch["n_layers"],
        out_dim=arch["out_dim"],
        key=jax.random.PRNGKey(0),
    )
    model = eqx.tree_deserialise_leaves(str(HERE / "classifier.eqx"), template)

    # Apply Fourier features + model forward on a (N,7) batch
    @jax.jit
    def forward_batch(X_raw_norm):
        X_lift = fourier_features(X_raw_norm, n_freqs=n_freqs)
        return jax.vmap(model)(X_lift)

    t0 = time.time()
    logits = forward_batch(jnp.asarray(X_norm))
    probs = jax.nn.softmax(logits, axis=-1)
    ld = -jnp.sum(probs * jnp.log(probs + 1e-12), axis=-1)
    ld_np = np.asarray(ld).reshape(FINE_N, FINE_N)
    print(
        f"LD field computed in {time.time() - t0:.2f}s: "
        f"mean={ld_np.mean():.3f}, max={ld_np.max():.3f} "
        f"(log(4) = {np.log(4):.3f})"
    )

    # Mask out extreme-E cells (same filter used in training: |E| < 100)
    E_mask = np.abs(E) < 100.0
    print(f"Fraction of cells with |E|<100: {E_mask.mean():.4f}")

    out = HERE / "ld_field.npz"
    np.savez(
        out,
        ld=ld_np,
        u_grid=us,
        fine_n=FINE_N,
        E_field=E,
        E_mask=E_mask,
        notes="softmax entropy; max log(4) is approximately 1.386",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    compute_ld_field()
