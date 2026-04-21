"""Train a 4-class classifier on the equilateral-section basin map.

Target: val_acc >= 0.95 on held-out 20% of cells.

IMPORTANT FINDING: the basin's 4-neighbor label-agreement is only ~14%,
meaning most cells sit on fractal Wada boundaries. Under an iid random
80/20 split on a 256x256 grid a classifier cannot generalize to 95%
because each held-out cell has training neighbors with different labels.
We still train a strong model (BasinMLP + Fourier features, normalized
inputs, mini-batch Adam with cosine decay), save all the artefacts the
downstream ld_growth animation needs, and report the empirical ceiling.

Saves model checkpoint + training curves + intermediate snapshots at
epochs 1, 10, 25, 40, 50.

Uses JAX/equinox (the established ecosystem in this codebase — see
mega3bp/ml.BasinMLP and experiments/orbit_discovery/*). PyTorch is not
installed on this pod, but JAX+CUDA is.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from mega3bp.ml import N_CLASSES, BasinMLP, fourier_features

HERE = Path(__file__).resolve().parent
SNAPSHOT_EPOCHS = [1, 10, 25, 40, 50]
N_EPOCHS = 50
BATCH_SIZE = 512
N_FOURIER = 8  # harmonics per raw feature
FEATURE_NAMES = ["u_x", "u_y", "E", "|u|", "u_x*u_y", "u_x^2", "u_y^2"]
HIDDEN = 256
N_LAYERS = 5
WEIGHT_DECAY = 1e-3


def featurize(u_x: np.ndarray, u_y: np.ndarray, E: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            u_x,
            u_y,
            E,
            np.sqrt(u_x * u_x + u_y * u_y),
            u_x * u_y,
            u_x * u_x,
            u_y * u_y,
        ],
        axis=-1,
    ).astype(np.float32)


def build_full_grid_features() -> tuple[np.ndarray, tuple[int, int]]:
    d = np.load(HERE / "basin_map.npz")
    us_x = d["u_x_grid"]
    us_y = d["u_y_grid"]
    U_x, U_y = np.meshgrid(us_x, us_y, indexing="ij")
    E = d["energies"].astype(np.float32)
    X_full = featurize(U_x, U_y, E).reshape(-1, 7)
    return X_full, (len(us_x), len(us_y))


def load_data():
    d = np.load(HERE / "basin_map.npz")
    us_x = d["u_x_grid"]
    us_y = d["u_y_grid"]
    U_x, U_y = np.meshgrid(us_x, us_y, indexing="ij")
    E = d["energies"].astype(np.float32)
    labels = d["labels"].astype(np.int64)
    X = featurize(U_x, U_y, E).reshape(-1, 7)
    y = labels.flatten()
    keep = np.abs(X[:, 2]) < 100.0
    X, y = X[keep], y[keep]
    return X, y


def compute_normalization(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_norm(X: np.ndarray | jnp.ndarray, mu, sd):
    return (X - mu) / sd


def predict_logits(model: BasinMLP, X_norm: jnp.ndarray) -> jnp.ndarray:
    X_lift = fourier_features(X_norm, n_freqs=N_FOURIER)
    return jax.vmap(model)(X_lift)


def train() -> float:
    print(f"JAX devices: {jax.devices()}")
    X, y = load_data()
    dist = {int(k): int((y == k).sum()) for k in range(N_CLASSES)}
    print(f"Dataset after E-filter: {len(y)} samples, label dist: {dist}")

    # Neighborhood-agreement diagnostic — reports fractal-ness of labels.
    d = np.load(HERE / "basin_map.npz")
    L = d["labels"]
    same = 0
    total = 0
    for i in range(1, L.shape[0] - 1):
        for j in range(1, L.shape[1] - 1):
            total += 1
            c = L[i, j]
            if (
                L[i - 1, j] == c
                and L[i + 1, j] == c
                and L[i, j - 1] == c
                and L[i, j + 1] == c
            ):
                same += 1
    print(f"4-neighbor label-agreement: {same / total:.4f} (fractal basin diagnostic)")

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    n_train = int(0.8 * len(y))
    Xtr_raw, ytr = X[perm[:n_train]], y[perm[:n_train]]
    Xva_raw, yva = X[perm[n_train:]], y[perm[n_train:]]

    mu, sd = compute_normalization(Xtr_raw)
    print(f"Feature normalization mu[:3]={mu[:3]} sd[:3]={sd[:3]}")
    Xtr = apply_norm(Xtr_raw, mu, sd)
    Xva = apply_norm(Xva_raw, mu, sd)

    Xtr_j = jnp.asarray(Xtr)
    ytr_j = jnp.asarray(ytr)
    Xva_j = jnp.asarray(Xva)
    yva_j = jnp.asarray(yva)

    in_dim = 7 * 2 * N_FOURIER
    key = jax.random.PRNGKey(42)
    model = BasinMLP(in_dim=in_dim, hidden=HIDDEN, n_layers=N_LAYERS, out_dim=N_CLASSES, key=key)
    n_params = sum(int(x.size) for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(
        f"Model: BasinMLP(in_dim={in_dim}, hidden={HIDDEN}, n_layers={N_LAYERS}, "
        f"out_dim={N_CLASSES}) fourier(n_freqs={N_FOURIER}) — {n_params} params"
    )

    steps_per_epoch = max(1, n_train // BATCH_SIZE)
    total_steps = N_EPOCHS * steps_per_epoch
    schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=total_steps, alpha=0.1)
    optimizer = optax.adamw(schedule, weight_decay=WEIGHT_DECAY)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def loss_fn(model, X, y):
        logits = predict_logits(model, X)
        log_p = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(log_p[jnp.arange(y.shape[0]), y])

    @eqx.filter_jit
    def train_step(model, opt_state, X, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, X, y)
        updates, opt_state_new = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model_new = eqx.apply_updates(model, updates)
        return model_new, opt_state_new, loss

    @eqx.filter_jit
    def eval_acc(model, X, y):
        logits = predict_logits(model, X)
        return jnp.mean(jnp.argmax(logits, axis=-1) == y)

    @eqx.filter_jit
    def predict_probs(model, X):
        logits = predict_logits(model, X)
        return jax.nn.softmax(logits, axis=-1)

    X_full_np, grid_shape = build_full_grid_features()
    X_full_norm = apply_norm(X_full_np, mu, sd)
    X_full = jnp.asarray(X_full_norm)

    losses: list[float] = []
    tr_accs: list[float] = []
    va_accs: list[float] = []
    snapshots_probs: dict[int, np.ndarray] = {}

    t0 = time.perf_counter()
    best_va = 0.0
    for epoch in range(1, N_EPOCHS + 1):
        perm_e = jax.random.permutation(jax.random.PRNGKey(epoch), n_train)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm_e[i : i + BATCH_SIZE]
            model, opt_state, loss = train_step(model, opt_state, Xtr_j[idx], ytr_j[idx])
            epoch_loss += float(loss)
            n_batches += 1
        epoch_loss /= max(1, n_batches)
        tr_acc = float(eval_acc(model, Xtr_j, ytr_j))
        va_acc = float(eval_acc(model, Xva_j, yva_j))
        losses.append(epoch_loss)
        tr_accs.append(tr_acc)
        va_accs.append(va_acc)
        best_va = max(best_va, va_acc)

        if epoch in SNAPSHOT_EPOCHS:
            ckpt_path = HERE / f"classifier_epoch_{epoch}.eqx"
            eqx.tree_serialise_leaves(str(ckpt_path), model)
            probs = np.asarray(predict_probs(model, X_full))
            snapshots_probs[epoch] = probs.reshape(grid_shape[0], grid_shape[1], N_CLASSES)

        if epoch % 5 == 0 or epoch in SNAPSHOT_EPOCHS or epoch == 1:
            print(
                f"epoch {epoch:3d}: loss={epoch_loss:.4f} "
                f"tr_acc={tr_acc:.4f} va_acc={va_acc:.4f}"
            )

    elapsed = time.perf_counter() - t0
    print(f"Total training time: {elapsed:.1f}s")

    eqx.tree_serialise_leaves(str(HERE / "classifier.eqx"), model)
    with open(HERE / "classifier.pkl", "wb") as f:
        pickle.dump(
            {
                "losses": losses,
                "tr_accs": tr_accs,
                "va_accs": va_accs,
                "best_va_acc": best_va,
                "snapshot_epochs": SNAPSHOT_EPOCHS,
                "feature_names": FEATURE_NAMES,
                "feature_mu": mu,
                "feature_sd": sd,
                "arch": {
                    "in_dim": in_dim,
                    "hidden": HIDDEN,
                    "n_layers": N_LAYERS,
                    "out_dim": N_CLASSES,
                    "fourier_n_freqs": N_FOURIER,
                },
                "n_epochs": N_EPOCHS,
                "batch_size": BATCH_SIZE,
                "n_train": int(n_train),
                "n_val": int(len(y) - n_train),
                "neighbor_agreement": same / total,
            },
            f,
        )

    np.savez(
        HERE / "classifier_snapshots.npz",
        **{f"epoch_{e}": snapshots_probs[e] for e in SNAPSHOT_EPOCHS},
    )
    print(f"final val_acc = {va_accs[-1]:.4f}  best val_acc during training = {best_va:.4f}")
    return va_accs[-1]


if __name__ == "__main__":
    final_va = train()
    if final_va < 0.95:
        print(
            f"NOTE: val_acc {final_va:.4f} below 0.95 target — ceiling is set by "
            "the fractal basin boundary (neighbor-agreement ~14%); see STATUS."
        )
