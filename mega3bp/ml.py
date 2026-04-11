"""Baseline ML for the 3BP basin classification task.

BasinMLP is a small equinox MLP that reads (n_1, n_2, n_3) on the Montgomery
shape sphere and predicts (escape_label, log_escape_time). The classification
head is 4-way (bound / body1_escape / body2_escape / body3_escape); ambiguous
trajectories are excluded from training and evaluation per the scoping
contract (they have no trusted label).

Design choices
--------------

Joint head. 5 output units: 4 classification logits + 1 log_escape_time
regression. Both trained simultaneously; the regression loss is masked to
only contribute on escape trajectories (where escape_time is finite).

Class-weighted cross-entropy. The dataset is ~94% bound, so a trivial
predict-bound model scores 0.935 accuracy. That clears the contract's 0.90
gate trivially without learning anything. We weight each class inversely to
its frequency so per-class F1 becomes the meaningful metric.

Raw shape-sphere coordinates as input. A 3-input MLP is the simplest baseline.
If spectral bias around fractal boundaries plateaus the model, Fourier
featurization (sin/cos harmonics of the coordinates) is the next step — see
fourier_features below.
"""
from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

Array = jnp.ndarray

N_CLASSES = 4  # bound + 3 escape classes


class BasinMLP(eqx.Module):
    """Simple feed-forward MLP over shape-sphere coordinates.

    Produces a 5-vector per sample: logits for the 4 classes + predicted
    log escape time (only meaningful where the true label is an escape).
    """

    layers: list

    def __init__(
        self,
        in_dim: int = 3,
        hidden: int = 256,
        n_layers: int = 5,
        out_dim: int = N_CLASSES + 1,
        *,
        key,
    ) -> None:
        keys = jax.random.split(key, n_layers + 1)
        sizes = [in_dim] + [hidden] * n_layers + [out_dim]
        self.layers = [
            eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i]) for i in range(len(sizes) - 1)
        ]

    def __call__(self, x: Array) -> Array:
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)


def fourier_features(x: Array, n_freqs: int = 8) -> Array:
    """Map input coordinates to sin/cos harmonics up to frequency n_freqs.

    Given x of shape (..., d), returns an array of shape (..., d * 2 * n_freqs)
    containing [sin(2^k * pi * x), cos(2^k * pi * x)] for k = 0..n_freqs-1.
    """
    freqs = 2.0 ** jnp.arange(n_freqs, dtype=x.dtype) * jnp.pi
    args = x[..., None] * freqs  # shape (..., d, n_freqs)
    sins = jnp.sin(args).reshape(*x.shape[:-1], -1)
    coss = jnp.cos(args).reshape(*x.shape[:-1], -1)
    return jnp.concatenate([sins, coss], axis=-1)


# ---- Loss ------------------------------------------------------------------

def loss_fn(
    model: BasinMLP,
    x: Array,                # (B, in_dim)
    y: Array,                # (B,) int in [0, 3]
    log_t: Array,            # (B,) regression target, nan-safe where bound
    escape_mask: Array,      # (B,) bool, True where regression target is valid
    class_weights: Array,    # (4,)
    reg_weight: float = 0.3,
) -> tuple[Array, dict]:
    logits_and_reg = jax.vmap(model)(x)        # (B, 5)
    logits = logits_and_reg[..., :N_CLASSES]
    log_t_pred = logits_and_reg[..., N_CLASSES]

    # Weighted cross-entropy
    log_p = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_p, y[:, None], axis=-1).squeeze(-1)   # (B,)
    weights = class_weights[y]
    cls_loss = jnp.sum(weights * nll) / jnp.sum(weights)

    # Masked MSE on log escape time
    #   Only apply where escape_mask is True (i.e. trajectory actually escapes).
    err = log_t_pred - log_t
    mse_terms = (err * err) * escape_mask
    denom = jnp.sum(escape_mask)
    mse_loss = jnp.where(denom > 0, jnp.sum(mse_terms) / jnp.maximum(denom, 1.0), 0.0)

    total = cls_loss + reg_weight * mse_loss

    pred = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(pred == y)
    return total, {
        "cls_loss": cls_loss,
        "mse_loss": mse_loss,
        "total_loss": total,
        "accuracy": acc,
    }


# ---- Inference helpers -----------------------------------------------------

@eqx.filter_jit
def predict(model: BasinMLP, x: Array) -> tuple[Array, Array]:
    """Returns (class_pred, log_escape_pred) for a batch of inputs."""
    out = jax.vmap(model)(x)
    logits = out[..., :N_CLASSES]
    log_t = out[..., N_CLASSES]
    return jnp.argmax(logits, axis=-1), log_t


def predict_proba(model: BasinMLP, x: Array) -> Array:
    out = jax.vmap(model)(x)
    return jax.nn.softmax(out[..., :N_CLASSES], axis=-1)


# ---- Data split utility ----------------------------------------------------

def stratified_split(
    labels: jnp.ndarray,
    key,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (train_idx, val_idx, test_idx) arrays over the input labels.

    Stratified: each output set preserves the class distribution of the input.
    Uses a single shuffle per class to preserve reproducibility.
    """
    assert train_frac + val_frac < 1.0
    n = labels.shape[0]
    train_idx = []
    val_idx = []
    test_idx = []
    for lab in range(N_CLASSES):
        mask = labels == lab
        idx = jnp.nonzero(mask, size=int(mask.sum()))[0]
        key, sub = jax.random.split(key)
        perm = jax.random.permutation(sub, idx.shape[0])
        shuffled = idx[perm]
        n_i = shuffled.shape[0]
        n_train = int(round(n_i * train_frac))
        n_val = int(round(n_i * val_frac))
        train_idx.append(shuffled[:n_train])
        val_idx.append(shuffled[n_train : n_train + n_val])
        test_idx.append(shuffled[n_train + n_val :])
    return (
        jnp.concatenate(train_idx),
        jnp.concatenate(val_idx),
        jnp.concatenate(test_idx),
    )
