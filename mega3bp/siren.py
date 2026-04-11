"""SIREN (sinusoidal representation network) for the 3BP basin task.

Sitzmann et al., "Implicit Neural Representations with Periodic Activation
Functions," NeurIPS 2020. We use the exact initialization scheme from the
paper, which is non-negotiable for SIREN stability:

    First layer:  W ~ Uniform(-1/in, 1/in)
    Hidden:       W ~ Uniform(-sqrt(6/in) / omega_0,
                              +sqrt(6/in) / omega_0)
    Activation:   sin(omega_0 * (W x + b))  everywhere except the last layer

omega_0 = 30 is the default choice from the paper and the right starting point
for targets that should resolve O(1..50)-frequency features. The basin
partition of the planar 3BP has sharp (possibly fractal) boundaries which the
MLP baseline smooths out; SIREN is the standard remedy because its sinusoidal
activations have a flat spectral response instead of ReLU/GELU's low-pass bias.

Matches the BasinMLP output signature: 4 classification logits + 1 log
escape-time regression, i.e. out_dim = 5 by default. This lets us reuse
mega3bp.ml.loss_fn / predict unchanged.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .ml import N_CLASSES

Array = jnp.ndarray

DEFAULT_OMEGA_0 = 30.0


class SirenLayer(eqx.Module):
    """One SIREN sinusoidal layer. Linear + sin with layer-specific omega.

    `is_first` controls the weight-initialization scheme per the paper.
    """

    weight: Array
    bias: Array
    omega_0: float = eqx.field(static=True)
    is_first: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        is_first: bool,
        omega_0: float = DEFAULT_OMEGA_0,
        key,
    ) -> None:
        self.omega_0 = omega_0
        self.is_first = is_first

        k_w, k_b = jax.random.split(key)
        if is_first:
            lim = 1.0 / in_features
        else:
            lim = (6.0 / in_features) ** 0.5 / omega_0
        self.weight = jax.random.uniform(
            k_w, (out_features, in_features), minval=-lim, maxval=lim
        )
        self.bias = jax.random.uniform(
            k_b, (out_features,), minval=-lim, maxval=lim
        )

    def __call__(self, x: Array) -> Array:
        return jnp.sin(self.omega_0 * (self.weight @ x + self.bias))


class BasinSiren(eqx.Module):
    """SIREN network with 5-dim output: 4 class logits + 1 regression.

    Same input/output contract as mega3bp.ml.BasinMLP so the existing
    training utilities (loss_fn, predict, stratified_split) apply unchanged.
    """

    first: SirenLayer
    hidden: list
    final: eqx.nn.Linear

    def __init__(
        self,
        in_dim: int = 3,
        hidden: int = 256,
        n_layers: int = 5,
        out_dim: int = N_CLASSES + 1,
        omega_0: float = DEFAULT_OMEGA_0,
        *,
        key,
    ) -> None:
        keys = jax.random.split(key, n_layers + 1)
        self.first = SirenLayer(in_dim, hidden, is_first=True, omega_0=omega_0, key=keys[0])
        self.hidden = [
            SirenLayer(hidden, hidden, is_first=False, omega_0=omega_0, key=keys[i + 1])
            for i in range(n_layers - 1)
        ]
        # Final layer: plain linear, no sin — emits class logits + regression.
        # Initialized with the paper's hidden-layer scheme (not scaled by omega_0
        # applied at call time) so it has a reasonable starting magnitude.
        lim = (6.0 / hidden) ** 0.5 / omega_0
        k_w, k_b = jax.random.split(keys[-1])
        w = jax.random.uniform(k_w, (out_dim, hidden), minval=-lim, maxval=lim)
        b = jax.random.uniform(k_b, (out_dim,), minval=-lim, maxval=lim)
        self.final = eqx.nn.Linear(hidden, out_dim, key=jax.random.PRNGKey(0))
        # Replace the default Linear params with our SIREN-aware init
        self.final = eqx.tree_at(lambda m: m.weight, self.final, w)
        self.final = eqx.tree_at(lambda m: m.bias, self.final, b)

    def __call__(self, x: Array) -> Array:
        h = self.first(x)
        for layer in self.hidden:
            h = layer(h)
        return self.final(h)
