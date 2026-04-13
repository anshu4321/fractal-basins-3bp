"""S₃-equivariant architectures for the 3BP basin classification.

The planar equal-mass 3BP has S₃ permutation symmetry: relabeling bodies
1↔2↔3 gives the same physical system but permutes the escape labels.

S₃ action on the 7D input (n₁,n₂,n₃,π₁ₓ,π₁ᵧ,π₂ₓ,π₂ᵧ):
  Shape sphere: (n₁,n₂) carries the standard rep, n₃ the sign rep.
  Jacobi momenta: each spatial component (πᵢₓ,π₂ₓ) and (π₁ᵧ,π₂ᵧ)
  carries the standard rep. Full 7D = 3·std ⊕ sign.

S₃ action on the output (bound, esc₁, esc₂, esc₃, log_t):
  bound and log_t are invariant. Escape logits permute with bodies.

Two architectures:
  S3ReynoldsNet — wraps any model with exact equivariance via group averaging.
  S3BodyNet — per-body feature architecture with built-in equivariance.
"""
from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

from .dynamics import angular_momentum as compute_L, kinetic_energy, potential_energy
from .ml import N_CLASSES, BasinMLP
from .shape_sphere import jacobi_momenta_to_body, shape_to_config

Array = jnp.ndarray

S3 = jnp.sqrt(jnp.asarray(3.0)) / 2.0  # √3/2

# ---- S₃ group elements (R_shape, M_jacobi, escape_perm) --------------------
#
# R acts on (n₁,n₂,n₃). M acts on each spatial component of Jacobi momenta
# [(π₁,π₂)_x and (π₁,π₂)_y separately]. escape_perm is the 0-indexed
# permutation of escape logits under σ: new_logit[i] = old_logit[σ(i)].

_R = {}
_M = {}
_P = {}

_R["e"] = jnp.eye(3)
_M["e"] = jnp.eye(2)
_P["e"] = jnp.array([0, 1, 2])

_R["(12)"] = jnp.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=jnp.float32)
_M["(12)"] = jnp.array([[-1, 0], [0, 1]], dtype=jnp.float32)
_P["(12)"] = jnp.array([1, 0, 2])

_R["(13)"] = jnp.array([[-0.5, -S3, 0], [-S3, 0.5, 0], [0, 0, -1]], dtype=jnp.float32)
_M["(13)"] = jnp.array([[0.5, -S3], [-S3, -0.5]], dtype=jnp.float32)
_P["(13)"] = jnp.array([2, 1, 0])

_R["(23)"] = jnp.array([[-0.5, S3, 0], [S3, 0.5, 0], [0, 0, -1]], dtype=jnp.float32)
_M["(23)"] = jnp.array([[0.5, S3], [S3, -0.5]], dtype=jnp.float32)
_P["(23)"] = jnp.array([0, 2, 1])

_R["(123)"] = jnp.array([[-0.5, S3, 0], [-S3, -0.5, 0], [0, 0, 1]], dtype=jnp.float32)
_M["(123)"] = jnp.array([[-0.5, -S3], [S3, -0.5]], dtype=jnp.float32)
_P["(123)"] = jnp.array([1, 2, 0])

_R["(132)"] = jnp.array([[-0.5, -S3, 0], [S3, -0.5, 0], [0, 0, 1]], dtype=jnp.float32)
_M["(132)"] = jnp.array([[-0.5, S3], [-S3, -0.5]], dtype=jnp.float32)
_P["(132)"] = jnp.array([2, 0, 1])

S3_NAMES = list(_R.keys())
R_ALL = jnp.stack([_R[k] for k in S3_NAMES])          # (6, 3, 3)
M_ALL = jnp.stack([_M[k] for k in S3_NAMES])          # (6, 2, 2)
PERM_ALL = jnp.stack([_P[k] for k in S3_NAMES])       # (6, 3) int

# Label permutation: for label in {0,1,2,3}, label 0 (bound) is invariant,
# labels 1-3 permute as 1-indexed bodies.
LABEL_PERM_ALL = jnp.concatenate(
    [jnp.zeros((6, 1), dtype=jnp.int32), PERM_ALL + 1], axis=-1
)  # (6, 4)


# ---- Transformations --------------------------------------------------------

def apply_s3_input(x: Array, idx: int) -> Array:
    """Apply S₃ element idx to a 7D input vector."""
    R = R_ALL[idx]
    M = M_ALL[idx]
    n = x[:3]
    pi = x[3:].reshape(2, 2)  # [[pi1x, pi1y], [pi2x, pi2y]]
    n_new = R @ n
    pi_new = jnp.stack([M @ pi[:, 0], M @ pi[:, 1]], axis=-1)
    return jnp.concatenate([n_new, pi_new.reshape(-1)])


def apply_s3_output(y: Array, idx: int) -> Array:
    """Apply σ⁻¹ to a 5D output (bound, esc0, esc1, esc2, log_t).

    The Reynolds operator needs: for each σ, untransform the output by σ⁻¹.
    Since we iterate over all 6 elements and S₃ is a group, σ⁻¹ is also in
    the table.  But it's simpler to use the direct formula:
        untransformed_escape[i] = transformed_escape[σ(i)]
    """
    perm = PERM_ALL[idx]
    bound = y[0:1]
    esc = y[1:4][perm]
    log_t = y[4:5]
    return jnp.concatenate([bound, esc, log_t])


def augment_single(x: Array, label: Array, idx: int) -> tuple[Array, Array]:
    """Apply S₃ element idx to one (x, label) pair."""
    x_new = apply_s3_input(x, idx)
    label_new = LABEL_PERM_ALL[idx, label]
    return x_new, label_new


def augment_batch(key: jax.Array, x: Array, labels: Array) -> tuple[Array, Array]:
    """Random S₃ augmentation: independently sample a group element per sample."""
    B = x.shape[0]
    idxs = jax.random.randint(key, (B,), 0, 6)
    x_aug = jax.vmap(apply_s3_input)(x, idxs)
    labels_aug = jax.vmap(lambda l, i: LABEL_PERM_ALL[i, l])(labels, idxs)
    return x_aug, labels_aug


# ---- S3ReynoldsNet ----------------------------------------------------------

class S3ReynoldsNet(eqx.Module):
    """Wraps any base model with exact S₃ equivariance via Reynolds averaging.

    At inference: averages f(σ·x) with output un-permuted, over all 6 σ ∈ S₃.
    Cost: 6× forward passes (trivial for MLPs).
    """

    base: eqx.Module

    def __call__(self, x: Array) -> Array:
        def one_element(idx):
            x_t = apply_s3_input(x, idx)
            y_t = self.base(x_t)
            return apply_s3_output(y_t, idx)

        outputs = jax.vmap(one_element)(jnp.arange(6))  # (6, 5)
        return jnp.mean(outputs, axis=0)


# ---- Per-body features for S3BodyNet ----------------------------------------

D_BODY = 8
D_GLOBAL = 4


def _body_features_single(x: Array) -> tuple[Array, Array]:
    """Compute S₃-equivariant per-body features from a single 7D input.

    Returns:
        body_feats:   (3, D_BODY)
        global_feats: (D_GLOBAL,)
    """
    n = x[:3]
    pi1 = x[3:5]
    pi2 = x[5:7]

    q = shape_to_config(n)                    # (3, 2)
    p = jacobi_momenta_to_body(pi1, pi2)      # (3, 2)

    d01 = q[1] - q[0]
    d02 = q[2] - q[0]
    d12 = q[2] - q[1]
    r01 = jnp.linalg.norm(d01)
    r02 = jnp.linalg.norm(d02)
    r12 = jnp.linalg.norm(d12)

    T = 0.5 * jnp.sum(p * p, axis=-1)        # (3,)

    def _feat(i, j, k, r_jk, r_ij, r_ik):
        q_cm = (q[j] + q[k]) / 2.0
        d_cm = q[i] - q_cm
        r_cm = jnp.linalg.norm(d_cm)
        v_rel = 1.5 * p[i]
        safe_r = jnp.maximum(r_cm, 1e-10)
        v_rad = jnp.dot(v_rel, d_cm) / safe_r
        L_i = d_cm[0] * v_rel[1] - d_cm[1] * v_rel[0]
        return jnp.array([r_jk, r_ij + r_ik, r_ij * r_ik,
                          T[i], T[j] + T[k], r_cm, v_rad, L_i])

    f0 = _feat(0, 1, 2, r12, r01, r02)
    f1 = _feat(1, 0, 2, r02, r01, r12)
    f2 = _feat(2, 0, 1, r01, r02, r12)
    body_feats = jnp.stack([f0, f1, f2])      # (3, 8)

    E = jnp.sum(T) - 1.0 / r01 - 1.0 / r02 - 1.0 / r12
    L = jnp.sum(q[:, 0] * p[:, 1] - q[:, 1] * p[:, 0])
    s2 = r01**2 * r02**2 + r01**2 * r12**2 + r02**2 * r12**2
    s3 = (r01 * r02 * r12) ** 2
    global_feats = jnp.array([E, L, s2, s3])  # (4,)

    return body_feats, global_feats


# ---- S3BodyNet --------------------------------------------------------------

class S3BodyNet(eqx.Module):
    """Per-body equivariant architecture with S₃ symmetry built in.

    Each body gets rotation-invariant features describing its relationship
    to the other two.  A shared MLP processes each body's features (weight
    tying enforces equivariance).  Pooled features produce the bound logit
    and log-escape-time; per-body features produce escape logits.

    Output: (bound, esc0, esc1, esc2, log_t) — same as BasinMLP.
    """

    body_net: BasinMLP
    bound_head: eqx.nn.Linear
    escape_head: eqx.nn.Linear
    reg_head: eqx.nn.Linear

    def __init__(
        self,
        body_hidden: int = 256,
        body_layers: int = 4,
        body_out: int = 128,
        *,
        key,
    ) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        in_dim = D_BODY + D_GLOBAL
        self.body_net = BasinMLP(
            in_dim=in_dim, hidden=body_hidden, n_layers=body_layers,
            out_dim=body_out, key=k1,
        )
        self.bound_head = eqx.nn.Linear(body_out, 1, key=k2)
        self.escape_head = eqx.nn.Linear(body_out, 1, key=k3)
        self.reg_head = eqx.nn.Linear(body_out, 1, key=k4)

    def __call__(self, x: Array) -> Array:
        body_feats, global_feats = _body_features_single(x)
        g = jnp.broadcast_to(global_feats, (3, D_GLOBAL))
        inp = jnp.concatenate([body_feats, g], axis=-1)  # (3, D_BODY+D_GLOBAL)

        h = jax.vmap(self.body_net)(inp)                  # (3, body_out)
        h_pool = jnp.mean(h, axis=0)                      # (body_out,)

        bound = self.bound_head(h_pool)                    # (1,)
        esc = jax.vmap(self.escape_head)(h).squeeze(-1)    # (3,)
        reg = self.reg_head(h_pool)                        # (1,)

        return jnp.concatenate([bound, esc, reg])          # (5,)
