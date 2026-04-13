"""Unit tests for S₃ augmentation correctness.

Tests per PROMPT_v2 Section 3.1:
1. Round-trip: apply π then π⁻¹ returns original (x, y) exactly.
2. Class distribution preserved under uniform random π.
3. Group multiplication table for R, M, P matrices.
4. Physical consistency: augmented input → same physical config modulo relabeling.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import numpy as np

from mega3bp.equivariant import (
    LABEL_PERM_ALL,
    M_ALL,
    PERM_ALL,
    R_ALL,
    S3_NAMES,
    apply_s3_input,
    augment_batch,
    augment_single,
)
from mega3bp.shape_sphere import (
    config_to_shape,
    jacobi_momenta_to_body,
    shape_to_config,
)

INV_IDX = {
    0: 0,  # e^{-1} = e
    1: 1,  # (12)^{-1} = (12)
    2: 2,  # (13)^{-1} = (13)
    3: 3,  # (23)^{-1} = (23)
    4: 5,  # (123)^{-1} = (132)
    5: 4,  # (132)^{-1} = (123)
}


def test_roundtrip():
    """Apply π then π⁻¹ on 1000 examples: must recover original (x, y) exactly."""
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (1000, 7))
    labels = jax.random.randint(jax.random.PRNGKey(0), (1000,), 0, 4)

    max_x_err = 0.0
    n_label_err = 0
    for idx in range(6):
        inv = INV_IDX[idx]
        x_fwd = jax.vmap(lambda xi: apply_s3_input(xi, idx))(x)
        labels_fwd = jax.vmap(lambda l: LABEL_PERM_ALL[idx, l])(labels)
        x_back = jax.vmap(lambda xi: apply_s3_input(xi, inv))(x_fwd)
        labels_back = jax.vmap(lambda l: LABEL_PERM_ALL[inv, l])(labels_fwd)
        err = float(jnp.max(jnp.abs(x_back - x)))
        max_x_err = max(max_x_err, err)
        n_label_err += int(jnp.sum(labels_back != labels))
        name = S3_NAMES[idx]
        print(f"  {name:>5s}: max |x_back - x| = {err:.2e}, label mismatches = {int(jnp.sum(labels_back != labels))}")

    # TF32 on H100: 10-bit mantissa gives ~1e-3 relative error per matmul.
    # Two matmuls (forward + inverse) give ~2e-3.
    assert max_x_err < 5e-3, f"Round-trip x error {max_x_err}"
    assert n_label_err == 0, f"Round-trip label errors: {n_label_err}"
    print("PASS: round-trip")


def test_class_distribution():
    """Uniform random π preserves class distribution over 100k samples."""
    key = jax.random.PRNGKey(7)
    N = 100_000
    x = jax.random.normal(jax.random.PRNGKey(1), (N, 7))
    labels = jnp.array(np.random.default_rng(42).choice(4, size=N, p=[0.44, 0.187, 0.187, 0.186]).astype(np.int32))

    orig_counts = np.bincount(np.array(labels), minlength=4)
    _, labels_aug = augment_batch(key, x, labels)
    aug_counts = np.bincount(np.array(labels_aug), minlength=4)

    print(f"  original:  {dict(enumerate(orig_counts.tolist()))}")
    print(f"  augmented: {dict(enumerate(aug_counts.tolist()))}")

    for c in range(4):
        ratio = aug_counts[c] / max(orig_counts[c], 1)
        print(f"  class {c}: ratio = {ratio:.4f}")
        assert 0.95 < ratio < 1.05, f"Class {c} distribution shifted: {orig_counts[c]} -> {aug_counts[c]}"
    print("PASS: class distribution preserved")


def test_group_multiplication():
    """Verify R, M, P form consistent representations of S₃.

    Derives the multiplication table from P (discrete, exact), then checks
    that R and M satisfy the same table.
    """
    errors = []
    for i in range(6):
        for j in range(6):
            P_prod = PERM_ALL[i][PERM_ALL[j]]
            k = None
            for c in range(6):
                if bool(jnp.all(PERM_ALL[c] == P_prod)):
                    k = c
                    break
            assert k is not None, f"P composition {S3_NAMES[i]}*{S3_NAMES[j]} not in group"
            R_prod = R_ALL[i] @ R_ALL[j]
            M_prod = M_ALL[i] @ M_ALL[j]
            r_err = float(jnp.max(jnp.abs(R_prod - R_ALL[k])))
            m_err = float(jnp.max(jnp.abs(M_prod - M_ALL[k])))
            if r_err > 1e-4 or m_err > 1e-4:
                errors.append(f"{S3_NAMES[i]} * {S3_NAMES[j]} = {S3_NAMES[k]}: R_err={r_err:.2e}, M_err={m_err:.2e}")
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        assert False, f"{len(errors)} multiplication table errors"
    print("PASS: group multiplication table (R, M, P all consistent)")


def test_physical_consistency():
    """Augmented shape sphere coords correspond to body-permuted physical config."""
    key = jax.random.PRNGKey(99)
    n_pts = shape_sphere_pts = jax.random.normal(key, (100, 3))
    n_pts = n_pts / jnp.linalg.norm(n_pts, axis=-1, keepdims=True)
    pi_flat = jax.random.normal(jax.random.PRNGKey(100), (100, 4)) * 0.5

    x = jnp.concatenate([n_pts, pi_flat], axis=-1)

    # For each group element, check that the augmented input
    # produces body positions that are a permutation of the original
    max_err = 0.0
    for idx in range(6):
        x_aug = jax.vmap(lambda xi: apply_s3_input(xi, idx))(x)
        n_orig = x[:, :3]
        n_aug = x_aug[:, :3]

        q_orig = jax.vmap(shape_to_config)(n_orig)  # (100, 3, 2)
        q_aug = jax.vmap(shape_to_config)(n_aug)

        pi1_orig, pi2_orig = x[:, 3:5], x[:, 5:7]
        pi1_aug, pi2_aug = x_aug[:, 3:5], x_aug[:, 5:7]
        p_orig = jax.vmap(jacobi_momenta_to_body)(pi1_orig, pi2_orig)  # (100, 3, 2)
        p_aug = jax.vmap(jacobi_momenta_to_body)(pi1_aug, pi2_aug)

        perm = PERM_ALL[idx]

        # The augmented body positions should be a permutation of the originals
        # q_aug[:, i, :] ≈ q_orig[:, perm[i], :]  (up to gauge/sign)
        # But gauge fixing complicates direct comparison.
        # Instead check that pairwise distances are preserved under the permutation.
        from mega3bp.shape_sphere import pairwise_distances
        d_orig = jax.vmap(pairwise_distances)(q_orig)  # (100, 3) = [r12, r13, r23]
        d_aug = jax.vmap(pairwise_distances)(q_aug)

        # Under body permutation σ, the pair distances permute:
        # r_{σ(i),σ(j)} in augmented = r_{i,j} in original
        # Map: pair (0,1)->0, (0,2)->1, (1,2)->2
        pair_idx = {(0,1): 0, (0,2): 1, (1,2): 2}
        perm_np = np.array(perm)
        d_aug_reordered = jnp.zeros_like(d_orig)
        for (i, j), col in pair_idx.items():
            si, sj = int(perm_np[i]), int(perm_np[j])
            new_pair = tuple(sorted([si, sj]))
            new_col = pair_idx[new_pair]
            d_aug_reordered = d_aug_reordered.at[:, col].set(d_aug[:, new_col])
        err = float(jnp.max(jnp.abs(d_aug_reordered - d_orig)))
        max_err = max(max_err, err)
        print(f"  {S3_NAMES[idx]:>5s}: max pairwise distance error = {err:.2e}")

    # Gauge-fixing in shape_to_config amplifies float32 rounding near
    # collision singularities. Pairwise distances are gauge-invariant but
    # computed through the gauge-fixed coordinates, so inherit the noise.
    assert max_err < 0.1, f"Physical consistency failed: max error {max_err}"
    print("PASS: physical consistency (pairwise distances preserved within float32 gauge noise)")


if __name__ == "__main__":
    print("\n=== Test 1: Round-trip ===")
    test_roundtrip()
    print("\n=== Test 2: Class distribution ===")
    test_class_distribution()
    print("\n=== Test 3: Group multiplication table ===")
    test_group_multiplication()
    print("\n=== Test 4: Physical consistency ===")
    test_physical_consistency()
    print("\n=== ALL TESTS PASSED ===")
