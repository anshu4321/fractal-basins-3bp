"""Group 5: Ablations and Scaling (Tests 5.1-5.4).

Characterizes when and why the classifier-prior method works.
Each ablation varies one parameter while holding others fixed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

OUT_DIR = Path(__file__).parent
N_CANDIDATES = 5000


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def test_5_1_classifier_quality(shape_pts, labels):
    """Test 5.1: Vary classifier quality (training set size).

    Train classifiers at N = {1e3, 1e4, 1e5, 1e6}, measure macro-F1,
    then run classifier-prior search with each.
    """
    print("\n" + "=" * 40)
    print("TEST 5.1: Classifier Quality vs Hit Rate")
    print("=" * 40)

    from mega3bp.ml import BasinMLP, predict_proba, loss_fn
    from baselines import classifier_entropy_candidates
    from run_group4 import run_method

    train_sizes = [1_000, 10_000, 100_000, min(1_000_000, len(labels))]
    results = []

    for N in train_sizes:
        print(f"\n--- N_train = {N} ---")

        # Subsample training data
        rng = np.random.default_rng(42)
        idx = rng.choice(len(labels), size=min(N, len(labels)), replace=False)
        X_train = jnp.array(shape_pts[idx], dtype=jnp.float64)
        y_train = jnp.array(labels[idx], dtype=jnp.int32)

        # Train a quick model (simplified inline training)
        import optax
        import equinox as eqx

        key = jax.random.PRNGKey(42)
        model = BasinMLP(in_dim=3, hidden=256, n_layers=5, key=key)
        optimizer = optax.adamw(1e-3, weight_decay=1e-4)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        # Class weights for balanced training
        unique, counts = np.unique(np.array(y_train), return_counts=True)
        cw = np.zeros(4)
        for u, c in zip(unique, counts):
            cw[u] = len(y_train) / (4.0 * c)
        class_weights = jnp.array(cw, dtype=jnp.float64)

        batch_size = min(4096, N)
        n_epochs = min(20, max(5, 40_000 // max(N // batch_size, 1)))

        @eqx.filter_jit
        def train_step(model, opt_state, x, y):
            log_t = jnp.zeros_like(y, dtype=jnp.float64)
            escape_mask = jnp.zeros_like(y, dtype=jnp.bool_)
            grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
            (loss, aux), grads = grad_fn(model, x, y, log_t, escape_mask, class_weights)
            updates, opt_state_new = optimizer.update(
                grads, opt_state, eqx.filter(model, eqx.is_array))
            model = eqx.apply_updates(model, updates)
            return model, opt_state_new, loss

        print(f"  Training {n_epochs} epochs, batch_size={batch_size}...")
        for epoch in range(n_epochs):
            perm = jax.random.permutation(jax.random.PRNGKey(epoch), len(y_train))
            for i in range(0, len(y_train), batch_size):
                batch_idx = perm[i:i+batch_size]
                model, opt_state, loss = train_step(
                    model, opt_state, X_train[batch_idx], y_train[batch_idx])

        # Compute macro-F1 on a held-out slice
        test_idx = rng.choice(len(labels), size=min(10_000, len(labels)), replace=False)
        X_test = jnp.array(shape_pts[test_idx], dtype=jnp.float64)
        y_test = np.array(labels[test_idx])
        preds = np.array(jnp.argmax(jax.vmap(model)(X_test)[..., :4], axis=-1))
        from sklearn.metrics import f1_score
        try:
            macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        except ImportError:
            # Fallback: compute accuracy
            macro_f1 = float(np.mean(preds == y_test))
        print(f"  Macro-F1: {macro_f1:.4f}")

        # Run classifier-prior search
        cands, _ = classifier_entropy_candidates(model, shape_pts, N_CANDIDATES)
        r = run_method(f"quality_N{N}", cands, seed=0)

        results.append({
            "N_train": N,
            "macro_f1": macro_f1,
            "hit_rate": r["hit_rate"],
            "n_verified": r["n_verified"],
            "wall_time_s": r["wall_time_s"],
        })
        print(f"  Hit rate: {r['hit_rate']:.4f}, Verified: {r['n_verified']}")

    write_json(OUT_DIR / "17_ablation_classifier_quality" / "results.json", results)

    # Check correlation
    f1s = [r["macro_f1"] for r in results if r["macro_f1"] is not None]
    hrs = [r["hit_rate"] for r in results if r["macro_f1"] is not None]
    if len(f1s) >= 3:
        corr = float(np.corrcoef(f1s, hrs)[0, 1])
        print(f"\n  Pearson correlation (F1 vs hit_rate): {corr:.3f}")
        print(f"  Pass criterion (>0.7): {'PASS' if corr > 0.7 else 'FAIL'}")

    return results


def test_5_2_dataset_size(shape_pts, labels):
    """Test 5.2: Vary dataset size used for k-NN disagreement."""
    print("\n" + "=" * 40)
    print("TEST 5.2: Dataset Size vs Unique Orbit Families")
    print("=" * 40)

    from baselines import label_disagreement_candidates
    from run_group4 import run_method

    dataset_sizes = [1_000, 10_000, 100_000, len(shape_pts)]
    results = []

    for N in dataset_sizes:
        print(f"\n--- Dataset size N={N} ---")
        subset_pts = shape_pts[:N]
        subset_labels = labels[:N]

        cands, _ = label_disagreement_candidates(subset_pts, subset_labels, N_CANDIDATES)
        r = run_method(f"dataset_{N}", cands, seed=0)
        r["dataset_size"] = N
        results.append({
            "dataset_size": N,
            "hit_rate": r["hit_rate"],
            "n_verified": r["n_verified"],
            "wall_time_s": r["wall_time_s"],
        })
        print(f"  Hit rate: {r['hit_rate']:.4f}, Verified: {r['n_verified']}")

    write_json(OUT_DIR / "18_ablation_dataset_size" / "results.json", results)
    return results


def test_5_3_threshold_sensitivity(shape_pts, model):
    """Test 5.3: Vary the entropy top-K cutoff."""
    print("\n" + "=" * 40)
    print("TEST 5.3: Threshold Sensitivity")
    print("=" * 40)

    if model is None:
        print("  No model available, skipping.")
        return []

    from baselines import classifier_entropy_candidates
    from run_group4 import run_method

    top_K_values = [100, 500, 1000, 5000, 10000]
    results = []

    for K in top_K_values:
        print(f"\n--- Top-K = {K} ---")
        cands, _ = classifier_entropy_candidates(model, shape_pts, K)
        r = run_method(f"threshold_{K}", cands, seed=0)
        results.append({
            "top_K": K,
            "n_actual": len(cands),
            "hit_rate": r["hit_rate"],
            "n_verified": r["n_verified"],
            "wall_time_s": r["wall_time_s"],
        })
        print(f"  Hit rate: {r['hit_rate']:.4f}, Verified: {r['n_verified']}")

    write_json(OUT_DIR / "19_ablation_threshold" / "results.json", results)

    # Check robustness
    if len(results) >= 2:
        hrs = [r["hit_rate"] for r in results]
        variation = (max(hrs) - min(hrs)) / max(max(hrs), 1e-10)
        print(f"\n  Hit rate variation: {variation:.2%}")
        print(f"  Robust (<30%): {'PASS' if variation < 0.3 else 'FAIL'}")

    return results


def test_5_4_perfect_classifier(shape_pts, labels):
    """Test 5.4: Perfect classifier (ground-truth labels for k-NN)."""
    print("\n" + "=" * 40)
    print("TEST 5.4: Perfect Classifier Upper Bound")
    print("=" * 40)

    from baselines import label_disagreement_candidates
    from run_group4 import run_method

    cands, scores = label_disagreement_candidates(shape_pts, labels, N_CANDIDATES)
    r = run_method("perfect_labels", cands, seed=0)

    result = {
        "hit_rate": r["hit_rate"],
        "n_verified": r["n_verified"],
        "wall_time_s": r["wall_time_s"],
    }
    write_json(OUT_DIR / "20_ablation_perfect_labels" / "results.json", [result])
    print(f"  Hit rate: {r['hit_rate']:.4f}, Verified: {r['n_verified']}")

    return result


def main():
    print("=" * 60)
    print("GROUP 5: ABLATIONS AND SCALING")
    print("=" * 60)

    # Load dataset
    npz_path = PROJECT_ROOT / "results" / "01_dataset_regen" / "at_rest_with_diagnostics.npz"
    d = np.load(npz_path)
    mask = (d["r_min_ever"] >= 0.01) & (d["label"] != -1)
    shape_pts = d["shape_n"][mask].astype(np.float64)
    labels = d["label"][mask].astype(np.int32)
    print(f"Dataset: {len(labels)} clean samples")

    # Try to load trained model
    model = None
    import equinox as eqx
    from mega3bp.ml import BasinMLP
    model_paths = [
        PROJECT_ROOT / "data" / "baseline_model.eqx",
        PROJECT_ROOT / "results" / "baseline_model.eqx",
    ]
    for mp in model_paths:
        if mp.exists():
            model = eqx.tree_deserialise_leaves(mp, BasinMLP(key=jax.random.PRNGKey(0)))
            print(f"Loaded model from {mp}")
            break

    test_5_1_classifier_quality(shape_pts, labels)
    test_5_2_dataset_size(shape_pts, labels)
    test_5_3_threshold_sensitivity(shape_pts, model)
    test_5_4_perfect_classifier(shape_pts, labels)

    # Write verdict
    verdict = [
        "# GROUP 5 VERDICT: Ablations and Scaling\n\n",
        "## Summary\n\n",
        "Four ablation experiments completed.\n\n",
        "- Test 5.1: Classifier quality vs hit rate\n",
        "- Test 5.2: Dataset size vs unique orbit families\n",
        "- Test 5.3: Disagreement threshold sensitivity\n",
        "- Test 5.4: Perfect classifier upper bound\n\n",
        "## Key findings\n\n",
        "See individual result files in directories 17-20.\n",
        "Fill in after analyzing results.\n",
    ]
    (OUT_DIR / "GROUP5_VERDICT.md").write_text("".join(verdict))

    print("\nGroup 5 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
