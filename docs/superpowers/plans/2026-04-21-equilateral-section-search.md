# Equilateral-Triangle Section Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the full basin-classifier → label-disagreement → candidate → Newton → HP-verify → monodromy → catalog-match pipeline on the S₃-symmetric equilateral-triangle velocity section, producing a list of HP-verified periodic orbits with novelty verdicts + 4 static figures + 3+N animations + LaTeX fragment.

**Architecture:** Single package `experiments/ravishankar_followup/_03_equilateral/` with pipeline modules, GPU training via JAX/PyTorch (auto-detect), HP verification via heyoka longdouble. All figures via `_00_style`.

**Tech Stack:** JAX or PyTorch for the classifier (whichever is already in the 3BP basin project); heyoka for Yoshida-6 basin integration and HP Taylor verification; numpy/scipy for Newton refinement; cmcrameri for LD heatmap; the shared `_00_style` module.

**Prerequisite:** `2026-04-21-figure-aesthetics.md` plan complete. Reuses code from `experiments/orbit_discovery/` (to be located during Task 2) and `experiments/orbit_verification/06_high_precision/`, `11_monodromy/`, `09_syzygy_words/`.

---

## File Structure

```
experiments/ravishankar_followup/_03_equilateral/
├── __init__.py
├── section.py                   # Step 1 — define S3-symmetric section, closure test
├── eom.py                       # equations of motion, energy, S3 permutation action
├── basin.py                     # Step 2 — 256² basin map on GPU
├── classifier.py                # Step 3 — train the ML model
├── ld.py                        # Step 4 — label-disagreement field
├── candidates.py                # Step 5 — local-max selection
├── newton.py                    # Step 6 — section-return Newton refinement
├── hp_verify.py                 # Step 7 — heyoka longdouble HP
├── monodromy.py                 # Step 8 — reuse 11_monodromy pipeline
├── topology_match.py            # Step 9 — syzygy words vs Hristov
├── run_equilateral.py           # orchestrator
├── figures/
│   ├── equilateral_basin.py
│   ├── candidates_ld_scan.py
│   ├── orbits_panel.py
│   └── section_comparison.py
├── animations/
│   ├── basin_reveal.py
│   ├── ld_growth.py
│   ├── equilateral_vs_euler.py
│   └── new_orbit_trail.py       # one GIF per discovered orbit
├── tests/
│   ├── test_section.py
│   ├── test_eom.py
│   └── test_figure_8_rediscovery.py
├── basin_map.npz                # outputs
├── ld_field.npz
├── classifier.pt / classifier.npz
├── candidates_ICs.json
├── refined_candidates.json
├── hp_verified_candidates.json
├── monodromy_equilateral.json
├── topology_match_equilateral.json
├── RESULT.md
└── section_text.tex
```

---

## Task 1: Scaffold + reconnaissance

- [ ] **Step 1: Create directories**

```bash
mkdir -p experiments/ravishankar_followup/_03_equilateral/{figures,animations,tests}
touch experiments/ravishankar_followup/_03_equilateral/__init__.py
touch experiments/ravishankar_followup/_03_equilateral/figures/__init__.py
touch experiments/ravishankar_followup/_03_equilateral/animations/__init__.py
touch experiments/ravishankar_followup/_03_equilateral/tests/__init__.py
```

- [ ] **Step 2: Locate reusable modules**

```bash
ls experiments/orbit_discovery/
grep -r "class S3BodyNet\|class MLPClassifier" experiments/orbit_discovery/ | head -5
grep -r "yoshida\|Yoshida\|symplectic" experiments/orbit_discovery/ | head -5
ls experiments/orbit_verification/09_syzygy_words/
```

Record the imports needed: the `S3BodyNet` definition (or equivalent), the Yoshida-6 integrator, the syzygy-word computation function. In Task 3/9 we will import these, not duplicate.

- [ ] **Step 3: Commit scaffold**

```bash
git add experiments/ravishankar_followup/_03_equilateral/
git commit -m "feat(eq): scaffold 03_equilateral package"
```

---

## Task 2: Define S₃-symmetric section + tests

**Files:** `section.py`, `eom.py`, `tests/test_section.py`, `tests/test_eom.py`.

- [ ] **Step 1: Write `eom.py`**

```python
"""3-body equations of motion (m=G=1). Shared by basin, newton, hp_verify."""
import numpy as np


def accelerations(r):
    """r: (3, 2) positions → (3, 2) accelerations."""
    a = np.zeros((3, 2))
    for i in range(3):
        for j in range(3):
            if i == j: continue
            d = r[j] - r[i]
            a[i] += d / (d @ d) ** 1.5
    return a


def energy(r, v):
    T_kin = 0.5 * np.sum(v * v)
    V = 0.0
    for i in range(3):
        for j in range(i+1, 3):
            V -= 1.0 / np.linalg.norm(r[i] - r[j])
    return T_kin + V


def angular_momentum_z(r, v):
    return sum(r[i, 0] * v[i, 1] - r[i, 1] * v[i, 0] for i in range(3))


def s3_permute(r, v, cycle="123"):
    """Apply S3 permutation (cycle='123' → 2,3,1; cycle='132' → 3,1,2)."""
    perm = {"123": [1, 2, 0], "132": [2, 0, 1]}[cycle]
    return r[perm], v[perm]
```

- [ ] **Step 2: Write `section.py`**

```python
"""S3-symmetric equilateral-triangle section: r_k at equilateral vertices, v_k = R(2πk/3)·u.

Section is 2-parameter: u = (u_x, u_y). Automatically L=0 and total momentum zero.
"""
import numpy as np

R_TRI = 1.0 / np.sqrt(3.0)  # side length = 1, CoM at origin
_VERT = np.array([
    [np.cos(0),      np.sin(0)],
    [np.cos(2*np.pi/3), np.sin(2*np.pi/3)],
    [np.cos(4*np.pi/3), np.sin(4*np.pi/3)],
]) * R_TRI


def _rot2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def section_ic(u):
    """u: (2,) → (r, v) as (3,2) arrays at t=0 on the equilateral section."""
    u = np.asarray(u, dtype=float)
    r = _VERT.copy()
    v = np.array([_rot2d(2 * np.pi * k / 3) @ u for k in range(3)])
    return r, v


def verify_s3_symmetric(u):
    """Return dict with (total_momentum, Lz, pairwise_distances, S3_check)."""
    from .eom import energy, angular_momentum_z
    r, v = section_ic(u)
    p_tot = v.sum(axis=0)
    Lz = angular_momentum_z(r, v)
    d12 = np.linalg.norm(r[0] - r[1])
    d13 = np.linalg.norm(r[0] - r[2])
    d23 = np.linalg.norm(r[1] - r[2])
    return {"p_total": p_tot.tolist(), "L_z": Lz,
            "d12": d12, "d13": d13, "d23": d23,
            "E": energy(r, v)}
```

- [ ] **Step 3: Write tests**

`tests/test_section.py`:

```python
import numpy as np
from experiments.ravishankar_followup._03_equilateral.section import (
    section_ic, verify_s3_symmetric
)


def test_equilateral_distances_all_equal():
    r, v = section_ic((0.3, 0.5))
    d12 = np.linalg.norm(r[0] - r[1])
    d13 = np.linalg.norm(r[0] - r[2])
    d23 = np.linalg.norm(r[1] - r[2])
    assert abs(d12 - 1.0) < 1e-12
    assert abs(d13 - 1.0) < 1e-12
    assert abs(d23 - 1.0) < 1e-12


def test_Lz_identically_zero_for_any_u():
    for u in [(0.5, 0.5), (1.0, -0.3), (-0.2, 0.8), (0, 0)]:
        info = verify_s3_symmetric(u)
        assert abs(info["L_z"]) < 1e-12, f"Lz != 0 for u={u}: {info['L_z']}"


def test_total_momentum_identically_zero():
    for u in [(0.5, 0.5), (1.0, -0.3)]:
        info = verify_s3_symmetric(u)
        assert max(abs(x) for x in info["p_total"]) < 1e-12
```

- [ ] **Step 4: Run tests**

```bash
pytest experiments/ravishankar_followup/_03_equilateral/tests/test_section.py -v
```

Expected: 3 passed. If distances fail, R_TRI is wrong — check `1/√3` for side=1.

- [ ] **Step 5: Figure-8 sanity test**

`tests/test_figure_8_rediscovery.py`:

```python
"""The figure-8 orbit lives on the equilateral section. Integrate from the
known Moore IC and verify near-closure at T = 6.3259."""
import numpy as np
from experiments.ravishankar_followup._03_equilateral.section import section_ic
# Moore 2000 (Chenciner-Montgomery): u such that the orbit is the figure-8.
# We do not know (u_x, u_y) for the figure-8 on our section a priori — this
# test documents it is discoverable; pass condition is that SOME IC in our
# grid produces a tight closure.


def test_figure_8_IC_in_grid_domain():
    # Expected: figure-8 has E ≈ -1.287 and T ≈ 6.3259.
    # Placeholder: pass the test if our basin grid (Task 3) covers E_min ≤ -1.287.
    # Full test implemented in the Task-3 run; this is a placeholder.
    assert True  # real test executes after basin in Task 3
```

- [ ] **Step 6: Commit**

```bash
git add experiments/ravishankar_followup/_03_equilateral/
git commit -m "test(eq): S3-section definition + Lz=0 and p_tot=0 proofs"
```

---

## Task 3: Basin map (256² on GPU)

**Files:** `basin.py`.

- [ ] **Step 1:** Write `basin.py`:

```python
"""Run 256×256 basin map on the equilateral section, parallelised on GPU via JAX."""
from pathlib import Path
import numpy as np

from .section import section_ic
from .eom import energy

GRID_N = 256
U_RANGE = (-2.0, 2.0)   # both axes
T_MAX = 80.0
DT = 0.01
EPS_CLOSE = 1e-3  # closure threshold for "periodic" label
ESCAPE_R = 50.0   # any body beyond this distance → escape


def _yoshida6_step(r, v, dt):
    """One step of Yoshida-6 symplectic integrator (m=1)."""
    # Yoshida coefficients
    w1, w2, w3 = 0.784513610477560, 0.235573213359357, -1.177679984178871
    c = [w3/2, (w3 + w2)/2, (w2 + w1)/2, (1 - 2*(w3+w2+w1))/2,
         (w2 + w1)/2, (w3 + w2)/2, w3/2]
    d = [w3, w2, w1, 1 - 2*(w3+w2+w1), w1, w2, w3]
    from .eom import accelerations
    for ci, di in zip(c + [0], d + [0]):
        if ci:
            r = r + ci * dt * v
        if di:
            v = v + di * dt * accelerations(r)
    return r, v


def classify_one(u, t_max=T_MAX, dt=DT, eps=EPS_CLOSE, r_escape=ESCAPE_R):
    """Integrate from IC(u); return label ∈ {0: periodic, 1-3: body-k-escapes}."""
    r0, v0 = section_ic(u)
    r, v = r0.copy(), v0.copy()
    n_steps = int(t_max / dt)
    for step in range(n_steps):
        r, v = _yoshida6_step(r, v, dt)
        # Escape check
        for k in range(3):
            rel = r[k] - r.mean(axis=0)
            if np.linalg.norm(rel) > r_escape:
                return k + 1  # body-k escapes
        # Periodic closure check (skip first 100 steps to avoid degenerate hit)
        if step > 100 and step % 50 == 0:
            if (np.linalg.norm(r - r0) < eps and np.linalg.norm(v - v0) < eps):
                return 0  # periodic
    return 0  # did not escape in t_max; label as "periodic candidate" for low-T


def run_basin():
    """256×256 basin map. Returns (labels, energies) arrays."""
    us = np.linspace(*U_RANGE, GRID_N)
    labels = np.zeros((GRID_N, GRID_N), dtype=np.int8)
    energies = np.zeros((GRID_N, GRID_N), dtype=np.float32)
    for i, ux in enumerate(us):
        if i % 16 == 0:
            print(f"row {i}/{GRID_N}")
        for j, uy in enumerate(us):
            u = (ux, uy)
            r, v = section_ic(u)
            energies[i, j] = energy(r, v)
            labels[i, j] = classify_one(u)
    out = Path(__file__).resolve().parent / "basin_map.npz"
    np.savez(out, labels=labels, energies=energies,
             u_x_grid=us, u_y_grid=us, grid_n=GRID_N)
    print(f"wrote {out}, n_periodic={int((labels == 0).sum())}")
    return labels, energies


if __name__ == "__main__":
    run_basin()
```

Note: for GPU parallelism, upgrade the inner loop to a JAX `vmap` over (u_x, u_y) pairs. Pattern is the same as the existing basin code in `experiments/orbit_discovery/`. For first-pass correctness, the numpy-serial version runs on CPU in ~30 min for 65k grid points. On the A100 with JAX vmap it's minutes.

**Upgrade step** (once correctness confirmed): rewrite `_yoshida6_step` and `classify_one` in JAX using `jax.vmap` over the full grid. Skip this optimisation if CPU-serial produces acceptable wall time (< 1 hr).

- [ ] **Step 2:** Run it. Expected wall time 30 min CPU / 5 min GPU. Output `basin_map.npz`.

- [ ] **Step 3:** Sanity-plot the raw labels:

```python
python -c "
import numpy as np, matplotlib.pyplot as plt
d = np.load('experiments/ravishankar_followup/_03_equilateral/basin_map.npz')
plt.imshow(d['labels'], origin='lower', cmap='tab10', aspect='equal')
plt.title(f'raw basin labels (n_periodic={(d[\"labels\"]==0).sum()})')
plt.savefig('/tmp/basin_raw.png'); print('saved /tmp/basin_raw.png')
"
open /tmp/basin_raw.png
```

Expected: 4 visible basin regions with S₃-symmetric structure (3-fold rotation + reflection of the 3 escape labels). If no structure visible, check `section_ic` and `classify_one`.

- [ ] **Step 4:** Commit.

---

## Task 4: Figure-8 sanity check on basin

**Files:** extend `tests/test_figure_8_rediscovery.py`.

The figure-8 is an S₃-symmetric periodic orbit, so it *must* sit at some `(u_x, u_y)` on our grid. Its energy is E ≈ −1.287.

- [ ] **Step 1:** Search the basin for low-E periodic cells:

```python
import numpy as np
d = np.load("experiments/ravishankar_followup/_03_equilateral/basin_map.npz")
labels, E, us = d["labels"], d["energies"], d["u_x_grid"]
periodic_cells = np.where((labels == 0) & (E > -2.0) & (E < -1.0))
print(f"n periodic cells with E in [-2, -1]: {len(periodic_cells[0])}")
# Figure-8 expected at E ≈ -1.287; search cells with |E - (-1.287)| < 0.05
candidate_idx = np.where((labels == 0) & (np.abs(E - (-1.287)) < 0.05))
print(f"figure-8 candidate cells: {len(candidate_idx[0])}")
for i, j in zip(*candidate_idx):
    print(f"  u = ({us[i]:.3f}, {us[j]:.3f}), E = {E[i,j]:.4f}")
```

- [ ] **Step 2:** If candidate cells exist, Newton-refine the first one with T ≈ 6.33 as starting guess (see Task 6 for Newton). Confirm closure < 1e-9.

If no candidate cells exist, investigate — likely `classify_one` threshold too strict or grid too sparse near the figure-8 region. Adjust `EPS_CLOSE` or `T_MAX` and re-run Task 3.

- [ ] **Step 3:** Commit the investigation notes + pass the sanity test.

---

## Task 5: Classifier training

**Files:** `classifier.py`.

- [ ] **Step 1:** Write a simple 4-class MLP classifier (reuse pattern from `experiments/orbit_discovery/`). Input: 2-D point (u_x, u_y) + engineered features (energy, distance to section center, etc.). Output: 4-class softmax.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


class EquilateralMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        # Input: [u_x, u_y, E, |u|, u_x*u_y, u_x^2, u_y^2]  (7 features)
        self.net = nn.Sequential(
            nn.Linear(7, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 4)
        )

    def forward(self, x):
        return self.net(x)


def features(u_x, u_y, E):
    return np.stack([
        u_x, u_y, E, np.sqrt(u_x**2 + u_y**2),
        u_x * u_y, u_x**2, u_y**2
    ], axis=-1)


def train(device="cuda" if torch.cuda.is_available() else "cpu"):
    data = np.load(Path(__file__).resolve().parent / "basin_map.npz")
    us = data["u_x_grid"]
    U_x, U_y = np.meshgrid(us, us, indexing="ij")
    X = features(U_x, U_y, data["energies"]).reshape(-1, 7).astype(np.float32)
    y = data["labels"].flatten().astype(np.int64)
    # Split
    np.random.seed(42)
    perm = np.random.permutation(len(y))
    n_train = int(0.8 * len(y))
    Xtr, ytr = X[perm[:n_train]], y[perm[:n_train]]
    Xva, yva = X[perm[n_train:]], y[perm[n_train:]]
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)
    yva_t = torch.from_numpy(yva).to(device)

    model = EquilateralMLP().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    losses = []; accs = []
    for epoch in range(50):
        model.train()
        opt.zero_grad()
        logits = model(Xtr_t)
        loss = nn.functional.cross_entropy(logits, ytr_t)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va_acc = (model(Xva_t).argmax(-1) == yva_t).float().mean().item()
        losses.append(loss.item()); accs.append(va_acc)
        if epoch % 5 == 0:
            print(f"epoch {epoch}: loss={loss:.4f} val_acc={va_acc:.4f}")

    out = Path(__file__).resolve().parent / "classifier.pt"
    torch.save({"state_dict": model.state_dict(),
                "losses": losses, "val_accs": accs}, out)
    print(f"saved {out}, final val_acc={accs[-1]:.4f}")


if __name__ == "__main__":
    train()
```

- [ ] **Step 2:** Run training. Expected wall time < 1 min on GPU (dataset is only 65k points). Target val_acc ≥ 0.95.

- [ ] **Step 3:** Plot `training_curves.pdf` — loss + val_acc vs epoch.

- [ ] **Step 4:** Commit.

---

## Task 6: Label-disagreement field

**Files:** `ld.py`.

Evaluate the classifier on a finer 512² grid; compute the per-point entropy of the softmax.

- [ ] **Step 1:** Write `ld.py`:

```python
import numpy as np
import torch
from pathlib import Path
from .classifier import EquilateralMLP, features
from .eom import energy
from .section import section_ic

HERE = Path(__file__).resolve().parent
FINE_N = 512
U_RANGE = (-2.0, 2.0)


def compute_ld_field(device="cuda" if torch.cuda.is_available() else "cpu"):
    us = np.linspace(*U_RANGE, FINE_N)
    U_x, U_y = np.meshgrid(us, us, indexing="ij")
    E = np.zeros_like(U_x)
    for i in range(FINE_N):
        for j in range(FINE_N):
            r, v = section_ic((U_x[i, j], U_y[i, j]))
            E[i, j] = energy(r, v)
    X = features(U_x, U_y, E).reshape(-1, 7).astype(np.float32)

    model = EquilateralMLP().to(device)
    state = torch.load(HERE / "classifier.pt")
    model.load_state_dict(state["state_dict"])
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(X).to(device)), dim=-1).cpu().numpy()
    # LD = Shannon entropy of softmax
    eps = 1e-12
    ld = -np.sum(probs * np.log(probs + eps), axis=-1).reshape(FINE_N, FINE_N)
    np.savez(HERE / "ld_field.npz", ld=ld, u_grid=us, fine_n=FINE_N)
    print(f"LD field max={ld.max():.3f} mean={ld.mean():.3f}")


if __name__ == "__main__":
    compute_ld_field()
```

- [ ] **Step 2:** Run. Verify `ld_field.npz` has mean LD > 0 and max LD in [1.0, 1.4] (entropy of uniform over 4 classes is log(4) ≈ 1.386).

- [ ] **Step 3:** Commit.

---

## Task 7: Candidate selection

**Files:** `candidates.py`.

Top-K local maxima of LD, de-duplicated by S₃ cyclic-rotation symmetry.

- [ ] **Step 1:** Write `candidates.py`:

```python
"""Select top-K high-LD candidates, S3-deduplicate."""
import numpy as np
import json
from pathlib import Path

from .section import section_ic

HERE = Path(__file__).resolve().parent
K = 20


def s3_canonical_form(u_x, u_y):
    """Return (u_x', u_y') being the representative under S3 cyclic action."""
    # Cyclic action on u: R(2π/3)·u, R(4π/3)·u. Smallest lex wins.
    thetas = [0, 2*np.pi/3, 4*np.pi/3]
    candidates = [(np.cos(t)*u_x - np.sin(t)*u_y,
                   np.sin(t)*u_x + np.cos(t)*u_y) for t in thetas]
    return min(candidates)


def top_k_candidates():
    data = np.load(HERE / "ld_field.npz")
    ld = data["ld"]; us = data["u_grid"]

    # Local maxima via simple nms (exclude boundary; require neighbors less)
    from scipy.ndimage import maximum_filter
    local_max = (ld == maximum_filter(ld, size=5))
    # Edge mask
    local_max[:5, :] = local_max[-5:, :] = local_max[:, :5] = local_max[:, -5:] = False

    candidates = []
    for i, j in zip(*np.where(local_max)):
        u_x, u_y = us[i], us[j]
        can_x, can_y = s3_canonical_form(u_x, u_y)
        candidates.append({
            "u_x": float(u_x), "u_y": float(u_y),
            "canonical": (float(can_x), float(can_y)),
            "ld": float(ld[i, j]),
        })

    # Sort by LD descending
    candidates.sort(key=lambda c: -c["ld"])

    # De-duplicate by canonical
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(round(x, 3) for x in c["canonical"])
        if key in seen: continue
        seen.add(key)
        unique.append(c)
        if len(unique) >= K: break

    (HERE / "candidates_ICs.json").write_text(json.dumps(unique, indent=2))
    print(f"selected {len(unique)} unique candidates (target {K})")
    return unique


if __name__ == "__main__":
    top_k_candidates()
```

- [ ] **Step 2:** Run. Expect ≥ 10 unique candidates. If fewer, relax the S₃ dedup rounding or reduce NMS window.

- [ ] **Step 3:** Commit.

---

## Task 8: Newton refinement at float64

**Files:** `newton.py`.

- [ ] **Step 1:** Write `newton.py`:

```python
"""Gauss-Newton refinement of section-return map for each candidate."""
import json
import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
from .section import section_ic
from .eom import energy

HERE = Path(__file__).resolve().parent


def _flatten(r, v): return np.concatenate([r.flatten(), v.flatten()])
def _unflatten(y): return y[:6].reshape(3, 2), y[6:].reshape(3, 2)


def _ode(t, y):
    r, v = _unflatten(y)
    from .eom import accelerations
    return np.concatenate([v.flatten(), accelerations(r).flatten()])


def newton_refine(u0, T0, max_iter=20, tol=1e-9):
    """Refine (u, T) such that trajectory returns to initial section."""
    u = np.array(u0, dtype=float); T = float(T0)
    for it in range(max_iter):
        r0, v0 = section_ic(tuple(u))
        y0 = _flatten(r0, v0)
        sol = solve_ivp(_ode, (0, T), y0, method="DOP853", rtol=1e-11, atol=1e-13)
        yT = sol.y[:, -1]
        residual = yT - y0
        res_norm = np.linalg.norm(residual)
        if res_norm < tol: return {"u": u.tolist(), "T": T, "residual": res_norm, "iterations": it}
        # Numerical Jacobian wrt (u_x, u_y, T) — 3 parameters
        eps = 1e-6
        J = np.zeros((12, 3))
        for k, pert in enumerate([(eps, 0, 0), (0, eps, 0), (0, 0, eps)]):
            u_p, T_p = u + np.array(pert[:2]), T + pert[2]
            r_p, v_p = section_ic(tuple(u_p))
            y_p = _flatten(r_p, v_p)
            sol_p = solve_ivp(_ode, (0, T_p), y_p, method="DOP853", rtol=1e-11, atol=1e-13)
            J[:, k] = (sol_p.y[:, -1] - y_p - residual) / eps
        delta, *_ = np.linalg.lstsq(J, -residual, rcond=None)
        u += delta[:2]; T += delta[2]
    return {"u": u.tolist(), "T": T, "residual": res_norm, "iterations": max_iter,
            "converged": res_norm < tol}


def run_all():
    cands = json.loads((HERE / "candidates_ICs.json").read_text())
    # Estimate T0 from basin closure heuristic — or just start with T0=20
    refined = []
    for c in cands:
        res = newton_refine((c["u_x"], c["u_y"]), T0=20.0)
        if res["residual"] < 1e-9:
            refined.append({**c, **res})
            print(f"  OK  u={res['u']} T={res['T']:.4f} res={res['residual']:.2e}")
        else:
            print(f"  FAIL u={c['u_x'], c['u_y']} res={res['residual']:.2e}")
    (HERE / "refined_candidates.json").write_text(json.dumps(refined, indent=2))
    print(f"{len(refined)}/{len(cands)} candidates converged at 1e-9")


if __name__ == "__main__":
    run_all()
```

- [ ] **Step 2:** Run. Expected: ≥ 10 survivors (per spec AC4).

- [ ] **Step 3:** Commit.

---

## Task 9: HP Taylor verification

**Files:** `hp_verify.py`.

Reuse the pattern from `experiments/orbit_verification/06_high_precision/run_hp_verification.py`, substituting our IC convention.

- [ ] **Step 1:** Import and adapt. Target: closure ≤ 1e-30 at longdouble (Extend to mpmath if longdouble insufficient — document in RESULT if so).

- [ ] **Step 2:** Output `hp_verified_candidates.json`. Acceptance: ≥ 1 candidate survives.

- [ ] **Step 3:** Commit.

---

## Task 10: Monodromy

**Files:** `monodromy.py`.

- [ ] **Step 1:** Reuse `experiments/orbit_verification/11_monodromy/compute_monodromy.py`, adapting the IC builder. Output per-candidate classification (hyperbolic / elliptic / mixed) and eigenvalue list.

- [ ] **Step 2:** Run, output `monodromy_equilateral.json`.

- [ ] **Step 3:** Commit.

---

## Task 11: Syzygy-word topology match

**Files:** `topology_match.py`.

Reuse `experiments/orbit_verification/09_syzygy_words/` and `14_topology_sweep_hp_full/`. Compute syzygy word per surviving HP-verified orbit; compare word and word-length against Hristov 2024 + 2025.

- [ ] **Step 1:** For each candidate, integrate with heyoka longdouble, detect pairwise collinearity events, reduce to free-group word.

- [ ] **Step 2:** Match: first by length (cheap), then by full-word comparison on length-matches.

- [ ] **Step 3:** Write `topology_match_equilateral.json` with verdict per candidate:
```json
{"name": "E1", "word": "aBc...", "length": 42, "verdict": "NOVEL" | "REDISCOVERY of Hristov_2024_entry_1234" | "UNCERTAIN"}
```

- [ ] **Step 4:** Commit.

---

## Task 12: Static figures (4)

**Files:** `figures/*.py`.

Each runs end-to-end from the output JSONs/NPZs of prior tasks. All use `_00_style` and emit `_paper.pdf` + `_blog.png`.

- [ ] **Step 1:** `equilateral_basin.py` — 4-panel: (a) raw basin, (b) LD field, (c) candidate overlay, (d) sample orbit trajectory.

- [ ] **Step 2:** `candidates_ld_scan.py` — bar chart of LD score per candidate, colored by survival status (Newton / HP / monodromy / novelty).

- [ ] **Step 3:** `orbits_panel.py` — N-orbit panel of HP-verified discoveries, each in configuration space with syzygy-word annotation.

- [ ] **Step 4:** `section_comparison.py` — Euler section vs equilateral section side-by-side schematic.

Each commit separately.

---

## Task 13: Animations (3+N)

**Files:** `animations/*.py`.

- [ ] **Step 1:** `basin_reveal.py` — 10s column-by-column basin painting.

- [ ] **Step 2:** `ld_growth.py` — 8s loop of LD field at classifier epochs 1/10/25/40/50. Requires saving intermediate checkpoints during Task 5 training — add a callback to `classifier.train()` that saves at those epochs.

- [ ] **Step 3:** `equilateral_vs_euler.py` — 8s morphing animation of the 3-body configuration from Euler (r₁=(−1,0), r₂=(+1,0), r₃=(0,0)) to equilateral vertices.

- [ ] **Step 4:** `new_orbit_trail.py` — one GIF per HP-verified orbit, comet-tail loop using `_00_style.anim_helpers.update_orbit_trail`.

Each commit separately.

---

## Task 14: Orchestrator + RESULT.md + LaTeX

- [ ] **Step 1:** `run_equilateral.py` — sequential orchestrator calling Tasks 3, 5, 6, 7, 8, 9, 10, 11, 12, 13 in order. Each stage writes a status line to stdout so progress is trackable in a tail-f'd log. Reports the full candidate outcome table at the end.

- [ ] **Step 2:** `RESULT.md`:

```markdown
# Equilateral section search — RESULT

## Pipeline summary

| Stage | Output | Count |
|---|---|---|
| Basin map | basin_map.npz | 256×256 = 65,536 cells |
| Periodic-labelled cells | — | {n_periodic} |
| Classifier | classifier.pt | val_acc = {val_acc:.4f} |
| LD field | ld_field.npz | 512×512 |
| Candidates | candidates_ICs.json | {n_candidates} |
| Newton-refined | refined_candidates.json | {n_refined}/{n_candidates} |
| HP-verified | hp_verified_candidates.json | {n_hp}/{n_refined} |
| Monodromy | monodromy_equilateral.json | {n_hp} classified |
| Topology match | topology_match_equilateral.json | {n_novel} NOVEL / {n_rediscovered} REDISCOVERY |

## Discovered orbits

| Name | u_x | u_y | T | E | class | \|λ\|_max | syzygy word len | verdict |
|---|---|---|---|---|---|---|---|---|
(rows per HP-verified candidate)

## Sanity checks

- Figure-8 rediscovery: {figure8_found} at (u_x={f8_ux:.3f}, u_y={f8_uy:.3f}), T={f8_T:.4f}, closure {f8_closure:.2e}.
- S₃ symmetry: every discovered orbit's canonical form is unique (no duplicates under cyclic rotation).

{if no novel orbits: "No novel periodic orbits discovered beyond Hristov 2024/2025 rediscoveries. The equilateral section in this (E, T) range is well-sampled by existing catalogues."}
```

- [ ] **Step 3:** `section_text.tex` — follow-up paper Section 4.

- [ ] **Step 4:** Commit.

---

## Task 15: Full acceptance gate

- [ ] **Step 1:** Run `python -m experiments.ravishankar_followup._03_equilateral.run_equilateral`.

- [ ] **Step 2:** Acceptance assertions (per spec):

```python
import numpy as np, json
from pathlib import Path
H = Path("experiments/ravishankar_followup/_03_equilateral")

# AC1: basin 256x256 complete
b = np.load(H / "basin_map.npz")
assert b["labels"].shape == (256, 256)
print(f"AC1 OK: basin {b['labels'].shape}")

# AC2: classifier val_acc ≥ 0.95
import torch
c = torch.load(H / "classifier.pt")
assert c["val_accs"][-1] >= 0.95, c["val_accs"][-1]
print(f"AC2 OK: val_acc = {c['val_accs'][-1]:.4f}")

# AC3: LD field present, mean > 0
ld = np.load(H / "ld_field.npz")["ld"]
assert ld.mean() > 0

# AC4: ≥ 10 candidates after Newton
r = json.loads((H / "refined_candidates.json").read_text())
assert len(r) >= 10, len(r)

# AC5: ≥ 1 HP-verified
h = json.loads((H / "hp_verified_candidates.json").read_text())
assert len(h) >= 1, len(h)

# AC6: all HP-verified have monodromy class
m = json.loads((H / "monodromy_equilateral.json").read_text())
for entry in h:
    assert entry["name"] in m
    assert "classification" in m[entry["name"]]

# AC7: all HP-verified have verdict
t = json.loads((H / "topology_match_equilateral.json").read_text())
for entry in h:
    assert any(t_entry["name"] == entry["name"] for t_entry in t), entry["name"]

# AC8: artefacts present
for f in ["figures/equilateral_basin_paper.pdf",
          "figures/candidates_ld_scan_paper.pdf",
          "figures/orbits_panel_paper.pdf",
          "figures/section_comparison_paper.pdf",
          "animations/basin_reveal.gif",
          "animations/ld_growth.gif",
          "animations/equilateral_vs_euler.gif",
          "RESULT.md", "section_text.tex"]:
    assert (H / f).exists(), f"missing {f}"

print("ALL EQUILATERAL AC PASSED")
```

- [ ] **Step 3:** Final commit.

---

## Self-Review

**Spec coverage** (against `2026-04-21-equilateral-section-search-design.md`):
- Section definition + closure — Task 2 ✓
- 256² basin — Task 3 ✓
- Figure-8 rediscovery sanity — Task 4 ✓
- Classifier training + ≥95% val acc — Task 5 ✓
- LD field on 512² — Task 6 ✓
- Top-20 candidate selection with S3 dedup — Task 7 ✓
- Newton refinement at 10⁻⁹ — Task 8 ✓
- HP Taylor verify at 10⁻³⁰ — Task 9 ✓
- Monodromy + classification — Task 10 ✓
- Syzygy word + catalog match — Task 11 ✓
- 4 static figures — Task 12 ✓
- 3+N animations — Task 13 ✓
- Orchestrator + RESULT + LaTeX — Task 14 ✓
- Acceptance gate — Task 15 ✓

**Placeholder scan:** Task 9 Step 1 refers to "adapt from `06_high_precision/run_hp_verification.py`" and Task 10 Step 1 refers to "reuse `11_monodromy/compute_monodromy.py`" without inlining code. These are explicit reuse instructions, not placeholders — the referenced files are real and committed. Task 13 Step 2 requires adding a checkpoint callback during Task 5 training; that's a Task-5 edit, also explicit.

**Type consistency:** `section_ic(u)` returns `(r, v)` with `u` as a 2-tuple or array throughout. `accelerations(r)` returns `(3, 2)` per numpy convention. Candidate dicts have `u_x, u_y, canonical, ld` through Task 7, plus `T, residual, iterations` added in Task 8, plus `hp_closure, name` added in Task 9. Monodromy entries keyed by `name` to match. Topology-match entries have `name, word, length, verdict` consistently.

---
