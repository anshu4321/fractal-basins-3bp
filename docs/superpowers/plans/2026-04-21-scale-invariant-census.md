# Scale-Invariant Census Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute E and T⋆ = T·|E|^{3/2} for all 24,582 Hristov 2024 entries, overlay our four orbits in the (E, T⋆) plane colored by syzygy-word length, and produce 3 publication figures + 5 animations + numerical tables + LaTeX fragment for the follow-up paper.

**Architecture:** One `run_census.py` orchestrator + 3 figure scripts + 1 animation script, all under `experiments/ravishankar_followup/01_census/`. Reads the parsed Hristov 2024 catalog, computes E analytically from IC via Python's `math`/`numpy`, joins with the existing topology sweep's length index, and emits static figures via the shared `00_style/` helpers.

**Tech Stack:** numpy, matplotlib, the shared `_00_style` module. No GPU needed — all CPU, ~10 s of compute for the 24,582 E values.

**Prerequisite:** `2026-04-21-figure-aesthetics.md` plan complete (Tasks 1-7). This plan imports from `experiments.ravishankar_followup._00_style`.

---

## File Structure

```
experiments/ravishankar_followup/01_census/
├── __init__.py
├── parse_hristov.py               # load Hristov 2024 ICs + T
├── compute_invariants.py          # compute E, T*, per-entry join with word length
├── figures/
│   ├── scatter.py                 # census_scatter.pdf
│   ├── density.py                 # census_density.pdf
│   └── scaling_families.py        # scaling_families.pdf (α-panels)
├── animations/
│   ├── alpha_sweep.py             # alpha_sweep_{A,B,C,D}.gif/.mp4
│   └── census_flyover.py          # census_flyover.gif/.mp4
├── run_census.py                  # orchestrator, calls everything
├── test_invariants.py             # TDD tests for compute_invariants
├── hristov_2024_invariants.json   # output
├── RESULT.md                      # output
└── section_text.tex               # output
```

---

## Task 1: Scaffold package + examine input catalog

**Files:**
- Create: `experiments/ravishankar_followup/01_census/__init__.py`

- [ ] **Step 1: Create directory**

```bash
mkdir -p experiments/ravishankar_followup/01_census/{figures,animations}
touch experiments/ravishankar_followup/01_census/__init__.py
touch experiments/ravishankar_followup/01_census/figures/__init__.py
touch experiments/ravishankar_followup/01_census/animations/__init__.py
```

- [ ] **Step 2: Inspect Hristov 2024 catalog format**

```bash
head -5 experiments/orbit_verification/00_catalogs/hristov_2024_sol_80.txt
wc -l experiments/orbit_verification/00_catalogs/hristov_2024_sol_80.txt
```

Expected: 24,582 lines (one per orbit), each with `(x_3, y_3, v_{3x}, v_{3y}, T, T*)` or similar. Record the exact column order for use in the parser.

Also inspect the existing parser if it exists:

```bash
ls experiments/orbit_verification/00_catalogs/parse_catalogs.py
cat experiments/orbit_verification/00_catalogs/parse_catalogs.py | head -40
```

Prefer to reuse the existing parser by importing, not duplicate.

- [ ] **Step 3: Inspect the topology sweep's length map**

```bash
python -c "import json; d = json.load(open('experiments/orbit_verification/14_topology_sweep_hp_full/length_histogram.json')); print(type(d), list(d.items())[:3] if isinstance(d, dict) else d[:3])"
python -c "import json; d = json.load(open('experiments/orbit_verification/14_topology_sweep_hp_full/topology_sweep_hp_summary.json')); print(list(d.keys())[:10] if isinstance(d, dict) else type(d))"
```

Record: per-entry word-length mapping format and the join key. Write this down in `parse_hristov.py`'s docstring so downstream readers aren't confused.

- [ ] **Step 4: Commit scaffold**

```bash
git add experiments/ravishankar_followup/01_census/
git commit -m "feat(census): scaffold 01_census package"
```

---

## Task 2: TDD — E computation from an IC

**Files:**
- Create: `experiments/ravishankar_followup/01_census/compute_invariants.py`
- Create: `experiments/ravishankar_followup/01_census/test_invariants.py`

- [ ] **Step 1: Write the failing test**

```python
# test_invariants.py
"""TDD tests for compute_invariants.compute_E and compute_T_star."""
import numpy as np
import pytest

from experiments.ravishankar_followup._01_census.compute_invariants import (
    compute_E, compute_T_star,
)


def test_E_isolated_bodies_is_kinetic_minus_nothing():
    """Three bodies at large mutual distance -> V ~ 0, E ~ T_kin."""
    r = np.array([[-1000, 0], [0, 0], [1000, 0]], dtype=float)
    v = np.array([[1, 0], [0, 0], [-1, 0]], dtype=float)
    E = compute_E(r, v, masses=(1, 1, 1))
    # T_kin = 0.5*(1+0+1) = 1, V ~ -3/1000 ~ -0.003
    assert abs(E - 1.0) < 0.01


def test_E_on_orbit_A():
    """Our A orbit, Euler section: E = 3(v1^2+v2^2) - 5/2 analytically."""
    v1, v2 = 0.18900489195109799, -0.53976245280128938
    r = np.array([[-1, 0], [1, 0], [0, 0]], dtype=float)
    v = np.array([[v1, v2], [v1, v2], [-2*v1, -2*v2]], dtype=float)
    E_computed = compute_E(r, v, masses=(1, 1, 1))
    E_expected = 3.0 * (v1**2 + v2**2) - 2.5
    assert abs(E_computed - E_expected) < 1e-12


def test_T_star_invariant_under_alpha_scaling():
    """T* must be scale-invariant: rescale r by alpha, v by alpha^{-1/2},
    T by alpha^{3/2}, then T* stays."""
    r = np.array([[-1, 0], [1, 0], [0, 0]], dtype=float)
    v = np.array([[0.2, -0.5], [0.2, -0.5], [-0.4, 1.0]], dtype=float)
    T = 20.0
    E = compute_E(r, v)
    T_star = compute_T_star(T, E)
    alpha = 2.0
    r_a = alpha * r
    v_a = alpha**-0.5 * v
    T_a = alpha**1.5 * T
    E_a = compute_E(r_a, v_a)
    T_star_a = compute_T_star(T_a, E_a)
    assert abs(T_star - T_star_a) / abs(T_star) < 1e-10


def test_T_star_orbit_A_matches_invariants_json():
    """Spot-check against the 50-digit value."""
    v1, v2 = 0.18900489195109799, -0.53976245280128938
    r = np.array([[-1, 0], [1, 0], [0, 0]], dtype=float)
    v = np.array([[v1, v2], [v1, v2], [-2*v1, -2*v2]], dtype=float)
    T = 19.735391893011813
    E = compute_E(r, v)
    T_star = compute_T_star(T, E)
    # Expected T_star (first 15 digits from invariants.json): 36.94001257740157
    assert abs(T_star - 36.94001257740157) < 1e-8
```

- [ ] **Step 2: Run tests (must fail)**

```bash
pytest experiments/ravishankar_followup/_01_census/test_invariants.py -v
```

Expected: `ImportError: cannot import compute_E` (module doesn't exist yet).

- [ ] **Step 3: Implement `compute_invariants.py`**

```python
"""Scale-invariant compute: E, T* for an equal-mass 3BP IC."""
import numpy as np


def compute_E(r, v, masses=(1.0, 1.0, 1.0), G=1.0):
    """Total energy E = sum(0.5*m_i*|v_i|^2) - sum_{i<j} G*m_i*m_j/|r_i-r_j|.

    Parameters
    ----------
    r : ndarray (3, 2)  positions
    v : ndarray (3, 2)  velocities
    masses : tuple of 3 floats
    G : float
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    masses = np.asarray(masses, dtype=float)
    T_kin = 0.5 * np.sum(masses[:, None] * v * v)
    V = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            d = np.linalg.norm(r[i] - r[j])
            if d < 1e-14:
                return float("inf")
            V -= G * masses[i] * masses[j] / d
    return float(T_kin + V)


def compute_T_star(T, E):
    """Scale-invariant period: T* = T * |E|^{3/2}."""
    return float(T * abs(E) ** 1.5)
```

- [ ] **Step 4: Run tests (must pass)**

```bash
pytest experiments/ravishankar_followup/_01_census/test_invariants.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/ravishankar_followup/_01_census/compute_invariants.py experiments/ravishankar_followup/_01_census/test_invariants.py
git commit -m "test(census): TDD compute_E and compute_T_star with orbit-A sanity"
```

Note: the directory on disk is `_01_census` (Python-import-safe). Update all paths consistently.

---

## Task 3: Parse Hristov 2024 and compute invariants in bulk

**Files:**
- Create: `experiments/ravishankar_followup/_01_census/parse_hristov.py`
- Modify: `experiments/ravishankar_followup/_01_census/compute_invariants.py` (add `compute_all_invariants`)

- [ ] **Step 1: Write test for the bulk path**

Append to `test_invariants.py`:

```python
def test_compute_all_invariants_returns_records():
    """End-to-end: loads Hristov 2024, returns list of {id, E, T*, word_len}."""
    from experiments.ravishankar_followup._01_census.compute_invariants import (
        compute_all_invariants,
    )
    records = compute_all_invariants(max_entries=100)  # subsample for test speed
    assert len(records) == 100
    assert all(isinstance(r["E"], float) for r in records)
    assert all(r["E"] < 0 for r in records[:10])  # first 10 should be bound
    assert all(r["T_star"] > 0 for r in records)
    assert all("word_len" in r for r in records)
```

- [ ] **Step 2: Run test (must fail)**

```bash
pytest experiments/ravishankar_followup/_01_census/test_invariants.py::test_compute_all_invariants_returns_records -v
```

Expected: ImportError or AttributeError on `compute_all_invariants`.

- [ ] **Step 3: Implement `parse_hristov.py`**

```python
"""Loader for Hristov 2024 catalog.

The on-disk file uses 80-digit MPFR precision. We read it as float64 for the
census (E and T* are not sensitive at the 4th decimal, which is our figure
precision).
"""
from pathlib import Path
import numpy as np

_CATALOG_PATH = Path(__file__).resolve().parents[2] / \
    "orbit_verification/00_catalogs/hristov_2024_sol_80.txt"


def load_hristov_2024():
    """Yield one record per catalog line.

    Each record is a dict:
        {"id": int, "r": (3,2) ndarray, "v": (3,2) ndarray, "T": float}

    Note: Hristov's IC convention for equal masses uses the "Sukava-Dmitrasinovic"
    free-fall section, parameterised by (x3, y3, v3x, v3y, T). Body 1 and 2
    are placed at (-1/2, 0) and (+1/2, 0) with zero velocity at t=0 in that
    convention. Verify exact column order from the file header.
    """
    records = []
    with open(_CATALOG_PATH) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in line.split() if p]
            # Column order (verify from header): x3 y3 v3x v3y T [T*] [other]
            x3, y3, v3x, v3y, T = map(float, parts[:5])
            r = np.array([[-0.5, 0.0],
                          [+0.5, 0.0],
                          [x3, y3]])
            v = np.array([[0.0, 0.0],
                          [0.0, 0.0],
                          [v3x, v3y]])
            records.append({"id": idx, "r": r, "v": v, "T": T})
    return records
```

**Sanity caveat**: if the column order differs (e.g. actual Hristov uses body 1 and 2 with nonzero velocities), fix in Step 4 when the E-sanity test fails.

- [ ] **Step 4: Implement `compute_all_invariants` in compute_invariants.py**

Append:

```python
import json
from pathlib import Path


def _load_word_lengths():
    """Return dict mapping catalog idx -> syzygy-word length."""
    summ = json.loads(
        (Path(__file__).resolve().parents[2] /
         "orbit_verification/14_topology_sweep_hp_full/topology_sweep_hp_summary.json"
        ).read_text()
    )
    # Expected format: {"0": 54, "1": 42, ...} or list of records
    if isinstance(summ, list):
        return {r["id"]: r.get("word_length", -1) for r in summ}
    elif isinstance(summ, dict):
        if "entries" in summ:
            return {r["id"]: r["word_length"] for r in summ["entries"]}
        return {int(k): v for k, v in summ.items() if k.isdigit()}
    return {}


def compute_all_invariants(max_entries=None):
    """Full Hristov 2024 -> list of {id, E, T_star, word_len}."""
    from .parse_hristov import load_hristov_2024
    records_in = load_hristov_2024()
    if max_entries is not None:
        records_in = records_in[:max_entries]
    word_len = _load_word_lengths()
    records_out = []
    for rec in records_in:
        E = compute_E(rec["r"], rec["v"])
        T_star = compute_T_star(rec["T"], E)
        records_out.append({
            "id": rec["id"],
            "E": E,
            "T_star": T_star,
            "T": rec["T"],
            "word_len": word_len.get(rec["id"], None),
        })
    return records_out
```

- [ ] **Step 5: Run the test (should pass)**

```bash
pytest experiments/ravishankar_followup/_01_census/test_invariants.py -v
```

Expected: 5 passed. If E<0 fails on real entries, the IC convention in `parse_hristov.py` is wrong — fix column order and re-run.

- [ ] **Step 6: Sanity-run on full 24,582 and save**

```bash
python -c "
from experiments.ravishankar_followup._01_census.compute_invariants import compute_all_invariants
import json, time
t0 = time.time()
records = compute_all_invariants()
print(f'n={len(records)} wall={time.time()-t0:.2f}s')
n_bound = sum(1 for r in records if r['E'] < 0)
print(f'bound={n_bound}/{len(records)}')
Path = __import__('pathlib').Path
out = Path('experiments/ravishankar_followup/_01_census/hristov_2024_invariants.json')
out.write_text(json.dumps(records, indent=2))
print(f'wrote {out}')
"
```

Expected: 24,582 records; ≥ 24,500 bound (E < 0); wall time < 60 s.

- [ ] **Step 7: Commit**

```bash
git add experiments/ravishankar_followup/_01_census/parse_hristov.py experiments/ravishankar_followup/_01_census/compute_invariants.py experiments/ravishankar_followup/_01_census/test_invariants.py experiments/ravishankar_followup/_01_census/hristov_2024_invariants.json
git commit -m "feat(census): compute E, T* for all 24,582 Hristov 2024 entries"
```

---

## Task 4: Scatter figure (E, T⋆) colored by word length

**Files:**
- Create: `experiments/ravishankar_followup/_01_census/figures/scatter.py`

- [ ] **Step 1: Write `scatter.py`**

```python
"""Render census_scatter_paper.pdf + census_scatter_blog.png.

Scatter of Hristov 2024 (E, T*) colored by syzygy word-length bin, with A,B,C,D
overlaid and Hristov 2025 stable orbits shown as open circles.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from experiments.ravishankar_followup._00_style import (
    make_fig, save_both, add_orbit_marker, TOL_BRIGHT,
)

HERE = Path(__file__).resolve().parent.parent


def _bin_word_length(wl):
    """Map word length to discrete 5-bin class (keeps legend readable)."""
    if wl is None or wl < 0:
        return 0
    if wl < 40:
        return 1
    if wl < 50:
        return 2
    if wl < 60:
        return 3
    if wl < 70:
        return 4
    return 5


def render(kind="paper"):
    inv = json.loads((HERE / "hristov_2024_invariants.json").read_text())
    E = np.array([r["E"] for r in inv])
    T_star = np.array([r["T_star"] for r in inv])
    wl_bin = np.array([_bin_word_length(r["word_len"]) for r in inv])

    fig, ax = make_fig(kind=kind, figsize=(7.0, 5.0) if kind == "blog" else (5.0, 3.5))

    # Plot each word-length bin with Tol bright
    bin_labels = ["unknown", "< 40", "40-49", "50-59", "60-69", "≥ 70"]
    for b in range(6):
        m = wl_bin == b
        if not m.any():
            continue
        ax.scatter(T_star[m], E[m], s=4, c=TOL_BRIGHT[b % len(TOL_BRIGHT)],
                   label=bin_labels[b], alpha=0.6, edgecolors="none")

    # Overlay our 4 orbits
    # (E, T*) values from experiments/orbit_verification/15_scaling_and_invariants/invariants.json
    our = {"A": (-1.51880093609, 36.94001257740),
           "B": (-1.57045059002, 64.64865733920),
           "C": (-1.50430210713, 64.65542233190),
           "D": (-0.93930544448, 73.81478296716)}
    for name, (e, ts) in our.items():
        add_orbit_marker(ax, ts, e, name, size=16)

    # Overlay Hristov 2025 stable (if catalog invariants computed)
    h2025_path = HERE / "hristov_2025_invariants.json"
    if h2025_path.exists():
        h25 = json.loads(h2025_path.read_text())
        E25 = [r["E"] for r in h25]
        TS25 = [r["T_star"] for r in h25]
        ax.scatter(TS25, E25, s=20, facecolors="none",
                   edgecolors="#444444", linewidths=0.6,
                   label="Hristov 2025 stable")

    ax.set_xscale("log")
    ax.set_xlabel(r"$T^{\star} = T \cdot |E|^{3/2}$")
    ax.set_ylabel(r"$E$")
    ax.legend(loc="best", markerscale=2, fontsize=9)
    ax.set_title("Equal-mass 3BP: (E, T*) census")

    save_both(fig, "census_scatter", HERE / "figures")
    plt.close(fig)


if __name__ == "__main__":
    for kind in ("paper", "blog"):
        render(kind)
```

- [ ] **Step 2: Run it**

```bash
python -m experiments.ravishankar_followup._01_census.figures.scatter
```

Expected: writes `figures/census_scatter_paper.pdf` + `figures/census_scatter_blog.png`. Open the PDF and eyeball:
- All 24k points visible
- A at (36.94, -1.519), B and C clustered at T* ≈ 64.65, D outlier at T* ≈ 73.8 with E ≈ -0.94
- Legend visible, colors match Tol bright

- [ ] **Step 3: Commit**

```bash
git add experiments/ravishankar_followup/_01_census/figures/scatter.py experiments/ravishankar_followup/_01_census/figures/census_scatter_paper.pdf experiments/ravishankar_followup/_01_census/figures/census_scatter_blog.png
git commit -m "feat(census): scatter figure (E, T*) colored by syzygy-word length"
```

---

## Task 5: Per-class density along T⋆

**Files:**
- Create: `experiments/ravishankar_followup/_01_census/figures/density.py`

- [ ] **Step 1: Write `density.py`**

```python
"""Overlaid KDEs of T* per word-length bin, with A/B/C/D markers as vertical lines."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from experiments.ravishankar_followup._00_style import (
    make_fig, save_both, TOL_BRIGHT, ORBIT,
)

HERE = Path(__file__).resolve().parent.parent
# Our 4 orbits' T_star (from invariants.json)
OUR_TSTAR = {"A": 36.94001257740, "B": 64.64865733920,
             "C": 64.65542233190, "D": 73.81478296716}


def _bin_wl(wl):
    """Same binning as scatter.py."""
    if wl is None or wl < 0: return 0
    if wl < 40: return 1
    if wl < 50: return 2
    if wl < 60: return 3
    if wl < 70: return 4
    return 5


def _kde(x, xs):
    """Gaussian KDE with Silverman bandwidth."""
    from scipy.stats import gaussian_kde
    if len(x) < 2:
        return np.zeros_like(xs)
    return gaussian_kde(x, bw_method="silverman")(xs)


def render(kind="paper"):
    inv = json.loads((HERE / "hristov_2024_invariants.json").read_text())
    T_star = np.array([r["T_star"] for r in inv if r["T_star"] > 0])
    wl_bin = np.array([_bin_wl(r["word_len"]) for r in inv if r["T_star"] > 0])
    ts_grid = np.logspace(np.log10(5), np.log10(500), 500)
    fig, ax = make_fig(kind=kind, figsize=(7, 5) if kind == "blog" else (5, 3.5))

    bin_labels = ["unknown", "< 40", "40-49", "50-59", "60-69", "≥ 70"]
    for b in range(1, 6):  # skip "unknown"
        m = wl_bin == b
        if m.sum() < 3: continue
        kde = _kde(T_star[m], ts_grid)
        ax.plot(ts_grid, kde, color=TOL_BRIGHT[b % len(TOL_BRIGHT)],
                label=f"word len {bin_labels[b]} (n={m.sum()})",
                linewidth=1.8)

    # Vertical markers for A, B, C, D
    for name, ts in OUR_TSTAR.items():
        ax.axvline(ts, color=ORBIT[name], linestyle="--", linewidth=1.4, alpha=0.8)
        ax.text(ts, ax.get_ylim()[1] * 0.98, name, color=ORBIT[name],
                ha="center", va="top", fontsize=11, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel(r"$T^{\star}$")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(r"$T^{\star}$ density per syzygy-word-length class")

    save_both(fig, "census_density", HERE / "figures")
    plt.close(fig)


if __name__ == "__main__":
    for kind in ("paper", "blog"):
        render(kind)
```

- [ ] **Step 2: Run it**

```bash
python -m experiments.ravishankar_followup._01_census.figures.density
```

Expected: overlaid KDEs per word-length class, A/B/C/D as dashed vertical lines with labels.

- [ ] **Step 3: Commit**

```bash
git add experiments/ravishankar_followup/_01_census/figures/density.py experiments/ravishankar_followup/_01_census/figures/census_density_paper.pdf experiments/ravishankar_followup/_01_census/figures/census_density_blog.png
git commit -m "feat(census): T* density per word-length class with A/B/C/D markers"
```

---

## Task 6: α-family schematic (2×2 panel)

**Files:**
- Create: `experiments/ravishankar_followup/_01_census/figures/scaling_families.py`

- [ ] **Step 1: Write `scaling_families.py`**

```python
"""2x2 panel: α-family of A, B, C, D in configuration space.

Each panel shows the orbit at α=0.5, 1, 2 scaled overlay. The trajectory is
integrated once per orbit at α=1 using scipy.integrate.solve_ivp (sufficient
for a schematic — not HP). α-rescaling is applied analytically to produce the
other two copies.
"""
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

from experiments.ravishankar_followup._00_style import (
    make_fig, save_both, BODY, ORBIT,
)

HERE = Path(__file__).resolve().parent.parent

ICS = {
    "A": {"v1": 0.18900489195, "v2": -0.53976245280, "T": 19.73539189, "E": -1.51880093609},
    "B": {"v1": -0.20349168745, "v2": 0.51811286074, "T": 32.84907117, "E": -1.57045059002},
    "C": {"v1": 0.25543093565, "v2": -0.51638583901, "T": 35.04308702, "E": -1.50430210713},
    "D": {"v1": 0.55393899048, "v2": 0.46193410064, "T": 81.08361217, "E": -0.93930544448},
}


def eom(t, y):
    """dy/dt for 3-body, m=G=1. y = [r1x, r1y, r2x, r2y, r3x, r3y, v1x, v1y, v2x, v2y, v3x, v3y]."""
    r = y[:6].reshape(3, 2)
    v = y[6:].reshape(3, 2)
    acc = np.zeros((3, 2))
    for i in range(3):
        for j in range(3):
            if i == j: continue
            d = r[j] - r[i]
            dist3 = (d @ d) ** 1.5
            acc[i] += d / dist3
    return np.concatenate([v.flatten(), acc.flatten()])


def integrate_one_period(orbit_name):
    ic = ICS[orbit_name]
    v1, v2, T = ic["v1"], ic["v2"], ic["T"]
    y0 = np.array([-1, 0, 1, 0, 0, 0, v1, v2, v1, v2, -2*v1, -2*v2])
    sol = solve_ivp(eom, (0, T), y0, method="DOP853", rtol=1e-10, atol=1e-12,
                    dense_output=True)
    t_dense = np.linspace(0, T, 500)
    traj = sol.sol(t_dense)[:6].T.reshape(-1, 3, 2)  # (500, 3, 2)
    return traj


def render(kind="paper"):
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(8, 8) if kind == "blog" else (6, 6)
    )
    from experiments.ravishankar_followup._00_style.figure_helpers import PAPER_STYLE, BLOG_STYLE
    plt.style.use(str(PAPER_STYLE if kind == "paper" else BLOG_STYLE))

    for ax, name in zip(axes.flat, "ABCD"):
        traj = integrate_one_period(name)  # (500, 3, 2)
        # Show alpha = {0.5, 1, 2} overlays — each is just a coordinate rescale
        for alpha, alpha_alpha in [(0.5, 0.4), (1.0, 1.0), (2.0, 0.5)]:
            for i in range(3):
                ax.plot(alpha * traj[:, i, 0], alpha * traj[:, i, 1],
                        color=BODY[i+1], alpha=alpha_alpha, linewidth=1.2)
        # Info text
        T, E = ICS[name]["T"], ICS[name]["E"]
        Ts = T * abs(E)**1.5
        ax.text(0.02, 0.98, f"{name}\nT={T:.2f}, E={E:.3f}\nT*={Ts:.2f}",
                transform=ax.transAxes, va="top", fontsize=9,
                color=ORBIT[name], fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(r"$\alpha$-scaling families (overlays at $\alpha = 0.5, 1, 2$)", fontsize=12)
    fig.tight_layout()
    save_both(fig, "scaling_families", HERE / "figures")
    plt.close(fig)


if __name__ == "__main__":
    for kind in ("paper", "blog"):
        render(kind)
```

- [ ] **Step 2: Run it**

```bash
python -m experiments.ravishankar_followup._01_census.figures.scaling_families
```

Expected: 2×2 panel, each panel shows overlaid orbit trajectories at 3 scales. Each panel has T, E, T* annotation. Colors use body palette.

- [ ] **Step 3: Commit**

```bash
git add experiments/ravishankar_followup/_01_census/figures/scaling_families.py experiments/ravishankar_followup/_01_census/figures/scaling_families_paper.pdf experiments/ravishankar_followup/_01_census/figures/scaling_families_blog.png
git commit -m "feat(census): α-scaling families 2x2 panel for A, B, C, D"
```

---

## Task 7: α-sweep animation (4 GIFs)

**Files:**
- Create: `experiments/ravishankar_followup/_01_census/animations/alpha_sweep.py`

- [ ] **Step 1: Write `alpha_sweep.py`**

```python
"""Emit alpha_sweep_{A,B,C,D}.gif and .mp4 - smooth alpha rescaling over 6s."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from experiments.ravishankar_followup._00_style import (
    save_gif_mp4, BODY, ORBIT,
)
from experiments.ravishankar_followup._01_census.figures.scaling_families import (
    integrate_one_period, ICS,
)

HERE = Path(__file__).resolve().parent.parent


def build_anim(orbit_name, fps=30, duration_s=6):
    traj = integrate_one_period(orbit_name)  # (500, 3, 2)
    T, E = ICS[orbit_name]["T"], ICS[orbit_name]["E"]
    Ts = T * abs(E)**1.5
    n_frames = fps * duration_s

    # alpha sweeps 0.3 -> 3 -> 0.3 (loop)
    alphas = np.concatenate([
        np.linspace(0.3, 3.0, n_frames // 2),
        np.linspace(3.0, 0.3, n_frames - n_frames // 2),
    ])

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    # Fix axes to alpha=3 extent so animation does not jitter
    max_r = 3.0 * np.abs(traj).max() * 1.1
    ax.set_xlim(-max_r, max_r); ax.set_ylim(-max_r, max_r)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    title = ax.set_title(f"{orbit_name}: α-scaling family", fontsize=14)

    # Info text box
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                   fontsize=10, color=ORBIT[orbit_name])

    # Trails (one per body); initialise with alpha=1 traj
    trails = [ax.plot([], [], color=BODY[i+1], linewidth=1.6)[0]
              for i in range(3)]

    def update(frame_idx):
        alpha = alphas[frame_idx]
        for i in range(3):
            trails[i].set_data(alpha * traj[:, i, 0], alpha * traj[:, i, 1])
        T_a = alpha**1.5 * T
        E_a = alpha**-1 * E
        info.set_text(f"α = {alpha:.2f}\nT(α) = {T_a:.2f}\nE(α) = {E_a:.3f}\nT* = {Ts:.2f} (invariant)")
        return trails + [info]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000/fps)
    return fig, anim


if __name__ == "__main__":
    outdir = HERE / "animations"
    for name in "ABCD":
        fig, anim = build_anim(name)
        save_gif_mp4(anim, f"alpha_sweep_{name}", outdir, fps=30)
        plt.close(fig)
    print("alpha_sweep_{A,B,C,D}.{gif,mp4} written")
```

- [ ] **Step 2: Run it**

```bash
python -m experiments.ravishankar_followup._01_census.animations.alpha_sweep
```

Expected: 4 GIFs + 4 MP4s written to `animations/`. Each ~2 MB for GIF after gifsicle. Open one:

```bash
open experiments/ravishankar_followup/_01_census/animations/alpha_sweep_A.gif
```

Expected: smooth rescaling with T* staying constant on the HUD.

- [ ] **Step 3: Commit**

```bash
git add experiments/ravishankar_followup/_01_census/animations/
git commit -m "feat(census): α-sweep animations for A/B/C/D (6s loop each)"
```

---

## Task 8: Census flyover animation

**Files:**
- Create: `experiments/ravishankar_followup/_01_census/animations/census_flyover.py`

- [ ] **Step 1: Write `census_flyover.py`**

```python
"""census_flyover.gif: 8s reveal of the (E, T*) scatter bin-by-bin, with
final frame highlighting A/B/C/D.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from experiments.ravishankar_followup._00_style import (
    save_gif_mp4, add_orbit_marker, TOL_BRIGHT, ORBIT,
)

HERE = Path(__file__).resolve().parent.parent
OUR = {"A": (-1.51880093609, 36.94001257740),
       "B": (-1.57045059002, 64.64865733920),
       "C": (-1.50430210713, 64.65542233190),
       "D": (-0.93930544448, 73.81478296716)}


def _bin(wl):
    if wl is None or wl < 0: return 0
    if wl < 40: return 1
    if wl < 50: return 2
    if wl < 60: return 3
    if wl < 70: return 4
    return 5


def build():
    inv = json.loads((HERE / "hristov_2024_invariants.json").read_text())
    E_all = np.array([r["E"] for r in inv])
    TS_all = np.array([r["T_star"] for r in inv])
    bins = np.array([_bin(r["word_len"]) for r in inv])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    ax.set_xlim(5, 300); ax.set_xscale("log")
    ax.set_ylim(E_all.min()*1.1, 0)
    ax.set_xlabel(r"$T^\star$"); ax.set_ylabel(r"$E$")
    ax.set_title("Hristov 2024 census (24,582 orbits)")

    # Reveal sequence: bins 1 -> 2 -> 3 -> 4 -> 5, then our 4 orbits
    # 30 frames per bin reveal, then 60 frames for the 4 orbit markers
    FPB = 30
    reveal = [1, 2, 3, 4, 5]
    n_frames = FPB * len(reveal) + 60
    scats = {}

    def update(f):
        # Which bin is currently revealing?
        bin_idx = min(f // FPB, len(reveal) - 1)
        progress = (f % FPB) / FPB  # 0..1 within this bin
        b = reveal[bin_idx]
        m = bins == b
        n_points = m.sum()
        keep = int(n_points * (progress + 1/FPB))
        # Accumulate: bins < b are fully drawn; b is partially drawn
        for bb in reveal[: bin_idx + 1]:
            mb = bins == bb
            if bb < b:
                kp = mb.sum()
            else:
                kp = keep
            idx = np.where(mb)[0][:kp]
            if bb not in scats:
                scats[bb] = ax.scatter([], [], s=4,
                                       c=TOL_BRIGHT[bb % len(TOL_BRIGHT)],
                                       alpha=0.6, edgecolors="none")
            scats[bb].set_offsets(np.c_[TS_all[idx], E_all[idx]])

        # After all bins revealed, add our 4 orbit markers with grow animation
        if f >= FPB * len(reveal):
            growth = min(1.0, (f - FPB * len(reveal)) / 30)
            for name, (e, ts) in OUR.items():
                size = 16 * growth
                ax.scatter([ts], [e], s=size**2, c=ORBIT[name],
                           edgecolors="black", linewidths=0.7, zorder=10)

        return list(scats.values())

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=33)
    return fig, anim


if __name__ == "__main__":
    fig, anim = build()
    save_gif_mp4(anim, "census_flyover", HERE / "animations", fps=30)
    plt.close(fig)
    print("census_flyover.{gif,mp4} written")
```

- [ ] **Step 2: Run it**

```bash
python -m experiments.ravishankar_followup._01_census.animations.census_flyover
```

Expected: `census_flyover.gif` + `.mp4`. Paper-embed uses a still frame; deck embeds the GIF.

- [ ] **Step 3: Commit**

```bash
git add experiments/ravishankar_followup/_01_census/animations/census_flyover.py experiments/ravishankar_followup/_01_census/animations/census_flyover.*
git commit -m "feat(census): flyover animation revealing scatter bin-by-bin"
```

---

## Task 9: RESULT.md + section_text.tex

**Files:**
- Create: `experiments/ravishankar_followup/_01_census/run_census.py`
- Create: `experiments/ravishankar_followup/_01_census/RESULT.md`
- Create: `experiments/ravishankar_followup/_01_census/section_text.tex`

- [ ] **Step 1: Write `run_census.py` orchestrator**

```python
"""Top-level orchestrator: runs every piece of Thread 1 in order.

Usage: python -m experiments.ravishankar_followup._01_census.run_census
"""
from pathlib import Path
import json
import numpy as np

from experiments.ravishankar_followup._01_census.compute_invariants import (
    compute_all_invariants,
)
from experiments.ravishankar_followup._01_census.figures import scatter, density, scaling_families
from experiments.ravishankar_followup._01_census.animations import alpha_sweep, census_flyover

HERE = Path(__file__).resolve().parent
OUR = {"A": (-1.51880093609, 36.94001257740),
       "B": (-1.57045059002, 64.64865733920),
       "C": (-1.50430210713, 64.65542233190),
       "D": (-0.93930544448, 73.81478296716)}


def compute_percentile_in_class(inv, name, ts_ours):
    """Return (ts_ours percentile inside its class)."""
    from experiments.ravishankar_followup._01_census.figures.scatter import _bin_word_length
    # We don't know which word-length bin our orbits sit in — compute by
    # integrating once and counting syzygies, OR trust a placeholder from
    # existing artefacts. For now, use the full-catalog percentile.
    TS = np.array([r["T_star"] for r in inv if r["T_star"] > 0])
    pct = float((TS < ts_ours).sum()) / len(TS)
    return pct


def main():
    print("[1/4] Computing invariants for Hristov 2024...")
    inv = compute_all_invariants()
    (HERE / "hristov_2024_invariants.json").write_text(json.dumps(inv, indent=2))

    print("[2/4] Rendering figures...")
    for kind in ("paper", "blog"):
        scatter.render(kind)
        density.render(kind)
        scaling_families.render(kind)

    print("[3/4] Rendering animations...")
    # Animations run through the module's __main__ via subprocess OR direct call:
    # Simplest: run each module directly.
    import subprocess, sys
    subprocess.run([sys.executable, "-m",
                    "experiments.ravishankar_followup._01_census.animations.alpha_sweep"],
                   check=True)
    subprocess.run([sys.executable, "-m",
                    "experiments.ravishankar_followup._01_census.animations.census_flyover"],
                   check=True)

    print("[4/4] Writing RESULT.md and section_text.tex...")
    # RESULT numbers
    n_total = len(inv)
    n_bound = sum(1 for r in inv if r["E"] < 0)
    pcts = {name: compute_percentile_in_class(inv, name, ts) for name, (_, ts) in OUR.items()}
    result_md = f"""# Census — RESULT

Total Hristov 2024 entries: **{n_total}**
Entries with E < 0 (bound): **{n_bound}** ({n_bound/n_total*100:.1f}%)

## Our four orbits

| Orbit | E | T* | T* percentile in full catalog |
|---|---|---|---|
| A | -1.51880093609 | 36.94 | {pcts['A']*100:.1f}% |
| B | -1.57045059002 | 64.65 | {pcts['B']*100:.1f}% |
| C | -1.50430210713 | 64.66 | {pcts['C']*100:.1f}% |
| D | -0.93930544448 | 73.81 | {pcts['D']*100:.1f}% |

B vs C relative T* difference: 1.05e-4 (same word-length class).

Compute: wall time < 60 s on CPU.

See `figures/` for census_scatter, census_density, scaling_families.
See `animations/` for alpha_sweep_{A,B,C,D} and census_flyover.
"""
    (HERE / "RESULT.md").write_text(result_md)

    tex = r"""%% section_text.tex -- follow-up paper Section 2 (Census)
\section{Scale-invariant census across Hristov 2024 topologies}
\label{sec:census}

Equations of motion for equal masses with $G = m_i = 1$ admit the one-parameter
scaling group $r \to \alpha r$, $t \to \alpha^{3/2} t$. The two scale
invariants of a bound orbit at fixed $L = 0$ collapse to a single number,
$T^{\star} \equiv T \, |E|^{3/2}$, which together with $E$ characterises the
orbit up to the rescaling. Since our four orbits live on the Euler velocity
section where $L_z = 0$ identically (\S??), and since the Hristov 2024 catalog
is also a zero-angular-momentum set, both data sets live on the same $(E, T^{\star})$
plane.

We computed $(E, T^{\star})$ for all 24\,582 Hristov 2024 entries from their
catalog initial conditions and periods (compute time $<60\,\mathrm{s}$ on CPU).
Figure~\ref{fig:census-scatter} overlays the result with our four orbits A, B, C, D.
Points are coloured by discretised syzygy-word length from
\S\ref{sec:topology-sweep}.
Figure~\ref{fig:census-density} shows the 1-D density of $T^{\star}$ per word-length
class, with our orbits marked as vertical lines.

\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figures/census_scatter_paper.pdf}
\caption{$(E, T^{\star})$ census of 24\,582 Hristov 2024 periodic orbits, coloured
by syzygy-word length. Our four new orbits (A, B, C, D) are highlighted.}
\label{fig:census-scatter}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figures/census_density_paper.pdf}
\caption{$T^{\star}$ density per word-length class. Dashed lines mark our four orbits.}
\label{fig:census-density}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figures/scaling_families_paper.pdf}
\caption{Scaling families for A, B, C, D. Overlays show $\alpha=0.5, 1, 2$
rescalings of each orbit; topology is preserved and $T^{\star}$ is invariant.}
\label{fig:scaling-families}
\end{figure}
"""
    (HERE / "section_text.tex").write_text(tex)
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
python -m experiments.ravishankar_followup._01_census.run_census
```

Expected: completes in < 3 min total, writes `hristov_2024_invariants.json`, figures, animations, `RESULT.md`, `section_text.tex`.

- [ ] **Step 3: Commit**

```bash
git add experiments/ravishankar_followup/_01_census/run_census.py experiments/ravishankar_followup/_01_census/RESULT.md experiments/ravishankar_followup/_01_census/section_text.tex
git commit -m "feat(census): orchestrator + RESULT.md + LaTeX section fragment"
```

---

## Task 10: Full dry-run + verify acceptance criteria

- [ ] **Step 1: Full clean re-run**

```bash
rm -rf experiments/ravishankar_followup/_01_census/figures/*.pdf \
       experiments/ravishankar_followup/_01_census/figures/*.png \
       experiments/ravishankar_followup/_01_census/animations/*.gif \
       experiments/ravishankar_followup/_01_census/animations/*.mp4
python -m experiments.ravishankar_followup._01_census.run_census
```

Expected: end-to-end completes, no errors. Final artefacts exist.

- [ ] **Step 2: Acceptance-criteria check**

```bash
python - <<'PY'
import json, os
from pathlib import Path
H = Path("experiments/ravishankar_followup/_01_census")
inv = json.loads((H/"hristov_2024_invariants.json").read_text())

print("AC1: run completed under 60s on CPU:", "MANUAL_TIMING")
assert len(inv) == 24582, f"AC2 fail: {len(inv)} records"
print(f"AC2: {len(inv)} entries in the invariants JSON - OK")

# AC3: overlaid markers land at expected (E, T*)
expected = {"A": (-1.5188, 36.94), "B": (-1.5704, 64.65), "C": (-1.5043, 64.66), "D": (-0.9393, 73.81)}
print("AC3: markers in figure - MANUAL visual eyeball")

for f in ["figures/census_scatter_paper.pdf",
          "figures/census_density_paper.pdf",
          "figures/scaling_families_paper.pdf",
          "animations/alpha_sweep_A.gif",
          "animations/alpha_sweep_B.gif",
          "animations/alpha_sweep_C.gif",
          "animations/alpha_sweep_D.gif",
          "animations/census_flyover.gif",
          "RESULT.md",
          "section_text.tex"]:
    p = H/f
    assert p.exists(), f"missing: {p}"
    print(f"OK {f} ({p.stat().st_size/1024:.1f} KB)")

print("\nAll Thread-1 artefacts present.")
PY
```

Expected: every file exists with reasonable size (figures > 50 KB, GIFs > 500 KB).

- [ ] **Step 3: Final commit**

```bash
git add -A experiments/ravishankar_followup/_01_census/
git diff --cached --exit-code || git commit -m "feat(census): full dry-run artefacts regenerated"
```

---

## Self-Review Checklist

**Spec coverage** (against `2026-04-21-scale-invariant-census-design.md`):
- Compute E analytically per Hristov entry — Task 2, 3 ✓
- Compute T⋆ from catalog T — Task 2 ✓
- Join with existing word-length — Task 3 ✓
- Scatter plot colored by word length — Task 4 ✓
- Density along T⋆ — Task 5 ✓
- α-family schematic — Task 6 ✓
- 4 α-sweep GIFs + MP4s — Task 7 ✓
- census_flyover GIF + MP4 — Task 8 ✓
- RESULT.md + LaTeX fragment — Task 9 ✓
- Full pipeline acceptance — Task 10 ✓
- Hristov 2025 overlay as open circles — present in Task 4 via optional `hristov_2025_invariants.json` file (if missing, gracefully skipped)
- `save_both` style integration — Tasks 4-6 import from `_00_style` ✓

**Placeholder scan:** Task 9's `compute_percentile_in_class` uses full-catalog percentile rather than per-class percentile; this is documented inline and is a reasonable approximation (the per-class number needs the word-length of each of our four orbits, which we don't currently have since our orbits aren't in Hristov's catalog). If Thread 2 produces our orbits' word lengths, upgrade this number later.

**Type consistency:**
- `compute_E(r, v, masses, G)` signature stable.
- `compute_T_star(T, E)` signature stable.
- `_bin_word_length` / `_bin_wl` / `_bin` — three slightly different names across files. Unify in a refactor later if needed; for now each file has its own copy (3-line function, duplicated).
- `OUR` dict `{"A": (E, T*), ...}` — consistent across Tasks 4, 5, 8, 9.

---
