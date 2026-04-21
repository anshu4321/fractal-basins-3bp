"""Figure-8 sanity: search basin for low-E periodic cells consistent with C-M orbit."""
import json
from pathlib import Path
import numpy as np

from .section import section_ic
from .eom import energy

HERE = Path(__file__).resolve().parent


def search():
    data = np.load(HERE / "basin_map.npz")
    labels = data["labels"]
    E = data["energies"]
    us_x = data["u_x_grid"]
    us_y = data["u_y_grid"]

    # Figure-8 reference: E ≈ -1.287. Widen window a bit to account for our
    # integration's symplectic drift.
    periodic_mask = (labels == 0)
    # Also restrict E to a physically sensible bounded range
    good_E = (E > -5.0) & (E < 0.0)
    candidate_mask = periodic_mask & good_E

    print(f"periodic cells (label=0): {periodic_mask.sum()}")
    print(f"periodic + E ∈ [-5, 0]: {candidate_mask.sum()}")

    # Look for cells near E = -1.287
    E_target = -1.287
    best_cells = []
    if candidate_mask.any():
        delta = np.abs(E - E_target)
        delta[~candidate_mask] = np.inf
        # Top 10 closest in E
        flat_idx = np.argsort(delta.ravel())[:10]
        for fi in flat_idx:
            i, j = divmod(fi, E.shape[1])
            best_cells.append({"i": int(i), "j": int(j),
                               "u_x": float(us_x[i]),
                               "u_y": float(us_y[j]),
                               "E": float(E[i, j]),
                               "delta_E": float(delta[i, j])})

    # Also look for the fully-symmetric u_y = 0 (L_z = 0) subspace
    # Find the row of the grid closest to u_y = 0
    row_uy0 = int(np.argmin(np.abs(us_y)))
    uy0_periodic = [{"i": int(i), "j": row_uy0,
                     "u_x": float(us_x[i]), "u_y": float(us_y[row_uy0]),
                     "E": float(E[i, row_uy0])}
                    for i in range(len(us_x))
                    if labels[i, row_uy0] == 0 and -3 < E[i, row_uy0] < 0]

    out = {
        "E_target_figure_8": E_target,
        "n_periodic_cells": int(periodic_mask.sum()),
        "n_periodic_bounded": int(candidate_mask.sum()),
        "top_10_by_delta_E": best_cells,
        "u_y0_subspace_periodic_bounded": uy0_periodic[:30],
        "note": (
            "Figure-8 has E ≈ -1.287, T ≈ 6.326. A periodic cell within ΔE < 0.05 "
            "and ready for Newton refinement validates the pipeline. Without Newton "
            "refinement we cannot claim rediscovery yet — this is a basin-level hit."
        ),
    }
    (HERE / "figure_8_sanity.json").write_text(json.dumps(out, indent=2))
    print(f"wrote figure_8_sanity.json")
    if best_cells:
        print(f"Closest to figure-8: (u_x, u_y)=({best_cells[0]['u_x']:.3f}, "
              f"{best_cells[0]['u_y']:.3f}), E={best_cells[0]['E']:.4f}, "
              f"ΔE={best_cells[0]['delta_E']:.4f}")
    print(f"u_y=0 subspace periodic bounded cells: {len(uy0_periodic)}")


if __name__ == "__main__":
    search()
