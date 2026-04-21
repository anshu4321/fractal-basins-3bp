"""LD per candidate, colored by survival status (Newton / HP / topology verdict)."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from experiments.ravishankar_followup._00_style import make_fig, save_both

HERE = Path(__file__).resolve().parent.parent

STATUS_COLORS = {
    "NOVEL": "#27AE60",         # green
    "HP_only": "#F39C12",       # amber: refined but not HP-verified — (n/a here, but kept for clarity)
    "Newton_only": "#3498DB",   # blue
    "candidate_only": "#95A5A6",  # grey
}


def render(kind="paper"):
    cands = json.loads((HERE / "candidates_ICs.json").read_text())
    refined = json.loads((HERE / "refined_candidates.json").read_text())
    hp = json.loads((HERE / "hp_verified_candidates.json").read_text())
    topo = json.loads((HERE / "topology_match_equilateral.json").read_text())

    # Build lookups keyed by candidate grid u = (u_x, u_y) rounded
    def key(u_x, u_y):
        return (round(float(u_x), 3), round(float(u_y), 3))

    refined_keys = {key(r["u_x"], r["u_y"]): r for r in refined}
    hp_keys = {key(r["u_x"], r["u_y"]): r for r in hp}
    # topology is keyed by name -> entry with u_x, u_y
    topo_by_key = {key(v["u_x"], v["u_y"]): v for v in topo.values()}

    fig, ax = make_fig(kind=kind, figsize=(9, 5) if kind == "blog" else (6.5, 3.5))
    for i, c in enumerate(cands):
        k = key(c["u_x"], c["u_y"])
        if k in hp_keys:
            if k in topo_by_key:
                verdict = topo_by_key[k].get("verdict", "")
                v_head = verdict.split()[0] if verdict else ""
                color = STATUS_COLORS.get(v_head, "#F39C12")
            else:
                color = "#F39C12"  # HP-only (no topology entry)
        elif k in refined_keys and refined_keys[k].get("converged"):
            color = STATUS_COLORS["Newton_only"]
        else:
            color = STATUS_COLORS["candidate_only"]
        ax.bar([i], [c["ld"]], color=color, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("candidate index"); ax.set_ylabel("LD (entropy)")
    ax.set_title(f"Candidates by LD, colored by survival status "
                 f"(n_total={len(cands)}, n_HP_NOVEL={len(hp)})")

    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(color=STATUS_COLORS["candidate_only"], label="candidate only"),
        Patch(color=STATUS_COLORS["Newton_only"], label="Newton converged"),
        Patch(color="#F39C12", label="HP-verified"),
        Patch(color=STATUS_COLORS["NOVEL"], label="HP-verified + NOVEL"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=8 if kind == "paper" else 10)

    save_both(fig, "candidates_ld_scan", HERE / "figures")
    plt.close(fig)


if __name__ == "__main__":
    for kind in ("paper", "blog"):
        render(kind)
    print("candidates_ld_scan_ok")
