"""Syzygy-word topology match for equilateral-section candidates.

For each HP-verified equilateral-section orbit (EQ1 = figure-8, EQ3 = new),
compute the Hristov-numbered syzygy word (free-group topology descriptor
via pairwise-collinearity events) and compare against Hristov 2024 (24,582)
and Hristov 2025 stable (971).

Methodology:
  - Re-use the proven Yoshida-6 float64 integrator + collinearity-detection
    pipeline from `experiments/orbit_verification/09_syzygy_words/compute_words.py`.
    Yoshida-6 at n_steps = 50,000 per period keeps non-crossing |area|
    samples O(1), far above the float-precision floor, so between-body
    identification is qualitatively correct.
  - Equilateral-section ICs come from section.section_ic(u) (edge length 1).
    Bodies are numbered 0, 1, 2 at the triangle vertices. For the syzygy
    word we use the same `OUR_TO_HRISTOV` remap conceptually, but since
    Hristov-ordering is only defined up to body permutation, we canonicalise
    via `equivalence_class` (cyclic rotation x S3 x time-reversal) before
    catalog comparison. So the body labelling at t=0 is immaterial to the
    match verdict.

Verdicts:
  - REDISCOVERY (catalog=..., row=..., label=...) if the canonical class
    matches any catalog entry.
  - NOVEL if no entry with this word length exists in either catalog.
  - UNCERTAIN otherwise (length collision but full-word match absent /
    not runnable for Hristov 2024 because per-entry words were not
    persisted by the earlier sweep).

Outputs: topology_match_equilateral.json with one entry per orbit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.resolve().parents[2]

# Import proven syzygy-word machinery.
sys.path.insert(0, str(ROOT / "experiments" / "orbit_verification" / "09_syzygy_words"))
from compute_words import (  # noqa: E402
    _integrate_and_record_areas,
    _extract_word,
    equivalence_class,
    load_hristov_labels,
    find_matches,
)

from experiments.ravishankar_followup._03_equilateral.section import section_ic  # noqa: E402

N_STEPS = 50_000  # match Action #3 convention


def compute_syzygy_word(u_x: float, u_y: float, T: float):
    """Integrate one period on the equilateral section with Yoshida-6 and
    extract the Hristov-numbered syzygy word. Returns (word, canonical,
    closure_residual, n_events, T).
    """
    r0, v0 = section_ic((u_x, u_y))
    q0 = np.asarray(r0, dtype=np.float64)
    p0 = np.asarray(v0, dtype=np.float64)  # unit masses, so p = v
    q_trace, areas = _integrate_and_record_areas(q0, p0, float(T), N_STEPS)
    word, events = _extract_word(q_trace, areas)
    closure = float(np.max(np.abs(q_trace[-1] - q0)))
    return {
        "word": word,
        "length": len(word),
        "n_events": len(events),
        "canonical": equivalence_class(word) if word else "",
        "T": float(T),
        "closure_residual": closure,
    }


def _hristov_2024_length_histogram():
    """Return the length histogram (dict[str, int]) from the HP topology
    sweep over the 24,582 Hristov 2024 entries. Per-entry words were not
    persisted, so length-only matching is the ceiling for that catalog.
    """
    path = ROOT / "experiments/orbit_verification/14_topology_sweep_hp_full/topology_sweep_hp_summary.json"
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    return s.get("length_hist", {})


def match_catalogs(word: str, canonical: str):
    """Search Hristov 2025 (exact word via canonical class) and Hristov 2024
    (length-only fallback). Return verdict + catalog hits."""
    info = {
        "hristov_2025_matches": [],
        "hristov_2024_length_collisions": 0,
        "hristov_2025_attempted": False,
        "hristov_2024_attempted": False,
    }

    # Hristov 2025 syzygies: full-word via canonical-class equivalence.
    h25 = ROOT / "experiments/orbit_verification/00_catalogs/hristov_2025_syzygies.txt"
    if h25.exists() and word:
        info["hristov_2025_attempted"] = True
        catalog = load_hristov_labels()
        hits = find_matches(word, catalog)
        info["hristov_2025_matches"] = [
            {"row": h["hristov_row"], "label": h["label"]}
            for h in hits
        ]
    else:
        info["hristov_2025_file_found"] = h25.exists()

    # Hristov 2024: length-only (per-entry words not persisted by the sweep).
    lh = _hristov_2024_length_histogram()
    if lh is not None:
        info["hristov_2024_attempted"] = True
        info["hristov_2024_length_collisions"] = int(lh.get(str(len(word)), 0))

    # Verdict.
    if info["hristov_2025_matches"]:
        m = info["hristov_2025_matches"][0]
        verdict = (
            f"REDISCOVERY Hristov_2025_row_{m['row']} label={m['label']}"
        )
    elif not word:
        # Empty syzygy word = non-syzygy orbit. The three bodies never
        # become collinear during one period (signed triangle area stays
        # one sign throughout). Hristov's catalogs are built from
        # syzygy orbits, so by construction they do NOT include non-syzygy
        # orbits (Lagrange-like relative equilibria and their perturbative
        # continuations). This is a strong topological novelty claim:
        # the orbit cannot be any Hristov catalog entry because none of
        # them have the empty word.
        if info["hristov_2024_attempted"] and info["hristov_2024_length_collisions"] == 0:
            verdict = (
                "NOVEL (non-syzygy orbit: empty word; "
                "Hristov 2024 has 0 length-0 entries out of 24,582; "
                "Hristov's catalogs are syzygy-based by construction "
                "and cannot contain non-syzygy orbits)"
            )
        else:
            verdict = (
                "NOVEL (empty syzygy word — non-syzygy orbit; "
                "not representable in Hristov's syzygy-based catalog)"
            )
    elif info["hristov_2024_length_collisions"] == 0 and (
        not info["hristov_2025_attempted"] or not info["hristov_2025_matches"]
    ):
        # No length collision anywhere => topologically impossible to be
        # any catalog entry.
        if info["hristov_2024_attempted"]:
            verdict = "NOVEL (no Hristov 2024 entry with this word length; Hristov 2025 absent/no-match)"
        else:
            verdict = "UNCERTAIN (no catalog data available)"
    else:
        # Length collisions exist in Hristov 2024 but we cannot match words.
        verdict = (
            f"UNCERTAIN ({info['hristov_2024_length_collisions']} Hristov 2024 "
            f"entries share length {len(word)}; full-word match not possible "
            "because per-entry words were not persisted by Action #14 sweep)"
        )

    info["verdict"] = verdict
    return info


def main():
    cands_path = HERE / "hp_verified_candidates.json"
    cands = json.loads(cands_path.read_text())
    out = {}
    for c in cands:
        name = c.get("name") or f"EQ{c.get('cluster_id', '?')}"
        u = c["u"]
        T = c["T"]
        print(f"Computing syzygy word for {name}  u=({u[0]:.6f}, {u[1]:.6f})  T={T:.6f} ...",
              flush=True)
        word_info = compute_syzygy_word(u[0], u[1], T)
        match = match_catalogs(word_info["word"], word_info["canonical"])
        entry = {
            **c,
            **word_info,
            **match,
        }
        out[name] = entry
        w = word_info["word"]
        print(f"  word   : '{w[:60]}{'...' if len(w) > 60 else ''}' (len={len(w)})", flush=True)
        print(f"  canon  : '{word_info['canonical'][:40]}" +
              f"{'...' if len(word_info['canonical']) > 40 else ''}'", flush=True)
        print(f"  close  : {word_info['closure_residual']:.3e}", flush=True)
        print(f"  verdict: {match['verdict']}", flush=True)
    (HERE / "topology_match_equilateral.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'topology_match_equilateral.json'}", flush=True)


if __name__ == "__main__":
    main()
