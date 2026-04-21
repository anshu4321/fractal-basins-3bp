# Census — RESULT

Total Hristov 2024 entries: **24582**
Bound (E < 0): **24582** (100.0%)
Wall time: 233.9 s

## Our four orbits vs. catalog T*-position

| Orbit | E | T* | Percentile in full catalog (T* < ours) |
|---|---|---|---|
| A | -1.51880 | 36.94 | 1.0% |
| B | -1.57045 | 64.65 | 26.7% |
| C | -1.50430 | 64.66 | 26.8% |
| D | -0.93931 | 73.81 | 60.5% |

## Striking observations

- All four orbits sit at **E ≈ -1 to -2**, in the sparse high-E region of the
  catalog. The Hristov 2024 density concentrates at E ≲ -4 (typical range
  E ∈ [-15, -3]); our orbits are 3-10× higher in energy.
- Orbit A (T* = 36.94) sits well left of the Hristov density peak (~70-80);
  percentile ~1%.
- Orbits B and C differ by |ΔT*| / T* ≈ 1.0 × 10⁻⁴ — same word-length class
  but distinct bifurcation branches. Hristov 2024 entries in the interval
  [64.645, 64.659]: **14**.

## Artefacts

- `hristov_2024_invariants.json` — per-entry invariants (24,582 records).
- `figures/census_scatter_*.{pdf,png}` — scatter with heatmap + A/B/C/D.
- `figures/census_density_*.{pdf,png}` — KDE of T*.
- `figures/scaling_families_*.{pdf,png}` — 2×2 α-panel for A, B, C, D.
- `animations/alpha_sweep_{A,B,C,D}.{gif,mp4}` — α-sweep loops.
- `animations/census_flyover.{gif,mp4}` — 8s reveal.

## TODO for follow-up

Per-entry syzygy word-lengths are not currently saved by
`14_topology_sweep_hp_full/` and appear as `word_len=None` in the JSON. A
future task will regenerate them (reusing the HP Taylor sweep pipeline) and
extend `figures/scatter.py` and `figures/density.py` to colour by class.
