# Thread 1 — scaffolding notes

## Hristov 2024 catalog format

Path: `experiments/orbit_verification/00_catalogs/hristov_2024_sol_80.txt`
Lines: 24582
Sample lines 1-5:
```
0.10081e-3  0.86410e0  0.27592e1  0.14361e2
0.20815e-4  0.18872e0  0.13906e1  0.14361e2
0.26853e-2  0.42051e0  0.17803e1  0.14571e2
0.13664e-2  0.16960e0  0.14874e1  0.15583e2
0.61578e-3  0.64969e-1  0.14078e1  0.15583e2
```
(abbreviated; full lines are ~160 chars of high-precision mpmath-style decimal)

Column order: `x3  y3  T  T_star`
- `x3`, `y3`: third-body initial position in free-fall canonical form (bodies 1, 2 fixed at (+-0.5, 0); zero initial velocities)
- `T`: period in Hristov units
- `T_star`: scale-invariant period T|E|^(3/2), energy normalised so E ~ -1
- Values in high-precision decimal notation (mpmath eN exponent form, ~80 significant figures)

Existing parser: PRESENT at `experiments/orbit_verification/00_catalogs/parse_catalogs.py`
- Also embedded in `experiments/orbit_verification/08_hristov_2024_sweep/sweep_AB_vs_hristov_2024.py` (load_catalog function)
- Both parsers use `enumerate(f, 1)` so row numbers are 1-based

## Syzygy word-length map

`topology_sweep_hp_summary.json` schema: dict
Top-level keys: n_total_in_window, n_processed, n_errors, max_closure_over_sweep,
  n_matches_A, n_matches_B, length_hist, window, wall_sweep_s, config
- n_total_in_window = 24582 (all catalog entries processed, zero errors)
- length_hist is a nested copy of the histogram (string int keys)

`length_histogram.json` schema: dict mapping syzygy-word length (str) -> count (int)
- e.g. {"2": 66, "3": 102, "4": 22, ...}
- Covers all 24582 catalog entries

`matches.json` schema: dict with keys A, B, a_canonical, b_canonical
- A and B are lists of matching catalog row indices (both empty -- no match found for our orbits)

## Join key

Use `row` (position in hristov_2024_sol_80.txt, **1-based**) as the join key --
matches the topology sweep indexing used by run_full.py and sweep_AB_vs_hristov_2024.py.
Note: the task plan calls this `entry_index` (0-based); actual implementation
uses 1-based row numbers (`enumerate(f, 1)`). Use 1-based `row` throughout _01_census
to stay consistent with existing experiment code.
