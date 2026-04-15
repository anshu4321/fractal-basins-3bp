# Syzygy-word verdict (Action #3)

Status: **DONE**

Integrator: float64 Yoshida-6, n_steps=50000 per period, Pool(4).

Validation: C PASS, D PASS

(Validation uses row-aligned Hristov 2025 syzygy labels: stable-catalog row 6 = our C = syzygy row 6 (length 42); stable-catalog row 11 = our D = syzygy row 11 (length 48).)

## Orbit words (Hristov-numbered)

| orbit | length | word (first 60 chars) | canon (first 40) |
|-------|-------:|-----------------------|------------------|
| A | 24 | `132132132132132132132132` | `123123123123123123123123` |
| B | 42 | `312312312312312312312312312312312312312312` | `1231231231231231231231231231231231231231...` |
| C | 42 | `132132132132132132132132132132132132132132` | `1231231231231231231231231231231231231231...` |
| D | 48 | `132132132132132132132132132132132132132132132132` | `1231231231231231231231231231231231231231...` |

## Matches for orbit A

NO MATCH. Orbit A's syzygy word (length 24) is not equivalent (under cyclic rotation + S3 body permutation + time reversal) to any of the 971 Hristov 2025 syzygy labels.

## Matches for orbit B

STRONG TOPOLOGICAL MATCH: 2 equivalent entries in Hristov 2025 syzygies.txt.

| row | label | length |
|----:|-------|-------:|
| 6 | `213213213213213213213213213213213213213213` | 42 |
| 7 | `213213213213213213213213213213213213213213` | 42 |

## Interpretation

At least one of A, B topologically matches a published Hristov 2025 family. These are candidates for 'new member of known family' rather than 'new family'. IC-based identity is already settled by Actions #1-#2 (A, B are not in any Hristov 2024 row).