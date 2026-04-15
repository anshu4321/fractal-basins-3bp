"""Action #3 - Syzygy-word computation for orbits A, B, C, D.

Fast float64 Yoshida-6 path. Collinear detection does not need high precision;
float64 at n_steps = 50,000 per period gives area values at sign changes that
are far from the float-precision floor (~1e-10), so midpoint identification is
qualitatively correct. This script targets < 30 s total wall on GCP.

Conventions (aligned with Hristov 2025 syzygies.txt, which is row-aligned
with the stable-orbits catalog hristov_2025_stable.txt):

  - At each collinear configuration (signed-area = 0 crossing) during one
    period, assign a digit 1, 2, or 3 = index of the body that is
    SPATIALLY BETWEEN the other two along the collinear line.
  - Hristov's body numbering (IC eq. 3 in arXiv:2510.22802): body 1 at
    (-1, 0), body 2 at (0, 0), body 3 at (+1, 0). Our internal numbering
    places body 0 at (-1, 0), body 1 at (+1, 0), body 2 at (0, 0), so we
    relabel our (0, 1, 2) -> Hristov (1, 3, 2).
  - Two words are topologically equivalent iff one is a cyclic rotation
    of a body-permutation (or time-reversal) of the other.

Validation gate (row-aligned Hristov 2025 labels for C, D):
  - Stable-catalog row 6 = our C = syzygy row 6 label
    "213213213213213213213213213213213213213213" (length 42, (213)^14).
  - Stable-catalog row 11 = our D = syzygy row 11 label
    "213213213213213213213213213213213213213213213213" (length 48, (213)^16).

  NOTE: The earlier prompt outlined short labels (length 16 for C, length 34
  for D); those are the labels at file lines 2 and 5 of the syzygy file but
  those lines correspond to different stable-catalog rows, not ours. The
  row-aligned labels are the definitive gate and match our HP-validated words.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from mega3bp.gcp_telemetry import Telemetry  # noqa: E402

HP_JSON = REPO / "experiments/orbit_verification/06_high_precision/hp_heyoka_newton_results.json"
SYZ_CAT = REPO / "experiments/orbit_verification/00_catalogs/hristov_2025_syzygies.txt"

# Row-aligned Hristov labels (see module docstring).
HRISTOV_LABEL = {
    "C": "213213213213213213213213213213213213213213",        # row 6, length 42
    "D": "213213213213213213213213213213213213213213213213",  # row 11, length 48
}

# our body index 0,1,2 -> Hristov body 1,3,2 (see module docstring).
OUR_TO_HRISTOV = {0: 1, 1: 3, 2: 2}

N_STEPS = 50_000  # float64 Yoshida-6 steps per period


# ---------------------------------------------------------------------------
# Yoshida-6 integrator in pure NumPy (float64).
# ---------------------------------------------------------------------------

# Yoshida 1990 Solution A, 6th order, symmetric composition of Verlet.
_W1 = -1.17767998417887
_W2 = 0.235573213359357
_W3 = 0.784513610477560
_W0 = 1.0 - 2.0 * (_W1 + _W2 + _W3)
_YOSHIDA6_WEIGHTS = np.array([_W3, _W2, _W1, _W0, _W1, _W2, _W3], dtype=np.float64)


def _force(q: np.ndarray) -> np.ndarray:
    """F_i = sum_{j != i} (q_j - q_i) / |q_j - q_i|^3 for equal unit masses."""
    q1, q2, q3 = q[0], q[1], q[2]
    d12 = q2 - q1
    d13 = q3 - q1
    d23 = q3 - q2
    r12_3 = (d12 @ d12) ** -1.5
    r13_3 = (d13 @ d13) ** -1.5
    r23_3 = (d23 @ d23) ** -1.5
    F1 = d12 * r12_3 + d13 * r13_3
    F2 = -d12 * r12_3 + d23 * r23_3
    F3 = -d13 * r13_3 - d23 * r23_3
    out = np.empty_like(q)
    out[0] = F1
    out[1] = F2
    out[2] = F3
    return out


def _verlet_step(q: np.ndarray, p: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    q_half = q + 0.5 * dt * p
    p_new = p + dt * _force(q_half)
    q_new = q_half + 0.5 * dt * p_new
    return q_new, p_new


def _yoshida6_step(q: np.ndarray, p: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    for w in _YOSHIDA6_WEIGHTS:
        q, p = _verlet_step(q, p, float(w) * h)
    return q, p


def _integrate_and_record_areas(q0: np.ndarray, p0: np.ndarray, T: float, n_steps: int):
    """Integrate for one period with Yoshida-6, recording signed area at each
    step along with the q-state (needed for midpoint identification).

    Returns (q_trace, areas) where:
      q_trace: shape (n_steps + 1, 3, 2) — full position history.
      areas:   shape (n_steps + 1,)       — signed area at each step.
    """
    h = T / n_steps
    q_trace = np.empty((n_steps + 1, 3, 2), dtype=np.float64)
    areas = np.empty(n_steps + 1, dtype=np.float64)
    q_trace[0] = q0
    areas[0] = _signed_area(q0)
    q, p = q0.copy(), p0.copy()
    for k in range(n_steps):
        q, p = _yoshida6_step(q, p, h)
        q_trace[k + 1] = q
        areas[k + 1] = _signed_area(q)
    return q_trace, areas


def _signed_area(Q: np.ndarray) -> float:
    """Signed area = (q1 - q0) x (q2 - q0). Zero when bodies are collinear."""
    a = Q[1] - Q[0]
    b = Q[2] - Q[0]
    return float(a[0] * b[1] - a[1] * b[0])


# ---------------------------------------------------------------------------
# Collinear event extraction.
# ---------------------------------------------------------------------------

# Threshold below which a sample is treated as numerically indistinguishable
# from zero. Float64 Yoshida-6 on these orbits gives |area| at non-crossing
# samples >> 1e-6 and |area| ~ 0 only near true crossings (interpolatable).
# 1e-10 is well below the typical |area| samples (O(1)) and well above the
# float64 noise floor (O(1e-14)).
AREA_ZERO_EPS = 1e-10


def _between_body(q_near: np.ndarray) -> int:
    """Return the index (internal 0..2) of the body that is spatially BETWEEN
    the other two along the line they define.

    At a true collinear configuration, exactly one body lies between the
    other two. We identify it by the projection parameter t = (q_m - q_i) .
    (q_j - q_i) / ||q_j - q_i||^2: body m is "between" iff 0 <= t <= 1.
    We pick the candidate with the smallest perpendicular distance to the line
    among those with 0 <= t <= 1.

    Fallback (for near-misses from float round-off): pick the candidate
    whose t is closest to 0.5.
    """
    candidates = []
    for m in range(3):
        others = [i for i in range(3) if i != m]
        vij = q_near[others[1]] - q_near[others[0]]
        vim = q_near[m] - q_near[others[0]]
        denom = float(np.dot(vij, vij))
        if denom < 1e-20:
            continue
        t_param = float(np.dot(vim, vij)) / denom
        proj = q_near[others[0]] + t_param * vij
        perp = float(np.linalg.norm(q_near[m] - proj))
        candidates.append({"m": m, "t": t_param, "perp": perp})
    inside = [c for c in candidates if 0.0 <= c["t"] <= 1.0]
    if inside:
        return min(inside, key=lambda c: c["perp"])["m"]
    return min(candidates, key=lambda c: abs(c["t"] - 0.5))["m"]


def _extract_word(q_trace: np.ndarray, areas: np.ndarray) -> tuple[str, list]:
    """Build the Hristov-numbered syzygy word from the trajectory.

    The orbit is periodic; syzygies are the zero-crossings of the signed
    area on the circle S^1 = R / T*Z. We build a reduced sequence of
    samples with |area| > AREA_ZERO_EPS and walk it cyclically — each
    consecutive pair with opposite signs is one syzygy. For each crossing
    we take the sample with the smaller |area| and identify the between-body
    at that sample's position. For the wrap-around pair we register the
    crossing at t=0 / t=T (the Euler half-twist starting configuration).
    """
    n = len(areas)
    nz_idx = [k for k in range(n) if abs(areas[k]) > AREA_ZERO_EPS]
    if not nz_idx:
        return "", []

    digits = []
    events = []
    for i in range(len(nz_idx)):
        k0 = nz_idx[i]
        k1 = nz_idx[(i + 1) % len(nz_idx)]
        a0 = areas[k0]
        a1 = areas[k1]
        if (a0 > 0) == (a1 > 0):
            continue  # no sign change between these samples

        if k1 > k0:
            k_near = k0 if abs(a0) <= abs(a1) else k1
            kind = "crossing"
        else:
            # Wrap-around: crossing happens at t=0/t=T. We evaluate
            # body-identification at grid index 0 (t=0 Euler config).
            k_near = 0
            kind = "wrap"

        m_internal = _between_body(q_trace[k_near])
        m_hristov = OUR_TO_HRISTOV[m_internal]
        digits.append(str(m_hristov))
        events.append({
            "kind": kind,
            "k0": int(k0),
            "k1": int(k1),
            "body_internal": int(m_internal),
            "body_hristov": int(m_hristov),
        })

    return "".join(digits), events


# ---------------------------------------------------------------------------
# Worker: one orbit, one process.
# ---------------------------------------------------------------------------

def integrate_and_get_word(args) -> dict:
    """Worker entrypoint: receive (name, v1, v2, T, n_steps) and return a
    complete per-orbit result dict."""
    name, v1, v2, T, n_steps = args
    t0 = time.perf_counter()
    q0 = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    p0 = np.array([[v1, v2], [v1, v2], [-2.0 * v1, -2.0 * v2]], dtype=np.float64)
    q_trace, areas = _integrate_and_record_areas(q0, p0, T, n_steps)
    word, events = _extract_word(q_trace, areas)
    closure = float(np.max(np.abs(q_trace[-1] - q0)))
    wall = time.perf_counter() - t0
    return {
        "name": name,
        "word": word,
        "length": len(word),
        "n_events": len(events),
        "T": T,
        "closure_residual": closure,
        "wall_time_s": wall,
        "canonical": equivalence_class(word),
    }


# ---------------------------------------------------------------------------
# Word equivalence: cyclic rotation + S3 body permutation + time reversal.
# ---------------------------------------------------------------------------

_S3_PERMS = [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]


def cyclic_rotations(s: str):
    return [s[i:] + s[:i] for i in range(len(s))]


def body_permutation(s: str, perm) -> str:
    return "".join(str(perm[int(c) - 1]) for c in s)


def equivalence_class(word: str) -> str:
    if not word:
        return word
    candidates = set()
    for perm in _S3_PERMS:
        permuted = body_permutation(word, perm)
        for rot in cyclic_rotations(permuted):
            candidates.add(rot)
            candidates.add(rot[::-1])
    return min(candidates)


def are_equivalent(s1: str, s2: str) -> bool:
    return len(s1) == len(s2) and equivalence_class(s1) == equivalence_class(s2)


# ---------------------------------------------------------------------------
# IO.
# ---------------------------------------------------------------------------

def load_orbit_ics():
    data = json.loads(HP_JSON.read_text())
    out = {}
    for entry in data:
        name = entry.get("name")
        if name in ("A", "B", "C", "D"):
            out[name] = {
                "v1": float(entry["v1_HP"]),
                "v2": float(entry["v2_HP"]),
                "T": float(entry["T_HP"]),
            }
    return out


def load_hristov_labels():
    """Parse hristov_2025_syzygies.txt with format `N)   DIGITS   LENGTH`.
    Returns a list of {row, label, length} dicts."""
    rows = []
    for line in SYZ_CAT.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        row_tok = parts[0].rstrip(")")
        try:
            row = int(row_tok)
        except ValueError:
            continue
        label = parts[1]
        if set(label) - set("123"):
            continue
        try:
            length = int(parts[2])
        except ValueError:
            length = len(label)
        rows.append({"row": row, "label": label, "length": length})
    return rows


def find_matches(word: str, catalog):
    if not word:
        return []
    target_canon = equivalence_class(word)
    matches = []
    for entry in catalog:
        if entry["length"] != len(word):
            continue
        if equivalence_class(entry["label"]) == target_canon:
            matches.append({
                "hristov_row": entry["row"],
                "label": entry["label"],
                "our_word": word,
                "length": entry["length"],
            })
    return matches


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> int:
    t_total = time.perf_counter()
    tele = Telemetry(HERE / "status.json", total=4, stage="init")

    print("[WORDS] loading ICs...", flush=True)
    ics = load_orbit_ics()
    args_list = [(name, ic["v1"], ic["v2"], ic["T"], N_STEPS)
                 for name, ic in ics.items()]
    tele.update(progress=0, stage="integrating", n_steps=N_STEPS, integrator="yoshida6_float64")

    print(f"[WORDS] parallel integrate (Pool(4), n_steps={N_STEPS})...", flush=True)
    per_orbit: dict[str, dict] = {}
    word_lengths: dict[str, int] = {}

    with Pool(processes=4) as pool:
        for i, res in enumerate(pool.imap_unordered(integrate_and_get_word, args_list)):
            per_orbit[res["name"]] = res
            word_lengths[res["name"]] = res["length"]
            print(f"[WORDS] {res['name']}: len={res['length']} "
                  f"(events={res['n_events']}), closure={res['closure_residual']:.3e}, "
                  f"wall={res['wall_time_s']:.1f}s",
                  flush=True)
            print(f"[WORDS] {res['name']}: word = {res['word'][:60]}"
                  f"{'...' if len(res['word']) > 60 else ''}", flush=True)
            tele.update(progress=i + 1, stage="integrating",
                        word_lengths=dict(word_lengths))

    tele.update(progress=4, stage="computing_words", word_lengths=dict(word_lengths))

    # Validate C, D against row-aligned Hristov labels.
    tele.update(stage="validating")
    validation = {}
    for name in ("C", "D"):
        expected = HRISTOV_LABEL[name]
        ok = are_equivalent(per_orbit[name]["word"], expected)
        validation[name] = "PASS" if ok else "FAIL"
        per_orbit[name]["validation_expected_label"] = expected
        per_orbit[name]["validation_result"] = validation[name]
    print(f"[WORDS] validation: C {validation['C']}, D {validation['D']}", flush=True)

    # Build words.json regardless.
    words_out = {
        "orbits": {
            name: {
                "word": per_orbit[name]["word"],
                "length": per_orbit[name]["length"],
                "equivalence_canonical": per_orbit[name]["canonical"],
                "T": per_orbit[name]["T"],
                "closure_residual": per_orbit[name]["closure_residual"],
                "wall_time_s": per_orbit[name]["wall_time_s"],
                "n_events": per_orbit[name]["n_events"],
                "validation_expected_label": per_orbit[name].get("validation_expected_label"),
                "validation_result": per_orbit[name].get("validation_result"),
            }
            for name in ("A", "B", "C", "D")
        },
        "config": {
            "integrator": "yoshida6_float64",
            "n_steps": N_STEPS,
            "our_to_hristov_body_map": OUR_TO_HRISTOV,
            "area_zero_eps": AREA_ZERO_EPS,
            "notes": (
                "Validation gate: row-aligned Hristov 2025 labels. "
                "Stable-catalog row 6 = our C = syzygy row 6 (length 42). "
                "Stable-catalog row 11 = our D = syzygy row 11 (length 48)."
            ),
        },
    }
    (HERE / "words.json").write_text(json.dumps(words_out, indent=2))

    if validation["C"] != "PASS" or validation["D"] != "PASS":
        tele.error("validation failed", validation=validation,
                    words={k: per_orbit[k]["word"] for k in ("A", "B", "C", "D")})
        verdict = {
            "status": "BLOCKED",
            "validation": validation,
            "reason": "C or D syzygy word does not match Hristov 2025 row-aligned label; abort before A/B matching.",
            "words": {name: per_orbit[name]["word"] for name in ("A", "B", "C", "D")},
            "matches": {"A": [], "B": []},
        }
        (HERE / "verdict.json").write_text(json.dumps(verdict, indent=2))
        _write_result_md(verdict, per_orbit)
        print("[WORDS] BLOCKED: skipping A/B catalog match.", flush=True)
        return 1

    # Match A, B against full Hristov 2025 syzygy catalog.
    tele.update(stage="matching")
    catalog = load_hristov_labels()
    print(f"[WORDS] loaded {len(catalog)} Hristov 2025 syzygy entries", flush=True)

    matches = {}
    for name in ("A", "B"):
        matches[name] = find_matches(per_orbit[name]["word"], catalog)
    print(f"[WORDS] matching: A {len(matches['A'])} matches, B {len(matches['B'])} matches",
          flush=True)

    wall_total = time.perf_counter() - t_total
    verdict = {
        "status": "DONE",
        "validation": validation,
        "matches": matches,
        "words": {name: per_orbit[name]["word"] for name in ("A", "B", "C", "D")},
        "wall_time_s_total": wall_total,
    }
    (HERE / "verdict.json").write_text(json.dumps(verdict, indent=2))
    _write_result_md(verdict, per_orbit)

    tele.done(stage="done",
              validation=validation,
              word_lengths=word_lengths,
              matches_A=len(matches["A"]),
              matches_B=len(matches["B"]),
              wall_time_s=wall_total)
    print(f"[WORDS] done in {wall_total:.1f}s", flush=True)
    return 0


def _write_result_md(verdict: dict, per_orbit: dict):
    lines = []
    lines.append("# Syzygy-word verdict (Action #3)")
    lines.append("")
    lines.append(f"Status: **{verdict['status']}**")
    lines.append("")
    lines.append(f"Integrator: float64 Yoshida-6, n_steps={N_STEPS} per period, Pool(4).")
    lines.append("")
    lines.append(f"Validation: C {verdict['validation']['C']}, D {verdict['validation']['D']}")
    lines.append("")
    lines.append("(Validation uses row-aligned Hristov 2025 syzygy labels: "
                 "stable-catalog row 6 = our C = syzygy row 6 (length 42); "
                 "stable-catalog row 11 = our D = syzygy row 11 (length 48).)")
    lines.append("")
    lines.append("## Orbit words (Hristov-numbered)")
    lines.append("")
    lines.append("| orbit | length | word (first 60 chars) | canon (first 40) |")
    lines.append("|-------|-------:|-----------------------|------------------|")
    for name in ("A", "B", "C", "D"):
        r = per_orbit[name]
        w = r["word"]
        c = r["canonical"]
        lines.append(
            f"| {name} | {len(w)} | `{w[:60]}{'...' if len(w) > 60 else ''}` | "
            f"`{c[:40]}{'...' if len(c) > 40 else ''}` |")
    lines.append("")

    if verdict["status"] == "BLOCKED":
        lines.append("## Validation failed")
        lines.append("")
        lines.append(f"Reason: {verdict.get('reason', 'unknown')}")
        lines.append("")
        lines.append("Abort. A/B matching skipped.")
    else:
        for name in ("A", "B"):
            ms = verdict["matches"][name]
            lines.append(f"## Matches for orbit {name}")
            lines.append("")
            if not ms:
                lines.append(
                    f"NO MATCH. Orbit {name}'s syzygy word "
                    f"(length {len(verdict['words'][name])}) is not equivalent "
                    "(under cyclic rotation + S3 body permutation + time reversal) "
                    "to any of the 971 Hristov 2025 syzygy labels.")
            else:
                lines.append(f"STRONG TOPOLOGICAL MATCH: {len(ms)} equivalent "
                             f"entries in Hristov 2025 syzygies.txt.")
                lines.append("")
                lines.append("| row | label | length |")
                lines.append("|----:|-------|-------:|")
                for m in ms[:10]:
                    lines.append(f"| {m['hristov_row']} | `{m['label']}` | {m['length']} |")
                if len(ms) > 10:
                    lines.append(f"| ... | {len(ms) - 10} more | ... |")
            lines.append("")

        lines.append("## Interpretation")
        lines.append("")
        if not verdict["matches"]["A"] and not verdict["matches"]["B"]:
            lines.append("Both A and B have syzygy words not conjugate to any published "
                         "Hristov 2025 label under cyclic rotation, S3 body permutation, or "
                         "time reversal. **Topological evidence that A and B are genuinely "
                         "new families.**")
        else:
            lines.append("At least one of A, B topologically matches a published Hristov 2025 "
                         "family. These are candidates for 'new member of known family' rather "
                         "than 'new family'. IC-based identity is already settled by Actions "
                         "#1-#2 (A, B are not in any Hristov 2024 row).")

    (HERE / "RESULT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
