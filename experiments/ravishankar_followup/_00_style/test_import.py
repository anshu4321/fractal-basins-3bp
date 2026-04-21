"""Smoke-tests that every thread can import from the style package."""
from experiments.ravishankar_followup._00_style import (
    make_fig, save_both, add_orbit_marker, save_gif_mp4,
    update_orbit_trail, BODY, ORBIT, ORBIT_EXT, TOL_BRIGHT,
)


def test_imports_resolve():
    assert len(BODY) == 3
    assert set(ORBIT.keys()) == {"A", "B", "C", "D"}
    assert len(TOL_BRIGHT) == 7
    assert len(ORBIT_EXT) >= 4


def test_make_fig_returns_fig_ax():
    fig, ax = make_fig("paper")
    assert fig is not None
    assert ax is not None


if __name__ == "__main__":
    test_imports_resolve()
    test_make_fig_returns_fig_ax()
    print("All style-package smoke tests PASS")
