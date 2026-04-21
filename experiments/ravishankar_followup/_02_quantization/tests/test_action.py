"""Tests for action integral computation."""
from experiments.ravishankar_followup._02_quantization.action import compute_action


def test_action_scaling_invariance():
    """Under scaling alpha, S scales as alpha^{1/2}."""
    S_A_1 = compute_action("A")
    S_A_2 = compute_action("A", alpha=2.0)
    # S(alpha) = alpha^{1/2} * S(1)
    assert abs(S_A_2 - 2.0 ** 0.5 * S_A_1) / abs(S_A_1) < 1e-10


def test_trap_vs_gl_agree_to_6_digits():
    """Gauss-Legendre and trapezoidal should agree to at least 6 digits."""
    S_trap = compute_action("A", method="trapezoidal")
    S_gl = compute_action("A", method="gauss_legendre")
    rel = abs(S_trap - S_gl) / abs(S_gl)
    assert rel < 1e-6, f"Trap/GL disagreement: {rel}"
