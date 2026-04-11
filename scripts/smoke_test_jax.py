"""Smoke test: JAX sees the H100, fp64 is enabled, and a real fp64 matmul lands on the GPU.

This is the minimal proof that the Phase 1-2 compute substrate works. If any assertion
here fails, downstream work is pointless until the failure is fixed.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def main() -> None:
    print("=== jax build info ===")
    print(f"jax version       : {jax.__version__}")
    print(f"jaxlib version    : {jax.lib.__version__}")
    print(f"jax_enable_x64    : {jax.config.read('jax_enable_x64')}")
    print(f"default backend   : {jax.default_backend()}")
    print(f"devices           : {jax.devices()}")
    print()

    assert jax.default_backend() == "gpu", "JAX did not find a GPU backend"
    devices = jax.devices()
    assert any(d.platform == "gpu" for d in devices), "no GPU devices reported"

    print("=== fp64 smoke test on GPU ===")
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2048, 2048), dtype=jnp.float64)
    y = jax.random.normal(key, (2048, 2048), dtype=jnp.float64)

    assert x.dtype == jnp.float64, f"expected float64, got {x.dtype}"
    print(f"matrix dtype      : {x.dtype}")
    print(f"matrix device     : {list(x.devices())}")

    z = (x @ y).block_until_ready()
    print(f"product dtype     : {z.dtype}")
    print(f"product device    : {list(z.devices())}")
    print(f"product fro norm  : {float(jnp.linalg.norm(z)):.6e}")

    print()
    print("=== fp64 round-trip invariance ===")
    a = jnp.asarray(1.0 + 1e-12, dtype=jnp.float64)
    b = jnp.asarray(1.0, dtype=jnp.float64)
    delta = float(a - b)
    print(f"float64 resolves 1e-12 : {delta:.3e}")
    assert 5e-13 < delta < 2e-12, (
        f"fp64 seems broken: expected ~1e-12, got {delta:.3e}"
    )

    print()
    print("=== OK: jax + cuda12 + fp64 on H100 ===")


if __name__ == "__main__":
    main()
