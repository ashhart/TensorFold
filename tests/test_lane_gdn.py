"""The lane gated-delta recurrence equals the stock state recurrence."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.models.gated_delta import gated_delta_ops  # noqa: E402

from tensorfold.kernels.lane_gdn import LaneGDNCache  # noqa: E402


def _reference(s0, qs, ks, vs, log_gs, betas):
    # [B, T, ...] through the stock ops path, float32
    g = mx.exp(log_gs)
    y, state = gated_delta_ops(qs, ks, vs, g, betas, mx.broadcast_to(s0[None], (qs.shape[0], *s0.shape)))
    return y, state


@pytest.mark.parametrize("use_kernel,version", [(False, 2), (True, 1), (True, 2)])
def test_lane_recurrence_matches_state_recurrence(use_kernel, version):
    if use_kernel and not mx.metal.is_available():
        pytest.skip("no Metal")
    key = mx.random.key(0)
    lanes, steps, hk, hv, d = 3, 7, 2, 6, 128
    k1, k2, k3, k4, k5, k6 = mx.random.split(key, 6)
    s0 = mx.random.normal((hv, d, d), key=k1) * 0.05
    qs = mx.random.normal((lanes, steps, hk, d), key=k2) * 0.1
    ks = mx.random.normal((lanes, steps, hk, d), key=k3) * 0.1
    vs = mx.random.normal((lanes, steps, hv, d), key=k4)
    log_gs = -mx.abs(mx.random.normal((lanes, steps, hv), key=k5)) * 0.2
    betas = mx.sigmoid(mx.random.normal((lanes, steps, hv), key=k6))
    want, _ = _reference(s0, qs, ks, vs, log_gs, betas)

    cache = LaneGDNCache(mx.zeros((lanes, 3, 8)), s0, key_heads=hk, capacity=4)
    cache.use_kernel = use_kernel
    cache.kernel_version = version
    got = [cache.step(qs[:, t], ks[:, t], vs[:, t], log_gs[:, t], betas[:, t]) for t in range(steps)]
    got = mx.stack(got, axis=1).astype(mx.float32)
    assert cache.t == steps - 1  # the last entry is written on the next step
    assert mx.allclose(got, want, atol=2e-3, rtol=2e-2).item()


def test_padded_step_changes_nothing():
    lanes, hk, hv, d = 2, 1, 3, 128
    s0 = mx.random.normal((hv, d, d), key=mx.random.key(1)) * 0.05
    cache = LaneGDNCache(mx.zeros((lanes, 3, 8)), s0, key_heads=hk, capacity=8)
    cache.use_kernel = False
    q = mx.random.normal((lanes, hk, d), key=mx.random.key(2)) * 0.1
    v = mx.random.normal((lanes, hv, d), key=mx.random.key(3))
    zero = mx.zeros((lanes, hv))
    before = cache.step(q, q, v, mx.full((lanes, hv), -0.1), mx.full((lanes, hv), 0.5))
    # decay 1, beta 0: a padded position
    cache.step(q, q, v, zero, zero)
    again = cache.step(q, q, v, zero, zero)
    assert before.shape == again.shape
    assert mx.allclose(cache.log_g, mx.full((lanes, hv), -0.1)).item()
