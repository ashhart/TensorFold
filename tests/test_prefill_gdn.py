"""The prompt Gated DeltaNet scan keeps mlx_lm's kernel bits and hands other calls to it."""

import pytest

mx = pytest.importorskip("mlx.core")
gd = pytest.importorskip("mlx_lm.models.gated_delta")

from tensorfold.kernels.qwen.dense.v1 import prefill_gdn

STOCK = prefill_gdn._STOCK or gd.gated_delta_kernel


def _inputs(B, T, hk, hv, dk, dv, dtype, seed):
    keys = mx.random.split(mx.random.key(seed), 6)
    q = (mx.random.normal((B, T, hk, dk), key=keys[0]) * 0.08).astype(dtype)
    k = (mx.random.normal((B, T, hk, dk), key=keys[1]) * 0.09).astype(dtype)
    v = mx.random.normal((B, T, hv, dv), key=keys[2]).astype(dtype)
    g = mx.random.uniform(0.8, 1.0, (B, T, hv), key=keys[3]).astype(mx.float32)
    beta = mx.sigmoid(mx.random.normal((B, T, hv), key=keys[4])).astype(dtype)
    state = (mx.random.normal((B, hv, dv, dk), key=keys[5]) * 0.1).astype(mx.float32)
    return q, k, v, g, beta, state


@pytest.mark.parametrize("B,T,hk,hv,dk,dv", [(1, 2048, 16, 48, 128, 128), (1, 1, 16, 48, 128, 128),
                                             (1, 3, 16, 48, 128, 128), (2, 129, 4, 8, 64, 128),
                                             (1, 77, 8, 8, 128, 64)])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_scan_matches_mlx_lm_kernel_bits(B, T, hk, hv, dk, dv, dtype):
    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    args = _inputs(B, T, hk, hv, dk, dv, dtype, 7 + T)
    assert prefill_gdn.takes(args[0], args[1], args[2], args[3], None)
    want_y, want_s = STOCK(*args)
    y, s = prefill_gdn.scan(*args)
    mx.eval(want_y, want_s, y, s)
    assert bool(mx.array_equal(y, want_y).item()) and bool(mx.array_equal(s, want_s).item())


def test_vector_gates_and_masks_stay_mlx_lm(monkeypatch):
    calls = []
    monkeypatch.setattr(prefill_gdn, "_STOCK", lambda *args: calls.append(args) or "stock")
    q, k, v, g, beta, state = _inputs(1, 4, 4, 8, 128, 128, mx.bfloat16, 3)
    g4 = mx.broadcast_to(g[..., None], (*g.shape, 128))
    assert prefill_gdn.scan(q, k, v, g4, beta, state) == "stock"
    assert prefill_gdn.scan(q, k, v, g, beta, state, mx.ones((1, 4), dtype=mx.bool_)) == "stock"
    assert len(calls) == 2


def test_install_routes_gated_delta_update():
    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    prefill_gdn.install()
    try:
        assert gd.gated_delta_kernel is prefill_gdn.scan
        q, k, v, _, _, state = _inputs(1, 33, 16, 48, 128, 128, mx.bfloat16, 11)
        a = mx.random.normal((1, 33, 48), key=mx.random.key(1)).astype(mx.bfloat16)
        b = mx.random.normal((1, 33, 48), key=mx.random.key(2)).astype(mx.bfloat16)
        a_log, dt = mx.zeros((48,), dtype=mx.float32), mx.ones((48,), dtype=mx.bfloat16)
        got = gd.gated_delta_update(q, k, v, a, b, a_log, dt, state)
        prefill_gdn.uninstall()
        want = gd.gated_delta_update(q, k, v, a, b, a_log, dt, state)
        mx.eval(got, want)
        assert all(bool(mx.array_equal(x, w).item()) for x, w in zip(got, want))
    finally:
        prefill_gdn.uninstall()
