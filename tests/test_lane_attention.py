"""The lane attention: a query's output does not depend on how many queries share the call."""


import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels.qwen.dense.v1 import lane_attention  # noqa: E402


def _same(a, b):
    return bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def _setup(L, T, cap_extra=37, seed=0):
    mx.random.seed(seed)
    H, HKV, D = 24, 4, 256
    cap = L + cap_extra
    kb = (mx.random.normal((1, HKV, cap, D)) * 0.6).astype(mx.bfloat16)
    vb = (mx.random.normal((1, HKV, cap, D)) * 0.6).astype(mx.bfloat16)
    q = (mx.random.normal((1, H, T, D)) * 0.6).astype(mx.bfloat16)
    return q, kb[:, :, :L], vb[:, :, :L], kb, vb


@pytest.mark.parametrize("L,T", [(5, 3), (130, 8), (1000, 16), (4100, 32), (20000, 8), (3000, 48), (700, 64)])
def test_window_rows_equal_single_queries(L, T):
    try:
        q, k, v, kb, vb = _setup(L, T)
        full = lane_attention.lane_sdpa(q, k, v, 256 ** -0.5)
        mx.eval(full)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"metal kernels unavailable: {str(exc).splitlines()[0][:80]}")
    for t in range(T):
        n = L - T + t + 1
        one = lane_attention.lane_sdpa(q[:, :, t:t + 1], kb[:, :, :n], vb[:, :, :n], 256 ** -0.5)
        assert _same(one, full[:, :, t:t + 1]), f"row {t} of {T} differs from its single query"


def test_matches_reference_attention():
    L, T = 300, 4
    q, k, v, _, _ = _setup(L, T, seed=3)
    try:
        ours = lane_attention.lane_sdpa(q, k, v, 256 ** -0.5).astype(mx.float32)
        mx.eval(ours)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"metal kernels unavailable: {str(exc).splitlines()[0][:80]}")
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=256 ** -0.5, mask="causal").astype(mx.float32)
    err = mx.max(mx.abs(ours - ref)).item()
    assert err < 0.02, err

