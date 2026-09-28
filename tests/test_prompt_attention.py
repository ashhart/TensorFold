"""Bound prompt attention without changing fused head-size dispatch or causal offsets."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels.qwen.dense.v1 import exact_attention, prompt_attention


def test_fused_prompt_heads_keep_stock_dispatch(monkeypatch):
    q = SimpleNamespace(shape=(1, 24, 129, 128))
    k = SimpleNamespace(shape=(1, 4, 8192, 128))
    calls = []
    monkeypatch.setattr(exact_attention, "_STOCK", lambda *args: calls.append("stock"))
    monkeypatch.setattr(prompt_attention, "attend", lambda *args: calls.append("parts"))
    exact_attention.exact_sdpa(q, k, k, None, 0.1, "causal")
    assert calls == ["stock"]


@pytest.mark.parametrize("rows,total", [(129, 4097), (256, 8192), (257, 8192), (272, 8192), (273, 8192), (300, 5000),
                                        (400, 8192), (385, 6001)])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_bounded_prompt_matches_stock_causal_bits(rows, total, dtype):
    q = mx.random.normal((1, 24, rows, 256), key=mx.random.key(41)).astype(dtype)
    k = mx.random.normal((1, 4, total, 256), key=mx.random.key(42)).astype(dtype)
    v = mx.random.normal((1, 4, total, 256), key=mx.random.key(43)).astype(dtype)
    stock = mx.fast.scaled_dot_product_attention(q, k, v, scale=256**-0.5, mask="causal")
    mx.eval(stock)
    parts = prompt_attention.attend(q, k, v, 256**-0.5)
    mx.eval(parts)
    assert bool(mx.array_equal(parts.view(mx.uint16), stock.view(mx.uint16)).item())
