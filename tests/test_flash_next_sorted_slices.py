"""A prompt chunk's sorted expert rows go to the gather in balanced slices of at most MAX_SORTED_ROWS, same bits."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm  # noqa: E402


def test_long_calls_split_into_balanced_slices_in_order(monkeypatch):
    calls = []

    def gather(x, w, s, b, rhs_indices=None, **kw):
        calls.append(int(x.shape[0]))
        return x[..., :1] + rhs_indices.astype(x.dtype)[:, None, None]

    monkeypatch.setattr(prefill_mm, "MAX_SORTED_ROWS", 100)
    monkeypatch.setattr(prefill_mm, "tiles", lambda: False)
    monkeypatch.setattr(prefill_mm.mx, "gather_qmm", gather)
    layer = SimpleNamespace(weight=mx.zeros((4, 8, 8), dtype=mx.uint32), scales=None, biases=None)
    x = mx.zeros((250, 64), dtype=mx.float32)
    idx = mx.array(np.sort(np.arange(250) % 4).astype(np.uint32))
    out = prefill_mm._experts(x, layer, idx)
    assert calls == [84, 84, 82] and out.shape == (250, 1)
    assert np.array_equal(np.array(out[:, 0]), np.array(idx).astype(np.float32))


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")
def test_sliced_rows_equal_one_call_bit_for_bit(monkeypatch):
    rng = np.random.default_rng(3)
    w = mx.array(rng.normal(size=(8, 64, 256)).astype(np.float32) * 0.05)
    wq, scales, biases = mx.quantize(w, group_size=32, bits=4)
    layer = SimpleNamespace(weight=wq, scales=scales.astype(mx.bfloat16), biases=biases.astype(mx.bfloat16))
    x = mx.array(rng.normal(size=(400, 256)).astype(np.float32)).astype(mx.bfloat16)
    idx = mx.array(np.sort(rng.integers(0, 8, size=400)).astype(np.uint32))
    whole = prefill_mm._experts(x, layer, idx)
    monkeypatch.setattr(prefill_mm, "MAX_SORTED_ROWS", 150)
    sliced = prefill_mm._experts(x, layer, idx)
    assert bool(mx.array_equal(whole, sliced).item())
