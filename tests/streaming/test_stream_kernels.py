"""Streamed experts read through slots give the resident kernels' bits: picks, weights, activations and outputs."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from types import SimpleNamespace  # noqa: E402

from tensorfold.kernels.qwen.flash_next.v1 import experts, stream_experts  # noqa: E402


def metal():
    try:
        return mx.metal.is_available()
    except Exception:  # noqa: BLE001 - no Metal: the kernels do not apply
        return False


pytestmark = pytest.mark.skipif(not metal(), reason="needs a Metal GPU")


def quantized(rng, n_experts, rows, cols):
    w = mx.array(rng.normal(0, 0.05, (n_experts, rows, cols)).astype(np.float32)).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=32, bits=4)
    return SimpleNamespace(weight=q, scales=s, biases=b)


def test_slot_kernels_equal_the_resident_kernels_bit_for_bit():
    rng = np.random.default_rng(17)
    ne, top, d, width, rows = 64, 10, 512, 640, 5          # Flash Next's width: the down kernel reads 512+ a row
    gate, up, down = quantized(rng, ne, width, d), quantized(rng, ne, width, d), quantized(rng, ne, d, width)
    sgate, sup, sdown = (quantized(rng, 1, width, d), quantized(rng, 1, width, d), quantized(rng, 1, d, width))
    shared = tuple(SimpleNamespace(weight=m.weight[0], scales=m.scales[0], biases=m.biases[0])
                   for m in (sgate, sup, sdown))
    x = mx.array(rng.normal(0, 1, (rows, d)).astype(np.float32)).astype(mx.bfloat16)
    logits = mx.array(rng.normal(0, 1, (rows, ne + 1)).astype(np.float32))
    act, picks, weights = experts.expert_gateup(x, logits, top, ne, gate, up, shared=shared[:2])
    y = experts.expert_down_y(act, picks, down, shared[2])
    # a pool of 3 * ne slots holding the experts at scattered slots, and a two-layer slot table
    perm = rng.permutation(3 * ne)[:ne]
    def pooled(m):
        parts = []
        for part in ("weight", "scales", "biases"):
            a = getattr(m, part)
            full = mx.zeros((3 * ne, *a.shape[1:]), dtype=a.dtype)
            full[mx.array(perm)] = a
            parts.append(full)
        return SimpleNamespace(weight=parts[0], scales=parts[1], biases=parts[2])
    pool = SimpleNamespace(gate=pooled(gate), up=pooled(up), down=pooled(down))
    table = np.zeros((2, ne), dtype=np.int32)
    table[1] = perm
    slot_of, layer = mx.array(table.reshape(-1)), mx.array([1] + [0] * 7, dtype=mx.int32)
    box = mx.zeros((rows * top,), dtype=mx.uint32)
    mx.eval(box)
    token = stream_experts.route(logits, top, box)
    act2, picks2, weights2 = stream_experts.gateup(x, logits, top, ne, pool, slot_of, layer, shared[:2])
    y2 = stream_experts.down(act2, picks2, ne, pool, slot_of, layer, shared[2])
    mx.eval(token, y, y2)
    assert np.array_equal(np.array(box).reshape(rows, top), np.array(picks))
    for a, b in ((picks, picks2), (weights, weights2), (act, act2), (y, y2)):
        assert bool(mx.array_equal(a, b).item())
