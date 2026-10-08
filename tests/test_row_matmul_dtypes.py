"""The row decoder's activation dtype: projections normalize to bf16 at the boundary.

row_glue's hand-written Metal sources read and write bfloat and MLX's quantized matmul
propagates the scales' dtype, so an fp16-metadata checkpoint used to hand fp16 activations
to a bfloat kernel and the JIT build failed (#279). These tests pin the boundary: bf16
metadata stays bit-identical, fp16 metadata rounds the fp32 accumulator through fp16 into
bf16, and the cast is exactly that double rounding. The backend under test is the generic
MLX path (what fp16-metadata and other non-4-bit packs run on hosts without tensor units);
Metal backends write bf16 themselves, so the cast is a no-op there.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

from tensorfold.kernels.qwen.dense.v1 import row_matmul


class MlxBackend:
    """The generic matmul as pure MLX ops: the dtype propagation under test, no Metal."""

    name = "mlx_generic"
    max_rows = 128

    def __call__(self, x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array,
                 group_size: int, bits: int = 4) -> mx.array:
        return mx.quantized_matmul(x, weight, scales, biases, transpose=True,
                                   group_size=group_size, bits=bits)


@pytest.fixture(autouse=True)
def generic_backend(monkeypatch):
    monkeypatch.setattr(row_matmul, "BACKEND", MlxBackend())


def quantized(dtype=mx.bfloat16, *, rows: int = 64, cols: int = 256,
              bias: bool = False) -> nn.QuantizedLinear:
    """A deterministic quantized linear with scales and biases in ``dtype`` metadata."""
    mx.random.seed(11)
    w = mx.random.normal((rows, cols))
    m = nn.QuantizedLinear(cols, rows, group_size=64, bits=4, bias=bias)
    q, s, b = mx.quantize(w, group_size=64, bits=4)
    m.weight = q
    m.scales = s.astype(dtype)
    m.biases = (b if bias else mx.zeros((rows, cols // 64))).astype(dtype)
    return m


def raw(module: Any, x: mx.array) -> mx.array:
    """The projection without the boundary cast (what MLX's dtype propagation alone gives)."""
    y = mx.quantized_matmul(x, module["weight"], module["scales"], module["biases"],
                            transpose=True, group_size=module.group_size, bits=module.bits)
    if "bias" in module:
        y = y + module["bias"]
    return y


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_project_normalizes_to_bf16_and_matches_the_double_rounding(dtype):
    m = quantized(dtype=dtype)
    x = mx.random.normal((1, 8, 256)).astype(mx.bfloat16)
    mx.eval(x)
    y = row_matmul.project(m, x)
    assert y.dtype == mx.bfloat16
    want = raw(m, x).astype(mx.bfloat16)
    assert mx.array_equal(y, want)


def test_project_bf16_metadata_is_bit_identical_to_the_uncast_path():
    m = quantized(dtype=mx.bfloat16)
    x = mx.random.normal((1, 8, 256)).astype(mx.bfloat16)
    mx.eval(x)
    assert mx.array_equal(row_matmul.project(m, x), raw(m, x))


def test_project_stack_normalizes_and_members_agree():
    members = [quantized(dtype=mx.float16, rows=32), quantized(dtype=mx.float16, rows=32)]
    stack = row_matmul.Stack(members)
    x = mx.random.normal((1, 4, 256)).astype(mx.bfloat16)
    mx.eval(x)
    y = row_matmul.project_stack(stack, x)
    assert y.dtype == mx.bfloat16
    singles = mx.concatenate([row_matmul.project(m, x) for m in members], axis=-1)
    assert mx.array_equal(y, singles)


def test_logits_and_embeddings_normalize_to_bf16():
    head = quantized(dtype=mx.float16, rows=128)
    x = mx.random.normal((1, 2, 256)).astype(mx.bfloat16)
    mx.eval(x)
    assert row_matmul.logits(head, x).dtype == mx.bfloat16

    e = nn.QuantizedEmbedding(128, 256, group_size=64, bits=4)
    qw, s, _ = mx.quantize(mx.random.normal((128, 256)), group_size=64, bits=4)
    e.weight = qw
    e.scales = s.astype(mx.float16)
    ids = mx.array([[3, 7, 42]], dtype=mx.uint32)
    assert row_matmul._bf16(e(ids)).dtype == mx.bfloat16
