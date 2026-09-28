"""Multimodal rotary positions use NumPy doubles without loading MLX or Torch."""

from __future__ import annotations

import sys
from contextvars import Context
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from tensorfold.vision import rotary


@pytest.fixture(autouse=True)
def block_accelerators(monkeypatch):
    for name in ("mlx", "mlx.core", "mlx.nn", "mlx_lm", "mlx_vlm", "torch", "triton"):
        monkeypatch.setitem(sys.modules, name, None)


def test_frequency_axes_interleave_height_and_width_with_temporal_remainder():
    axes = rotary.frequency_axes(16, (4, 2, 2))
    assert axes == [0, 1, 2, 0, 1, 2, 0, 0]
    assert [axes.count(axis) for axis in range(3)] == [4, 2, 2]


@pytest.mark.parametrize("dims,sections", [(0, (0, 0, 0)), (3, (1, 0, 0)), (8, (1, 1)),
                                           (8, (2, 1, 0)), (8, (4, 1, -1)), (8, (2, True, 1))])
def test_invalid_frequency_sections_are_rejected(dims, sections):
    with pytest.raises(ValueError):
        rotary.frequency_axes(dims, sections)


def test_decode_positions_adjust_each_stream_without_mutating_input():
    positions = [10, 11, 20, 21, 22]
    caches = [NS(vision_rope_delta=-3), NS(vision_rope_delta=5)]
    assert rotary.decode_positions(positions, caches, [2, 3]) == [7, 8, 25, 26, 27]
    assert positions == [10, 11, 20, 21, 22]
    assert rotary.decode_positions(positions, [NS(), NS()], [2, 3]) is positions


@pytest.mark.parametrize("caches,widths", [([NS(vision_rope_delta=1)], [1]),
                                         ([NS(vision_rope_delta=1)], [1, 1])])
def test_decode_positions_reject_mismatched_stream_windows(caches, widths):
    with pytest.raises(ValueError, match="cover each stream"):
        rotary.decode_positions([1, 2], caches, widths)


def test_position_context_is_nested_exception_safe_and_context_local():
    assert rotary._positions.get() is None
    outer, inner = object(), object()
    with rotary.vision_positions(outer):
        assert rotary._positions.get() is outer
        assert Context().run(rotary._positions.get) is None
        with pytest.raises(RuntimeError):
            with rotary.vision_positions(inner):
                assert rotary._positions.get() is inner
                raise RuntimeError("exit")
        assert rotary._positions.get() is outer
    assert rotary._positions.get() is None


@pytest.fixture
def numpy_mlx(monkeypatch):
    mx, nn, mlx = ModuleType("mlx.core"), ModuleType("mlx.nn"), ModuleType("mlx")
    mx.array, mx.int32, mx.where, mx.concatenate = np.array, np.int32, np.where, np.concatenate
    nn.Module = object
    mlx.core, mlx.nn = mx, nn
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)


class PositionRoPE:
    dims, traditional = 16, False

    def __init__(self):
        self.calls = []

    def __call__(self, values, offset=0):
        self.calls.append((values.shape, offset))
        offset = np.asarray(offset)
        added = offset if offset.ndim == 0 else offset[:, None, None, None]
        return values + added


def installed():
    inner = PositionRoPE()
    attention = NS(rope=inner)
    core = NS(layers=[NS(_layer=NS(self_attn=attention)), NS(self_attn=None)])
    rotary.install_rotary(core, (4, 2, 2))
    return inner, attention.rope


def test_installed_rotary_delegates_text_unchanged(numpy_mlx):
    inner, rope = installed()
    values = np.zeros((1, 2, 3, 20), np.float32)
    np.testing.assert_array_equal(rope(values, offset=7), values + 7)
    assert inner.calls == [(values.shape, 7)]


def test_installed_rotary_selects_axes_per_frequency_and_preserves_tail(numpy_mlx):
    inner, rope = installed()
    values = np.arange(120, dtype=np.float32).reshape(1, 2, 3, 20)
    positions = np.array([[[10, 11, 12]], [[20, 21, 22]], [[30, 31, 32]]])
    with rotary.vision_positions(positions):
        actual = rope(values)
    expected = values.copy()
    for frequency, axis in enumerate([0, 1, 2, 0, 1, 2, 0, 0] * 2):
        expected[..., frequency] += positions[axis, 0]
    np.testing.assert_array_equal(actual, expected)
    assert [shape for shape, offset in inner.calls] == [(3, 2, 1, 20)] * 3
    np.testing.assert_array_equal(rope(values, offset=2), values + 2)


@pytest.mark.parametrize("shape,positions", [((2, 2, 3, 20), (3, 1, 3)), ((1, 2, 3, 20), (3, 1, 2)),
                                            ((1, 2, 3), (3, 1, 3))])
def test_rotary_refuses_incompatible_request_shapes(numpy_mlx, shape, positions):
    _, rope = installed()
    with rotary.vision_positions(np.zeros(positions)):
        with pytest.raises(ValueError, match="prefill chunk"):
            rope(np.zeros(shape))


def test_rotary_refuses_traditional_ordering(numpy_mlx):
    inner = PositionRoPE()
    inner.traditional = True
    with pytest.raises(ValueError, match="nontraditional"):
        rotary.install_rotary(NS(layers=[NS(self_attn=NS(rope=inner))]), (4, 2, 2))
