"""Request-local multimodal rotary positions without changing text-only arithmetic."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Sequence

_positions = ContextVar('tensorfold_vision_positions', default=None)


def frequency_axes(dims: int, sections: Sequence[int]) -> list[int]:
    if dims <= 0 or dims % 2 or len(sections) != 3:
        raise ValueError('multimodal rotary dimensions require three valid sections')
    if any(type(n) is not int or n < 0 for n in sections) or sum(sections) != dims // 2:
        raise ValueError('multimodal rotary sections must cover the rotary frequencies')
    return [1 if i % 3 == 1 and i < 3 * sections[1] else
            2 if i % 3 == 2 and i < 3 * sections[2] else 0 for i in range(dims // 2)]


@contextmanager
def vision_positions(positions: Any):
    token = _positions.set(positions)
    try:
        yield
    finally:
        _positions.reset(token)


def decode_positions(positions, caches, widths):
    if not any(getattr(cache, 'vision_rope_delta', 0) for cache in caches):
        return positions
    if len(caches) != len(widths) or sum(widths) != len(positions):
        raise ValueError('vision positions must cover each stream window')
    out, start = [], 0
    for cache, width in zip(caches, widths):
        delta = int(getattr(cache, 'vision_rope_delta', 0))
        out.extend(int(p) + delta for p in positions[start:start + width])
        start += width
    return out


def install_rotary(core: Any, sections: Sequence[int]) -> None:
    import mlx.core as mx
    import mlx.nn as nn

    class MultimodalRoPE(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.dims = int(inner.dims)
            self.axes = frequency_axes(self.dims, sections)
            if getattr(inner, 'traditional', False):
                raise ValueError('this vision adapter requires nontraditional rotary ordering')

        def __call__(self, x, offset=0):
            positions = _positions.get()
            if positions is None:
                return self.inner(x, offset=offset)
            if x.ndim != 4 or x.shape[0] != 1 or tuple(positions.shape) != (3, 1, x.shape[2]):
                raise ValueError('vision rotary positions do not match the prefill chunk')
            rows = x.transpose(2, 1, 0, 3)
            rotated = [self.inner(rows, offset=positions[axis, 0]) for axis in range(3)]
            axes = mx.array(self.axes + self.axes, dtype=mx.int32)
            prefix = mx.where(axes == 1, rotated[1][..., :self.dims],
                              mx.where(axes == 2, rotated[2][..., :self.dims], rotated[0][..., :self.dims]))
            return mx.concatenate([prefix, rows[..., self.dims:]], axis=-1).transpose(2, 1, 0, 3)

    for layer in core.layers:
        inner = getattr(layer, '_layer', layer)
        attention = getattr(inner, 'self_attn', None)
        if attention is not None:
            attention.rope = MultimodalRoPE(attention.rope)
