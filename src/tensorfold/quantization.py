"""Checkpoint affine formats resolved with MLX's per-module override semantics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

AFFINE_BITS = (2, 3, 4, 5, 6, 8)
AFFINE_GROUPS = (32, 64, 128)


@dataclass(frozen=True)
class AffineSpec:
    bits: int
    group_size: int = 64
    mode: str = 'affine'

    def __post_init__(self):
        if type(self.bits) is not int or self.bits not in AFFINE_BITS:
            raise ValueError('affine weights require 2, 3, 4, 5, 6 or 8 bits')
        if type(self.group_size) is not int or self.group_size not in AFFINE_GROUPS:
            raise ValueError('affine group size must be 32, 64 or 128')
        if self.mode != 'affine':
            raise ValueError('these kernels support affine quantization only')


def quantization_block(config: dict) -> dict | None:
    for source in (config, config.get('text_config') or {}):
        for key in ('quantization', 'quantization_config'):
            value = source.get(key)
            if isinstance(value, dict) and value:
                return value
    return None


def canonical_path(path: str) -> str:
    path = path[:-7] if path.endswith('.weight') else path
    for prefix in ('model.language_model.', 'language_model.', 'text_model.', 'model.'):
        if path.startswith(prefix):
            path = path[len(prefix):]
    return path


def _spec(value: dict, *, default_bits: int | None = None) -> AffineSpec:
    bits = value.get('bits', default_bits)
    if bits is None:
        raise ValueError('affine quantization metadata must declare bits')
    return AffineSpec(bits, value.get('group_size', 64), value.get('mode') or 'affine')


def resolve_affine(config: dict, path: str | None = None) -> AffineSpec | None:
    block = quantization_block(config)
    if block is None:
        return None
    method = block.get('quant_method')
    if method not in (None, 'mlx', 'affine'):
        raise ValueError(f'{method} is not an MLX affine checkpoint')
    global_spec = _spec(block)
    if path is None:
        return global_spec
    wanted = canonical_path(path)
    matches = [value for key, value in block.items() if canonical_path(str(key)) == wanted]
    if not matches:
        return global_spec
    specs = []
    for value in matches:
        if value is False or value == {}:
            specs.append(None)
        elif value is True:
            specs.append(global_spec)
        elif isinstance(value, dict):
            specs.append(_spec(value, default_bits=4))
        else:
            raise ValueError(f'invalid per-module quantization metadata: {path}')
    if any(spec != specs[0] for spec in specs[1:]):
        raise ValueError(f'conflicting quantization aliases for {path}')
    return specs[0]


def validate_shapes(weight_shape: Sequence[int], scale_shape: Sequence[int], bias_shape: Sequence[int],
                    spec: AffineSpec) -> tuple[int, int]:
    shapes = [tuple(shape) for shape in (weight_shape, scale_shape, bias_shape)]
    if any(len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape) for shape in shapes):
        raise ValueError('packed affine matrices and metadata must have positive two-dimensional shapes')
    weights, scales, biases = shapes
    n, k = scales[0], scales[1] * spec.group_size
    if scales != biases or weights != (n, k * spec.bits // 32) or k * spec.bits % 32:
        raise ValueError('packed weight, scales and biases disagree with the declared affine format')
    return n, k


def checkpoint_specs(config: dict) -> dict[str, AffineSpec | None]:
    block = quantization_block(config)
    if block is None:
        return {}
    result: dict[str, AffineSpec | None] = {'': resolve_affine(config)}
    for path, value in block.items():
        if isinstance(value, dict) or type(value) is bool:
            result[str(path)] = resolve_affine(config, str(path))
    return result
