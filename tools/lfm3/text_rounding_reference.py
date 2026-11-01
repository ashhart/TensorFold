"""Independent replay of the text RMS normalization's prescribed FP32 partial and lane reductions."""
import numpy as np
from fp64_reference import bf


def sum32(values):
    result = np.asarray(values, dtype=np.float32)
    if result.shape[-1] != 32:
        raise ValueError('Expected exactly 32 reduction lanes')
    for distance in (1, 2, 4, 8, 16):
        result = result + result[..., np.arange(32) ^ distance]
    return result[..., 0]


def norm_rows(values, gain, epsilon, elementary_backend=None):
    original, gain = np.asarray(values), np.asarray(gain)
    if original.ndim == 0 or not np.isfinite(original).all() or not np.array_equal(original, bf(original)):
        raise ValueError('Expected finite BF16 normalization inputs')
    values = np.asarray(original, dtype=np.float32)
    width = values.shape[-1]
    if gain.shape != (width,) or not np.isfinite(gain).all() or not np.array_equal(gain, bf(gain)):
        raise ValueError('Expected one finite BF16 normalization gain per column')
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Expected positive finite normalization epsilon')
    if width == 2048:
        partial = values.reshape(*values.shape[:-1], 256, 8)
        square = np.zeros(partial.shape[:-1], dtype=np.float32)
        for at in range(8):
            square = (partial[..., at].astype(np.float64) ** 2 + square).astype(np.float32)
        groups = sum32(square.reshape(*values.shape[:-1], 8, 32))
        padded = np.zeros((*values.shape[:-1], 32), dtype=np.float32)
        padded[..., :8] = groups
        total = sum32(padded)
    elif width == 128:
        partial = values.reshape(*values.shape[:-1], 4, 32)
        square = np.zeros((*values.shape[:-1], 32), dtype=np.float32)
        for at in range(4):
            square = (partial[..., at, :].astype(np.float64) ** 2 + square).astype(np.float32)
        total = sum32(square)
    else:
        raise ValueError('Expected a text hidden row or attention head')
    variance = total / np.float32(width) + np.float32(epsilon)
    inverse = (1 / np.sqrt(variance.astype(np.float64))).astype(np.float32)[..., None]
    if elementary_backend is not None:
        from elementary_reference import elementary
        inverse = elementary(variance, 'rsqrt', elementary_backend)[..., None]
    first = bf(values * inverse).astype(np.float32)
    result = first * np.asarray(gain, dtype=np.float32)
    if elementary_backend == 'metal':
        # Metal's FP32 arithmetic flushes subnormal products, before the BF16 store.
        result = np.where(np.abs(result) < np.finfo(np.float32).tiny, np.copysign(np.float32(0), result), result)
    return bf(result)
