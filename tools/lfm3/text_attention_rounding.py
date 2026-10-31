"""Independent bounded FP32 replay of decoder rotary and one-partition attention."""
import math
import numpy as np
from fp64_reference import bf
from elementary_reference import elementary, fma
from text_rounding_reference import sum32


def rotary(values, spec, backend):
    values = np.asarray(values, dtype=np.float32)
    pairs = int(values.shape[-1] * spec.get('partial_rotary_factor', 1)) // 2
    frequency = np.arange(64, dtype=np.float32)
    exponent = -frequency / np.float32(64) * np.float32(math.log2(spec['rope_theta']))
    frequency = elementary(exponent, 'exp2', backend)
    frequency[pairs:] = 0
    angle = np.arange(len(values), dtype=np.float32)[:, None] * frequency
    cosine = bf(elementary(angle, 'cos', backend)).astype(np.float32)[:, None]
    sine = bf(elementary(angle, 'sin', backend)).astype(np.float32)[:, None]
    a, b = values[..., :64], values[..., 64:]
    return bf(np.concatenate((fma(a, cosine, -(b * sine)), fma(b, cosine, a * sine)), axis=-1))


def attention(q, k, v, window, backend):
    q, k, v = (np.asarray(value, dtype=np.float32) for value in (q, k, v))
    heads, rows, width = q.shape
    if width != 128 or rows > 1024 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError('Named text attention replay requires at most 1024 equally expanded 128-channel rows')
    output = np.empty_like(q)
    for first in range(0, rows, 16):
        query = q[:, first:first + 16]
        products = np.zeros((heads, len(query[0]), rows, 32), dtype=np.float32)
        for part in range(4):
            products = fma(query[:, :, None, part * 32:(part + 1) * 32],
                           k[:, None, :, part * 32:(part + 1) * 32], products)
        scores = sum32(products) * np.float32(1 / math.sqrt(128))
        distance = np.arange(first, first + len(query[0]))[:, None] - np.arange(rows)[None]
        allowed = distance >= 0
        if window:
            allowed &= distance < window
        scores = np.where(allowed[None], scores, np.float32(-np.inf))
        maxima = np.maximum.accumulate(scores, axis=-1)
        previous = np.concatenate((np.full((*scores.shape[:2], 1), -np.inf, np.float32), maxima[..., :-1]), axis=-1)
        with np.errstate(invalid='ignore'):
            alpha = elementary(previous - maxima, 'exp', backend)
            probability = elementary(scores - maxima, 'exp', backend)
        alpha = np.where(np.isneginf(previous), np.float32(0), alpha)
        probability = np.where(np.isneginf(scores), np.float32(0), probability)
        total = np.zeros(scores.shape[:2], dtype=np.float32)
        result = np.zeros((*total.shape, 128), dtype=np.float32)
        for key in range(rows):
            active = allowed[None, :, key]
            next_total = fma(total, alpha[..., key], probability[..., key])
            next_result = fma(result, alpha[..., key, None], probability[..., key, None] * v[:, key, None])
            total = np.where(active, next_total, total)
            result = np.where(active[..., None], next_result, result)
        output[:, first:first + len(query[0])] = result / total[..., None]
    return bf(output)
