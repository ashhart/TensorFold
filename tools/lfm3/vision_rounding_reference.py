"""Independent FP32 replay of named vision operators; ideal FP64 remains separate."""
import numpy as np
from fp64_reference import bf
from text_rounding_reference import sum32
from elementary_reference import elementary, fma


def total256(partial):
    groups = sum32(partial.reshape(*partial.shape[:-1], 8, 32))
    padded = np.zeros((*groups.shape[:-1], 32), dtype=np.float32)
    padded[..., :8] = groups
    return sum32(padded)


def rms_rows(values, gain, epsilon, elementary_backend='numpy'):
    values = np.asarray(values, dtype=np.float32)
    width = values.shape[-1]
    if width == 1024:
        partial = values.reshape(*values.shape[:-1], 4, 256)
        square = np.zeros(partial.shape[:-2] + (256,), dtype=np.float32)
        for at in range(4):
            square = fma(partial[..., at, :], partial[..., at, :], square)
        total = total256(square)
    elif width == 64:
        a, b = values[..., :32], values[..., 32:]
        total = sum32(fma(a, a, b * b))
    else:
        raise ValueError('Expected 1024 vision channels or a 64-channel head')
    variance = total / np.float32(width) + np.float32(epsilon)
    inverse = elementary(variance, 'rsqrt', elementary_backend)[..., None]
    return bf(bf(values * inverse).astype(np.float32) * np.asarray(gain, dtype=np.float32))


def convolution(grid, kernel):
    grid, kernel = np.asarray(grid, dtype=np.float32), np.asarray(kernel, dtype=np.float32)
    height, width, _ = grid.shape
    padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)))
    result = np.zeros_like(grid)
    for row in range(3):
        for column in range(3):
            result = fma(padded[row:row + height, column:column + width], kernel[:, row, column], result)
    return bf(result)



def rotary(values, coordinates, elementary_backend='numpy'):
    values = np.asarray(values, dtype=np.float32)
    lane = np.arange(32)
    frequency = elementary(-np.asarray(lane % 16, dtype=np.float32) / np.float32(16)
                        * np.float32(19.931568569324174), 'exp2', elementary_backend)
    coordinate = np.where(lane < 16, coordinates[:, 0, None], coordinates[:, 1, None]).astype(np.float32)
    angle = coordinate * frequency
    cosine = elementary(angle, 'cos', elementary_backend)[:, None, :]
    sine = elementary(angle, 'sin', elementary_backend)[:, None, :]
    first, second = values[..., :32], values[..., 32:]
    return bf(np.concatenate((fma(first, cosine, -(second * sine)),
                              fma(second, cosine, first * sine)), axis=-1))


def attention(q, k, v, elementary_backend='numpy'):
    """Replay noncausal FP32 attention; elementary function behavior is explicitly selected."""
    q, k, v = (np.asarray(value, dtype=np.float32) for value in (q, k, v))
    heads, rows, width = q.shape
    if width != 64 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError('Expected equally expanded 64-channel Q/K/V')
    output = np.empty_like(q)
    for first in range(0, rows, 16):
        query = q[:, first:first + 16]
        products = query[:, :, None, :32] * k[:, None, :, :32]
        products = fma(query[:, :, None, 32:], k[:, None, :, 32:], products)
        scores = sum32(products) * np.float32(0.125)
        maxima = np.maximum.accumulate(scores, axis=-1)
        previous = np.concatenate((np.full((*scores.shape[:2], 1), -np.inf, dtype=np.float32),
                                   maxima[..., :-1]), axis=-1)
        alphas = elementary(previous - maxima, 'exp', elementary_backend)
        probabilities = elementary(scores - maxima, 'exp', elementary_backend)
        maximum = np.full(scores.shape[:2], -np.inf, dtype=np.float32)
        total = np.zeros_like(maximum)
        result = np.zeros((*maximum.shape, 64), dtype=np.float32)
        for key in range(rows):
            score = scores[..., key]
            next_maximum = np.maximum(maximum, score)
            alpha = alphas[..., key]
            probability = probabilities[..., key]
            total = fma(total, alpha, probability)
            result = fma(result, alpha[..., None], probability[..., None] * v[:, key, None, :])
            maximum = next_maximum
        output[:, first:first + 16] = result / total[..., None]
    return bf(output)


def layer_norm(values, gain, bias, epsilon, elementary_backend='numpy'):
    values = np.asarray(values, dtype=np.float32)
    partial = values.reshape(len(values), 4, 256)
    summed = np.zeros((len(values), 256), dtype=np.float32)
    for at in range(4):
        summed = summed + partial[:, at]
    mean = total256(summed) / np.float32(1024)
    centered = partial - mean[:, None, None]
    square = np.zeros_like(summed)
    for at in range(4):
        square = fma(centered[:, at], centered[:, at], square)
    variance = total256(square) / np.float32(1024) + np.float32(epsilon)
    inverse = elementary(variance, 'rsqrt', elementary_backend)
    scaled = centered.reshape(values.shape) * inverse[:, None]
    return bf(fma(scaled, np.asarray(gain, np.float32), np.asarray(bias, np.float32)))


def gelu(values, elementary_backend='numpy'):
    values = np.asarray(values, dtype=np.float32)
    a = np.abs(values * np.float32(0.70710678118654752))
    t = np.float32(1) / fma(np.float32(0.3275911), a, np.float32(1))
    polynomial = fma(t, np.float32(1.061405429), np.float32(-1.453152027))
    for coefficient in (1.421413741, -0.284496736, 0.254829592):
        polynomial = fma(t, polynomial, np.float32(coefficient))
    polynomial = t * polynomial
    erf = fma(-polynomial, elementary(-a * a, 'exp', elementary_backend), np.float32(1))
    erf = np.where(values < 0, -erf, erf)
    return bf((np.float32(0.5) * values) * (np.float32(1) + erf))
