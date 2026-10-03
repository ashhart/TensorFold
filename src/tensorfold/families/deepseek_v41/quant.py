"""DeepSeek-V4.1's FP8/FP4 round trips and the checkpoint's linears, after oMLX's ``quantization.py`` (MIT)."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.deepseek.v41 import rows as KV

FP8_MAX = 448.0
FP4_MAX = 6.0


def _pow2(e: mx.array) -> mx.array:
    """2 ** e (int32 e in [-126, 127]) built from its bits: exact."""

    return ((e.astype(mx.int32) + 127).astype(mx.uint32) << 23).view(mx.float32)


def _floor_log2(a: mx.array) -> mx.array:
    """floor(log2(a)) of positive normal fp32 values, from the exponent bits."""

    return ((a.view(mx.uint32) >> 23) & 255).astype(mx.int32) - 127


def _ceil_log2(a: mx.array) -> mx.array:
    bits = a.view(mx.uint32)
    return _floor_log2(a) + ((bits & 0x7FFFFF) != 0).astype(mx.int32)


def round_fp8(x: mx.array) -> mx.array:
    """fp32 x to its E4M3FN value (round to nearest even, saturating at 448), in fp32."""

    a = mx.minimum(mx.abs(x.astype(mx.float32)), FP8_MAX)
    step = _pow2(mx.maximum(_floor_log2(mx.maximum(a, 2.0 ** -9)) - 3, -9))
    q = mx.minimum(mx.round(a / step) * step, FP8_MAX)
    return mx.where(x < 0, -q, q)


def round_fp4(a: mx.array) -> mx.array:
    """|scaled| values to E2M1 magnitudes: midpoint ties go to the code with an even low bit."""

    q = mx.zeros_like(a)
    for threshold, value, inclusive in ((0.25, 0.5, False), (0.75, 1.0, True), (1.25, 1.5, False),
                                        (1.75, 2.0, True), (2.5, 3.0, False), (3.5, 4.0, True), (5.0, 6.0, False)):
        q = mx.where(a >= threshold if inclusive else a > threshold, value, q)
    return q


def quantize_activation(x: mx.array, bits: int = 8, group: int = 32, e4m3_scale: bool = False) -> mx.array:
    """x rounded through FP8 or FP4 and back a ``group`` at a time (UE8M0 scales, or E4M3 with ``e4m3_scale``)."""

    dtype, shape = x.dtype, x.shape
    if shape[-1] % group:
        raise ValueError(f"activation width {shape[-1]} does not divide by {group}")
    g = x.astype(mx.float32).reshape(*shape[:-1], shape[-1] // group, group)
    limit = FP8_MAX if bits == 8 else FP4_MAX
    floor = limit * (2.0 ** -9 if e4m3_scale else 2.0 ** -126)
    amax = mx.maximum(mx.max(mx.abs(g), axis=-1, keepdims=True), floor)
    if e4m3_scale:
        scale = round_fp8(amax / limit)
    else:
        scale = _pow2(mx.maximum(_ceil_log2(amax / limit), -126))
    scaled = mx.clip(g / scale, -limit, limit)
    if bits == 8:
        q = round_fp8(scaled)
    elif bits == 4:
        q = round_fp4(mx.abs(scaled))
        q = mx.where(scaled < 0, -q, q)
    else:
        raise ValueError("only FP8 and FP4 round trips are defined")
    return (q * scale).reshape(shape).astype(dtype)


def fp8(x: mx.array) -> mx.array:
    """The official FP8 activation of a quantized projection's input (E4M3, a UE8M0 scale per 32)."""

    if KV.fp8_fits(x):
        return KV.fp8(x)
    return quantize_activation(x, 8, 32)


def swiglu_fp8(gate: mx.array, up: mx.array, weights: mx.array | None, limit: float, dtype: Any) -> mx.array:
    """oMLX's fused SwiGLU tail: clamp, g * sigmoid(g) * u in fp32 (times the route weight), to ``dtype``, then FP8."""

    if KV.fp8_fits(gate) and gate.shape == up.shape:
        return KV.swiglu_fp8(gate, up, weights, limit, dtype)
    g, u = gate.astype(mx.float32), up.astype(mx.float32)
    if limit:
        g = mx.minimum(g, limit)
        u = mx.clip(u, -limit, limit)
    neg = 1.0 / (1.0 + KV.fexp(mx.abs(g)))
    sig = mx.where(g < 0, neg, 1.0 - neg)
    value = (g * sig) * u
    if weights is not None:
        value = value * weights
    return fp8(value.astype(dtype))


class Linear:
    """A linear as the checkpoint stores it (mxfp8, mxfp4, affine or bf16), FP8-rounding its input if quantized."""

    def __init__(self, weight: mx.array, scales: mx.array | None = None, biases: mx.array | None = None, *,
                 bits: int | None = None, group: int = 32, mode: str = "affine", fp8_input: bool = True) -> None:
        self.weight, self.scales, self.biases = weight, scales, biases
        self.bits, self.group, self.mode = bits, int(group), str(mode)
        self.fp8_input = bool(fp8_input) and scales is not None
        if scales is not None:
            if bits is None:
                raise ValueError("a quantized linear needs its bits")
            if mode == "affine" and biases is None:
                raise ValueError("an affine linear needs its biases")
            if mode != "affine" and biases is not None:
                raise ValueError(f"{mode} linears have no biases")
            if int(weight.shape[-1]) * 32 != int(scales.shape[-1]) * self.group * int(bits):
                raise ValueError(f"{bits}-bit {mode} weights {tuple(weight.shape)} do not fit scales "
                                 f"{tuple(scales.shape)} in groups of {group}")

    @property
    def quantized(self) -> bool:
        return self.scales is not None

    @property
    def outs(self) -> int:
        return int(self.weight.shape[-2])

    @property
    def ins(self) -> int:
        if not self.quantized:
            return int(self.weight.shape[-1])
        return int(self.weight.shape[-1]) * 32 // int(self.bits)

    def arrays(self) -> list[mx.array]:
        return [a for a in (self.weight, self.scales, self.biases) if a is not None]

    def __call__(self, x: mx.array, *, prequantized: bool = False) -> mx.array:
        if not self.quantized:
            return KV.matmul(x, self.weight, x.dtype)
        if self.fp8_input and not prequantized:
            x = fp8(x)
        if KV.rows_mode() and int(x.shape[0]) > 1:
            if self.mode in ("mxfp8", "mxfp4") and self.group == 32 and \
                    KV.fpqmv_rows_fits(x, int(self.bits), self.ins, self.outs):
                return KV.fpqmv_rows(x, self.weight, self.scales, int(self.bits), self.ins, self.outs)
            return mx.concatenate([self._qmm(x[r:r + 1]) for r in range(int(x.shape[0]))])
        return self._qmm(x)

    def _qmm(self, x: mx.array) -> mx.array:
        return mx.quantized_matmul(x, self.weight, self.scales, self.biases, transpose=True, group_size=self.group,
                                   bits=self.bits, mode=self.mode)

    def dequantized(self) -> mx.array:
        if not self.quantized:
            return self.weight
        return mx.dequantize(self.weight, self.scales, self.biases, group_size=self.group, bits=self.bits,
                             mode=self.mode)


class Experts:
    """A stack of routed experts' matrices ``[E, N, K]`` in MLX's quantized layout, read by ``gather_qmm``."""

    def __init__(self, weight: mx.array, scales: mx.array, biases: mx.array | None, *, bits: int, group: int,
                 mode: str) -> None:
        if int(weight.shape[-1]) * 32 != int(scales.shape[-1]) * group * bits or weight.shape[:-1] != scales.shape[:-1]:
            raise ValueError(f"{mode} expert codes {tuple(weight.shape)} do not fit scales {tuple(scales.shape)}")
        self.weight, self.scales, self.biases = weight, scales, biases
        self.bits, self.group, self.mode = int(bits), int(group), str(mode)

    @property
    def count(self) -> int:
        return int(self.weight.shape[0])

    def arrays(self) -> list[mx.array]:
        return [a for a in (self.weight, self.scales, self.biases) if a is not None]

    def __call__(self, x: mx.array, ids: mx.array, sort: bool) -> mx.array:
        return mx.gather_qmm(x, self.weight, self.scales, self.biases, rhs_indices=ids, transpose=True,
                             group_size=self.group, bits=self.bits, mode=self.mode, sorted_indices=sort)


def per_row(fn: Any, x: mx.array, rows_exact: bool) -> mx.array:
    """fn over x's rows: one call (prompt rows), or each row alone (decode rows: a row's own one-row bits)."""

    rows = int(x.shape[0])
    if rows == 1 or not rows_exact:
        return fn(x)
    return mx.concatenate([fn(x[r:r + 1]) for r in range(rows)])
