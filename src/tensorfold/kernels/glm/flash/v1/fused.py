"""The fused blocks' Metal header (precise exp, no fma in sums of squares: mlx-vlm #2105) and kernel cache."""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx

from tensorfold.kernels.glm.flash.v1 import kernels as K

MAX_ROWS = 16
SQ_FMA = 0

_HEADER = K._HEADER + r"""
template <typename U>
inline U sigmoid_precise(U x) {
  U e = static_cast<U>(metal::precise::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}
// nn.silu is an mx.compile'd x * sigmoid(x) on bf16: MLX's Sigmoid in bfloat arithmetic with the JIT's (fast) exp
template <typename U>
inline U sigmoid_fast(U x) {
  U e = static_cast<U>(metal::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}
#pragma clang fp contract(off)
// MLX's rms_norm accumulates acc += x * x in its prebuilt library: FMA 1 if that contracts to an fma there
template <int FMA>
inline float sq_acc(float acc, float v) { return FMA ? fma(v, v, acc) : v * v + acc; }
inline float mul_add(float acc, float a, float b) { return a * b + acc; }
inline float add_nc(float a, float b) { return a + b; }
#pragma clang fp contract(on)
"""

_kernels: dict[str, Any] = {}


def metal() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def _kernel(name: str, source: str, inputs: list[str], outputs: list[str]) -> Any:
    kernel = _kernels.get(name)
    if kernel is None:
        digest = hashlib.sha256((_HEADER + source).encode()).hexdigest()[:10]
        kernel = mx.fast.metal_kernel(name=f"tf_glm5_fused_{name}_{digest}", input_names=inputs,
                                      output_names=outputs, source=source, header=_HEADER)
        _kernels[name] = kernel
    return kernel


