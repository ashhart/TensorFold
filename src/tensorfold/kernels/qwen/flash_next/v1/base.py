"""What every Flash Next kernel module shares: the quantized-dot header, the kernel cache, small int inputs."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import mlx.core as mx

from tensorfold.kernels.inputs import ints, padded  # noqa: F401  (8+ elements, one source a kernel name)

MAX_ROWS = 16
# streams a per-stream kernel call takes: Metal binds at most 31 buffers and a stream brings two
MAX_STREAMS = 8

QDOT_HEADER = r"""
// sum_i w_i x_i over one 32-value group: w = scale * q + bias
inline float qgroup_dot(const device uint32_t* w, float scale, float bias, const thread float* x) {
  float dq = 0.0f, dx = 0.0f;
  for (int word = 0; word < 4; word++) {
    const uint32_t bits = w[word];
    for (int n = 0; n < 8; n++) {
      const float xv = x[word * 8 + n];
      dq = fma(float((bits >> (4 * n)) & 0xFu), xv, dq);
      dx += xv;
    }
  }
  return fma(scale, dq, bias * dx);
}
// Elementwise ops as the checkpoint's training framework does them on bf16 tensors: fp32 math, one rounding.
inline float bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
inline float bsilu(float x) { return float(bfloat(x / (1.0f + metal::exp(-x)))); }
inline float fsig(float x) { return 1.0f / (1.0f + metal::exp(-x)); }
inline float log1p_(float x) {
  const float u = 1.0f + x;
  return u == 1.0f ? x : x * (metal::log(u) / (u - 1.0f));
}
// softplus in fp32 (threshold 20, as torch.nn.functional.softplus)
inline float fsoftplus(float x) { return x > 20.0f ? x : log1p_(metal::exp(x)); }


// Rank-k expert of a row inside one simdgroup: lane l holds logits l, l + 32, ...; rounds of (largest logit,
// lowest id); returns the id picked in round k and, through ``picked``, the logits of rounds 0..k.
template <int NE>
inline int simd_topk(const device float* logits, int k, uint lane, thread float* picked) {
  float v[NE / 32];
  for (int j = 0; j < NE / 32; j++) v[j] = logits[j * 32 + int(lane)];
  int id = 0;
  for (int round = 0; round <= k; round++) {
    float best = -INFINITY;
    int bid = NE;
    for (int j = 0; j < NE / 32; j++) {
      const int e = j * 32 + int(lane);
      if (v[j] > best || (v[j] == best && e < bid)) { best = v[j]; bid = e; }
    }
    for (int off = 16; off > 0; off /= 2) {
      const float ob = simd_shuffle_xor(best, off);
      const int oi = simd_shuffle_xor(bid, off);
      if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; }
    }
    picked[round] = best;
    id = bid;
    if (int(lane) == bid % 32) v[bid / 32] = -INFINITY;
  }
  return id;
}
// simd_topk's rounds 0 .. TOPK-1 in one pass: ids[k] and logits picked[k] of each round
template <int NE, int TOPK>
inline void simd_topk_all(const device float* logits, uint lane, thread int* ids, thread float* picked) {
  float v[NE / 32];
  for (int j = 0; j < NE / 32; j++) v[j] = logits[j * 32 + int(lane)];
  for (int round = 0; round < TOPK; round++) {
    float best = -INFINITY;
    int bid = NE;
    for (int j = 0; j < NE / 32; j++) {
      const int e = j * 32 + int(lane);
      if (v[j] > best || (v[j] == best && e < bid)) { best = v[j]; bid = e; }
    }
    for (int off = 16; off > 0; off /= 2) {
      const float ob = simd_shuffle_xor(best, off);
      const int oi = simd_shuffle_xor(bid, off);
      if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; }
    }
    picked[round] = best;
    ids[round] = bid;
    if (int(lane) == bid % 32) v[bid / 32] = -INFINITY;
  }
}
// MLX's 4-bit qmv inner loop (quantized.h): 16 inputs a lane, pre-divided by 1, 16, 256, 4096 so the masked
// nibbles need no shift; w = scale * q + bias gives scale * dot(q, x) + bias * sum(x). As in MLX, each run of 4
// inputs is summed in bf16 (its x[i] + x[i + 1] + ... on bfloat16_t) before the fp32 sum: with that, a row's
// result is bit for bit MLX's one-row quantized matmul.
inline float load16(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
  }
  return sum;
}
// one lane's 16 inputs times its 8 bytes of one weight row (qdot16 with the weights already loaded)
inline float qdot16w(const thread uint16_t* ws, const thread float* xt, float scale, float bias, float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  return scale * accum + sum * bias;
}
inline float qdot16(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  return scale * accum + sum * bias;
}
"""

# Each row's matrix-unit tile arithmetic is independent of its neighboring rows.
MMA_HEADER = r"""
// the bf16 at index i (0..7) of 8 packed bf16, as fp32
inline float bfv(uint4 v, int i) {
  const uint w = v[i / 2];
  return as_type<float>((i % 2) ? (w & 0xFFFF0000u) : (w << 16));
}
// 2^-4e: a nibble left in place times an input scaled by this is the nibble's value times the input, exactly
inline float pre4(int e) { return as_type<float>(uint(127 - 4 * e) << 23); }
// a group's input sums of rows fn / fn + 1: the lane's 8 left to right, then the group's 4 words by shuffles
inline void group_sums(thread const float* xa, thread const float* xc, thread float& v, thread float& u) {
  v = xa[0]; u = xc[0];
  for (int i = 1; i < 8; i++) { v += xa[i]; u += xc[i]; }
  v += simd_shuffle_xor(v, ushort(4)); u += simd_shuffle_xor(u, ushort(4));
  v += simd_shuffle_xor(v, ushort(16)); u += simd_shuffle_xor(u, ushort(16));
}
// one group: P = sum q x in 4 steps (nibbles in place), then acc = fma(bias, sum x, fma(scale, P, acc))
inline void mma_sums(uint word, thread const float* xa, thread const float* xc, float v, float u, int fm, float sc,
                     float bi, thread float& acc0, thread float& acc1) {
  simdgroup_matrix<float, 8, 8> P = simdgroup_matrix<float, 8, 8>(0.0f);
  for (int st = 0; st < 4; st++) {
    const int e = 2 * st + fm % 2;
    simdgroup_matrix<float, 8, 8> am, bm;
    am.thread_elements()[0] = float(word & (0xFu << (8 * st)));
    am.thread_elements()[1] = float(word & (0xFu << (8 * st + 4)));
    bm.thread_elements()[0] = xa[e] * pre4(e);
    bm.thread_elements()[1] = xc[e] * pre4(e);
    simdgroup_multiply_accumulate(P, am, bm, P);
  }
  acc0 = fma(bi, v, fma(sc, P.thread_elements()[0], acc0));
  acc1 = fma(bi, u, fma(sc, P.thread_elements()[1], acc1));
}
inline void mma_group(uint word, thread const float* xa, thread const float* xc, int fm, float sc, float bi,
                      thread float& acc0, thread float& acc1) {
  float v, u;
  group_sums(xa, xc, v, u);
  mma_sums(word, xa, xc, v, u, fm, sc, bi, acc0, acc1);
}
// lane l's place in a tile: output fm, rows fn and fn + 1
inline int tile_fm(int l) { return ((l / 4) & 4) + ((l / 2) % 4); }
inline int tile_fn(int l) { return ((l / 4) & 2) * 2 + (l % 2) * 2; }
"""

_kernels: dict[str, Any] = {}
_counts: dict[int, mx.array] = {}
consts: dict[Any, mx.array] = {}


class _Kernel:
    """Cache kernels per integer template with constants embedded in source to avoid per-call template regex work."""

    def __init__(self, name: str, source: Any, inputs: list[str], outputs: list[str], header: str) -> None:
        self.name, self.source, self.inputs, self.outputs, self.header = name, source, inputs, outputs, header
        self.compiled: dict[tuple, Any] = {}

    def __call__(self, *, template: Any = (), **kwargs: Any) -> Any:
        key = tuple(template)
        run = self.compiled.get(key)
        if run is None:
            if callable(self.source):
                self.source = self.source()
            text = "".join(f"  constexpr int {k} = {int(v)};\n" for k, v in key) + self.source
            digest = hashlib.sha256((self.header + text).encode()).hexdigest()[:16]
            run = self.compiled[key] = mx.fast.metal_kernel(name=f"{self.name}_{digest}", input_names=self.inputs,
                                                            output_names=self.outputs, source=text,
                                                            header=self.header)
        return run(**kwargs)


def kernel(name: str, source: Any, inputs: list[str], outputs: list[str], header: str = QDOT_HEADER) -> _Kernel:
    """The kernel for ``name`` (one source per name; ``source`` may be a callable, built once)."""

    found = _kernels.get(name)
    if found is None:
        found = _kernels[name] = _Kernel(name, source, inputs, outputs, header)
    return found


def count(rows: int) -> mx.array:
    """A one-element int input: always under 8 elements, so always ``constant``."""

    value = _counts.get(rows)
    if value is None:
        value = _counts[rows] = mx.array([rows], dtype=mx.int32)
    return value


def log2(base: float) -> mx.array:
    value = consts.get(("log2", base))
    if value is None:
        value = consts[("log2", base)] = mx.array([math.log2(base)], dtype=mx.float32)
    return value


class QWeights:
    """A 4-bit group-32 quantized matrix [N, K] as three arrays (words, scales, biases)."""

    def __init__(self, weight: mx.array, scales: mx.array, biases: mx.array) -> None:
        self.weight, self.scales, self.biases = weight, scales, biases
        self.rows = int(weight.shape[0])
        self.cols = int(weight.shape[1]) * 8

    @classmethod
    def of(cls, *linears: Any) -> "QWeights":
        """One matrix from quantized linears' rows, stacked in order."""

        for linear in linears:
            if getattr(linear, "bits", 4) != 4 or getattr(linear, "group_size", 32) != 32:
                raise ValueError("expected 4-bit weights in groups of 32")
        if len(linears) == 1:
            one = linears[0]
            return cls(one.weight, one.scales, one.biases)
        return cls(mx.concatenate([l.weight for l in linears]), mx.concatenate([l.scales for l in linears]),
                   mx.concatenate([l.biases for l in linears]))


def pick(name: str, count: int, index: str) -> str:
    """``index == 0 ? NAME0 : index == 1 ? NAME1 : ... NAME{count-1}`` (buffers are bound one per stream)."""

    expr = f"{name}{count - 1}"
    for b in range(count - 2, -1, -1):
        expr = f"({index} == {b} ? {name}{b} : {expr})"
    return expr
