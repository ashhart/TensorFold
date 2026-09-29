"""What every Flash Next kernel module shares: the quantized-dot header, the kernel cache, small int inputs."""

from __future__ import annotations

import functools
import hashlib
import math
from typing import Any

import mlx.core as mx

from tensorfold.kernels import threads
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
// float(v) for v < 2^23 by the exponent trick (an or and a subtract): the same value as a convert
inline float nib(uint v) { return as_type<float>(0x4B000000u | v) - 8388608.0f; }
// qdot16 and qgroup_dot with nib: the same bits, cheaper than the convert from two rows before M5, dearer at one
inline float qdot16x(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * nib(ws[i] & 0x000fu) + xt[4 * i + 1] * nib(ws[i] & 0x00f0u) +
             xt[4 * i + 2] * nib(ws[i] & 0x0f00u) + xt[4 * i + 3] * nib(ws[i] & 0xf000u);
  return scale * accum + sum * bias;
}
inline float qgroup_dotx(const device uint32_t* w, float scale, float bias, const thread float* x) {
  float dq = 0.0f, dx = 0.0f;
  for (int word = 0; word < 4; word++) {
    const uint32_t bits = w[word];
    for (int n = 0; n < 8; n++) {
      const float xv = x[word * 8 + n];
      dq = fma(nib((bits >> (4 * n)) & 0xFu), xv, dq);
      dx += xv;
    }
  }
  return fma(scale, dq, bias * dx);
}
// two 16-bit words' nibbles q (q < 1024 each half) as half with no convert: 1024 + q by the exponent trick, minus 1024
inline half2 nib2(uint q) { return as_type<half2>(0x64006400u | q) - half2(1024.0h); }
// load16 with the third and fourth inputs divided by 16 and 1: with qdot16h's nibbles, the same products as qdot16's
inline float load16h(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 16.0f; xt[i + 3] = float(d);
  }
  return sum;
}
// qdot16 over load16h's inputs with half nibbles (q, 16 q, 16 q, q): the same products in the same order
inline float qdot16h(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint32_t* ws = (const device uint32_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 2; i++) {
    const uint u = ws[i];
    const half2 a = nib2(u & 0x000F000Fu), b = nib2(u & 0x00F000F0u), c = nib2((u >> 4) & 0x00F000F0u),
                d = nib2((u >> 12) & 0x000F000Fu);
    accum += xt[8 * i] * float(a.x) + xt[8 * i + 1] * float(b.x) + xt[8 * i + 2] * float(c.x) +
             xt[8 * i + 3] * float(d.x);
    accum += xt[8 * i + 4] * float(a.y) + xt[8 * i + 5] * float(b.y) + xt[8 * i + 6] * float(c.y) +
             xt[8 * i + 7] * float(d.y);
  }
  return scale * accum + sum * bias;
}
// qgroup_dot with half nibbles: the same fmas in the same order
inline float qgroup_doth(const device uint32_t* w, float scale, float bias, const thread float* x) {
  float dq = 0.0f, dx = 0.0f;
  for (int word = 0; word < 4; word++) {
    const uint32_t bits = w[word];
    const half2 q04 = nib2(bits & 0x000F000Fu), q15 = nib2((bits >> 4) & 0x000F000Fu),
                q26 = nib2((bits >> 8) & 0x000F000Fu), q37 = nib2((bits >> 12) & 0x000F000Fu);
    const float q[8] = {float(q04.x), float(q15.x), float(q26.x), float(q37.x),
                        float(q04.y), float(q15.y), float(q26.y), float(q37.y)};
    for (int n = 0; n < 8; n++) {
      const float xv = x[word * 8 + n];
      dq = fma(q[n], xv, dq);
      dx += xv;
    }
  }
  return fma(scale, dq, bias * dx);
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

# Codes of any MLX affine width: a row is one little-endian bit stream and 32 values fill BITS words exactly.
AFFINE_HEADER = r"""
// values 8m .. 8m + 7 of the 32-value chunk at ``chunk`` (a chunk starts a word, and 8 values span at most 2 words)
template <int BITS>
inline void codes8(const device uint* chunk, int m, thread float* q) {
  const int bit = 8 * m * BITS, word = bit >> 5, shift = bit & 31;
  const uint hi = shift + 8 * BITS > 32 ? chunk[word + 1] : 0u;
  const ulong v = ((ulong(hi) << 32) | ulong(chunk[word])) >> shift;
  for (int i = 0; i < 8; i++) q[i] = float(uint(v >> (BITS * i)) & ((1u << BITS) - 1u));
}
// value i of a row's bit stream starting at ``row`` (for lookups: one value a thread)
template <int BITS>
inline uint code_at(const device uint* row, int i) {
  const int bit = i * BITS, word = bit >> 5, shift = bit & 31;
  uint v = row[word] >> shift;
  if (shift + BITS > 32) v |= row[word + 1] << (32 - shift);
  return v & ((1u << BITS) - 1u);
}
// qgroup_dot for any width: one 32-value chunk, codes in order, then scale and bias
template <int BITS>
inline float qchunk_dot(const device uint* chunk, float scale, float bias, const thread float* x) {
  float dq = 0.0f, dx = 0.0f;
  for (int m = 0; m < 4; m++) {
    float q[8];
    codes8<BITS>(chunk, m, q);
    for (int n = 0; n < 8; n++) {
      dq = fma(q[n], x[8 * m + n], dq);
      dx += x[8 * m + n];
    }
  }
  return fma(scale, dq, bias * dx);
}
"""

# A lane's step of MLX's qmv_fast loop for any width: VPT codes (16, or 8 at 6 and 8 bits) from one row.
LANE_CODES = r"""
constexpr int lane_values(int bits) { return bits == 6 || bits == 8 ? 8 : 16; }
// the VPT codes from value v0 of a row (v0 a multiple of VPT, so they start on a 2-byte boundary)
template <int BITS, int VPT>
inline void lane_codes(const device uint8_t* row, int v0, thread float* q) {
  constexpr int NH = VPT * BITS / 16;
  const device ushort* h = (const device ushort*)(row + v0 * BITS / 8);
  ushort u[NH + 1];
  for (int j = 0; j < NH; j++) u[j] = h[j];
  u[NH] = 0;
  for (int i = 0; i < VPT; i++) {
    const int bit = BITS * i, j = bit / 16, s = bit % 16;
    uint v = uint(u[j]) >> s;
    if (s + BITS > 16) v |= uint(u[j + 1]) << (16 - s);
    q[i] = float(v & ((1u << BITS) - 1u));
  }
}
"""

# The matrix-unit and scalar hyper-connection steps for any width (after MMA_HEADER and AFFINE_HEADER).
AFFINE_MMA_HEADER = r"""
// mma_sums with the lane's 8 codes as values (no nibbles in place, so no input scaling)
inline void mma_sums_q(thread const float* q, thread const float* xa, thread const float* xc, float v, float u, int fm,
                       float sc, float bi, thread float& acc0, thread float& acc1) {
  simdgroup_matrix<float, 8, 8> P = simdgroup_matrix<float, 8, 8>(0.0f);
  for (int st = 0; st < 4; st++) {
    const int e = 2 * st + fm % 2;
    simdgroup_matrix<float, 8, 8> am, bm;
    am.thread_elements()[0] = q[2 * st];
    am.thread_elements()[1] = q[2 * st + 1];
    bm.thread_elements()[0] = xa[e];
    bm.thread_elements()[1] = xc[e];
    simdgroup_multiply_accumulate(P, am, bm, P);
  }
  acc0 = fma(bi, v, fma(sc, P.thread_elements()[0], acc0));
  acc1 = fma(bi, u, fma(sc, P.thread_elements()[1], acc1));
}
// mma_group for any width: the chunk's group sums, then the lane's 8 codes (values 8 (fn / 2) ..) through the MMA
template <int BITS>
inline void mma_chunk(const device uint* chunk, thread const float* xa, thread const float* xc, int fm, int fn,
                      float sc, float bi, thread float& acc0, thread float& acc1) {
  float v, u, q[8];
  group_sums(xa, xc, v, u);
  codes8<BITS>(chunk, fn / 2, q);
  mma_sums_q(q, xa, xc, v, u, fm, sc, bi, acc0, acc1);
}
// scalar_group for any width: the MMA path's product sum of one chunk, in its order
template <int BITS>
inline float scalar_chunk(const device uint* chunk, const threadgroup float* x) {
  float q[32];
  for (int m = 0; m < 4; m++) codes8<BITS>(chunk, m, q + 8 * m);
  float p = 0.0f;
  for (int st = 0; st < 4; st++)
    for (int k = 0; k < 8; k++) {
      const int i = 8 * (k / 2) + 2 * st + k % 2;
      p = fma(q[i], x[i], p);
    }
  return p;
}
"""

_kernels: dict[str, Any] = {}
_counts: dict[int, mx.array] = {}
consts: dict[Any, mx.array] = {}


class _Kernel:
    """Cache kernels per integer template with constants embedded in source to avoid per-call template regex work."""

    def __init__(self, name: str, source: Any, inputs: list[str], outputs: list[str], header: str,
                 reserve: int) -> None:
        self.name, self.source, self.inputs, self.outputs, self.header = name, source, inputs, outputs, header
        self.reserve = reserve
        self.compiled: dict[tuple, Any] = {}

    def __call__(self, *, template: Any = (), **kwargs: Any) -> Any:
        tg = kwargs["threadgroup"]
        size = self.reserve or int(tg[0]) * int(tg[1]) * int(tg[2])
        size = size if size > threads.SAFE else 0             # a pipeline past SAFE reserves its threads everywhere
        key = (tuple(template), size)
        run = self.compiled.get(key)
        if run is None:
            if callable(self.source):
                self.source = self.source()
            text = "".join(f"  constexpr int {k} = {int(v)};\n" for k, v in key[0]) + self.source
            header = self.header + (threads.reserve(size) if size else "")
            digest = hashlib.sha256((header + text).encode()).hexdigest()[:16]
            run = self.compiled[key] = mx.fast.metal_kernel(name=f"{self.name}_{digest}", input_names=self.inputs,
                                                            output_names=self.outputs, source=text, header=header)
        return run(**kwargs)


@functools.cache
def _generation() -> int:
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    digits = "".join(ch for ch in str(info.get("architecture", "")).removeprefix("applegpu_g") if ch.isdigit())
    return int(digits) if digits else 0


def nib_rows() -> int:
    """Rows from which the 4-bit dots take nib: 2 before M5, 0 (never) on GPUs with tensor units."""

    return 0 if _generation() >= 17 else 2


def half_nibs() -> bool:
    """Whether those dots make nibbles in half (M3, M4: fp16 issues beside fp32) instead of by nib (M1, M2)."""

    return _generation() in (15, 16)


def by_rows(name: str, source: str, rows: int) -> tuple[str, str]:
    """A kernel's name and source for ``rows`` rows: from nib_rows() rows its dots skip the convert (same bits)."""

    if not nib_rows() or rows < nib_rows():
        return name, source
    if half_nibs():
        return name + "_h", (source.replace("load16(", "load16h(").replace("qdot16(", "qdot16h(")
                             .replace("qgroup_dot(", "qgroup_doth("))
    return name + "_x", source.replace("qdot16(", "qdot16x(").replace("qgroup_dot(", "qgroup_dotx(")


def kernel(name: str, source: Any, inputs: list[str], outputs: list[str], header: str = QDOT_HEADER, *,
           reserve: int = 0) -> _Kernel:
    """The kernel for ``name`` (one source per name, may be a callable); ``reserve`` fixes its largest threadgroup."""

    found = _kernels.get(name)
    if found is None:
        found = _kernels[name] = _Kernel(name, source, inputs, outputs, header, reserve)
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
    """An MLX affine matrix [N, K] (words, scales, biases) of ``bits`` in groups of ``group``."""

    def __init__(self, weight: mx.array, scales: mx.array, biases: mx.array, bits: int = 4, group: int = 32) -> None:
        self.weight, self.scales, self.biases = weight, scales, biases
        self.bits, self.group = int(bits), int(group)
        self.rows = int(weight.shape[0])
        self.cols = int(weight.shape[1]) * 32 // self.bits

    @property
    def q4(self) -> bool:
        """The 4-bit group-32 format the original kernels read."""

        return (self.bits, self.group) == (4, 32)

    @classmethod
    def of(cls, *linears: Any) -> "QWeights":
        """One matrix from quantized linears' rows, stacked in order, widened exactly to one format if they differ."""

        parts = [cls(l.weight, l.scales, l.biases, getattr(l, "bits", 4), getattr(l, "group_size", 32))
                 for l in linears]
        for linear, part in zip(linears, parts):
            if getattr(linear, "mode", "affine") != "affine" or part.bits not in AFFINE_BITS or part.group % 32:
                raise ValueError("expected MLX affine weights (2-8 bits, groups of 32, 64 or 128)")
        bits, group = max(p.bits for p in parts), min(p.group for p in parts)
        parts = [p.widened(bits, group) for p in parts]
        if len(parts) == 1:
            return parts[0]
        return cls(mx.concatenate([p.weight for p in parts]), mx.concatenate([p.scales for p in parts]),
                   mx.concatenate([p.biases for p in parts]), bits, group)

    def widened(self, bits: int, group: int) -> "QWeights":
        """The same dequantized values in ``bits``-bit groups of ``group``: codes kept, each scale repeated."""

        if (bits, group) == (self.bits, self.group):
            return self
        if bits < self.bits or self.group % group:
            raise ValueError(f"cannot widen {self.bits}-bit g{self.group} to {bits}-bit g{group} exactly")
        weight = self.weight if bits == self.bits else repack(self.weight, self.bits, bits)
        reps = self.group // group
        scales, biases = (mx.repeat(a, reps, axis=1) if reps > 1 else a for a in (self.scales, self.biases))
        return QWeights(weight, scales, biases, bits, group)


AFFINE_BITS = (2, 3, 4, 5, 6, 8)


def repack(weight: mx.array, bits: int, wider: int) -> mx.array:
    """MLX-packed codes of ``bits`` re-packed at ``wider`` bits, value for value (host-side, for small stacks)."""

    import numpy as np

    words = np.asarray(weight.astype(mx.uint32))
    n, k = words.shape[0], words.shape[1] * 32 // bits
    stream = np.unpackbits(words.view(np.uint8).reshape(n, -1), axis=1, bitorder="little")
    codes = stream.reshape(n, k, bits)
    out = np.zeros((n, k, wider), dtype=np.uint8)
    out[:, :, :bits] = codes
    packed = np.packbits(out.reshape(n, k * wider), axis=1, bitorder="little")
    return mx.array(packed.view(np.uint32).reshape(n, k * wider // 32))


def edited(source: str, edits: list[tuple[str, str]]) -> str:
    """A kernel's source with each (old, new) edit applied; every old text must appear exactly once."""

    for old, new in edits:
        if source.count(old) != 1:
            raise AssertionError(f"kernel source changed under an edit ({old[:60]!r})")
        source = source.replace(old, new)
    return source


def pick(name: str, count: int, index: str) -> str:
    """``index == 0 ? NAME0 : index == 1 ? NAME1 : ... NAME{count-1}`` (buffers are bound one per stream)."""

    expr = f"{name}{count - 1}"
    for b in range(count - 2, -1, -1):
        expr = f"({index} == {b} ? {name}{b} : {expr})"
    return expr
