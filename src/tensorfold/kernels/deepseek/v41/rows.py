"""DeepSeek-V4.1-Flash's row kernels; ``head_logits`` is oMLX's ``head.py`` kernel (MIT)."""

from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

_HEAD = r"""
    const uint row = thread_position_in_grid.x / KL;
    if (row >= N) return;
    const uint lane = thread_index_in_simdgroup % KL;
    const uint query = threadgroup_position_in_grid.y;
    float sum = 0;
    for (uint k = lane * 4; k < K; k += KL * 4) {
        const size_t offset = size_t(row) * K + k;
        const float4 a = float4(w[offset], w[offset + 1], w[offset + 2], w[offset + 3]);
        const float4 b = float4(x[query * K + k], x[query * K + k + 1], x[query * K + k + 2], x[query * K + k + 3]);
        sum += dot(a, b);
    }
    for (uint offset = KL / 2; offset > 0; offset /= 2) {
        sum += simd_shuffle_down(sum, offset);
    }
    if (lane == 0) y[query * N + row] = sum;
"""


def metal() -> bool:
    return mx.metal.is_available() and mx.default_device() == mx.gpu


@cache
def _head_kernel() -> Any:
    return mx.fast.metal_kernel(name="tf_dsv41_bf16_head_fp32", input_names=["x", "w"], output_names=["y"],
                                source=_HEAD)


def head_fits(weight: mx.array) -> bool:
    rows, width = weight.shape
    return metal() and weight.dtype == mx.bfloat16 and width % 4 == 0 and width >= 64


def head_logits(x: mx.array, weight: mx.array, rows_exact: bool) -> mx.array:
    """x [R, K] through the bf16 head [V, K]: fp32 logits [R, V]."""

    rows, width = weight.shape
    if head_fits(weight):
        lanes = 16
        return _head_kernel()(inputs=[mx.contiguous(x.astype(mx.float32)), weight],
                              template=[("N", rows), ("K", width), ("KL", lanes)],
                              grid=(((rows * lanes + 63) // 64) * 64, int(x.shape[0]), 1), threadgroup=(64, 1, 1),
                              output_shapes=[(int(x.shape[0]), rows)], output_dtypes=[mx.float32])[0]
    w = weight.astype(mx.float32)
    if not rows_exact or int(x.shape[0]) == 1:
        return x.astype(mx.float32) @ w.T
    return mx.concatenate([x[r:r + 1].astype(mx.float32) @ w.T for r in range(int(x.shape[0]))])


# -- hyper-connection boundaries (oMLX's ``hyper_connection.py`` kernels, MIT): one element or one row a thread group
_HC_POST = r"""
    const uint z = thread_position_in_grid.x;
    if (z >= ROWS * D) return;
    const uint row = z / D, d = z % D;
    float values[4];
    for (uint i = 0; i < 4; ++i)
        values[i] = float(residual[(row * 4 + i) * D + d]);
    const float value = float(x[z]);
    for (uint j = 0; j < 4; ++j) {
        float sum = 0.0f;
        for (uint i = 0; i < 4; ++i)
            sum = fma(comb[row * 16 + i * 4 + j], values[i], sum);
        y[(row * 4 + j) * D + d] = T(post[row * 4 + j] * value + sum);
    }
"""

_HC_PRE_NORM = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    threadgroup float sums[8];
    float values[(D + 255) / 256];
    float total = 0.0f;
    for (uint t = 0; t < (D + 255) / 256; ++t) {
        const uint d = tid + 256 * t;
        float value = 0.0f;
        if (d < D) {
            for (uint i = 0; i < 4; ++i)
                value = float(x[(row * 4 + i) * D + d]) * pre[row * 4 + i] + value;
            value = float(T(value));
        }
        values[t] = value;
        total = total + value * value;
    }
    total = simd_sum(total);
    if (lane == 0) sums[simd] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float sum = lane < 8 ? sums[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0) sums[0] = rsqrt(sum / float(D) + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < (D + 255) / 256; ++t) {
        const uint d = tid + 256 * t;
        if (d < D) y[row * D + d] = T((values[t] * sums[0]) * float(weight[d]));
    }
"""


@cache
def _hc_kernel(name: str) -> Any:
    source, inputs = {"post": (_HC_POST, ["x", "residual", "post", "comb"]),
                      "pre_norm": (_HC_PRE_NORM, ["x", "pre", "weight", "eps"])}[name]
    return mx.fast.metal_kernel(name=f"tf_dsv41_hc_{name}", input_names=inputs, output_names=["y"], source=source,
                                header="#pragma clang fp contract(off)\n")


def hc_post(f: mx.array, residual: mx.array, post: mx.array, comb: mx.array) -> mx.array:
    """New streams [R, 4, D]: post_j * f + sum_i comb[i, j] * residual_i (fp32, the sum by fma in i order)."""

    rows, dims = int(f.shape[0]), int(f.shape[-1])
    if metal() and residual.dtype == f.dtype:
        return _hc_kernel("post")(inputs=[mx.contiguous(f), mx.contiguous(residual), mx.contiguous(post),
                                          mx.contiguous(comb)],
                                  template=[("T", f.dtype), ("ROWS", rows), ("D", dims)], grid=(rows * dims, 1, 1),
                                  threadgroup=(256, 1, 1), output_shapes=[residual.shape], output_dtypes=[f.dtype])[0]
    r = residual.astype(mx.float32)
    out = []
    for j in range(4):
        s = comb[:, 0, j:j + 1] * r[:, 0]
        for i in range(1, 4):
            s = comb[:, i, j:j + 1] * r[:, i] + s
        out.append(post[:, j:j + 1] * f.astype(mx.float32) + s)
    return mx.stack(out, axis=1).astype(f.dtype)


def hc_pre_norm(x: mx.array, pre: mx.array, weight: mx.array, eps: float) -> mx.array:
    """RMSNorm(sum_i pre_i * x_i, rounded to x's dtype) times the weight: [R, D]."""

    rows, dims = int(x.shape[0]), int(x.shape[-1])
    if metal() and dims <= 8192:
        return _hc_kernel("pre_norm")(inputs=[mx.contiguous(x), mx.contiguous(pre.astype(mx.float32)), weight,
                                              mx.array([eps], dtype=mx.float32)],
                                      template=[("T", x.dtype), ("D", dims)], grid=(rows * 256, 1, 1),
                                      threadgroup=(256, 1, 1), output_shapes=[(rows, dims)],
                                      output_dtypes=[x.dtype])[0]
    xf = x.astype(mx.float32)
    v = xf[:, 0] * pre[:, 0:1]
    for i in range(1, 4):
        v = xf[:, i] * pre[:, i:i + 1] + v
    v = v.astype(x.dtype).astype(mx.float32)
    inv = mx.rsqrt(mx.sum(v * v, -1, keepdims=True) / dims + eps)
    return ((v * inv) * weight.astype(mx.float32)).astype(x.dtype)


@cache
def _exp_kernel() -> Any:
    return mx.fast.metal_kernel(name="tf_dsv41_fast_exp", input_names=["x"], output_names=["y"],
                                source="const uint i = thread_position_in_grid.x; if (i < N) y[i] = exp(x[i]);")


def fexp(x: mx.array) -> mx.array:
    """exp as oMLX's and DeepSeek's Metal kernels compute it (Metal's fast exp), MLX's exp elsewhere."""

    x = x.astype(mx.float32)
    if not metal() or x.size == 0:
        return mx.exp(x)
    return _exp_kernel()(inputs=[mx.contiguous(x)], template=[("N", x.size)], grid=(x.size, 1, 1),
                         threadgroup=(256, 1, 1), output_shapes=[x.shape], output_dtypes=[mx.float32])[0]


# -- attention: oMLX's ``packed_attention.py`` fused kernel and merge (MIT), a threadgroup per (query, head)
_ATTN_FUSED = r"""
    const uint lane = thread_index_in_simdgroup, block = simdgroup_index_in_threadgroup;
    const uint head = threadgroup_position_in_grid.x, query = threadgroup_position_in_grid.y;
    const int H = meta[0], W = meta[1], C = meta[2], NW = meta[3], NC = meta[4];
    threadgroup float scores[NB * CHUNK], maxima[NB], probabilities[NB * CHUNK];
    float qv[D / 32];
    for (int v = 0; v < D / 32; ++v) qv[v] = float(q[(query * H + head) * D + lane * (D / 32) + v]);
    float maximum = -1e30f;
    for (int j = 0; j < CHUNK; ++j) {
        const int slot = int(block) * CHUNK + j;
        const bool compressed = slot >= W;
        const int row = slot >= W + C ? -1 : compressed ? ci[query * C + slot - W] : wi[query * W + slot];
        float score = -INFINITY;
        if (row >= 0 && row < (compressed ? NC : NW)) {
            const int first = lane * (D / 32);
            float dot = 0.0f;
            for (int v = 0; v < D / 32; ++v) {
                const float kv = compressed ? float(pooled[size_t(row) * D + first + v])
                                            : float(window[size_t(row) * D + first + v]);
                dot = fma(qv[v], kv, dot);
            }
            score = simd_sum(dot) * scalep[0];
        }
        maximum = max(maximum, score);
        if (lane == 0) scores[block * CHUNK + j] = score;
    }
    if (lane == 0) maxima[block] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = -1e30f;
    for (uint b = lane; b <= min(block | uint(64 / CHUNK - 1), uint(NB - 1)); b += 32)
        maximum = max(maximum, maxima[b]);
    maximum = simd_max(maximum);
    float denominator = 0.0f;
    for (uint j = lane; j < CHUNK; j += 32) {
        const float p = exp(scores[block * CHUNK + j] - maximum);
        denominator += p;
        probabilities[block * CHUNK + j] = float(bfloat16_t(p));
    }
    denominator = simd_sum(denominator);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float acc[D / 32];
    for (int v = 0; v < D / 32; ++v) acc[v] = 0.0f;
    for (int j = 0; j < CHUNK; ++j) {
        const int slot = int(block) * CHUNK + j;
        const bool compressed = slot >= W;
        const int row = slot >= W + C ? -1 : compressed ? ci[query * C + slot - W] : wi[query * W + slot];
        if (row < 0 || row >= (compressed ? NC : NW)) continue;
        const int first = lane * (D / 32);
        const float p = probabilities[block * CHUNK + j];
        for (int v = 0; v < D / 32; ++v) {
            const float kv = compressed ? float(pooled[size_t(row) * D + first + v])
                                        : float(window[size_t(row) * D + first + v]);
            acc[v] = fma(p, kv, acc[v]);
        }
    }
    const size_t out_base = ((size_t(query) * H + head) * NB + block) * (D + 2);
    for (int v = 0; v < D / 32; ++v) partial[out_base + lane * (D / 32) + v] = acc[v];
    if (lane == 0) { partial[out_base + D] = maximum; partial[out_base + D + 1] = denominator; }
"""

_ATTN_MERGE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint part = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint head = part / (D / 32), d = (part % (D / 32)) * 32 + lane;
    const uint query = threadgroup_position_in_grid.y;
    const int H = meta[0], NB = meta[1];
    if (head >= H) return;
    const size_t base = (size_t(query) * H + head) * NB * (D + 2);
    float maximum = -1e30f, denominator = 0.0f, acc = 0.0f;
    for (int b = 0; b < NB; ++b) {
        const size_t pos = base + b * (D + 2);
        const float next_maximum = partial[pos + D];
        const float correction = (b % (64 / CHUNK) == 0) ? exp(maximum - next_maximum) : 1.0f;
        denominator = denominator * correction + partial[pos + D + 1];
        acc = acc * correction + partial[pos + d];
        maximum = next_maximum;
    }
    denominator += exp(float(sink[head]) - maximum);
    out[(query * H + head) * D + d] = T(acc / denominator);
"""


@cache
def _attn_kernel(stage: str) -> Any:
    if stage == "fused":
        return mx.fast.metal_kernel(name="tf_dsv41_attn_fused",
                                    input_names=["q", "window", "pooled", "wi", "ci", "meta", "scalep"],
                                    output_names=["partial"], source=_ATTN_FUSED)
    return mx.fast.metal_kernel(name="tf_dsv41_attn_merge", input_names=["partial", "sink", "meta"],
                                output_names=["out"], source=_ATTN_MERGE)


def attention_fits(slots: int, dims: int) -> bool:
    return metal() and 0 < slots <= 2048 and dims % 32 == 0 and dims <= 512


def attention(q: mx.array, window: mx.array, pooled: mx.array | None, wi: mx.array, ci: mx.array, sink: mx.array,
              scale: float) -> mx.array:
    """q [L, H, D] over window rows ``wi`` [L, W] and pool rows ``ci`` [L, C] (-1: none), with fp32 sinks."""

    length, heads, dim = (int(s) for s in q.shape)
    W, C = int(wi.shape[-1]), int(ci.shape[-1])
    count = W + C
    chunk = 32 if count <= 1024 else 64
    blocks = max(1, (count + chunk - 1) // chunk)
    if pooled is None or not int(pooled.size):
        pooled = mx.zeros((1, dim), dtype=mx.bfloat16)
    meta = mx.array([heads, W, C, int(window.shape[0]), int(pooled.shape[0])], dtype=mx.int32)
    partial = _attn_kernel("fused")(
        inputs=[mx.contiguous(q), mx.contiguous(window.astype(mx.float32)), mx.contiguous(pooled),
                mx.contiguous(wi.astype(mx.int32)) if W else mx.zeros((1,), mx.int32),
                mx.contiguous(ci.astype(mx.int32)) if C else mx.zeros((1,), mx.int32), meta,
                mx.array([scale], dtype=mx.float32)],
        template=[("D", dim), ("NB", blocks), ("CHUNK", chunk)], grid=(heads * blocks * 32, length, 1),
        threadgroup=(blocks * 32, 1, 1), output_shapes=[(length, heads, blocks, dim + 2)],
        output_dtypes=[mx.float32])[0]
    return _attn_kernel("merge")(
        inputs=[partial, sink.astype(mx.float32), mx.array([heads, blocks], dtype=mx.int32)],
        template=[("D", dim), ("T", q.dtype), ("CHUNK", chunk)],
        grid=((heads * (dim // 32) + 3) // 4 * 128, length, 1), threadgroup=(128, 1, 1),
        output_shapes=[(length, heads, dim)], output_dtypes=[q.dtype])[0]


# -- index scores: oMLX's ``kernels.py`` index kernel (MIT), reading the index keys' exact bf16 values
_INDEX = r"""
    const uint lane = thread_index_in_threadgroup % 32;
    const uint position = threadgroup_position_in_grid.x * 4 + thread_index_in_threadgroup / 32;
    const uint query = threadgroup_position_in_grid.y;
    const int H = HEADS, N = meta[1], M = meta[2], start = meta[3], ratio = RATIO;
    if (position >= M) return;
    const int row = CANDIDATES ? candidates[query * M + position] : int(position);
    if (row < 0 || row >= N || row >= (start + int(query) + 1) / ratio) {
        if (lane == 0) scores[query * M + position] = -INFINITY;
        return;
    }
    float key[D / 32];
    for (int v = 0; v < D / 32; ++v) key[v] = float(keys[size_t(row) * D + lane + v * 32]);
    float result = 0.0f;
    for (int h = 0; h < H; ++h) {
        float dot = 0.0f;
        for (int v = 0; v < D / 32; ++v)
            dot += float(q[(query * H + h) * D + lane + v * 32]) * key[v];
        result += max(simd_sum(dot), 0.0f) * weights[query * H + h];
    }
    if (lane == 0) scores[query * M + position] = result;
"""


@cache
def _index_kernel() -> Any:
    return mx.fast.metal_kernel(name="tf_dsv41_index_scores",
                                input_names=["q", "keys", "weights", "candidates", "meta"],
                                output_names=["scores"], source=_INDEX)


def index_fits(dims: int) -> bool:
    return metal() and dims % 32 == 0


def index_scores(q: mx.array, w: mx.array, keys: mx.array, rows: int, start: int, ratio: int,
                 candidates: mx.array | None = None) -> mx.array:
    """Index scores [L, M] of q [L, H, D] over the first ``rows`` keys or ``candidates``; unseen keys score -inf."""

    length, heads, dim = (int(s) for s in q.shape)
    width = rows if candidates is None else int(candidates.shape[-1])
    if not width:
        return mx.zeros((length, 0), dtype=mx.float32)
    return _index_kernel()(
        inputs=[mx.contiguous(q), mx.contiguous(keys), mx.contiguous(w.astype(mx.float32)),
                mx.zeros((1,), mx.int32) if candidates is None else mx.contiguous(candidates.astype(mx.int32)),
                mx.array([heads, rows, width, start, ratio, 0], dtype=mx.int32)],
        template=[("D", dim), ("CANDIDATES", candidates is not None), ("HEADS", heads), ("RATIO", ratio)],
        grid=((width + 3) // 4 * 128, length, 1), threadgroup=(128, 1, 1), output_shapes=[(length, width)],
        output_dtypes=[mx.float32])[0]


# -- FP8 activations: oMLX's ``activation.py`` kernels (MIT): one SIMD group a 32-value scale group
_ROUND = r"""
    // Clamp to 448 * 2^-126 so the power-of-two scale stays normal.
    const float amax = max(simd_max(abs(v)), 0x1.cp-118f);
    const int scale_exponent = max(int(ceil(log2(amax / 448.0f))), -126);
    const float scale = as_type<float>(uint(scale_exponent + 127) << 23);
    const float scaled = clamp(v / scale, -448.0f, 448.0f);
    const float a = abs(scaled);
    const int step_exponent = max(int(floor(log2(max(a, 0x1p-9f)))) - 3, -9);
    const float step = as_type<float>(uint(step_exponent + 127) << 23);
    const float q = sign(scaled) * min(rint(a / step) * step, 448.0f);
    if (i < n) y[i] = T(q * scale);
"""

_FP8 = r"""
    const uint i = thread_position_in_grid.x;
    const uint n = N;
    const float v = i < n ? float(x[i]) : 0.0f;
""" + _ROUND

_SWIGLU_FP8 = r"""
    const uint i = thread_position_in_grid.x;
    const uint n = N;
    float v = 0.0f;
    if (i < n) {
        float g = gate[i], u = up[i];
        if (limit[0] != 0.0f) {
            g = min(g, limit[0]);
            u = clamp(u, -limit[0], limit[0]);
        }
        const float neg_sigmoid = 1.0f / (1.0f + exp(abs(g)));
        const float sigmoid = g < 0 ? neg_sigmoid : 1.0f - neg_sigmoid;
        float value = (g * sigmoid) * u;
        if (WEIGHTED) value *= weights[i / D];
        v = float(T(value));
    }
""" + _ROUND


@cache
def _act_kernel(tail: bool) -> Any:
    if tail:
        return mx.fast.metal_kernel(name="tf_dsv41_swiglu_fp8", input_names=["gate", "up", "weights", "limit"],
                                    output_names=["y"], source=_SWIGLU_FP8)
    return mx.fast.metal_kernel(name="tf_dsv41_fp8", input_names=["x"], output_names=["y"], source=_FP8)


def fp8_fits(x: mx.array) -> bool:
    return metal() and x.size > 0 and int(x.shape[-1]) % 32 == 0 and x.dtype in (mx.float32, mx.float16,
                                                                                  mx.bfloat16)


def fp8(x: mx.array) -> mx.array:
    """The FP8 round trip (E4M3, a UE8M0 scale per 32) in one kernel."""

    return _act_kernel(False)(inputs=[mx.contiguous(x)], template=[("T", x.dtype), ("N", x.size)],
                              grid=(x.size, 1, 1), threadgroup=(256, 1, 1), output_shapes=[x.shape],
                              output_dtypes=[x.dtype])[0]


def swiglu_fp8(gate: mx.array, up: mx.array, weights: mx.array | None, limit: float, dtype: Any) -> mx.array:
    """oMLX's SwiGLU tail: clamp, g * sigmoid(g) * u (times each pick's weight), to ``dtype``, then FP8."""

    return _act_kernel(True)(
        inputs=[mx.contiguous(gate), mx.contiguous(up),
                mx.contiguous(weights.reshape(-1).astype(mx.float32)) if weights is not None else mx.ones((1,)),
                mx.array([float(limit or 0)], dtype=mx.float32)],
        template=[("T", dtype), ("N", gate.size), ("D", int(gate.shape[-1])), ("WEIGHTED", weights is not None)],
        grid=(gate.size, 1, 1), threadgroup=(256, 1, 1), output_shapes=[gate.shape], output_dtypes=[dtype])[0]


# -- Sinkhorn: oMLX's ``hyper_connection.py`` kernel (MIT): a row's 4 x 4 mix normalised in a fixed order
_SINKHORN = r"""
    const uint row = thread_position_in_grid.x;
    if (row >= ROWS) return;
    float values[16];
    for (int i = 0; i < 16; ++i) values[i] = x[row * 16 + i];
    for (int iteration = 0; iteration < ITERS; ++iteration) {
        if (iteration > 0) {
            for (int r = 0; r < 4; ++r) {
                float total = 0.0f;
                for (int c = 0; c < 4; ++c) total = values[r * 4 + c] + total;
                total = total + eps[0];
                for (int c = 0; c < 4; ++c) values[r * 4 + c] /= total;
            }
        }
        for (int c = 0; c < 4; ++c) {
            float total = 0.0f;
            for (int r = 0; r < 4; ++r) total = values[r * 4 + c] + total;
            total = total + eps[0];
            for (int r = 0; r < 4; ++r) values[r * 4 + c] /= total;
        }
    }
    for (int i = 0; i < 16; ++i) y[row * 16 + i] = values[i];
"""


@cache
def _sinkhorn_kernel() -> Any:
    return mx.fast.metal_kernel(name="tf_dsv41_sinkhorn", input_names=["x", "eps"], output_names=["y"],
                                source=_SINKHORN,
                                header="#pragma clang fp reassociate(off)\n#pragma clang fp contract(off)\n")


def sinkhorn(comb: mx.array, eps: float, iters: int) -> mx.array:
    """comb [..., 4, 4] fp32: a column normalisation, then ``iters - 1`` row-then-column ones."""

    if metal() and comb.size and comb.dtype == mx.float32 and tuple(comb.shape[-2:]) == (4, 4):
        rows = comb.size // 16
        return _sinkhorn_kernel()(inputs=[comb, mx.array([eps], dtype=mx.float32)],
                                  template=[("ROWS", rows), ("ITERS", max(1, iters))], grid=(rows, 1, 1),
                                  threadgroup=(32, 1, 1), output_shapes=[comb.shape], output_dtypes=[mx.float32])[0]
    comb = comb / (mx.sum(comb, -2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (mx.sum(comb, -1, keepdims=True) + eps)
        comb = comb / (mx.sum(comb, -2, keepdims=True) + eps)
    return comb


# -- decode rows' projections: MLX's one-row fp_qmv_fast (Apple, MIT), simdgroup r on row r, sharing weight reads
_FPQMV_HEADER = r"""
inline float tf_e4m3(uint8_t b) {
  uint16_t v = b & 127;
  uint16_t sign_bit = ((uint16_t)((b >> 7) & 1)) << 15;
  uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
  return float(as_type<half>(u) * half(256.0));
}
inline float tf_e2m1(uint8_t b) {
  half c = as_type<half>(ushort((b & 7) << 9));
  c *= half(16384.0);
  return float(b & 8 ? -c : c);
}
inline float tf_e8m0(uint8_t b) {
  uint32_t out = (b == 0 ? 0x400000 : (uint32_t(b) << 23));
  return as_type<float>(out);
}
"""

_FPQMV_ROWS = r"""
  const uint lane = thread_index_in_simdgroup;
  const int r = int(simdgroup_index_in_threadgroup);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int PF = 32 / BITS;
  constexpr int VPT = PF * 2;
  constexpr int BLOCK = VPT * 32;
  constexpr int SSTEP = 32 / VPT;
  constexpr int KW = K * BITS / 8;
  constexpr int KG = K / 32;
  const device uint8_t* ws = (const device uint8_t*)W + size_t(row0) * KW + lane * 8;
  const device uint8_t* sc = S + size_t(row0) * KG + lane / SSTEP;
  const device T* x = X + size_t(r) * K + lane * VPT;
  float xt[VPT];
  float result[RPS];
  for (int j = 0; j < RPS; ++j) result[j] = 0.0f;
  for (int k = 0; k < K; k += BLOCK) {
    for (int i = 0; i < VPT; ++i) xt[i] = float(x[i]);
    for (int j = 0; j < RPS; ++j) {
      const device uint8_t* wl = ws + j * KW;
      const float s = tf_e8m0(sc[j * KG]);
      float accum = 0.0f;
      if (BITS == 4) {
        const device uint16_t* w16 = (const device uint16_t*)wl;
        for (int i = 0; i < VPT / 4; ++i)
          accum += (xt[4 * i] * tf_e2m1(uint8_t(w16[i] & 15)) + xt[4 * i + 1] * tf_e2m1(uint8_t((w16[i] >> 4) & 15)) +
                    xt[4 * i + 2] * tf_e2m1(uint8_t((w16[i] >> 8) & 15)) +
                    xt[4 * i + 3] * tf_e2m1(uint8_t((w16[i] >> 12) & 15)));
      } else {
        for (int i = 0; i < VPT; ++i) accum += xt[i] * tf_e4m3(wl[i]);
      }
      result[j] += s * accum;
    }
    ws += BLOCK * BITS / 8;
    sc += BLOCK / 32;
    x += BLOCK;
  }
  for (int j = 0; j < RPS; ++j) {
    const float v = simd_sum(result[j]);
    if (lane == 0) OUT[size_t(r) * N + row0 + j] = T(v);
  }
"""

ROWS_MAX = 16
_ROWS_MODE = [False]


def rows_mode() -> bool:
    """True while a decode forward runs: every projection gives each row its one-row call's bits."""

    return _ROWS_MODE[0]


class decode_rows:
    """``with decode_rows(): ...`` runs a decode forward's projections row-exact."""

    def __enter__(self) -> None:
        self.previous = _ROWS_MODE[0]
        _ROWS_MODE[0] = True

    def __exit__(self, *exc: Any) -> None:
        _ROWS_MODE[0] = self.previous


@cache
def _fpqmv_kernel() -> Any:
    return mx.fast.metal_kernel(name="tf_dsv41_fpqmv_rows", input_names=["X", "W", "S"], output_names=["OUT"],
                                source=_FPQMV_ROWS, header=_FPQMV_HEADER)


def fpqmv_rows_fits(x: mx.array, bits: int, k: int, n: int) -> bool:
    rows = int(x.shape[0])
    block = 2 * 32 * 32 // bits
    return (metal() and x.ndim == 2 and 1 < rows <= ROWS_MAX and bits in (4, 8) and k % block == 0 and n % 4 == 0
            and x.dtype in (mx.bfloat16, mx.float16, mx.float32))


def fpqmv_rows(x: mx.array, weight: mx.array, scales: mx.array, bits: int, k: int, n: int) -> mx.array:
    """x [R, k] through mxfp8 / mxfp4 weights [n, ...] in groups of 32: each row with MLX's one-row bits."""

    rows = int(x.shape[0])
    return _fpqmv_kernel()(inputs=[mx.contiguous(x), weight, scales],
                           template=[("BITS", bits), ("K", k), ("N", n), ("RPS", 4), ("T", x.dtype)],
                           grid=(32 * rows, n // 4, 1), threadgroup=(32 * rows, 1, 1), output_shapes=[(rows, n)],
                           output_dtypes=[x.dtype])[0]


def matmul(x: mx.array, w: mx.array, out_dtype: Any = None, groups: int = 1) -> mx.array:
    """x @ w.T for unquantized (optionally grouped) w [N, K] as MLX computes it, decode rows one MLX call each."""

    out_dtype = out_dtype or mx.promote_types(x.dtype, w.dtype)

    def plain(a: mx.array) -> mx.array:
        if groups == 1:
            return mx.matmul(a, w.T).astype(out_dtype)
        rows = int(a.shape[0])
        g = a.reshape(rows, groups, -1)
        return mx.einsum("lgd,grd->lgr", g, w.reshape(groups, int(w.shape[0]) // groups, -1)).reshape(rows, -1) \
            .astype(out_dtype)

    if rows_mode() and int(x.shape[0]) > 1:
        return mx.concatenate([plain(x[r:r + 1]) for r in range(int(x.shape[0]))])
    return plain(x)