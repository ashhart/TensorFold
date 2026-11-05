// Copyright 2024 Mistral and the HuggingFace Inc. team. All rights reserved.

// SPDX-License-Identifier: Apache-2.0

// Reference: transformers v5.5.4 Pixtral/LightOnOcr and PyTorch v2.9.0 activation formulas (Apache-2.0/BSD).
#include "ops/common.hpp"

namespace lighton {
__device__ __forceinline__ float value(uint16_t x) { return bf16_to_float(x); }
__device__ __forceinline__ float rounded(float x) { return bf16_to_float(float_to_bf16(x)); }
__device__ __forceinline__ float mul(float a, float b) { return a * b; }
__device__ __forceinline__ float add(float a, float b) { return a + b; }
__device__ __forceinline__ float div(float a, float b) { return a / b; }
}

// inv holds the height frequencies first, then the distinct width frequencies; both halves rotate as rotate_half.
template <bool SWAP_AXES>
__device__ __forceinline__ void lo_tables(
    const float* __restrict__ inv, uint16_t* __restrict__ cs, uint16_t* __restrict__ sn,
    int gh, int gw, int dim) {
    const int half = dim / 2, quarter = dim / 4;
    const int64_t end = static_cast<int64_t>(gh) * gw * half;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < end;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t patch = i / half;
        const int j = i % half;
        const bool height = (j < quarter) != SWAP_AXES;
        const int position = height ? patch / gw : patch % gw;
        const float angle = lighton::mul(static_cast<float>(position), inv[j]);
        cs[i] = float_to_bf16(cosf(angle));
        sn[i] = float_to_bf16(sinf(angle));
    }
}

// Torch BF16 multiplies and rounds each rotary branch before their BF16 add; values only change layout.
template <bool ROUND_BRANCHES>
__device__ __forceinline__ void lo_rotary(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ v, const uint16_t* __restrict__ cs,
    const uint16_t* __restrict__ sn, uint16_t* __restrict__ qo,
    uint16_t* __restrict__ ko, uint16_t* __restrict__ vo, int rows, int heads, int dim) {
    const int half = dim / 2;
    const int64_t end = static_cast<int64_t>(rows) * heads * dim;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < end;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int j = i % dim, head = (i / dim) % heads;
        const int64_t row = i / (heads * dim);
        const int partner = j < half ? j + half : j - half;
        const int64_t p = (row * heads + head) * dim + partner;
        const int64_t out = (static_cast<int64_t>(head) * rows + row) * dim + j;
        const float c = lighton::value(cs[row * half + j % half]);
        const float s = lighton::value(sn[row * half + j % half]);
        const float sign = j < half ? -1.f : 1.f;
        float qa = lighton::mul(lighton::value(q[i]), c), qb = lighton::mul(sign * lighton::value(q[p]), s);
        float ka = lighton::mul(lighton::value(k[i]), c), kb = lighton::mul(sign * lighton::value(k[p]), s);
        if constexpr (ROUND_BRANCHES) {
            qa = lighton::rounded(qa);
            qb = lighton::rounded(qb);
            ka = lighton::rounded(ka);
            kb = lighton::rounded(kb);
        }
        qo[out] = float_to_bf16(lighton::add(qa, qb));
        ko[out] = float_to_bf16(lighton::add(ka, kb));
        vo[out] = v[i];
    }
}

// torch.nn.functional.unfold on NCHW: channel first, then dy/dx, with odd trailing patch rows/columns discarded.
template <bool CHANNEL_FIRST>
__device__ __forceinline__ void lo_merge(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ out, int gh, int gw, int width) {
    const int mh = gh / 2, mw = gw / 2;
    const int64_t end = static_cast<int64_t>(mh) * mw * width * 4;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < end;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int lane = CHANNEL_FIRST ? i % 4 : (i / width) % 4;
        const int ch = CHANNEL_FIRST ? (i / 4) % width : i % width;
        const int64_t token = i / (width * 4);
        const int64_t row = (token / mw) * 2 + lane / 2;
        const int64_t col = (token % mw) * 2 + lane % 2;
        out[i] = x[(row * gw + col) * width + ch];
    }
}

#define LO_TABLES_ARGS const float* inv, uint16_t* cs, uint16_t* sn, int gh, int gw, int dim
#define LO_ROTARY_ARGS const uint16_t* q, const uint16_t* k, const uint16_t* v, \
    const uint16_t* cs, const uint16_t* sn, uint16_t* qo, uint16_t* ko, \
    uint16_t* vo, int rows, int heads, int dim
#define LO_MERGE_ARGS const uint16_t* x, uint16_t* out, int gh, int gw, int width

extern "C" __global__ void __launch_bounds__(256) tf_lo_tables(LO_TABLES_ARGS) {
    lo_tables<false>(inv, cs, sn, gh, gw, dim);
}
extern "C" __global__ void __launch_bounds__(256) tf_lo_rotary(LO_ROTARY_ARGS) {
    lo_rotary<true>(q, k, v, cs, sn, qo, ko, vo, rows, heads, dim);
}
extern "C" __global__ void __launch_bounds__(256) tf_lo_merge(LO_MERGE_ARGS) {
    lo_merge<true>(x, out, gh, gw, width);
}

// Numerical contract controls use identical launches and pointers; each changes only its named operation.
extern "C" __global__ void __launch_bounds__(256) control_lo_tables_axes(LO_TABLES_ARGS) {
    lo_tables<true>(inv, cs, sn, gh, gw, dim);
}
extern "C" __global__ void __launch_bounds__(256) control_lo_rotary_unrounded(LO_ROTARY_ARGS) {
    lo_rotary<false>(q, k, v, cs, sn, qo, ko, vo, rows, heads, dim);
}
extern "C" __global__ void __launch_bounds__(256) control_lo_merge_patchmajor(LO_MERGE_ARGS) {
    lo_merge<false>(x, out, gh, gw, width);
}

template <bool TANH_FORM>
__device__ __forceinline__ void lo_gelu(const uint16_t* x, uint16_t* out, int64_t n) {
    for (int64_t i = gid(); i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const float v = bf16_to_float(x[i]);
        const float a = TANH_FORM ? tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))
                                 : erff(v * 0.7071067811865475244f);
        out[i] = float_to_bf16((v * 0.5f) * (1.f + a));
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_lo_gelu(const uint16_t* x, uint16_t* out, int64_t n) {
    lo_gelu<false>(x, out, n);
}
extern "C" __global__ void __launch_bounds__(256) control_lo_gelu_tanh(const uint16_t* x, uint16_t* out, int64_t n) {
    lo_gelu<true>(x, out, n);
}

// A block's fixed-order FP32 sum: wave32 XOR trees followed by its wave partials in ascending order.
__device__ __forceinline__ float lo_inverse(float part, int width, float eps) {
    __shared__ float waves[8];
    for (int m = 16; m; m >>= 1) part += __shfl_xor(part, m, 32);
    if ((threadIdx.x & 31) == 0) waves[threadIdx.x >> 5] = part;
    __syncthreads();
    float sum = 0.f;
    for (int w = 0; w < static_cast<int>(blockDim.x) / 32; ++w) sum += waves[w];
    return 1.f / sqrtf(sum / static_cast<float>(width) + eps);
}

// Torch's RMSNorm: BF16(x * inv) before multiplication by the exactly widened BF16 weight.
template <bool ROUND_ROW>
__device__ __forceinline__ void lo_rms(
    const uint16_t* x, const float* weight, uint16_t* out, int width, float eps) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * width;
    float sum = 0.f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float v = bf16_to_float(x[at + i]);
        sum += v * v;
    }
    const float inv = lo_inverse(sum, width, eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float normalized = bf16_to_float(x[at + i]) * inv;
        if constexpr (ROUND_ROW) normalized = lighton::rounded(normalized);
        out[at + i] = float_to_bf16(normalized * weight[i]);
    }
}

#define LO_RMS_ARGS const uint16_t* x, const float* weight, uint16_t* out, int width, float eps
extern "C" __global__ void __launch_bounds__(256) tf_lo_rms(LO_RMS_ARGS) { lo_rms<true>(x, weight, out, width, eps); }
extern "C" __global__ void __launch_bounds__(256) control_lo_rms_unrounded(LO_RMS_ARGS) {
    lo_rms<false>(x, weight, out, width, eps);
}

// Qwen3's head128 Q/K norm and Torch RoPE, with packed per-row BF16 outputs and V unchanged.
template <bool ROUND_ROPE>
__device__ __forceinline__ void lo_text_qk(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, const float* wq, const float* wk,
    const float* inv, const int* pos, uint16_t* qo, uint16_t* ko, uint16_t* vo,
    int h, int hk, int dim, float eps) {
    const int d = threadIdx.x, row = blockIdx.x, head = blockIdx.y;
    if (head >= h + hk) {
        const int64_t at = (static_cast<int64_t>(row) * hk + head - h - hk) * dim + d;
        vo[at] = v[at];
        return;
    }
    const bool is_k = head >= h;
    const int local = is_k ? head - h : head;
    const int nh = is_k ? hk : h;
    const uint16_t* src = is_k ? k : q;
    const float* w = is_k ? wk : wq;
    const int64_t at = (static_cast<int64_t>(row) * nh + local) * dim;
    const int half = dim / 2, partner = (d + half) % dim;
    const float x = bf16_to_float(src[at + d]);
    const float xp = bf16_to_float(src[at + partner]);
    const float norm = lo_inverse(x * x, dim, eps);
    const float y = lighton::rounded(lighton::rounded(x * norm) * w[d]);
    const float yp = lighton::rounded(lighton::rounded(xp * norm) * w[partner]);
    const float angle = static_cast<float>(pos[row]) * inv[d % half];
    const float cs = lighton::rounded(cosf(angle)), sn = lighton::rounded(sinf(angle));
    float a = y * cs, z = (d < half ? -yp : yp) * sn;
    if constexpr (ROUND_ROPE) {
        a = lighton::rounded(a);
        z = lighton::rounded(z);
    }
    (is_k ? ko : qo)[at + d] = float_to_bf16(a + z);
}

#define LO_TEXT_ARGS const uint16_t* q, const uint16_t* k, const uint16_t* v, const float* wq, const float* wk, \
    const float* inv, const int* pos, uint16_t* qo, uint16_t* ko, uint16_t* vo, int h, int hk, int dim, float eps
extern "C" __global__ void __launch_bounds__(128) tf_lo_text_qk(LO_TEXT_ARGS) {
    lo_text_qk<true>(q, k, v, wq, wk, inv, pos, qo, ko, vo, h, hk, dim, eps);
}
extern "C" __global__ void __launch_bounds__(128) control_lo_text_qk_unrounded(LO_TEXT_ARGS) {
    lo_text_qk<false>(q, k, v, wq, wk, inv, pos, qo, ko, vo, h, hk, dim, eps);
}

// Shared attention consumes FP32 head-major queries; projection inputs and outputs remain BF16.
extern "C" __global__ void __launch_bounds__(256) tf_lo_rows_to_heads(
    const uint16_t* src, float* out, int rows, int heads, int dim) {
    const int64_t end = static_cast<int64_t>(rows) * heads * dim;
    for (int64_t i = gid(); i < end; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int d = i % dim, head = (i / dim) % heads;
        const int64_t row = i / (heads * dim);
        out[(static_cast<int64_t>(head) * rows + row) * dim + d] = bf16_to_float(src[i]);
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_lo_heads_to_rows(
    const float* src, uint16_t* out, int rows, int heads, int dim) {
    const int64_t end = static_cast<int64_t>(rows) * heads * dim;
    for (int64_t i = gid(); i < end; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int d = i % dim, head = (i / dim) % heads;
        const int64_t row = i / (heads * dim);
        out[i] = float_to_bf16(src[(static_cast<int64_t>(head) * rows + row) * dim + d]);
    }
}

// Full attention within one image, 64 queries/keys a tile, BF16 inputs/outputs and bounded FP32 LDS/state.
template <bool ISOLATED>
__device__ __forceinline__ void lo_attention64(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out,
    int total, int heads, int start, int count, float scale) {
    __shared__ float ks[64][64], vs[64][64];
    const int head = blockIdx.y, qi = blockIdx.x * 64 + threadIdx.x;
    const bool live = qi < count;
    const int64_t base = static_cast<int64_t>(head) * total + start;
    const int64_t key_base = static_cast<int64_t>(head) * total + (ISOLATED ? start : 0);
    const int key_count = ISOLATED ? count : total;
    float qr[64], acc[64];
    float maximum = -INFINITY, denominator = 0.f;
    if (live) for (int d = 0; d < 64; ++d) {
        qr[d] = bf16_to_float(q[(base + qi) * 64 + d]) * scale;
        acc[d] = 0.f;
    }
    for (int begin = 0; begin < key_count; begin += 64) {
        const int size = min(64, key_count - begin);
        __syncthreads();
        for (int i = threadIdx.x; i < size * 64; i += blockDim.x) {
            const int row = i / 64, d = i % 64;
            ks[row][d] = bf16_to_float(k[(key_base + begin + row) * 64 + d]);
            vs[row][d] = bf16_to_float(v[(key_base + begin + row) * 64 + d]);
        }
        __syncthreads();
        if (!live) continue;
        for (int row = 0; row < size; ++row) {
            float score = 0.f;
            for (int d = 0; d < 64; ++d) score += qr[d] * ks[row][d];
            if (score == -INFINITY && maximum == -INFINITY) {
                for (int d = 0; d < 64; ++d) acc[d] += 0.f * vs[row][d];
                continue;
            }
            const float next = fmaxf(maximum, score);
            const float correction = expf(maximum - next), probability = expf(score - next);
            denominator = denominator * correction + probability;
            for (int d = 0; d < 64; ++d) acc[d] = acc[d] * correction + probability * vs[row][d];
            maximum = next;
        }
    }
    if (live) for (int d = 0; d < 64; ++d)
        out[(static_cast<int64_t>(start + qi) * heads + head) * 64 + d] = float_to_bf16(acc[d] / denominator);
}

#define LO_ATTN_ARGS const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out, \
    int total, int heads, int start, int count, float scale
extern "C" __global__ void __launch_bounds__(64) tf_lo_attention64(LO_ATTN_ARGS) {
    lo_attention64<true>(q, k, v, out, total, heads, start, count, scale);
}
extern "C" __global__ void __launch_bounds__(64) control_lo_attention_leak(LO_ATTN_ARGS) {
    lo_attention64<false>(q, k, v, out, total, heads, start, count, scale);
}
