// LightOnOCR-3 on HIP: Pixtral 2D rotary, attention, norms, merger, GELU, Qwen3 q/k norm; BF16 as the reference rounds.
#include "ops/common.hpp"
#include <math.h>

namespace {

__device__ __forceinline__ float lo_bf(float x) { return bf16_to_float(float_to_bf16(x)); }

// Patch `row` of a gw-wide grid: the first dim/4 angles follow its row, the rest its column (`swap`: the control).
__device__ __forceinline__ void lo_tables(const float* inv, uint16_t* cs, uint16_t* sn, int gh, int gw, int dim,
                                          bool swap) {
    const long long half = dim / 2;
    const long long i = gid();
    if (i >= static_cast<long long>(gh) * gw * half) return;
    const int row = static_cast<int>(i / half);
    const int j = static_cast<int>(i % half);
    const int y = row / gw;
    const int x = row % gw;
    const bool first = j < dim / 4;
    const int p = (first != swap) ? y : x;
    const float a = static_cast<float>(p) * inv[j];
    cs[i] = float_to_bf16(cosf(a));
    sn[i] = float_to_bf16(sinf(a));
}

// Rotate-half on one element: out = x*c + rot(x)*s, with rot(x) = (-x[d+h], x[d-h]).
__device__ __forceinline__ float lo_turn(float x, float partner, float c, float s, bool lower, bool rounded) {
    const float t = lower ? -partner : partner;
    if (!rounded) return x * c + t * s;
    return lo_bf(lo_bf(x * c) + lo_bf(t * s));
}

// Raster [rows][heads][dim] q/k/v to head-major [heads][rows][dim]; q and k rotated, v copied.
__device__ __forceinline__ void lo_rotary(const uint16_t* q, const uint16_t* k, const uint16_t* v, const uint16_t* cs,
                                          const uint16_t* sn, uint16_t* qo, uint16_t* ko, uint16_t* vo, int rows,
                                          int heads, int dim, bool rounded) {
    const long long i = gid();
    if (i >= static_cast<long long>(rows) * heads * dim) return;
    const int d = static_cast<int>(i % dim);
    const int r = static_cast<int>((i / dim) % rows);
    const int h = static_cast<int>(i / (static_cast<long long>(dim) * rows));
    const int half = dim / 2;
    const bool lower = d < half;
    const int pd = lower ? d + half : d - half;
    const long long base = (static_cast<long long>(r) * heads + h) * dim;
    const long long t = static_cast<long long>(r) * half + (lower ? d : d - half);
    const float c = bf16_to_float(cs[t]);
    const float s = bf16_to_float(sn[t]);
    qo[i] = float_to_bf16(lo_turn(bf16_to_float(q[base + d]), bf16_to_float(q[base + pd]), c, s, lower, rounded));
    ko[i] = float_to_bf16(lo_turn(bf16_to_float(k[base + d]), bf16_to_float(k[base + pd]), c, s, lower, rounded));
    vo[i] = v[base + d];
}

// Output token: channel ch of its 2x2 group at ch*4 + (2*dy + dx); the control stores patch-major (lane*width + ch).
__device__ __forceinline__ void lo_merge(const uint16_t* x, uint16_t* out, int gh, int gw, int width, bool channel_first) {
    const long long per = 4LL * width;
    const long long i = gid();
    const int across = gw / 2;
    if (i >= static_cast<long long>(gh / 2) * across * per) return;
    const int token = static_cast<int>(i / per);
    const long long within = i % per;
    const int lane = static_cast<int>(channel_first ? within % 4 : within / width);
    const long long ch = channel_first ? within / 4 : within % width;
    const int row = (token / across) * 2 + lane / 2;
    const int col = (token % across) * 2 + lane % 2;
    out[i] = x[(static_cast<long long>(row) * gw + col) * width + ch];
}

// Block-wide FP32 sum over 32-lane waves: butterfly inside each wave, then the wave partials in order.
template <int Waves>
__device__ __forceinline__ float lo_block_sum(float v, float* part) {
    for (int m = 16; m; m >>= 1) v += __shfl_xor(v, m, 32);
    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;
    if (lane == 0) part[wave] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < Waves; ++w) total += part[w];
    return total;
}

// RMSNorm per row: y = x / sqrt(mean(x^2) + eps), rounded to BF16, then times the weight, rounded again.
__device__ __forceinline__ void lo_rms(const uint16_t* x, const float* weight, uint16_t* out, int width, float eps,
                                       bool rounded) {
    __shared__ float part[8];
    const long long base = static_cast<long long>(blockIdx.x) * width;
    float sq = 0.0f;
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        const float v = bf16_to_float(x[base + c]);
        sq += v * v;
    }
    const float total = lo_block_sum<8>(sq, part);
    const float r = 1.0f / sqrtf(total / static_cast<float>(width) + eps);
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        const float y = bf16_to_float(x[base + c]) * r;
        out[base + c] = float_to_bf16(weight[c] * (rounded ? lo_bf(y) : y));
    }
}

// One row and head (q, then k, then v heads): q/k take per-head RMSNorm, then rotate-half with BF16 cos/sin(pos*inv).
__device__ __forceinline__ void lo_text_qk(const uint16_t* q, const uint16_t* k, const uint16_t* v, const float* wq,
                                           const float* wk, const float* inv, const int* pos, uint16_t* qo,
                                           uint16_t* ko, uint16_t* vo, int h, int hk, int dim, float eps, bool rounded) {
    __shared__ float part[8];
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int d = threadIdx.x;
    if (head >= h + hk) {
        const long long at = (static_cast<long long>(row) * hk + (head - h - hk)) * dim + d;
        vo[at] = v[at];
        return;
    }
    const bool is_q = head < h;
    const uint16_t* src = is_q ? q : k;
    uint16_t* dst = is_q ? qo : ko;
    const float* w = is_q ? wq : wk;
    const long long base = (static_cast<long long>(row) * (is_q ? h : hk) + (is_q ? head : head - h)) * dim;
    const float x = bf16_to_float(src[base + d]);
    const float total = lo_block_sum<4>(x * x, part);
    const float r = 1.0f / sqrtf(total / static_cast<float>(dim) + eps);
    const int half = dim / 2;
    const bool lower = d < half;
    const int pd = lower ? d + half : d - half;
    const float n = lo_bf(w[d] * lo_bf(x * r));
    const float np = lo_bf(w[pd] * lo_bf(bf16_to_float(src[base + pd]) * r));
    const float a = static_cast<float>(pos[row]) * inv[lower ? d : pd];
    const float c = lo_bf(cosf(a));
    const float s = lo_bf(sinf(a));
    dst[base + d] = float_to_bf16(lo_turn(n, np, c, s, lower, rounded));
}

// Windowed attention, one thread per query, per-key FP32 online softmax (arXiv 1805.02867); control ignores the window.
__device__ __forceinline__ void lo_attention(const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out,
                                             int total, int heads, int start, int count, float scale, bool leak) {
    constexpr int D = 64;
    __shared__ float keys[64][D];
    __shared__ float values[64][D];
    const int head = blockIdx.y;
    const int t = threadIdx.x;
    const int row = start + blockIdx.x * 64 + t;
    const bool live = row < start + count;
    const int lo = leak ? 0 : start;
    const int hi = leak ? total : start + count;
    const long long plane = static_cast<long long>(head) * total * D;
    float qr[D];
    for (int d = 0; d < D; ++d) qr[d] = live ? bf16_to_float(q[plane + static_cast<long long>(row) * D + d]) : 0.0f;
    float acc[D];
    for (int d = 0; d < D; ++d) acc[d] = 0.0f;
    float top = -INFINITY;
    float sum = 0.0f;
    for (int tile = lo; tile < hi; tile += 64) {
        const int n = min(64, hi - tile);
        __syncthreads();
        for (int e = t; e < 64 * D; e += 64) {
            const int j = e / D;
            const long long at = plane + static_cast<long long>(tile + j) * D + e % D;
            keys[j][e % D] = j < n ? bf16_to_float(k[at]) : 0.0f;
            values[j][e % D] = j < n ? bf16_to_float(v[at]) : 0.0f;
        }
        __syncthreads();
        for (int j = 0; j < n; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += qr[d] * keys[j][d];
            const float score = dot * scale;
            const float next = fmaxf(top, score);
            if (next != top) {
                const float shrink = expf(top - next);
                sum *= shrink;
                for (int d = 0; d < D; ++d) acc[d] *= shrink;
                top = next;
            }
            const float e = expf(score - top);
            sum += e;
            for (int d = 0; d < D; ++d) acc[d] += e * values[j][d];
        }
    }
    if (!live) return;
    const long long o = (static_cast<long long>(row) * heads + head) * D;
    for (int d = 0; d < D; ++d) out[o + d] = float_to_bf16(acc[d] / sum);
}

__device__ __forceinline__ float lo_gelu_erf(float x) { return x * (1.0f + erff(x * static_cast<float>(M_SQRT1_2))) * 0.5f; }

__device__ __forceinline__ float lo_gelu_tanh(float x) {
    const float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(u));
}

}  // namespace

#define LO_TABLES_ARGS const float* inv, uint16_t* cs, uint16_t* sn, int gh, int gw, int dim
#define LO_ROTARY_ARGS const uint16_t* q, const uint16_t* k, const uint16_t* v, \
    const uint16_t* cs, const uint16_t* sn, uint16_t* qo, uint16_t* ko, \
    uint16_t* vo, int rows, int heads, int dim
#define LO_MERGE_ARGS const uint16_t* x, uint16_t* out, int gh, int gw, int width
#define LO_RMS_ARGS const uint16_t* x, const float* weight, uint16_t* out, int width, float eps
#define LO_TEXT_ARGS const uint16_t* q, const uint16_t* k, const uint16_t* v, const float* wq, const float* wk, \
    const float* inv, const int* pos, uint16_t* qo, uint16_t* ko, uint16_t* vo, int h, int hk, int dim, float eps
#define LO_ATTN_ARGS const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out, \
    int total, int heads, int start, int count, float scale

extern "C" __global__ __launch_bounds__(256) void tf_lo_tables(LO_TABLES_ARGS) { lo_tables(inv, cs, sn, gh, gw, dim, false); }
extern "C" __global__ __launch_bounds__(256) void control_lo_tables_axes(LO_TABLES_ARGS) { lo_tables(inv, cs, sn, gh, gw, dim, true); }

extern "C" __global__ __launch_bounds__(256) void tf_lo_rotary(LO_ROTARY_ARGS) {
    lo_rotary(q, k, v, cs, sn, qo, ko, vo, rows, heads, dim, true);
}
extern "C" __global__ __launch_bounds__(256) void control_lo_rotary_unrounded(LO_ROTARY_ARGS) {
    lo_rotary(q, k, v, cs, sn, qo, ko, vo, rows, heads, dim, false);
}

extern "C" __global__ __launch_bounds__(256) void tf_lo_merge(LO_MERGE_ARGS) { lo_merge(x, out, gh, gw, width, true); }
extern "C" __global__ __launch_bounds__(256) void control_lo_merge_patchmajor(LO_MERGE_ARGS) { lo_merge(x, out, gh, gw, width, false); }

extern "C" __global__ __launch_bounds__(256) void tf_lo_gelu(const uint16_t* x, uint16_t* out, int64_t n) {
    const long long i = gid();
    if (i < n) out[i] = float_to_bf16(lo_gelu_erf(bf16_to_float(x[i])));
}
extern "C" __global__ __launch_bounds__(256) void control_lo_gelu_tanh(const uint16_t* x, uint16_t* out, int64_t n) {
    const long long i = gid();
    if (i < n) out[i] = float_to_bf16(lo_gelu_tanh(bf16_to_float(x[i])));
}

extern "C" __global__ __launch_bounds__(256) void tf_lo_rms(LO_RMS_ARGS) { lo_rms(x, weight, out, width, eps, true); }
extern "C" __global__ __launch_bounds__(256) void control_lo_rms_unrounded(LO_RMS_ARGS) { lo_rms(x, weight, out, width, eps, false); }

extern "C" __global__ __launch_bounds__(128) void tf_lo_text_qk(LO_TEXT_ARGS) {
    lo_text_qk(q, k, v, wq, wk, inv, pos, qo, ko, vo, h, hk, dim, eps, true);
}
extern "C" __global__ __launch_bounds__(128) void control_lo_text_qk_unrounded(LO_TEXT_ARGS) {
    lo_text_qk(q, k, v, wq, wk, inv, pos, qo, ko, vo, h, hk, dim, eps, false);
}

// [rows][heads][dim] BF16 to [heads][rows][dim] FP32, and back with BF16 rounding.
extern "C" __global__ __launch_bounds__(256) void tf_lo_rows_to_heads(const uint16_t* src, float* dst, int rows, int heads, int dim) {
    const long long i = gid();
    if (i >= static_cast<long long>(rows) * heads * dim) return;
    const long long col = i % dim;
    const long long row = (i / dim) % rows;
    const long long head = i / (static_cast<long long>(rows) * dim);
    dst[i] = bf16_to_float(src[(row * heads + head) * dim + col]);
}
extern "C" __global__ __launch_bounds__(256) void tf_lo_heads_to_rows(const float* src, uint16_t* dst, int rows, int heads, int dim) {
    const long long i = gid();
    if (i >= static_cast<long long>(rows) * heads * dim) return;
    const long long col = i % dim;
    const long long head = (i / dim) % heads;
    const long long row = i / (static_cast<long long>(heads) * dim);
    dst[i] = float_to_bf16(src[(head * rows + row) * dim + col]);
}

extern "C" __global__ __launch_bounds__(64) void tf_lo_attention64(LO_ATTN_ARGS) {
    lo_attention(q, k, v, out, total, heads, start, count, scale, false);
}
extern "C" __global__ __launch_bounds__(64) void control_lo_attention_leak(LO_ATTN_ARGS) {
    lo_attention(q, k, v, out, total, heads, start, count, scale, true);
}
