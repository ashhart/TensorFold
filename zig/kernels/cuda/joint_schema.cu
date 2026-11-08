// Clef's joint schema head, HIP's joint_schema.hip on CUDA: the same FP32 sums in order, one rounding at each output.

#include <math.h>
#include <stdint.h>

#include "common/reduce.cuh"

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ long long gid() { return static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; }

__device__ __forceinline__ float bf(uint16_t h) { return __uint_as_float(static_cast<uint32_t>(h) << 16); }

// c10::BFloat16's round to nearest even, NaN to 0x7fc0.
__device__ __forceinline__ uint16_t to_bf16(float f) {
    if (f != f) return 0x7fc0;
    const uint32_t u = __float_as_uint(f);
    return static_cast<uint16_t>((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

__device__ __forceinline__ float r16(float v) { return bf(to_bf16(v)); }

// A block's FP32 sum in one fixed order: each warp's xor tree, then the warps' sums in order (all threads get it).
__device__ float block_total(float v) {
    __shared__ float part[kWarps];
    return tf_common::block_sum_serial(v, part);
}

// A NaN-keeping maximum: fmaxf would drop a NaN where torch's max keeps it.
__device__ __forceinline__ float nan_max(float a, float b) { return (a != a || a > b) ? a : b; }

__device__ float block_max(float v) {
    __shared__ float part[kWarps];
    __shared__ float total;
    for (int o = 16; o > 0; o >>= 1) v = nan_max(v, __shfl_xor_sync(0xffffffffu, v, o, 32));
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = part[0];
        for (int w = 1; w < kWarps; ++w) m = nan_max(m, part[w]);
        total = m;
    }
    __syncthreads();
    const float out = total;
    __syncthreads();
    return out;
}

// Element (r, c) of a BF16 [n, k] weight packed as tf_pack_bf16_lane does: [n/64][k/64][8][4][32][4].
__device__ __forceinline__ long long lane_at(long long r, int c, int k) {
    const int w = c & 15;
    const long long tile = (r >> 6) * (k >> 6) + (c >> 6);
    const int lane = static_cast<int>(r & 7) * 4 + ((w & 7) >> 1);
    const int e = (w & 1) + 2 * (w >> 3);
    return tile * 4096 + ((r & 63) >> 3) * 512 + ((c & 63) >> 4) * 128 + lane * 4 + e;
}

}  // namespace

// y = bf16((x - mean) / sqrt(var + eps) w + b) over each row of `dim`: FP32 mean, then FP32 biased variance.
extern "C" __global__ void __launch_bounds__(kThreads)
    tf_js_layer_norm(const uint16_t* x, const uint16_t* w, const uint16_t* b, uint16_t* y, int dim, float eps) {
    const long long at = static_cast<long long>(blockIdx.x) * dim;
    float s = 0.0f;
    for (int i = threadIdx.x; i < dim; i += kThreads) s += bf(x[at + i]);
    const float mean = block_total(s) / static_cast<float>(dim);
    float q = 0.0f;
    for (int i = threadIdx.x; i < dim; i += kThreads) {
        const float d = bf(x[at + i]) - mean;
        q += d * d;
    }
    const float inv = 1.0f / sqrtf(block_total(q) / static_cast<float>(dim) + eps);
    for (int i = threadIdx.x; i < dim; i += kThreads) y[at + i] = to_bf16((bf(x[at + i]) - mean) * inv * bf(w[i]) + bf(b[i]));
}

// The exact GELU in place: bf16(x / 2 (1 + erf(x / sqrt 2))).
extern "C" __global__ void tf_js_gelu(uint16_t* x, long long n) {
    const long long i = gid();
    if (i >= n) return;
    const float v = bf(x[i]);
    x[i] = to_bf16(0.5f * v * (1.0f + erff(v * 0.70710678118654752f)));
}

// Each row of `cols` FP32 scores to exp(s - max) / sum in place, FP32 throughout.
extern "C" __global__ void __launch_bounds__(kThreads) tf_js_softmax_rows(float* s, int cols) {
    float* row = s + static_cast<long long>(blockIdx.x) * cols;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += kThreads) m = nan_max(row[i], m);
    m = block_max(m);
    float t = 0.0f;
    for (int i = threadIdx.x; i < cols; i += kThreads) t += expf(row[i] - m);
    const float sum = block_total(t);
    for (int i = threadIdx.x; i < cols; i += kThreads) row[i] = expf(row[i] - m) / sum;
}

extern "C" __global__ void tf_js_widen(const uint16_t* x, float* y, long long n) {
    const long long i = gid();
    if (i < n) y[i] = bf(x[i]);
}

extern "C" __global__ void tf_js_round(const float* x, uint16_t* y, long long n) {
    const long long i = gid();
    if (i < n) y[i] = to_bf16(x[i]);
}

// sums[s][d] += span s's rows in this piece in prompt row order (piece row p - first): FP32.
extern "C" __global__ void tf_js_span_sums(const uint16_t* rows, int first, int count, int dim, const int* spans,
                                           float* sums) {
    const int d = blockIdx.x * kThreads + threadIdx.x;
    if (d >= dim) return;
    const int span = blockIdx.y;
    const int lo = max(spans[2 * span], first);
    const int hi = min(spans[2 * span + 1], first + count);
    if (lo >= hi) return;
    float* out = sums + static_cast<long long>(span) * dim + d;
    float acc = *out;
    for (int p = lo; p < hi; ++p) acc += bf(rows[static_cast<long long>(p - first) * dim + d]);
    *out = acc;
}

// sums[s][d] += the packed lm_head rows of span s's prompt tokens ids[p], in prompt order: FP32 (dim % 64 == 0).
extern "C" __global__ void tf_js_lex_sums(const uint16_t* lane, const int* ids, int dim, const int* spans,
                                          float* sums) {
    const int d = blockIdx.x * kThreads + threadIdx.x;
    if (d >= dim) return;
    const int span = blockIdx.y;
    float* out = sums + static_cast<long long>(span) * dim + d;
    float acc = *out;
    for (int p = spans[2 * span]; p < spans[2 * span + 1]; ++p) acc += bf(lane[lane_at(ids[p], d, dim)]);
    *out = acc;
}

// out[s][d] = bf16(sums[s][d] / the span's length): torch's mean, one rounding.
extern "C" __global__ void tf_js_span_means(const float* sums, const int* spans, int dim, uint16_t* out) {
    const int d = blockIdx.x * kThreads + threadIdx.x;
    if (d >= dim) return;
    const long long at = static_cast<long long>(blockIdx.y) * dim + d;
    const float n = static_cast<float>(spans[2 * blockIdx.y + 1] - spans[2 * blockIdx.y]);
    out[at] = to_bf16(sums[at] / n);
}

// Step 5, question q, any number of options: w = bf16(softmax(bf16(bf16(r . f) / root))) in wts, sum bf16(w r).
extern "C" __global__ void __launch_bounds__(kThreads)
    tf_js_option_summary(const uint16_t* routed, const uint16_t* fields, const int* first, int width, float root,
                         float* wts, uint16_t* out) {
    const int q = blockIdx.x;
    const int j0 = first[q];
    const int n = first[q + 1] - j0;
    float* w = wts + j0;
    const uint16_t* f = fields + static_cast<long long>(q) * width;
    for (int j = 0; j < n; ++j) {
        const uint16_t* r = routed + static_cast<long long>(j0 + j) * width;
        float s = 0.0f;
        for (int i = threadIdx.x; i < width; i += kThreads) s += bf(r[i]) * bf(f[i]);
        const float dot = block_total(s);
        if (threadIdx.x == 0) w[j] = r16(r16(dot) / root);
    }
    __syncthreads();
    float m = -INFINITY;
    for (int j = threadIdx.x; j < n; j += kThreads) m = nan_max(w[j], m);
    m = block_max(m);
    float t = 0.0f;
    for (int j = threadIdx.x; j < n; j += kThreads) t += expf(w[j] - m);
    const float sum = block_total(t);
    for (int j = threadIdx.x; j < n; j += kThreads) w[j] = r16(expf(w[j] - m) / sum);
    __syncthreads();
    for (int i = threadIdx.x; i < width; i += kThreads) {
        float acc = 0.0f;
        for (int j = 0; j < n; ++j) acc += r16(w[j] * bf(routed[static_cast<long long>(j0 + j) * width + i]));
        out[static_cast<long long>(q) * width + i] = to_bf16(acc);
    }
}

// `F.normalize`'s divisor: bf16(max(bf16(||x||), eps)) from FP32 squares.
__device__ float norm_divisor(float squares, float eps) { return r16(fmaxf(r16(sqrtf(block_total(squares))), eps)); }

// Step 7 for option j: the prior, the cosine of f_q and o_j (BF16 values in FP32) and the scorer's input row.
extern "C" __global__ void __launch_bounds__(kThreads)
    tf_js_logits(const uint16_t* fields, const uint16_t* opts, const uint16_t* qv, const uint16_t* g,
                 const uint16_t* lex, const int* owner, int width, int hidden, float prior_scale, float* prior,
                 float* cosine, uint16_t* feat) {
    const int j = blockIdx.x;
    const int q = owner[j];
    const uint16_t* qk = qv + static_cast<long long>(q) * hidden;
    const uint16_t* ej = lex + static_cast<long long>(j) * hidden;
    float sa = 0.0f, se = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += kThreads) {
        const float a = r16(bf(qk[i]) + bf(g[i]));
        sa += a * a;
        se += bf(ej[i]) * bf(ej[i]);
    }
    const float da = norm_divisor(sa, 1e-12f);
    const float de = norm_divisor(se, 1e-12f);
    float dot = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += kThreads)
        dot += r16(bf(ej[i]) / de) * r16(r16(bf(qk[i]) + bf(g[i])) / da);
    const float p = r16(prior_scale * r16(block_total(dot)));
    const uint16_t* f = fields + static_cast<long long>(q) * width;
    const uint16_t* o = opts + static_cast<long long>(j) * width;
    float sf = 0.0f, so = 0.0f;
    for (int i = threadIdx.x; i < width; i += kThreads) {
        sf += bf(f[i]) * bf(f[i]);
        so += bf(o[i]) * bf(o[i]);
    }
    const float df = norm_divisor(sf, 1e-8f);
    const float dn = norm_divisor(so, 1e-8f);
    float c = 0.0f;
    uint16_t* out = feat + static_cast<long long>(j) * 4 * width;
    for (int i = threadIdx.x; i < width; i += kThreads) {
        const float x = bf(f[i]), y = bf(o[i]);
        c += r16(r16(x / df) * r16(y / dn));
        out[i] = f[i];
        out[width + i] = o[i];
        out[2 * width + i] = to_bf16(x * y);
        out[3 * width + i] = to_bf16(fabsf(r16(x - y)));
    }
    const float cs = r16(block_total(c));
    if (threadIdx.x == 0) {
        prior[j] = p;
        cosine[j] = cs;
    }
}

// logit_j = bf16(prior + bf16(gate bf16(bf16(js cos) + res))), the scalars already the head's BF16 values.
extern "C" __global__ void tf_js_combine(const float* prior, const float* cosine, const uint16_t* res, int n,
                                         float joint_scale, float gate, uint16_t* logits) {
    const int j = static_cast<int>(gid());
    if (j >= n) return;
    const float joint = r16(r16(joint_scale * cosine[j]) + bf(res[j]));
    logits[j] = to_bf16(prior[j] + r16(gate * joint));
}

// out = bf16(x + y) over n values: the head's residual adds.
extern "C" __global__ void tf_js_add(const uint16_t* x, const uint16_t* y, uint16_t* out, long long n) {
    const long long i = gid();
    if (i < n) out[i] = to_bf16(bf(x[i]) + bf(y[i]));
}

// `rows` copies of one row of n BF16 values (a bias before a product that adds into it, or a broadcast).
extern "C" __global__ void tf_js_broadcast(const uint16_t* row, uint16_t* out, int n, long long total) {
    const long long i = gid();
    if (i < total) out[i] = row[i % n];
}
