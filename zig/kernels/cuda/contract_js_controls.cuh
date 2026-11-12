// The joint schema head's planted controls: each breaks one tf_js_* kernel in a way its named gate must catch.
#pragma once

#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

namespace tf_ctl_js {

__device__ __forceinline__ long long gid() { return static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; }

__device__ __forceinline__ float x16(const uint16_t* x, long long i) { return __uint_as_float(uint32_t(x[i]) << 16); }

__device__ __forceinline__ uint16_t bf(float x) {
    const uint32_t u = __float_as_uint(x);
    return x != x ? 0x7fc0 : uint16_t((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

__device__ __forceinline__ float r16(float x) { return __uint_as_float(uint32_t(bf(x)) << 16); }

}  // namespace tf_ctl_js

// Gate 4: tf_js_layer_norm with both sums rounded to BF16 at every add, while its spec declares FP32 sums.
extern "C" __global__ void tf_ctl_js_norm_bf16(const uint16_t* x, const uint16_t* w, const uint16_t* b, uint16_t* y,
                                               int dim, float eps) {
    using namespace tf_ctl_js;
    if (threadIdx.x != 0) return;
    const long long at = static_cast<long long>(blockIdx.x) * dim;
    float s = 0.0f, q = 0.0f;
    for (int i = 0; i < dim; ++i) s = r16(s + x16(x, at + i));
    const float mean = s / float(dim);
    for (int i = 0; i < dim; ++i) {
        const float d = x16(x, at + i) - mean;
        q = r16(q + d * d);
    }
    const float inv = 1.0f / sqrtf(q / float(dim) + eps);
    for (int i = 0; i < dim; ++i) y[at + i] = bf((x16(x, at + i) - mean) * inv * x16(w, i) + x16(b, i));
}

// Gate 1: tf_js_softmax_rows that skips non-finite scores as masked, so a row holding NaN or infinity stays finite.
extern "C" __global__ void tf_ctl_js_softmax_safe(float* s, int cols) {
    if (threadIdx.x != 0) return;
    float* row = s + static_cast<long long>(blockIdx.x) * cols;
    float m = -INFINITY, t = 0.0f;
    for (int i = 0; i < cols; ++i)
        if (isfinite(row[i])) m = fmaxf(m, row[i]);
    for (int i = 0; i < cols; ++i) t += isfinite(row[i]) ? expf(row[i] - m) : 0.0f;
    for (int i = 0; i < cols; ++i) row[i] = isfinite(row[i]) ? expf(row[i] - m) / t : 0.0f;
}

// Gate 3: tf_js_combine without its three intermediate roundings, so its logits miss the head's BF16 points.
extern "C" __global__ void tf_ctl_js_combine_unrounded(const float* prior, const float* cosine, const uint16_t* res,
                                                       int n, float scale, float gate, uint16_t* out) {
    using namespace tf_ctl_js;
    const long long i = gid();
    if (i < n) out[i] = bf(prior[i] + gate * (scale * cosine[i] + x16(res, i)));
}

// Gate 3: GELU left as the identity.
extern "C" __global__ void tf_ctl_js_gelu_identity(uint16_t* x, long long n) {
    const long long i = tf_ctl_js::gid();
    if (i < n) x[i] = x[i];
}

// Gate 3: BF16 bits widened as FP16.
extern "C" __global__ void tf_ctl_js_widen_half(const uint16_t* x, float* y, long long n) {
    const long long i = tf_ctl_js::gid();
    if (i < n) y[i] = __half2float(reinterpret_cast<const __half*>(x)[i]);
}

// Gate 3: FP32 to BF16 by truncation, not round to nearest even.
extern "C" __global__ void tf_ctl_js_round_truncate(const float* x, uint16_t* y, long long n) {
    const long long i = tf_ctl_js::gid();
    if (i < n) y[i] = __float_as_uint(x[i]) >> 16;
}

// Gate 3: the residual add truncated to BF16.
extern "C" __global__ void tf_ctl_js_add_truncate(const uint16_t* x, const uint16_t* y, uint16_t* out, long long n) {
    using namespace tf_ctl_js;
    const long long i = gid();
    if (i < n) out[i] = __float_as_uint(x16(x, i) + x16(y, i)) >> 16;
}

// Gate 3: every broadcast value from the row's first entry.
extern "C" __global__ void tf_ctl_js_broadcast_first(const uint16_t* row, uint16_t* out, int n, long long total) {
    const long long i = tf_ctl_js::gid();
    if (i < total) out[i] = row[0];
}

// Gate 3: each span's last row in the piece left out of its sum.
extern "C" __global__ void tf_ctl_js_span_short(const uint16_t* rows, int first, int count, int dim, const int* spans,
                                                float* sums) {
    using namespace tf_ctl_js;
    const int d = blockIdx.x * 256 + threadIdx.x, s = blockIdx.y;
    if (d >= dim) return;
    const int lo = max(spans[2 * s], first), hi = min(spans[2 * s + 1], first + count);
    float& out = sums[static_cast<long long>(s) * dim + d];
    for (int p = lo; p < hi - 1; ++p) out += x16(rows, static_cast<long long>(p - first) * dim + d);
}

// Gate 3: the packed lm_head read as if it were row-major.
extern "C" __global__ void tf_ctl_js_lex_rowmajor(const uint16_t* lane, const int* ids, int dim, const int* spans,
                                                  float* sums) {
    using namespace tf_ctl_js;
    const int d = blockIdx.x * 256 + threadIdx.x, s = blockIdx.y;
    if (d >= dim) return;
    float& out = sums[static_cast<long long>(s) * dim + d];
    for (int p = spans[2 * s]; p < spans[2 * s + 1]; ++p) out += x16(lane, static_cast<long long>(ids[p]) * dim + d);
}

// Gate 3: span means with their divisor left out.
extern "C" __global__ void tf_ctl_js_means_undivided(const float* sums, const int* spans, int dim, uint16_t* out) {
    const int d = blockIdx.x * 256 + threadIdx.x;
    const long long i = static_cast<long long>(blockIdx.y) * dim + d;
    if (d < dim) out[i] = tf_ctl_js::bf(sums[i]);
}

// Gate 3: option summary weights all 1 / n, not the softmax of the scores.
extern "C" __global__ void tf_ctl_js_summary_uniform(const uint16_t* routed, const uint16_t* fields, const int* first,
                                                     int width, float root, float* wts, uint16_t* out) {
    using namespace tf_ctl_js;
    const int q = blockIdx.x, lo = first[q], n = first[q + 1] - lo;
    for (int i = threadIdx.x; i < width; i += 256) {
        float s = 0.0f;
        for (int j = 0; j < n; ++j) s += r16(r16(1.0f / n) * x16(routed, static_cast<long long>(lo + j) * width + i));
        out[static_cast<long long>(q) * width + i] = bf(s);
    }
}

// Gate 3: every option's logits outputs copied from the first option's.
extern "C" __global__ void tf_ctl_js_logits_first(float* prior, float* cosine, uint16_t* feat, int width, int n) {
    const long long i = tf_ctl_js::gid();
    if (i > 0 && i < n) {
        prior[i] = prior[0];
        cosine[i] = cosine[0];
    }
    if (i >= 4LL * width && i < 4LL * width * n) feat[i] = feat[i % (4LL * width)];
}
