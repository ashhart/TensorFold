// Qwen3.5 vision with BF16 operation boundaries and FP32 matrix, normalization and attention accumulators.
#include "ops/common.hpp"
#include <math.h>

__device__ __forceinline__ float qvh_rnd(float x) { return bf16_to_float(float_to_bf16(x)); }

extern "C" __global__ __launch_bounds__(32) void tf_qvh_linear(
    const uint16_t* x, const float* w, float* out, int rows, int n, int k) {
    const int col = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
    float sum = 0;
    for (int d = lane; d < k; d += 32) sum += bf16_to_float(x[(long long)row * k + d]) * w[(long long)col * k + d];
    for (int mask = 16; mask; mask >>= 1) sum += __shfl_down(sum, mask, 32);
    if (lane == 0) out[(long long)row * n + col] = sum;
}

extern "C" __global__ __launch_bounds__(256) void tf_qvh_embed(
    const float* acc, const float* bias, const float* pos, const int* index, const float* weight, uint16_t* out, int width) {
    const long long row = blockIdx.x;
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        float position[4];
        for (int t = 0; t < 4; ++t)
            position[t] = qvh_rnd(pos[(long long)index[row * 4 + t] * width + d] * qvh_rnd(weight[row * 4 + t]));
        const float p = qvh_rnd(qvh_rnd(qvh_rnd(position[0] + position[1]) + position[2]) + position[3]);
        out[row * width + d] = float_to_bf16(qvh_rnd(acc[row * width + d] + bias[d]) + p);
    }
}

extern "C" __global__ __launch_bounds__(256) void tf_qvh_layernorm(
    const uint16_t* x, const float* w, const float* b, uint16_t* out, int width, float eps) {
    __shared__ float part[8];
    __shared__ float stat;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const long long row = blockIdx.x;
    float sum = 0;
    for (int d = threadIdx.x; d < width; d += 256) sum += bf16_to_float(x[row * width + d]);
    for (int m = 16; m; m >>= 1) sum += __shfl_xor(sum, m, 32);
    if (lane == 0) part[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0;
        for (int i = 0; i < 8; ++i) s += part[i];
        stat = s / width;
    }
    __syncthreads();
    const float mean = stat;
    sum = 0;
    for (int d = threadIdx.x; d < width; d += 256) {
        const float v = bf16_to_float(x[row * width + d]) - mean;
        sum += v * v;
    }
    for (int m = 16; m; m >>= 1) sum += __shfl_xor(sum, m, 32);
    if (lane == 0) part[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0;
        for (int i = 0; i < 8; ++i) s += part[i];
        stat = 1.0f / sqrtf(s / width + eps);
    }
    __syncthreads();
    for (int d = threadIdx.x; d < width; d += 256)
        out[row * width + d] = float_to_bf16((bf16_to_float(x[row * width + d]) - mean) * stat * w[d] + b[d]);
}

template<bool Rounded> __device__ __forceinline__ float qvh_bias(float x, float b) {
    return Rounded ? qvh_rnd(x + b) : x + b;
}

template<bool Rounded> __device__ __forceinline__ void qvh_qkv_rope(
    const float* acc, const float* bias, const int* hw, const float* inv,
    uint16_t* q, uint16_t* k, uint16_t* v, int rows, int heads) {
    const int row = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const int width = heads * 64, half = 32, quarter = 16;
    const int partner = d < half ? d + half : d - half;
    const int j = d % half;
    const float angle = (float)hw[row * 2 + (j >= quarter)] * inv[j % quarter];
    const float cs = cosf(angle), sn = sinf(angle), sign = d < half ? -1.0f : 1.0f;
    const long long src = (long long)row * 3 * width;
    const int c = head * 64 + d, p = head * 64 + partner;
    const long long dst = ((long long)head * rows + row) * 64 + d;
    q[dst] = float_to_bf16(qvh_bias<Rounded>(acc[src + c], bias[c]) * cs + sign * qvh_bias<Rounded>(acc[src + p], bias[p]) * sn);
    k[dst] = float_to_bf16(qvh_bias<Rounded>(acc[src + width + c], bias[width + c]) * cs +
                         sign * qvh_bias<Rounded>(acc[src + width + p], bias[width + p]) * sn);
    v[dst] = float_to_bf16(acc[src + 2 * width + c] + bias[2 * width + c]);
}

#define QVH_QKV(Name, Rounded) \
extern "C" __global__ __launch_bounds__(64) void Name(const float* acc, const float* bias, const int* hw, const float* inv, \
    uint16_t* q, uint16_t* k, uint16_t* v, int rows, int heads) { qvh_qkv_rope<Rounded>(acc,bias,hw,inv,q,k,v,rows,heads); }
QVH_QKV(tf_qvh_qkv_rope, true)
QVH_QKV(tf_ctl_qvh_qkv_unrounded, false)
#undef QVH_QKV

__device__ __forceinline__ float qvh_score(const float* qr, const float* key, float scale) {
    float sum = 0;
    for (int d = 0; d < 64; ++d) sum += qr[d] * key[d];
    return qvh_rnd(qvh_rnd(sum) * scale);
}

extern "C" __global__ __launch_bounds__(64) void tf_qvh_attention64(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out,
    int rows, int heads, int start, int count, float scale) {
    __shared__ float keys[64][64];
    __shared__ float values[64][64];
    const int head = blockIdx.y, qi = blockIdx.x * 64 + threadIdx.x;
    const bool live = qi < count;
    const long long base = (long long)head * rows + start;
    float query[64], acc[64];
    if (live) for (int d = 0; d < 64; ++d) { query[d] = bf16_to_float(q[(base + qi) * 64 + d]); acc[d] = 0; }
    float maximum = -INFINITY, total = 0;
    bool bad = false;
    for (int pass = 0; pass < 2; ++pass) {
        for (int at = 0; at < count; at += 64) {
            const int n = min(64, count - at);
            __syncthreads();
            for (int i = threadIdx.x; i < n * 64; i += 64) {
                keys[i / 64][i % 64] = bf16_to_float(k[(base + at + i / 64) * 64 + i % 64]);
                if (pass == 1) values[i / 64][i % 64] = bf16_to_float(v[(base + at + i / 64) * 64 + i % 64]);
            }
            __syncthreads();
            if (!live) continue;
            for (int r = 0; r < n; ++r) {
                const float score = qvh_score(query, keys[r], scale);
                bad |= !isfinite(score);
                if (pass == 0) {
                    const float next = fmaxf(maximum, score);
                    total = total * expf(maximum - next) + expf(score - next);
                    maximum = next;
                } else {
                    const float p = qvh_rnd(expf(score - maximum) / total);
                    for (int d = 0; d < 64; ++d) acc[d] += p * values[r][d];
                }
            }
        }
    }
    if (live) for (int d = 0; d < 64; ++d)
        out[((long long)(start + qi) * heads + head) * 64 + d] = float_to_bf16(bad ? NAN : acc[d]);
}

extern "C" __global__ __launch_bounds__(256) void tf_qvh_bias_gelu(
    const float* acc, const float* bias, uint16_t* out, int width, int tanh_form) {
    const long long row = blockIdx.x;
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        const float x = qvh_rnd(acc[row * width + d] + bias[d]);
        const float g = tanh_form ? 0.5f * x * (1 + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)))
                                  : 0.5f * x * (1 + erff(x * 0.7071067811865476f));
        out[row * width + d] = float_to_bf16(g);
    }
}

extern "C" __global__ __launch_bounds__(256) void tf_qvh_bias_residual(
    const float* acc, const float* bias, uint16_t* x, int width) {
    const long long row = blockIdx.x;
    for (int d = threadIdx.x; d < width; d += blockDim.x)
        x[row * width + d] = float_to_bf16(bf16_to_float(x[row * width + d]) + qvh_rnd(acc[row * width + d] + bias[d]));
}

extern "C" __global__ __launch_bounds__(256) void tf_qvh_bias_cast(
    const float* acc, const float* bias, uint16_t* out, int width) {
    const long long row = blockIdx.x;
    for (int d = threadIdx.x; d < width; d += blockDim.x) out[row * width + d] = float_to_bf16(acc[row * width + d] + bias[d]);
}

template<bool Trig, bool Products> __device__ __forceinline__ void qvh_text_rope(
    const void* x, int kind, void* out, int out_kind, long long s_head, long long s_row,
    int rows, int heads, int dim, int rotary, const int* positions, float theta) {
    const long long at = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= (long long)rows * heads * dim) return;
    const int d = at % dim, h = (at / dim) % heads, row = at / ((long long)heads * dim);
    const long long base = ((long long)row * heads + h) * dim;
    float y = load(x, kind, base + d);
    if (d < rotary) {
        const int half = rotary / 2, j = d % half;
        const int axis = j % 3 == 1 && j < 33 ? 1 : (j % 3 == 2 && j < 30 ? 2 : 0);
        const float inverse = 1.0f / powf(theta, (float)j / half);
        const float angle = (float)positions[row * 3 + axis] * inverse;
        const float cs = Trig ? rounded(cosf(angle), kind) : cosf(angle);
        const float sn = Trig ? rounded(sinf(angle), kind) : sinf(angle);
        const float a = load(x, kind, base + j), b = load(x, kind, base + j + half);
        if constexpr (Products)
            y = d < half ? rounded(a * cs, kind) - rounded(b * sn, kind)
                         : rounded(b * cs, kind) + rounded(a * sn, kind);
        else
            y = d < half ? a * cs - b * sn : b * cs + a * sn;
        y = rounded(y, kind);
    }
    store(out, out_kind, (long long)h * s_head + (long long)row * s_row + d, y);
}

#define QVH_TEXT(Name, Trig, Products) \
extern "C" __global__ __launch_bounds__(256) void Name(const void* x, int kind, void* out, int out_kind, \
    long long s_head, long long s_row, int rows, int heads, int dim, int rotary, const int* positions, float theta) { \
    qvh_text_rope<Trig, Products>(x,kind,out,out_kind,s_head,s_row,rows,heads,dim,rotary,positions,theta); }
QVH_TEXT(tf_qvh_text_rope, true, true)
QVH_TEXT(tf_ctl_qvh_text_trig_unrounded, false, true)
QVH_TEXT(tf_ctl_qvh_text_products_unrounded, true, false)
#undef QVH_TEXT
