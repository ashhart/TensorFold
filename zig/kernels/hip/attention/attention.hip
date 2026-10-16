// Causal attention, fp32 scores and softmax; KV heads are read as head / (heads / kv_heads), never repeated.

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "attention_fa.hip"

namespace tf {
namespace rocm {

__device__ float lane0_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x += __shfl_xor(x, mask, 32);
    return __shfl(x, 0, 32);
}

template <typename Cache>
__device__ float row_dot(const float* qreg, const Cache* row, int d, int lane) {
    float partial = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = lane + i * 32;
        if (idx < d) partial += qreg[i] * static_cast<float>(row[idx]);
    }
    return lane0_sum(partial);
}

template <typename Cache>
__global__ void __launch_bounds__(32)
causal_kernel(const float* __restrict__ q, const Cache* __restrict__ k, const Cache* __restrict__ v,
              float* __restrict__ out, int qlen, int span, int heads, int kv_heads, int d, float scale, int q_pos0,
              long long k_sb, long long k_sh, long long k_ss, long long v_sb, long long v_sh, long long v_ss) {
    int qi = blockIdx.x;
    int head = blockIdx.y;
    int batch = blockIdx.z;
    if (qi >= qlen) return;
    int lane = threadIdx.x;
    int kv = head / (heads / kv_heads);
    int visible = q_pos0 + qi + 1;
    if (visible > span) visible = span;

    const float* qrow = q + ((static_cast<long long>(batch) * heads + head) * qlen + qi) * d;
    float qreg[8] = {};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = lane + i * 32;
        if (idx < d) qreg[i] = qrow[idx];
    }

    const Cache* kbase = k + batch * k_sb + kv * k_sh;
    const Cache* vbase = v + batch * v_sb + kv * v_sh;
    float max_score = -INFINITY;
    for (int key = 0; key < visible; ++key) {
        float score = row_dot(qreg, kbase + key * k_ss, d, lane) * scale;
        max_score = fmaxf(max_score, score);
    }
    float denom = 0.f;
    for (int key = 0; key < visible; ++key) {
        float score = row_dot(qreg, kbase + key * k_ss, d, lane) * scale;
        denom += expf(score - max_score);
    }
    float acc[8] = {};
    for (int key = 0; key < visible; ++key) {
        float score = row_dot(qreg, kbase + key * k_ss, d, lane) * scale;
        float weight = expf(score - max_score) / denom;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = lane + i * 32;
            if (idx < d) acc[i] += weight * static_cast<float>((vbase + key * v_ss)[idx]);
        }
    }
    float* orow = out + ((static_cast<long long>(batch) * heads + head) * qlen + qi) * d;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = lane + i * 32;
        if (idx < d) orow[idx] = acc[i];
    }
}

}  // namespace rocm
}  // namespace tf

// Decode: a warp scores a key, a block per head reduces the softmax, then V streams in 128-key tiles.
template <typename Cache>
__global__ void score_keys(const float* __restrict__ q, const Cache* __restrict__ k, float* __restrict__ scores,
                           int span, int heads, int kv_heads, int d, float scale, int visible, long long k_sb,
                           long long k_sh, long long k_ss, const int* pos) {
    if (pos != nullptr) visible = min(pos[blockIdx.z] + 1, span);
    int lane = threadIdx.x & 31;
    int key = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int head = blockIdx.y;
    int batch = blockIdx.z;
    const float* qrow = q + (static_cast<long long>(batch) * heads + head) * d;
    float qreg[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = lane + i * 32;
        qreg[i] = idx < d ? qrow[idx] : 0.f;
    }
    float partial = 0.f;
    if (key < visible) {
        int kv = head / (heads / kv_heads);
        const Cache* krow = k + batch * k_sb + static_cast<long long>(kv) * k_sh + static_cast<long long>(key) * k_ss;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = lane + i * 32;
            if (idx < d) partial += qreg[i] * static_cast<float>(krow[idx]);
        }
    }
    float dot = tf::rocm::lane0_sum(partial);
    if (lane == 0 && key < span) {
        float score = key < visible ? dot * scale : -INFINITY;
        scores[(static_cast<long long>(batch) * heads + head) * span + key] = score;
    }
}

extern "C" __global__ void tf_softmax_stats(const float* __restrict__ scores, float* __restrict__ stats, int span,
                                            int visible, const int* pos) {
    if (pos != nullptr) visible = min(pos[blockIdx.y] + 1, span);
    int head = blockIdx.x;
    int batch = blockIdx.y;
    const float* row = scores + (static_cast<long long>(batch) * gridDim.x + head) * span;
    float local = -INFINITY;
    for (int i = threadIdx.x; i < visible; i += blockDim.x) local = fmaxf(local, row[i]);
    __shared__ float buf[256];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (threadIdx.x < step) buf[threadIdx.x] = fmaxf(buf[threadIdx.x], buf[threadIdx.x + step]);
        __syncthreads();
    }
    float max_score = buf[0];
    __syncthreads();
    local = 0.f;
    for (int i = threadIdx.x; i < visible; i += blockDim.x) local += expf(row[i] - max_score);
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (threadIdx.x < step) buf[threadIdx.x] += buf[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float* dest = stats + (static_cast<long long>(batch) * gridDim.x + head) * 2;
        dest[0] = max_score;
        dest[1] = buf[0];
    }
}

template <typename Cache>
__global__ void apply_values(const float* __restrict__ scores, const float* __restrict__ stats,
                             const Cache* __restrict__ v, float* __restrict__ partials, int span, int heads,
                             int kv_heads, int d, int visible, long long v_sb, long long v_sh, long long v_ss,
                             const int* pos) {
    if (pos != nullptr) visible = min(pos[blockIdx.z] + 1, span);
    int tile = blockIdx.x;
    int head = blockIdx.y;
    int batch = blockIdx.z;
    int lane = threadIdx.x;
    int key0 = tile * 128;
    if (lane >= d) return;
    if (key0 >= visible) {  // a tile past the keys in use (the graph's grid covers the whole cache) adds zero
        partials[(((static_cast<long long>(batch) * heads + head) * gridDim.x) + tile) * d + lane] = 0.f;
        return;
    }
    int kv = head / (heads / kv_heads);
    const float* row = scores + (static_cast<long long>(batch) * heads + head) * span;
    float max_score = stats[(static_cast<long long>(batch) * heads + head) * 2];
    float denom = stats[(static_cast<long long>(batch) * heads + head) * 2 + 1];
    const Cache* base = v + batch * v_sb + static_cast<long long>(kv) * v_sh + lane;
    float acc = 0.f;
    int stop = key0 + 128;
    if (stop > visible) stop = visible;
    for (int key = key0; key < stop; ++key) {
        float weight = expf(row[key] - max_score) / denom;
        acc += weight * static_cast<float>(base[static_cast<long long>(key) * v_ss]);
    }
    int tiles = gridDim.x;
    partials[(((static_cast<long long>(batch) * heads + head) * tiles) + tile) * d + lane] = acc;
}

extern "C" __global__ void tf_sum_partials(const float* __restrict__ partials, float* __restrict__ out, int tiles,
                                           int d) {
    int head = blockIdx.x;
    int batch = blockIdx.y;
    int lane = threadIdx.x;
    if (lane >= d) return;
    const float* col = partials + ((static_cast<long long>(batch) * gridDim.x + head) * tiles) * d + lane;
    float acc = 0.f;
    for (int tile = 0; tile < tiles; ++tile) acc += col[static_cast<long long>(tile) * d];
    out[(static_cast<long long>(batch) * gridDim.x + head) * d + lane] = acc;
}

// The template kernels launches.zig resolves.
template __global__ void tf::rocm::causal_kernel<__half>(const float* __restrict__ q, const __half* __restrict__ k,
    const __half* __restrict__ v, float* __restrict__ out, int qlen, int span, int heads, int kv_heads, int d,
    float scale, int q_pos0, long long k_sb, long long k_sh, long long k_ss, long long v_sb, long long v_sh,
    long long v_ss);
template __global__ void tf::rocm::causal_kernel<hip_bfloat16>(const float* __restrict__ q,
    const hip_bfloat16* __restrict__ k, const hip_bfloat16* __restrict__ v, float* __restrict__ out, int qlen, int span,
    int heads, int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh, long long k_ss,
    long long v_sb, long long v_sh, long long v_ss);
template __global__ void tf::rocm::causal_kernel<float>(const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ out, int qlen, int span, int heads, int kv_heads, int d,
    float scale, int q_pos0, long long k_sb, long long k_sh, long long k_ss, long long v_sb, long long v_sh,
    long long v_ss);
template __global__ void score_keys<__half>(const float* __restrict__ q, const __half* __restrict__ k,
    float* __restrict__ scores, int span, int heads, int kv_heads, int d, float scale, int visible, long long k_sb,
    long long k_sh, long long k_ss, const int* pos);
template __global__ void score_keys<hip_bfloat16>(const float* __restrict__ q, const hip_bfloat16* __restrict__ k,
    float* __restrict__ scores, int span, int heads, int kv_heads, int d, float scale, int visible, long long k_sb,
    long long k_sh, long long k_ss, const int* pos);
template __global__ void score_keys<float>(const float* __restrict__ q, const float* __restrict__ k,
    float* __restrict__ scores, int span, int heads, int kv_heads, int d, float scale, int visible, long long k_sb,
    long long k_sh, long long k_ss, const int* pos);
template __global__ void apply_values<__half>(const float* __restrict__ scores, const float* __restrict__ stats,
    const __half* __restrict__ v, float* __restrict__ partials, int span, int heads, int kv_heads, int d, int visible,
    long long v_sb, long long v_sh, long long v_ss, const int* pos);
template __global__ void apply_values<hip_bfloat16>(const float* __restrict__ scores, const float* __restrict__ stats,
    const hip_bfloat16* __restrict__ v, float* __restrict__ partials, int span, int heads, int kv_heads, int d,
    int visible, long long v_sb, long long v_sh, long long v_ss, const int* pos);
template __global__ void apply_values<float>(const float* __restrict__ scores, const float* __restrict__ stats,
    const float* __restrict__ v, float* __restrict__ partials, int span, int heads, int kv_heads, int d, int visible,
    long long v_sb, long long v_sh, long long v_ss, const int* pos);
template __global__ void tf::rocm::fa_prefill<__half, 64>(const float* __restrict__ q, const __half* __restrict__ k,
    const __half* __restrict__ v, float* __restrict__ out, int qlen, int span, int heads, int kv_heads, int d,
    float scale, int q_pos0, long long k_sb, long long k_sh, long long k_ss, long long v_sb, long long v_sh,
    long long v_ss);
template __global__ void tf::rocm::fa_prefill<hip_bfloat16, 64>(const float* __restrict__ q,
    const hip_bfloat16* __restrict__ k, const hip_bfloat16* __restrict__ v, float* __restrict__ out, int qlen, int span,
    int heads, int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh, long long k_ss,
    long long v_sb, long long v_sh, long long v_ss);
template __global__ void tf::rocm::fa_prefill<float, 32>(const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ out, int qlen, int span, int heads, int kv_heads, int d,
    float scale, int q_pos0, long long k_sb, long long k_sh, long long k_ss, long long v_sb, long long v_sh,
    long long v_ss);
