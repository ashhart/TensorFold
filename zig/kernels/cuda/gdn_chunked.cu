// Chunkwise delta recurrence: BF16 input products, explicit FP32 state arithmetic, and serial exceptional heads.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "common/qmm_frag.cuh"
#include "common/gdn_chunk_layout.cuh"
#include "common/gdn_chunk_serial.cuh"

using namespace tf_gdn_chunk;

extern "C" __global__ __launch_bounds__(256) void tf_gdn_chunk_classify(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const float* decay, const __nv_bfloat16* beta, const float* state,
    unsigned* fallback, int rows, int hk, int hv) {
    __shared__ unsigned bad;
    if (threadIdx.x == 0) bad = 0;
    __syncthreads();
    const int head = blockIdx.x, qhead = head / (hv / hk);
    unsigned local = 0;
    for (int i = threadIdx.x; i < rows * Width; i += blockDim.x) {
        const int row = i / Width, column = i % Width;
        const float qv = __bfloat162float(q[(static_cast<size_t>(row) * hk + qhead) * Width + column]);
        const float kv = __bfloat162float(k[(static_cast<size_t>(row) * hk + qhead) * Width + column]);
        const float vv = __bfloat162float(v[(static_cast<size_t>(row) * hv + head) * Width + column]);
        local |= !isfinite(qv) || !isfinite(kv) || !isfinite(vv) || fabsf(qv) > 1.0f || fabsf(kv) > 1.0f || fabsf(vv) > 1e12f;
        local |= (qv != 0 && fabsf(qv) < 0x1p-126f) || (kv != 0 && fabsf(kv) < 0x1p-126f);
    }
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        const float g = decay[static_cast<size_t>(row) * hv + head];
        const float b = __bfloat162float(beta[static_cast<size_t>(row) * hv + head]);
        local |= !isfinite(g) || !isfinite(b) || g < 0 || g > 1 || b < 0 || b > 1;
    }
    for (int i = threadIdx.x; i < Width * Width; i += blockDim.x) {
        const float value = state[static_cast<size_t>(head) * Width * Width + i];
        local |= !isfinite(value) || fabsf(value) > 1e12f;
    }
    if (local) atomicOr(&bad, local);
    __syncthreads();
    if (threadIdx.x == 0) fallback[head] = bad;
}

extern "C" __global__ __launch_bounds__(128) void tf_gdn_chunk_scores(
    const __nv_bfloat16* q, const __nv_bfloat16* k, Scores* out, int rows, int hk) {
    __shared__ __align__(16) __nv_bfloat16 sq[Chunk][Width], sk[Chunk][Width];
    const int chunk = blockIdx.x, head = blockIdx.y, begin = chunk * Chunk;
    for (int i = threadIdx.x; i < Chunk * Width; i += blockDim.x) {
        const int row = begin + i / Width, key = i % Width;
        sq[i / Width][key] = row < rows ? q[(static_cast<size_t>(row) * hk + head) * Width + key] : __float2bfloat16_rn(0);
        sk[i / Width][key] = row < rows ? k[(static_cast<size_t>(row) * hk + head) * Width + key] : __float2bfloat16_rn(0);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int group = lane >> 2, pair = (lane & 3) * 2, column = (warp & 1) * 8 + group;
    float sum[4] = {};
    const __nv_bfloat16* a = warp < 2 ? &sk[0][0] : &sq[0][0];
#pragma unroll
    for (int key = 0; key < Width; key += 16) {
        uint32_t av[4];
        av[0] = *reinterpret_cast<const uint32_t*>(a + group * Width + key + pair);
        av[1] = *reinterpret_cast<const uint32_t*>(a + (group + 8) * Width + key + pair);
        av[2] = *reinterpret_cast<const uint32_t*>(a + group * Width + key + pair + 8);
        av[3] = *reinterpret_cast<const uint32_t*>(a + (group + 8) * Width + key + pair + 8);
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sk[column][key + pair]);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sk[column][key + pair + 8]);
        qmm_frag::mma(sum, av, b0, b1);
    }
    Scores* result = out + static_cast<size_t>(chunk) * hk + head;
    float* matrix = warp < 2 ? result->kk : result->qk;
    const int c0 = (warp & 1) * 8 + pair;
    matrix[group * Chunk + c0] = sum[0];
    matrix[group * Chunk + c0 + 1] = sum[1];
    matrix[(group + 8) * Chunk + c0] = sum[2];
    matrix[(group + 8) * Chunk + c0 + 1] = sum[3];
}

extern "C" __global__ __launch_bounds__(32) void tf_gdn_chunk_coefficients(
    const Scores* scores, const float* decay, const __nv_bfloat16* beta,
    const unsigned* fallback, Coefficients* out, int rows, int hk, int hv) {
    const int chunk = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    if (fallback[head]) return;
    __shared__ float g[Chunk], b[Chunk], lower[Chunk * Chunk];
    const int begin = chunk * Chunk, count = min(Chunk, rows - begin);
    Coefficients* result = out + static_cast<size_t>(chunk) * hv + head;
    const Scores* input = scores + static_cast<size_t>(chunk) * hk + head / (hv / hk);
    if (tid < Chunk) {
        g[tid] = tid < count ? decay[static_cast<size_t>(begin + tid) * hv + head] : 1;
        b[tid] = tid < count ? __bfloat162float(beta[static_cast<size_t>(begin + tid) * hv + head]) : 0;
    }
    __syncwarp();
    for (int at = tid; at < Chunk * Chunk; at += 32) {
        const int row = at / Chunk, column = at % Chunk;
        float factor = 1;
        for (int t = column + 1; t <= row; ++t) factor = __fmul_rn(factor, g[t]);
        lower[at] = row < count && column < row ? __fmul_rn(__fmul_rn(b[row], factor), input->kk[at]) : 0;
        result->attention[at] = row < count && column <= row ? __fmul_rn(factor, input->qk[at]) : 0;
        result->solve_beta[at] = 0;
    }
    __syncwarp();
    if (tid < Chunk) {
        float prefix = 1, suffix = 1;
        for (int t = 0; t <= tid; ++t) prefix = __fmul_rn(prefix, g[t]);
        for (int t = tid + 1; t < count; ++t) suffix = __fmul_rn(suffix, g[t]);
        result->prefix[tid] = prefix;
        result->suffix[tid] = suffix;
        float inverse[Chunk] = {};
        for (int row = tid; row < count; ++row) {
            float value = row == tid ? 1 : 0;
            for (int j = tid; j < row; ++j) value = __fmaf_rn(-lower[row * Chunk + j], inverse[j], value);
            inverse[row] = value;
            result->solve_beta[row * Chunk + tid] = __fmul_rn(value, b[tid]);
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 2) void tf_gdn_chunk_scan(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const Coefficients* coefficients, const unsigned* fallback, const float* initial,
    float* last, __nv_bfloat16* out, int rows, int hk, int hv) {
    const int head = blockIdx.x, first_value = blockIdx.y * Values, tid = threadIdx.x;
    if (fallback[head]) return;
    __shared__ ScanShared sm;
    for (int i = tid; i < Values * Width; i += 256)
        sm.state[i / Width][i % Width] = initial[(static_cast<size_t>(head) * Width + first_value + i / Width) * Width + i % Width];
    __syncthreads();
    const int value = tid / Chunk, row = tid % Chunk, qhead = head / (hv / hk);
    for (int begin = 0; begin < rows; begin += Chunk) {
        const int count = min(Chunk, rows - begin);
        for (int i = tid; i < Chunk * Width; i += 256) {
            const int t = i / Width, key = i % Width;
            const size_t at = (static_cast<size_t>(begin + t) * hk + qhead) * Width + key;
            sm.q[key][t] = t < count ? __bfloat162float(q[at]) : 0;
            sm.k[key][t] = t < count ? __bfloat162float(k[at]) : 0;
        }
        const float* source = reinterpret_cast<const float*>(coefficients + static_cast<size_t>(begin / Chunk) * hv + head);
        float* destination = reinterpret_cast<float*>(&sm.coefficients);
        for (int i = tid; i < sizeof(Coefficients) / sizeof(float); i += 256) destination[i] = source[i];
        __syncthreads();
        float prediction[4] = {}, query[4] = {};
#pragma unroll 1
        for (int key = 0; key < Width; key += 4) {
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                prediction[part] = __fmaf_rn(sm.state[value][key + part], sm.k[key + part][row], prediction[part]);
                query[part] = __fmaf_rn(sm.state[value][key + part], sm.q[key + part][row], query[part]);
            }
        }
        const float predicted = __fadd_rn(__fadd_rn(prediction[0], prediction[1]), __fadd_rn(prediction[2], prediction[3]));
        const float prior = __fadd_rn(__fadd_rn(query[0], query[1]), __fadd_rn(query[2], query[3]));
        const size_t at = (static_cast<size_t>(begin + row) * hv + head) * Width + first_value + value;
        sm.delta[value][row] = row < count ? __fsub_rn(__bfloat162float(v[at]), __fmul_rn(sm.coefficients.prefix[row], predicted)) : 0;
        __syncthreads();
        float delta = 0;
        for (int j = 0; j <= row && row < count; ++j)
            delta = __fmaf_rn(sm.coefficients.solve_beta[row * Chunk + j], sm.delta[value][j], delta);
        __syncthreads();
        sm.delta[value][row] = delta;
        __syncthreads();
        if (row < count) {
            float y = __fmul_rn(sm.coefficients.prefix[row], prior);
            for (int j = 0; j <= row; ++j) y = __fmaf_rn(sm.coefficients.attention[row * Chunk + j], sm.delta[value][j], y);
            out[at] = __float2bfloat16_rn(y);
        }
        for (int i = tid; i < Values * Width; i += 256) {
            const int vr = i / Width, key = i % Width;
            float state = __fmul_rn(sm.state[vr][key], sm.coefficients.prefix[count - 1]);
            for (int j = 0; j < count; ++j)
                state = __fmaf_rn(__fmul_rn(sm.coefficients.suffix[j], sm.delta[vr][j]), sm.k[key][j], state);
            sm.state[vr][key] = state;
        }
        __syncthreads();
    }
    for (int i = tid; i < Values * Width; i += 256)
        last[(static_cast<size_t>(head) * Width + first_value + i / Width) * Width + i % Width] = sm.state[i / Width][i % Width];
}

extern "C" __global__ __launch_bounds__(128) void tf_gdn_chunk_fallback(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const float* decay, const __nv_bfloat16* beta, const unsigned* fallback,
    const float* initial, float* last, __nv_bfloat16* out, int rows, int hk, int hv) {
    if (!fallback[blockIdx.x]) return;
    tf_gdn_chunk_serial::run<__nv_bfloat16, __nv_bfloat16, 64, 32>(q, k, v, decay, beta, initial, last, out, rows, hk, hv);
}
