// GDN state products on tensor cores with checked three-limb FP32 operands and explicit FP32 accumulation.
#include "gdn_chunked.cu"
#include "common/gdn_state_parts.cuh"

namespace tf_gdn_tensor {
using namespace tf_gdn_state_parts;
template<int V>
__device__ __forceinline__ void run(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const tf_gdn_chunk::Coefficients* coefficients, const unsigned* fallback, const float* initial,
    float* last, __nv_bfloat16* out, int rows, int hk, int hv) {
    constexpr int NT = V / 16, Work = V / 16;
    const int head = blockIdx.x, vfirst = blockIdx.y * V;
    if (fallback[head]) return;
    __shared__ Shared<V> sm;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / 2, wn = warp % 2, group = lane >> 2, pair_index = (lane & 3) * 2;
    const int qhead = head / (hv / hk);
    float state[2][NT][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const int key = wm * 32 + i * 16 + group + h * 8;
                    const int value_column = wn * (V / 2) + j * 8 + pair_index + p;
                    state[i][j][h * 2 + p] = initial[(static_cast<size_t>(head) * 128 + vfirst + value_column) * 128 + key];
                }
    for (int begin = 0; begin < rows; begin += 16) {
        const int count = min(16, rows - begin);
        if (tid == 0) sm.lost = 0;
        for (int index = tid; index < 16 * 128; index += 256) {
            const int t = index / 128, key = index % 128;
            const size_t at = (static_cast<size_t>(begin + t) * hk + qhead) * 128 + key;
            sm.q[t][key] = t < count ? reinterpret_cast<const uint16_t*>(q)[at] : 0;
            sm.k[t][key] = t < count ? reinterpret_cast<const uint16_t*>(k)[at] : 0;
        }
        const float* source = reinterpret_cast<const float*>(coefficients + static_cast<size_t>(begin / 16) * hv + head);
        float* target = reinterpret_cast<float*>(&sm.coefficients);
        for (int index = tid; index < sizeof(sm.coefficients) / 4; index += 256) target[index] = source[index];
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        const int key = wm * 32 + i * 16 + group + h * 8;
                        const int column = wn * (V / 2) + j * 8 + pair_index + p;
                        uint16_t parts[3];
                        if (!split(state[i][j][h * 2 + p], parts)) atomicOr(&sm.lost, 1u);
#pragma unroll
                        for (int limb = 0; limb < 3; ++limb) sm.data.state[limb][column][key] = parts[limb];
                    }
        __syncthreads();
        if (sm.lost) {
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int p = 0; p < 2; ++p) {
                            const int key = wm * 32 + i * 16 + group + h * 8;
                            const int column = wn * (V / 2) + j * 8 + pair_index + p;
                            sm.data.exact_state[column][key] = state[i][j][h * 2 + p];
                        }
            __syncthreads();
            for (int index = tid; index < V * 16; index += 256) {
                const int column = index / 16, t = index % 16;
                float ks[4] = {}, qs[4] = {};
#pragma unroll 1
                for (int key = 0; key < 128; key += 4)
#pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        const float s = sm.data.exact_state[column][key + n];
                        ks[n] = __fmaf_rn(s, tf_gdn_state_parts::value(sm.k[t][key + n]), ks[n]);
                        qs[n] = __fmaf_rn(s, tf_gdn_state_parts::value(sm.q[t][key + n]), qs[n]);
                    }
                const float prediction = __fadd_rn(__fadd_rn(ks[0], ks[1]), __fadd_rn(ks[2], ks[3]));
                sm.history[column][t] = __fadd_rn(__fadd_rn(qs[0], qs[1]), __fadd_rn(qs[2], qs[3]));
                const size_t at = (static_cast<size_t>(begin + t) * hv + head) * 128 + vfirst + column;
                sm.delta[column][t] = t < count ? __fsub_rn(__bfloat162float(v[at]), __fmul_rn(sm.coefficients.prefix[t], prediction)) : 0;
            }
        } else {
            const uint16_t (*input)[136] = warp < 4 ? sm.k : sm.q;
            for (int n = warp % 4; n < V / 8; n += 4) {
                float sums[3][4] = {};
#pragma unroll 1
                for (int key = 0; key < 128; key += 16) {
                    uint32_t a[4] = {
                        *reinterpret_cast<const uint32_t*>(&input[group][key + pair_index]),
                        *reinterpret_cast<const uint32_t*>(&input[group + 8][key + pair_index]),
                        *reinterpret_cast<const uint32_t*>(&input[group][key + pair_index + 8]),
                        *reinterpret_cast<const uint32_t*>(&input[group + 8][key + pair_index + 8]),
                    };
#pragma unroll
                    for (int limb = 0; limb < 3; ++limb) {
                        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sm.data.state[limb][n * 8 + group][key + pair_index]);
                        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sm.data.state[limb][n * 8 + group][key + pair_index + 8]);
                        qmm_frag::mma(sums[limb], a, b0, b1);
                    }
                }
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        const int t = group + h * 8, column = n * 8 + pair_index + p;
                        const float result = combine(sums, h * 2 + p);
                        if (warp < 4) {
                            const size_t at = (static_cast<size_t>(begin + t) * hv + head) * 128 + vfirst + column;
                            sm.delta[column][t] = t < count ? __fsub_rn(__bfloat162float(v[at]), __fmul_rn(sm.coefficients.prefix[t], result)) : 0;
                        } else sm.history[column][t] = result;
                    }
            }
        }
        __syncthreads();
        float correction[Work];
#pragma unroll
        for (int job = 0; job < Work; ++job) {
            const int index = tid + job * 256, column = index / 16, t = index % 16;
            float result = 0;
            for (int j = 0; j <= t && t < count; ++j)
                result = __fmaf_rn(sm.coefficients.solve_beta[t * 16 + j], sm.delta[column][j], result);
            correction[job] = result;
        }
        __syncthreads();
#pragma unroll
        for (int job = 0; job < Work; ++job) sm.delta[(tid + job * 256) / 16][tid % 16] = correction[job];
        __syncthreads();
        for (int index = tid; index < V * 16; index += 256) {
            const int column = index / 16, t = index % 16;
            if (t < count) {
                float y = __fmul_rn(sm.coefficients.prefix[t], sm.history[column][t]);
                for (int j = 0; j <= t; ++j) y = __fmaf_rn(sm.coefficients.attention[t * 16 + j], sm.delta[column][j], y);
                out[(static_cast<size_t>(begin + t) * hv + head) * 128 + vfirst + column] = __float2bfloat16_rn(y);
            }
        }
        __syncthreads();
        if (tid == 0) sm.lost = 0;
        __syncthreads();
        for (int index = tid; index < V * 16; index += 256) {
            const int column = index / 16, t = index % 16;
            const float d = __fmul_rn(sm.delta[column][t], sm.coefficients.suffix[t]);
            uint16_t parts[3];
            if (!split(d, parts)) atomicOr(&sm.lost, 1u);
#pragma unroll
            for (int limb = 0; limb < 3; ++limb) sm.data.scaled_delta[limb][column][t] = parts[limb];
        }
        __syncthreads();
        if (sm.lost) {
            for (int index = tid; index < V * 16; index += 256)
                sm.data.exact_delta[index / 16][index % 16] = __fmul_rn(sm.delta[index / 16][index % 16], sm.coefficients.suffix[index % 16]);
            __syncthreads();
        }
        const float decay = sm.coefficients.prefix[count - 1];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int key_base = wm * 32 + i * 16, value_base = wn * (V / 2) + j * 8;
                if (sm.lost) {
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int p = 0; p < 2; ++p) {
                            const int key = key_base + group + h * 8, column = value_base + pair_index + p;
                            float s = __fmul_rn(state[i][j][h * 2 + p], decay);
                            for (int t = 0; t < count; ++t) s = __fmaf_rn(sm.data.exact_delta[column][t], tf_gdn_state_parts::value(sm.k[t][key]), s);
                            state[i][j][h * 2 + p] = s;
                        }
                } else {
                    uint32_t a[4] = {
                        pair(sm.k[pair_index][key_base + group], sm.k[pair_index + 1][key_base + group]),
                        pair(sm.k[pair_index][key_base + group + 8], sm.k[pair_index + 1][key_base + group + 8]),
                        pair(sm.k[pair_index + 8][key_base + group], sm.k[pair_index + 9][key_base + group]),
                        pair(sm.k[pair_index + 8][key_base + group + 8], sm.k[pair_index + 9][key_base + group + 8]),
                    };
                    float update[3][4] = {};
#pragma unroll
                    for (int limb = 0; limb < 3; ++limb) {
                        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sm.data.scaled_delta[limb][value_base + group][pair_index]);
                        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sm.data.scaled_delta[limb][value_base + group][pair_index + 8]);
                        qmm_frag::mma(update[limb], a, b0, b1);
                    }
#pragma unroll
                    for (int p = 0; p < 4; ++p) state[i][j][p] = __fadd_rn(__fmul_rn(state[i][j][p], decay), combine(update, p));
                }
            }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const int key = wm * 32 + i * 16 + group + h * 8;
                    const int column = wn * (V / 2) + j * 8 + pair_index + p;
                    last[(static_cast<size_t>(head) * 128 + vfirst + column) * 128 + key] = state[i][j][h * 2 + p];
                }
}
}

#define TF_GDN_TENSOR(V) \
extern "C" __global__ __launch_bounds__(256,2) void tf_gdn_chunk_tensor_##V( \
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, \
    const tf_gdn_chunk::Coefficients* coefficients, const unsigned* fallback, const float* initial, \
    float* last, __nv_bfloat16* out, int rows, int hk, int hv) { \
    tf_gdn_tensor::run<V>(q,k,v,coefficients,fallback,initial,last,out,rows,hk,hv); \
}
TF_GDN_TENSOR(16)
TF_GDN_TENSOR(32)
#undef TF_GDN_TENSOR
