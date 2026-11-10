// Original serial FP32 GDN arithmetic with asynchronous BF16 shared-memory staging.
#include "gdn_prefill.cu"
#include "common/cp_async.cuh"

namespace tf_gdn_serial_async {
using tf_gdn_prefill::DK;
using tf_gdn_prefill::DV;
template<int ROWS, int STEPS> struct alignas(16) Stage {
    uint2 k[STEPS][DK / 4], q[STEPS][DK / 4];
    __nv_bfloat16 v[STEPS][ROWS];
    float g[STEPS];
    __nv_bfloat16 b[STEPS];
};
static_assert(2 * sizeof(Stage<32,16>) == 18624);
static_assert(2 * sizeof(Stage<32,32>) == 37248);
static_assert(2 * sizeof(Stage<64,16>) == 20672);
static_assert(2 * sizeof(Stage<64,32>) == 41344);

template<int ROWS, int STEPS>
__device__ __forceinline__ void issue(Stage<ROWS,STEPS>& dst, const __nv_bfloat16* q,
    const __nv_bfloat16* k, const __nv_bfloat16* v, const float* g, const __nv_bfloat16* beta,
    int begin, int W, int hk, int hv, int head, int key_head, int value_first) {
    constexpr int threads = 2 * ROWS;
    for (int index = threadIdx.x; index < STEPS * (DK / 8); index += threads) {
        const int step = index / (DK / 8), column = index % (DK / 8);
        const int row = min(begin + step, W - 1);
        const size_t at = (static_cast<size_t>(row) * hk + key_head) * DK + 8 * column;
        const bool valid = begin + step < W;
        tf_common::cp16z(&dst.k[step][2 * column], k + at, valid);
        tf_common::cp16z(&dst.q[step][2 * column], q + at, valid);
    }
    for (int index = threadIdx.x; index < STEPS * (ROWS / 8); index += threads) {
        const int step = index / (ROWS / 8), column = index % (ROWS / 8);
        const int row = min(begin + step, W - 1);
        const size_t at = (static_cast<size_t>(row) * hv + head) * DV + value_first + 8 * column;
        tf_common::cp16z(&dst.v[step][8 * column], v + at, begin + step < W);
    }
    if (threadIdx.x < STEPS) {
        const int row = begin + threadIdx.x;
        const size_t at = static_cast<size_t>(min(row, W - 1)) * hv + head;
        dst.g[threadIdx.x] = row < W ? g[at] : 0;
        dst.b[threadIdx.x] = row < W ? beta[at] : __float2bfloat16_rn(0);
    }
    tf_common::cp_commit();
}

template<int ROWS, int STEPS>
__device__ __forceinline__ void run(const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ g, const __nv_bfloat16* __restrict__ beta,
    const float* state, float* last, __nv_bfloat16* __restrict__ y, int W, int hk, int hv) {
    __shared__ Stage<ROWS,STEPS> stages[2];
    const int head = blockIdx.x, row = blockIdx.y * ROWS + (threadIdx.x >> 1), half = threadIdx.x & 1;
    const int key_head = head / (hv / hk);
    float s[64];
    const float* s0 = state + (static_cast<size_t>(head) * DV + row) * DK;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float4 t = *reinterpret_cast<const float4*>(s0 + 8 * j + 4 * half);
        s[4 * j] = t.x; s[4 * j + 1] = t.y; s[4 * j + 2] = t.z; s[4 * j + 3] = t.w;
    }
    issue(stages[0], q, k, v, g, beta, 0, W, hk, hv, head, key_head, blockIdx.y * ROWS);
    int slot = 0;
    for (int t0 = 0; t0 < W; t0 += STEPS, slot ^= 1) {
        tf_common::cp_wait<0>();
        __syncthreads();
        if (t0 + STEPS < W)
            issue(stages[slot ^ 1], q, k, v, g, beta, t0 + STEPS, W, hk, hv, head, key_head, blockIdx.y * ROWS);
        const int n = min(STEPS, W - t0);
        const auto& ks = stages[slot].k;
        const auto& qs = stages[slot].q;
        const auto& vs = stages[slot].v;
        const auto& gs = stages[slot].g;
        const auto& bs = stages[slot].b;
        for (int tt = 0; tt < n; ++tt) {
            const float gt = gs[tt], bt = tf_gdn_prefill::to_float(bs[tt]);
            const size_t vat = (static_cast<size_t>(t0 + tt) * hv + head) * DV + row;
            const float vt = __bfloat162float(vs[tt][threadIdx.x >> 1]);
            float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float4 kk = tf_gdn_prefill::widen(ks[tt][2 * j + half]);
                s[4 * j] = s[4 * j] * gt;
                s[4 * j + 1] = s[4 * j + 1] * gt;
                s[4 * j + 2] = s[4 * j + 2] * gt;
                s[4 * j + 3] = s[4 * j + 3] * gt;
                m[0] = __fmaf_rn(s[4 * j], kk.x, m[0]);
                m[1] = __fmaf_rn(s[4 * j + 1], kk.y, m[1]);
                m[2] = __fmaf_rn(s[4 * j + 2], kk.z, m[2]);
                m[3] = __fmaf_rn(s[4 * j + 3], kk.w, m[3]);
            }
            float mem = (m[0] + m[1]) + (m[2] + m[3]);
            mem = mem + __shfl_xor_sync(0xffffffffu, mem, 1);
            const float delta = (vt - mem) * bt;
            float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float4 kk = tf_gdn_prefill::widen(ks[tt][2 * j + half]), qq = tf_gdn_prefill::widen(qs[tt][2 * j + half]);
                s[4 * j] = __fmaf_rn(kk.x, delta, s[4 * j]);
                s[4 * j + 1] = __fmaf_rn(kk.y, delta, s[4 * j + 1]);
                s[4 * j + 2] = __fmaf_rn(kk.z, delta, s[4 * j + 2]);
                s[4 * j + 3] = __fmaf_rn(kk.w, delta, s[4 * j + 3]);
                o[0] = __fmaf_rn(s[4 * j], qq.x, o[0]);
                o[1] = __fmaf_rn(s[4 * j + 1], qq.y, o[1]);
                o[2] = __fmaf_rn(s[4 * j + 2], qq.z, o[2]);
                o[3] = __fmaf_rn(s[4 * j + 3], qq.w, o[3]);
            }
            float out = (o[0] + o[1]) + (o[2] + o[3]);
            out = out + __shfl_xor_sync(0xffffffffu, out, 1);
            if (half == 0) y[vat] = __float2bfloat16_rn(out);
        }
    }
    float* s1 = last + (static_cast<size_t>(head) * DV + row) * DK;
#pragma unroll
    for (int j = 0; j < 16; ++j)
        *reinterpret_cast<float4*>(s1 + 8 * j + 4 * half) = make_float4(s[4 * j], s[4 * j + 1], s[4 * j + 2],
                                                                        s[4 * j + 3]);
}
}

#define TF_GDN_ASYNC(R,S) \
extern "C" __global__ __launch_bounds__(2*R) void tf_gdn_async_##R##_##S( \
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, \
    const float* g, const __nv_bfloat16* beta, const float* initial, float* last, \
    __nv_bfloat16* output, int rows, int hk, int hv) { \
    tf_gdn_serial_async::run<R,S>(q,k,v,g,beta,initial,last,output,rows,hk,hv); \
}
TF_GDN_ASYNC(32,16)
TF_GDN_ASYNC(32,32)
TF_GDN_ASYNC(64,16)
TF_GDN_ASYNC(64,32)
#undef TF_GDN_ASYNC
