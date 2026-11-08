// Softmax top-k routing rows for every CUDA family (float64 reference: core/route_ref.zig), a row's bits its own.

// Router logits on mma tiles, K ascending from zero (a row's sums never depend on the row count), and top-k rows.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "common/glue_math.cuh"
#include "common/qmm_frag.cuh"
#include "common/router_mma.cuh"

using namespace tf_glue;

namespace tf_route {

constexpr int NT = 128;
constexpr int MOST = 8;  // logits a thread holds: up to NT * MOST experts

// The block's largest (value, index), lower index on ties; slots alternate by `parity`: one barrier an iteration.
__device__ __forceinline__ void block_best(float& v, int& i, float (*sv)[NT / 32], int (*si)[NT / 32], int parity) {
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, off);
        const int oi = __shfl_xor_sync(0xffffffffu, i, off);
        if (ov > v || (ov == v && oi < i)) {
            v = ov;
            i = oi;
        }
    }
    const int warp = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        sv[parity][warp] = v;
        si[parity][warp] = i;
    }
    __syncthreads();
    v = sv[parity][0];
    i = si[parity][0];
    for (int w = 1; w < NT / 32; ++w)
        if (sv[parity][w] > v || (sv[parity][w] == v && si[parity][w] < i)) {
            v = sv[parity][w];
            i = si[parity][w];
        }
}

}  // namespace tf_route

using namespace tf_route;

// route_logits over 32 experts a block (prompt chunks: the input rows are read once per 32 experts).
extern "C" __global__ void __launch_bounds__(128) tf_route_logits(const __nv_bfloat16* __restrict__ X, int x_ld,
                                                                  const __nv_bfloat16* __restrict__ W, float* __restrict__ OUT,
                                                                  int M, int D, int E) {
    tf_router_mma::logits<4, 2, 128>(X, x_ld, W, OUT, M, D, E);
}

// route_logits over 128 rows and 64 experts a block, eight warps (from 64 rows), the same sums; EFAST block order.
extern "C" __global__ void __launch_bounds__(256) tf_route_logits_wide(const __nv_bfloat16* __restrict__ X, int x_ld,
                                                                       const __nv_bfloat16* __restrict__ W, float* __restrict__ OUT,
                                                                       int M, int D, int E) {
    tf_router_mma::logits<8, 2, 64, 8, true>(X, x_ld, W, OUT, M, D, E);
}

// route_logits with three stages of 256 in flight (up to 16 rows, where the router rows' reads set its time).
extern "C" __global__ void __launch_bounds__(128) tf_route_logits_few(const __nv_bfloat16* __restrict__ X, int x_ld,
                                                                     const __nv_bfloat16* __restrict__ W, float* __restrict__ OUT,
                                                                     int M, int D, int E) {
    tf_router_mma::logits<4, 3, 256>(X, x_ld, W, OUT, M, D, E);
}

// A row's TOPK largest of NE logits (lower id on ties), bf16(e^(l_k - l_0) / sum); `shared`: slot TOPK gated by NE.
extern "C" __global__ void __launch_bounds__(NT) tf_route_topk(const float* __restrict__ L, int NL, int NE, int TOPK,
                                                               int* __restrict__ PICK, float* __restrict__ WTS, int SLOTS,
                                                               int shared) {
    __shared__ float sv[2][NT / 32];
    __shared__ int si[2][NT / 32];
    const int r = blockIdx.x;
    const float* row = L + static_cast<long long>(r) * NL;
    float v[MOST];
#pragma unroll
    for (int j = 0; j < MOST; ++j) {
        const int i = threadIdx.x + j * NT;
        v[j] = i < NE ? row[i] : -INFINITY;
    }
    float top = 0.0f, total = 0.0f, mine = 0.0f;
    int pick = 0;
    for (int k = 0; k < TOPK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < MOST; ++j) {
            const int i = threadIdx.x + j * NT;
            if (v[j] > bv || (v[j] == bv && i < bi)) {
                bv = v[j];
                bi = i;
            }
        }
        block_best(bv, bi, sv, si, k & 1);
        if (k == 0) top = bv;
#pragma unroll
        for (int j = 0; j < MOST; ++j)
            if (threadIdx.x + j * NT == bi) v[j] = -INFINITY;
        const float ex = tf_exp(__fsub_rn(bv, top));
        total = __fadd_rn(total, ex);
        if (threadIdx.x == k) {
            mine = ex;
            pick = bi;
        }
    }
    if (threadIdx.x < TOPK) {
        PICK[static_cast<long long>(r) * SLOTS + threadIdx.x] = pick;
        WTS[static_cast<long long>(r) * SLOTS + threadIdx.x] = rbf(__fdiv_rn(mine, total));
    }
    if (shared && threadIdx.x == 0) {
        PICK[static_cast<long long>(r) * SLOTS + TOPK] = NE;
        WTS[static_cast<long long>(r) * SLOTS + TOPK] = rbf(tf_sigmoid(rbf(row[NE])));
    }
}
