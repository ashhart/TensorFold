// Softmax top-k routing rows for every CUDA family (float64 reference: core/route_ref.zig), a row's bits its own.

// Router logits on mma tiles, K ascending from zero (a row's sums never depend on the row count), and top-k rows.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "common/glue_math.cuh"
#include "common/qmm_frag.cuh"

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

// OUT[m, e] = x[m] . w[e], 8 experts a warp, RT tiles of 16 rows, one mma chain over K, STAGES of KC (row sums fixed).
template <int WARPS, int STAGES, int KC, int RT = 1>
__device__ __forceinline__ void route_logits(const __nv_bfloat16* __restrict__ X, int x_ld, const __nv_bfloat16* __restrict__ W,
                                             float* __restrict__ OUT, int M, int D, int E) {
    constexpr int LD = KC + 8, EB = 8 * WARPS, XR = 16 * RT, ROWS = XR + EB;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto tile = reinterpret_cast<__nv_bfloat16 (*)[ROWS][LD]>(smem_raw);  // [STAGES] XR input rows, then EB expert rows
    const int m0 = blockIdx.x * XR, e0 = blockIdx.y * EB;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int stages = (D + KC - 1) / KC;
    const int xr = min(XR, M - m0);  // input rows past M stay zero: written once, never staged
    for (int v = threadIdx.x; v < STAGES * (XR - xr) * (KC / 8); v += blockDim.x) {
        const int per = (XR - xr) * (KC / 8);
        *reinterpret_cast<uint4*>(&tile[v / per][xr + (v % per) / (KC / 8)][8 * (v % (KC / 8))]) = make_uint4(0, 0, 0, 0);
    }
    auto stage = [&](int st) {
        const int k0 = st * KC, buf = st % STAGES;
        for (int v = threadIdx.x; v < (xr + EB) * (KC / 8); v += blockDim.x) {
            const int i = v / (KC / 8), k = k0 + 8 * (v % (KC / 8));
            const bool x_row = i < xr;
            const int row = x_row ? i : XR + i - xr;
            const int r = x_row ? m0 + i : e0 + i - xr;
            const bool read = k < D && (x_row || r < E);
            const __nv_bfloat16* src = x_row ? X + static_cast<long long>(r) * x_ld + (read ? k : 0)
                                             : W + static_cast<long long>(read ? r : 0) * D + (read ? k : 0);
            qmm_frag::cp16z(&tile[buf][row][8 * (v % (KC / 8))], src, read);
        }
    };
    float acc[RT][4] = {};
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < stages) stage(st);
        qmm_frag::commit();
    }
    for (int st = 0; st < stages; ++st) {
        if (st + STAGES - 1 < stages) stage(st + STAGES - 1);
        qmm_frag::commit();
        qmm_frag::wait<STAGES - 1>();
        __syncthreads();
        const int buf = st % STAGES;
        const int steps = min(KC, D - st * KC) / 16;
        for (int kk = 0; kk < steps; ++kk) {
            uint32_t b[2];
            qmm_frag::ldmatrix2(b, &tile[buf][XR + warp * 8 + lane % 8][kk * 16 + ((lane / 8) % 2) * 8]);
#pragma unroll
            for (int i = 0; i < RT; ++i) {
                uint32_t a[4];
                qmm_frag::ldmatrix4(a, &tile[buf][i * 16 + lane % 16][kk * 16 + (lane / 16) * 8]);
                qmm_frag::mma(acc[i], a, b[0], b[1]);
            }
        }
        __syncthreads();
    }
    const int e = e0 + warp * 8 + 2 * t;
#pragma unroll
    for (int i = 0; i < RT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = m0 + i * 16 + g + 8 * h;
            if (r >= M) continue;
            if (e < E) OUT[static_cast<long long>(r) * E + e] = acc[i][2 * h];
            if (e + 1 < E) OUT[static_cast<long long>(r) * E + e + 1] = acc[i][2 * h + 1];
        }
}

// route_logits over 32 experts a block (prompt chunks: the input rows are read once per 32 experts).
extern "C" __global__ void __launch_bounds__(128) tf_route_logits(const __nv_bfloat16* __restrict__ X, int x_ld,
                                                                  const __nv_bfloat16* __restrict__ W, float* __restrict__ OUT,
                                                                  int M, int D, int E) {
    route_logits<4, 2, 128>(X, x_ld, W, OUT, M, D, E);
}

// route_logits over 128 rows and 64 experts a block, eight warps (from 64 rows), the same sums.
extern "C" __global__ void __launch_bounds__(256) tf_route_logits_wide(const __nv_bfloat16* __restrict__ X, int x_ld,
                                                                       const __nv_bfloat16* __restrict__ W, float* __restrict__ OUT,
                                                                       int M, int D, int E) {
    route_logits<8, 2, 64, 8>(X, x_ld, W, OUT, M, D, E);
}

// route_logits with three stages of 256 in flight (up to 16 rows, where the router rows' reads set its time).
extern "C" __global__ void __launch_bounds__(128) tf_route_logits_few(const __nv_bfloat16* __restrict__ X, int x_ld,
                                                                     const __nv_bfloat16* __restrict__ W, float* __restrict__ OUT,
                                                                     int M, int D, int E) {
    route_logits<4, 3, 256>(X, x_ld, W, OUT, M, D, E);
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
