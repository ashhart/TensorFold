// Flash Next's hyper-connection and n-gram glue: each row's own fixed-order fp32 sums, IEEE division and square root.

// glue.py and qmm.py's arithmetic and bf16 rounding points, the 4-bit dots on mma tiles with K in ascending order.

#include <cuda_bf16.h>
#include <stdint.h>

#include "glue_math.cuh"
#include "qmm_frag.cuh"

using namespace tf_glue;

namespace fn_hc {

constexpr int TILE_N = 64; // columns a stored weight tile holds ([N/64][K/32][64][4] words)

__device__ __forceinline__ float bsig(float x) { return rbf(tf_sigmoid(x)); }
__device__ __forceinline__ float bsilu(float x) { return rbf(tf_silu(x)); }

// bf16x2 of nibbles i (low half) and i + 1 (high half) of an MLX word: the integers 0..15, exact.
__device__ __forceinline__ uint32_t nibbles(uint32_t w, int i) {
    const float lo = static_cast<float>((w >> (4 * i)) & 0xFu), hi = static_cast<float>((w >> (4 * i + 4)) & 0xFu);
    return (__float_as_uint(lo) >> 16) | (__float_as_uint(hi) & 0xFFFF0000u);
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    return static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(lo))) |
           (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(hi))) << 16);
}

// A [16, 32] bf16 tile at `x` (row stride `ld` elements) as the two k16 steps' A fragments.
__device__ __forceinline__ void a_frags(const __nv_bfloat16* x, int ld, uint32_t (&a)[2][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        const __nv_bfloat16* p = x + 16 * s + 2 * t;
        a[s][0] = *reinterpret_cast<const uint32_t*>(p + g * ld);
        a[s][1] = *reinterpret_cast<const uint32_t*>(p + (g + 8) * ld);
        a[s][2] = *reinterpret_cast<const uint32_t*>(p + g * ld + 8);
        a[s][3] = *reinterpret_cast<const uint32_t*>(p + (g + 8) * ld + 8);
    }
}

// p = x[16, 32] . q[32, 8]: one column's four words (32 nibbles), two chained k16 steps from zero.
__device__ __forceinline__ void group_dot(float (&p)[4], const uint32_t (&a)[2][4], uint4 words) {
    const int t = (threadIdx.x & 31) & 3;
    qmm_frag::mma0(p, a[0], nibbles(words.x, 2 * t), nibbles(words.y, 2 * t));
    qmm_frag::mma(p, a[1], nibbles(words.z, 2 * t), nibbles(words.w, 2 * t));
}

// A block's fixed-order sum: each warp's butterfly, then the warps' sums in warp order.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
    v = warp_sum(v);
    const int warps = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < warps; ++w) total = __fadd_rn(total, scratch[w]);
    __syncthreads();
    return total;
}

}  // namespace fn_hc

using namespace fn_hc;

// glue._hc_writeback, program (row, 256 columns): MODE 0 sums only, 2 MoE slots (fp32 y and weights), 4 K slices.
template <int MODE>
__device__ __forceinline__ void hc_writeback(const __nv_bfloat16* H, __nv_bfloat16* HOUT, float* __restrict__ PSS,
                                             const float* __restrict__ BR, const __nv_bfloat16* __restrict__ INJ,
                                             const float* __restrict__ Y, const float* __restrict__ WTS, long long RS, int D,
                                             int S, int TOPK, int SLOTS, int WORLD) {
    constexpr int MAXS = 8;
    __shared__ float red[8][MAXS];
    const int r = blockIdx.x, c = blockIdx.y, nc = D / 256;
    const int d = c * 256 + threadIdx.x;
    float branch = 0.0f;
    if constexpr (MODE == 4) {
        float acc = BR[static_cast<long long>(r) * D + d];
        for (int k = 1; k < WORLD; ++k) acc = __fadd_rn(acc, BR[k * RS + static_cast<long long>(r) * D + d]);
        branch = rbf(acc);
    } else if constexpr (MODE == 2) {
        float acc = 0.0f;
        for (int k = 0; k <= TOPK; ++k) {
            const long long at = static_cast<long long>(r) * SLOTS + k;
            acc = __fmaf_rn(Y[at * D + d], WTS[at], acc);
        }
        branch = rbf(acc);
    }
    const int warp = threadIdx.x >> 5;
    for (int s = 0; s < S; ++s) {
        const long long at = static_cast<long long>(r) * S * D + static_cast<long long>(s) * D + d;
        float hv = bf(H[at]);
        if constexpr (MODE != 0) {
            hv = rbf(__fadd_rn(hv, rbf(__fmul_rn(branch, bf(INJ[r * S + s])))));
            HOUT[at] = __float2bfloat16_rn(hv);
        }
        const float ss = warp_sum(__fmul_rn(hv, hv));
        if ((threadIdx.x & 31) == 0) red[warp][s] = ss;
    }
    __syncthreads();
    if (threadIdx.x < S) {
        float total = 0.0f;
        for (int w = 0; w < 8; ++w) total = __fadd_rn(total, red[w][threadIdx.x]);
        PSS[(static_cast<long long>(r) * nc + c) * S + threadIdx.x] = total;
    }
}

extern "C" __global__ void __launch_bounds__(256) fn_hc_writeback0(const __nv_bfloat16* H, float* PSS, int D, int S) {
    hc_writeback<0>(H, nullptr, PSS, nullptr, nullptr, nullptr, nullptr, 0, D, S, 0, 1, 1);
}

extern "C" __global__ void __launch_bounds__(256) fn_hc_writeback_moe(const __nv_bfloat16* H, __nv_bfloat16* HOUT, float* PSS,
                                                                      const __nv_bfloat16* INJ, const float* Y,
                                                                      const float* WTS, int D, int S, int TOPK, int SLOTS) {
    hc_writeback<2>(H, HOUT, PSS, nullptr, INJ, Y, WTS, 0, D, S, TOPK, SLOTS, 1);
}

extern "C" __global__ void __launch_bounds__(256) fn_hc_writeback_slices(const __nv_bfloat16* H, __nv_bfloat16* HOUT, float* PSS,
                                                                         const float* BR, const __nv_bfloat16* INJ,
                                                                         long long RS, int D, int S, int WORLD) {
    hc_writeback<4>(H, HOUT, PSS, BR, INJ, nullptr, nullptr, RS, D, S, 0, 1, WORLD);
}

// qmm._qmm_hcdown, block (16 rows, 64 columns, K slice): the slice's normed rows times the down rows, fp32 to PART.
extern "C" __global__ void __launch_bounds__(128) fn_hc_down(const __nv_bfloat16* __restrict__ H, const float* __restrict__ PSS,
                                                             const float* __restrict__ SCALE, __nv_bfloat16* __restrict__ NORMED,
                                                             const uint4* __restrict__ W, const __nv_bfloat16* __restrict__ S,
                                                             const __nv_bfloat16* __restrict__ B, float* __restrict__ PART,
                                                             int M, float eps, int N, int K, int D, int NC, int SS, int SK) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ float rinv[16];
    const int kg = K / 32, per = kg / SK, width = per * 32, ld = width + 8;
    __nv_bfloat16* xt = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // [16][width + 8] the slice's normed rows
    float* xsum = reinterpret_cast<float*>(smem_raw + 16 * ld * 2); // [16][per] their 32-group sums
    const int m0 = blockIdx.x * 16, n0 = blockIdx.y * TILE_N, slice = blockIdx.z, k0 = slice * width;
    const int st = k0 / D;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    if (threadIdx.x < 16) {
        const int r = m0 + threadIdx.x;
        float total = 0.0f;
        if (r < M)
            for (int c = 0; c < NC; ++c) total = __fadd_rn(total, PSS[(static_cast<long long>(r) * NC + c) * SS + st]);
        rinv[threadIdx.x] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(total, static_cast<float>(D)), eps)));
    }
    __syncthreads();
    const int vecs = width / 8;
    for (int e = threadIdx.x; e < 16 * vecs; e += blockDim.x) {
        const int rr = e / vecs, v = e % vecs, r = m0 + rr;
        uint4 out = make_uint4(0, 0, 0, 0);
        if (r < M) {
            const long long at = static_cast<long long>(r) * K + k0 + 8 * v;
            const uint4 raw = *reinterpret_cast<const uint4*>(H + at);
            const __nv_bfloat16* hv = reinterpret_cast<const __nv_bfloat16*>(&raw);
            float x[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) x[i] = rbf(__fmul_rn(__fmul_rn(bf(hv[i]), rinv[rr]), SCALE[k0 + 8 * v + i]));
            out = make_uint4(pack(x[0], x[1]), pack(x[2], x[3]), pack(x[4], x[5]), pack(x[6], x[7]));
            if (blockIdx.y == 0) *reinterpret_cast<uint4*>(NORMED + at) = out;
        }
        *reinterpret_cast<uint4*>(xt + rr * ld + 8 * v) = out;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 16 * per; e += blockDim.x) {
        const int rr = e / per, i = e % per;
        float sum = 0.0f;
        for (int k = 0; k < 32; ++k) sum = __fadd_rn(sum, bf(xt[rr * ld + 32 * i + k]));
        xsum[rr * per + i] = sum;
    }
    __syncthreads();
    float acc[2][4] = {};
    const uint4* tile = W + static_cast<long long>(n0 / TILE_N) * kg * TILE_N;
#pragma unroll 2
    for (int i = 0; i < per; ++i) {
        const int grp = slice * per + i;
        uint32_t a[2][4];
        a_frags(xt + 32 * i, ld, a);
        const float xs0 = xsum[g * per + i], xs1 = xsum[(g + 8) * per + i];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int local = warp * 16 + j * 8 + g;
            float p[4];
            group_dot(p, a, tile[static_cast<long long>(grp) * TILE_N + local]);
            const int n = n0 + warp * 16 + j * 8 + 2 * t;
            const float s0 = n < N ? bf(S[static_cast<long long>(grp) * N + n]) : 0.0f;
            const float s1 = n + 1 < N ? bf(S[static_cast<long long>(grp) * N + n + 1]) : 0.0f;
            const float b0 = n < N ? bf(B[static_cast<long long>(grp) * N + n]) : 0.0f;
            const float b1 = n + 1 < N ? bf(B[static_cast<long long>(grp) * N + n + 1]) : 0.0f;
            acc[j][0] = __fmaf_rn(xs0, b0, __fmaf_rn(p[0], s0, acc[j][0]));
            acc[j][1] = __fmaf_rn(xs0, b1, __fmaf_rn(p[1], s1, acc[j][1]));
            acc[j][2] = __fmaf_rn(xs1, b0, __fmaf_rn(p[2], s0, acc[j][2]));
            acc[j][3] = __fmaf_rn(xs1, b1, __fmaf_rn(p[3], s1, acc[j][3]));
        }
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int n = n0 + warp * 16 + j * 8 + 2 * t;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = m0 + g + 8 * h;
            if (r >= M) continue;
            float* out = PART + (static_cast<long long>(slice) * M + r) * N;
            if (n < N) out[n] = acc[j][2 * h];
            if (n + 1 < N) out[n + 1] = acc[j][2 * h + 1];
        }
    }
}

// glue._hc_reduce_act, a thread a column: SK (<= 32) slices in order, bf16 SiLU and its sums, the inject gates.
extern "C" __global__ void __launch_bounds__(1024) fn_hc_reduce_act(const float* __restrict__ PART, __nv_bfloat16* __restrict__ ACT,
                                                                    float* __restrict__ XS, __nv_bfloat16* __restrict__ INJ,
                                                                    int SK, int M, int S, int LOW, int HAS_INJ, int NDN) {
    __shared__ float a_s[1024];
    const int r = blockIdx.x, i = threadIdx.x;
    if (i < LOW + (HAS_INJ ? S : 0)) {
        float part[32];
#pragma unroll
        for (int s = 0; s < 32; ++s) part[s] = s < SK ? PART[(static_cast<long long>(s) * M + r) * NDN + i] : 0.0f;
        float v = part[0];
#pragma unroll
        for (int s = 1; s < 32; ++s)
            if (s < SK) v = __fadd_rn(v, part[s]);
        v = rbf(__fdiv_rn(rbf(v), static_cast<float>(S)));
        if (i < LOW) {
            const float a = bsilu(v);
            ACT[static_cast<long long>(r) * LOW + i] = __float2bfloat16_rn(a);
            a_s[i] = a;
        } else {
            INJ[static_cast<long long>(r) * S + (i - LOW)] = __float2bfloat16_rn(__fmul_rn(2.0f, bsig(v)));
        }
    }
    __syncthreads();
    if (i < LOW / 32) {
        float sum = 0.0f;
        for (int k = 0; k < 32; ++k) sum = __fadd_rn(sum, a_s[i * 32 + k]);
        XS[static_cast<long long>(r) * (LOW / 32) + i] = sum;
    }
}

// qmm._qmm_upmix, block (16 rows, 32 dims), two warps a stream (SS 4): the up rows, then the streams mixed in order.
extern "C" __global__ void __launch_bounds__(256) fn_hc_upmix(const __nv_bfloat16* __restrict__ X, const float* __restrict__ XS,
                                                              const uint4* __restrict__ W, const __nv_bfloat16* __restrict__ S,
                                                              const __nv_bfloat16* __restrict__ B, const __nv_bfloat16* __restrict__ NORMED,
                                                              __nv_bfloat16* __restrict__ MIXED, float* __restrict__ XSM,
                                                              int M, int N, int K, int D, int SS) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int kg = K / 32;
    const int ld = K + 8;
    __nv_bfloat16* xt = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // [16][K + 8]
    float* part = reinterpret_cast<float*>(smem_raw + 16 * ld * 2); // [SS][16][32] each stream's terms
    const int m0 = blockIdx.x * 16, j = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    for (int e = threadIdx.x; e < 16 * (K / 8); e += blockDim.x) {
        const int rr = e / (K / 8), v = e % (K / 8), r = m0 + rr;
        *reinterpret_cast<uint4*>(xt + rr * ld + 8 * v) = r < M ? *reinterpret_cast<const uint4*>(X + static_cast<long long>(r) * K + 8 * v) : make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    const int st = warp >> 1;
    const int row0 = st * D + j * 32;
    const int q0 = 2 * (warp & 1);  // this warp's two of the stream's four 8-column tiles
    float acc[4][4] = {};
#pragma unroll 5
    for (int grp = 0; grp < kg; ++grp) {
        uint32_t a[2][4];
        a_frags(xt + grp * 32, ld, a);
        const float xs0 = m0 + g < M ? XS[static_cast<long long>(m0 + g) * kg + grp] : 0.0f;
        const float xs1 = m0 + g + 8 < M ? XS[static_cast<long long>(m0 + g + 8) * kg + grp] : 0.0f;
#pragma unroll
        for (int q = q0; q < q0 + 2; ++q) {
            const int col = row0 + q * 8 + g;
            float p[4];
            group_dot(p, a, W[(static_cast<long long>(col / TILE_N) * kg + grp) * TILE_N + col % TILE_N]);
            const int n = row0 + q * 8 + 2 * t;
            const float s0 = bf(S[static_cast<long long>(grp) * N + n]), s1 = bf(S[static_cast<long long>(grp) * N + n + 1]);
            const float b0 = bf(B[static_cast<long long>(grp) * N + n]), b1 = bf(B[static_cast<long long>(grp) * N + n + 1]);
            acc[q][0] = __fmaf_rn(xs0, b0, __fmaf_rn(p[0], s0, acc[q][0]));
            acc[q][1] = __fmaf_rn(xs0, b1, __fmaf_rn(p[1], s1, acc[q][1]));
            acc[q][2] = __fmaf_rn(xs1, b0, __fmaf_rn(p[2], s0, acc[q][2]));
            acc[q][3] = __fmaf_rn(xs1, b1, __fmaf_rn(p[3], s1, acc[q][3]));
        }
    }
#pragma unroll
    for (int q = q0; q < q0 + 2; ++q)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int rr = g + 8 * (e >> 1), dd = q * 8 + 2 * t + (e & 1), r = m0 + rr;
            const float nv = r < M ? bf(NORMED[static_cast<long long>(r) * N + row0 + q * 8 + 2 * t + (e & 1)]) : 0.0f;
            part[(st * 16 + rr) * 32 + dd] = rbf(__fmul_rn(bsig(rbf(acc[q][e])), nv));
        }
    __syncthreads();
    __shared__ float mixed_s[16][32];
    for (int e = threadIdx.x; e < 16 * 32; e += blockDim.x) {
        const int rr = e >> 5, dd = e & 31;
        float total = 0.0f;
        for (int s = 0; s < SS; ++s) total = __fadd_rn(total, part[(s * 16 + rr) * 32 + dd]);
        const float m = rbf(__fdiv_rn(total, static_cast<float>(SS)));
        mixed_s[rr][dd] = m;
        if (m0 + rr < M) MIXED[static_cast<long long>(m0 + rr) * D + j * 32 + dd] = __float2bfloat16_rn(m);
    }
    __syncthreads();
    if (threadIdx.x < 16 && m0 + threadIdx.x < M) {
        float sum = 0.0f;
        for (int k = 0; k < 32; ++k) sum = __fadd_rn(sum, mixed_s[threadIdx.x][k]);
        XSM[static_cast<long long>(m0 + threadIdx.x) * (D / 32) + j] = sum;
    }
}

// glue._ple_gate, a block a row: per stream normed keys and queries, their dot's signed root gating the values.
extern "C" __global__ void __launch_bounds__(512) fn_ple_gate(const __nv_bfloat16* __restrict__ KEYS, const __nv_bfloat16* __restrict__ VALS,
                                                              const __nv_bfloat16* __restrict__ H, const float* __restrict__ NK,
                                                              const float* __restrict__ NQ, __nv_bfloat16* __restrict__ GATED,
                                                              float* __restrict__ PSS, float eps, int D, int S) {
    __shared__ float scratch[32];
    const int r = blockIdx.x;
    const float fd = static_cast<float>(D);
    for (int s = 0; s < S; ++s) {
        const long long row = static_cast<long long>(r) * S * D + static_cast<long long>(s) * D;
        float kss = 0.0f, qss = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            const float kv = bf(KEYS[row + d]), hv = bf(H[row + d]);
            kss = __fmaf_rn(kv, kv, kss);
            qss = __fmaf_rn(hv, hv, qss);
        }
        kss = block_sum(kss, scratch);
        qss = block_sum(qss, scratch);
        const float krinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(kss, fd), eps)));
        const float qrinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(qss, fd), eps)));
        float dot = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            const float kn = rbf(__fmul_rn(__fmul_rn(bf(KEYS[row + d]), krinv), NK[s * D + d]));
            const float qn = rbf(__fmul_rn(__fmul_rn(bf(H[row + d]), qrinv), NQ[s * D + d]));
            dot = __fadd_rn(dot, rbf(__fmul_rn(kn, qn)));
        }
        dot = block_sum(dot, scratch);
        float gate = rbf(__fdiv_rn(dot, __fsqrt_rn(fd)));
        const float mag = __fsqrt_rn(fmaxf(fabsf(gate), 1e-6f));
        gate = rbf(gate > 0.0f ? mag : (gate < 0.0f ? -mag : 0.0f));
        const float sg = bsig(gate);
        float gss = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            const float gv = rbf(__fmul_rn(sg, bf(VALS[static_cast<long long>(r) * D + d])));
            GATED[row + d] = __float2bfloat16_rn(gv);
            gss = __fmaf_rn(gv, gv, gss);
        }
        gss = block_sum(gss, scratch);
        if (threadIdx.x == 0) PSS[static_cast<long long>(r) * S + s] = gss;
    }
}

// glue._ple_conv, program (row, 512 columns): the dilated conv over [TAIL; rows], bf16 SiLU, the adds, NROW kept.
extern "C" __global__ void __launch_bounds__(512) fn_ple_conv(const __nv_bfloat16* __restrict__ GATED, const float* __restrict__ PSS,
                                                              const float* __restrict__ NC, const __nv_bfloat16* __restrict__ TAIL,
                                                              const __nv_bfloat16* __restrict__ CW, const __nv_bfloat16* H,
                                                              __nv_bfloat16* HOUT, __nv_bfloat16* __restrict__ NROW, float eps,
                                                              int D, int S, int TAPS, int DIL) {
    const int r = blockIdx.x;
    const int w = S * D, nt = (TAPS - 1) * DIL;
    const int e = blockIdx.y * blockDim.x + threadIdx.x;
    if (e >= w) return;
    const int s = e / D;
    const float fd = static_cast<float>(D);
    float acc = 0.0f;
    for (int t = 0; t < TAPS; ++t) {
        const int at = r + t * DIL;
        float xv;
        if (at < nt) {
            xv = bf(TAIL[static_cast<long long>(at) * w + e]);
        } else {
            const int rr = at - nt;
            const float rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(PSS[rr * S + s], fd), eps)));
            xv = rbf(__fmul_rn(__fmul_rn(bf(GATED[static_cast<long long>(rr) * w + e]), rinv), NC[e]));
        }
        acc = __fmaf_rn(bf(CW[static_cast<long long>(e) * TAPS + t]), xv, acc);
    }
    const float act = bsilu(rbf(acc));
    const float g = bf(GATED[static_cast<long long>(r) * w + e]);
    const float ple = rbf(__fadd_rn(g, act));
    const long long at = static_cast<long long>(r) * w + e;
    HOUT[at] = __float2bfloat16_rn(__fadd_rn(bf(H[at]), ple));
    const float rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(PSS[r * S + s], fd), eps)));
    NROW[at] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(g, rinv), NC[e]));
}

// glue._add_streams: out[r, s] = bf16(e[r] + hs[r, s]), the MTP head's input streams.
extern "C" __global__ void fn_add_streams(const __nv_bfloat16* __restrict__ E, const __nv_bfloat16* __restrict__ HS,
                                          __nv_bfloat16* __restrict__ OUT, int D, int S) {
    const int r = blockIdx.x, d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const float e = bf(E[static_cast<long long>(r) * D + d]);
    for (int s = 0; s < S; ++s) {
        const long long at = (static_cast<long long>(r) * S + s) * D + d;
        OUT[at] = __float2bfloat16_rn(__fadd_rn(e, bf(HS[at])));
    }
}

// Dense HC uses the same stored stream sums and FP32 gamma before one BF16 normalization rounding.
extern "C" __global__ void fn_hc_normalize(const __nv_bfloat16* H, const float* PSS, const float* SCALE,
                                            __nv_bfloat16* OUT, float eps, int D, int SS) {
    __shared__ float rinv;
    const int r = blockIdx.x, stream = blockIdx.y, nc = D / 256;
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int c = 0; c < nc; ++c) total = __fadd_rn(total, PSS[(static_cast<long long>(r) * nc + c) * SS + stream]);
        rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(total, static_cast<float>(D)), eps)));
    }
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const int col = stream * D + d;
        const long long at = static_cast<long long>(r) * SS * D + col;
        OUT[at] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(bf(H[at]), rinv), SCALE[col]));
    }
}

// Dense up-projection values are already BF16: sigmoid, product and stream mean keep their original rounding seams.
extern "C" __global__ void fn_hc_dense_mix(const __nv_bfloat16* UP, const __nv_bfloat16* NORMED,
                                            __nv_bfloat16* MIXED, float* XSM, int D, int SS) {
    __shared__ float row[32];
    const int r = blockIdx.x, group = blockIdx.y, d = group * 32 + threadIdx.x;
    float total = 0.0f;
    for (int stream = 0; stream < SS; ++stream) {
        const long long at = (static_cast<long long>(r) * SS + stream) * D + d;
        const float value = rbf(__fmul_rn(bsig(bf(UP[at])), bf(NORMED[at])));
        total = __fadd_rn(total, value);
    }
    const float mixed = rbf(__fdiv_rn(total, static_cast<float>(SS)));
    MIXED[static_cast<long long>(r) * D + d] = __float2bfloat16_rn(mixed);
    row[threadIdx.x] = mixed;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 32; ++i) sum = __fadd_rn(sum, row[i]);
        XSM[static_cast<long long>(r) * (D / 32) + group] = sum;
    }
}

// Shared expert uses its own BF16 projection outputs and rounding boundaries.
extern "C" __global__ void fn_shared_swiglu(const __nv_bfloat16* GATE, const __nv_bfloat16* UP,
                                             __nv_bfloat16* ACT, int W) {
    const int r = blockIdx.x, j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= W) return;
    const long long at = static_cast<long long>(r) * W + j;
    ACT[at] = __float2bfloat16_rn(__fmul_rn(bsilu(bf(GATE[at])), bf(UP[at])));
}

// Write only the final reserved slot; routed slots retain their FP32 partial sums.
extern "C" __global__ void fn_shared_scatter(const float* SHARED, float* Y, int D, int SLOTS) {
    const int r = blockIdx.x, j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j < D) Y[(static_cast<long long>(r) * SLOTS + SLOTS - 1) * D + j] = SHARED[static_cast<long long>(r) * D + j];
}
