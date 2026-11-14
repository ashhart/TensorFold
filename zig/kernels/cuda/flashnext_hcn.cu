// Flash Next's BF16 read-out on prompt tiles: the normalize folded into the down and the up+mix, the chain's bits.

#include <cuda_bf16.h>
#include <stdint.h>

#include "common/fold.cuh"
#include "common/glue_math.cuh"
#include "common/qmm_frag.cuh"

namespace hcn {
using namespace qmm_frag;
using tf_glue::bf;
using tf_glue::rbf;

__device__ __forceinline__ float bsig(float x) { return rbf(tf_glue::tf_sigmoid(x)); }

// fn_hc_normalize2's row scale: the stream's 256-column square sums in order, then tf_common::rms_scale
__device__ __forceinline__ float rinv_of(const float* __restrict__ pss, long long r, int st, int NC, int SS, int D, float eps) {
    const float total = tf_common::sum_rn(NC, [&](int c) { return pss[(r * NC + c) * SS + st]; });
    return tf_common::rms_scale(total, D, eps);
}

// down: [rows, K] normed streams x [N <= 336, K] bf16 weights -> fp32 [rows, N]; 16 MT rows a CTA
constexpr int DN_THREADS = 224;  // 7 warps, three n8 tiles each: columns [0, 160) and [160, 336)
constexpr int DN_SPLIT = 20;     // n8 tiles of the first column tile
constexpr int DN_SS = 8;         // streams at most

template <int MT, int STAGES>
struct DnShape {
    static constexpr int BM = 16 * MT;
    static constexpr int STAGE = BM * 128;
    static constexpr int SMEM = STAGES * STAGE + BM * DN_SS * 4;
};

template <int MT, int STAGES>
__device__ __forceinline__ void down_body(const __nv_bfloat16* __restrict__ H, const float* __restrict__ PSS,
                                          const float* __restrict__ SCALE, const uint2* __restrict__ W,
                                          float* __restrict__ PART, int M, int N, int K, int SK, int D, int NC, int SS,
                                          float eps) {
    using S = DnShape<MT, STAGES>;
    constexpr int BM = S::BM;
    extern __shared__ __align__(128) unsigned char buf[];
    float* rinv_s = reinterpret_cast<float*>(buf + STAGES * S::STAGE);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, t4 = lane & 3;
    const int KG = K / 64, per = KG / SK;
    const int ct = blockIdx.x & 1, m0 = (blockIdx.x >> 1) * BM;

    for (int i = tid; i < BM * SS; i += DN_THREADS)
        rinv_s[(i / SS) * DN_SS + i % SS] = rinv_of(PSS, min(m0 + i / SS, M - 1), i % SS, NC, SS, D, eps);

    const uint2* wp[3];
    bool valid[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int n8 = ct * DN_SPLIT + warp * 3 + j;
        valid[j] = n8 * 8 < N && (ct == 1 || n8 < DN_SPLIT);
        wp[j] = W + static_cast<size_t>(n8 >> 3) * KG * 1024 + (n8 & 7) * 128 + lane;
    }

    auto stage = [&](int s) { return buf + s * S::STAGE; };
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * 8; c += DN_THREADS) {
            const int r = c >> 3, ch = c & 7, row = min(m0 + r, M - 1);
            cp16z(p + r * 128 + swz<8>(r, ch) * 16, H + static_cast<size_t>(row) * K + g * 64 + ch * 8, m0 + r < M);
        }
    };
    auto loadb = [&](int g, uint2 (&b)[3][4]) {
#pragma unroll
        for (int j = 0; j < 3; ++j)
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) b[j][kt] = valid[j] ? __ldg(wp[j] + static_cast<size_t>(g) * 1024 + kt * 32) : make_uint2(0u, 0u);
    };

    float acc[MT][3][4], tot[MT][3][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KG) load(s, s);
        commit();
    }
    uint2 bufa[3][4], bufb[3][4];
    loadb(0, bufa);

    auto body = [&](int it, uint2 (&cur)[3][4], uint2 (&nxt)[3][4]) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < KG) load(next % STAGES, next);
        commit();
        if (it + 1 < KG) loadb(it + 1, nxt);
        unsigned char* p = stage(it % STAGES);
        {
            // x = bf16(h * rinv[row, stream] * scale[k]) as fn_hc_normalize2 rounds it, in place
            const int s_id = (it * 64) / D;
            for (int c = tid; c < BM * 8; c += DN_THREADS) {
                const int r = c >> 3, ch = c & 7;
                uint4* cell = reinterpret_cast<uint4*>(p + r * 128 + swz<8>(r, ch) * 16);
                const uint4 raw = *cell;
                const __nv_bfloat16* hv = reinterpret_cast<const __nv_bfloat16*>(&raw);
                const float rv = rinv_s[r * DN_SS + s_id];
                const float4 sa = __ldg(reinterpret_cast<const float4*>(SCALE + it * 64 + ch * 8));
                const float4 sb = __ldg(reinterpret_cast<const float4*>(SCALE + it * 64 + ch * 8 + 4));
                const float sv[8] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w};
                __nv_bfloat16 o8[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) o8[i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(bf(hv[i]), rv), sv[i]));
                *cell = *reinterpret_cast<const uint4*>(o8);
            }
        }
        __syncthreads();
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint32_t a[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * 128 + swz<8>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < 3; ++j)
                if (valid[j]) {
#pragma unroll
                    for (int i = 0; i < MT; ++i) mma(acc[i][j], a[i], cur[j][kt].x, cur[j][kt].y);
                }
        }
        if ((it + 1) % per == 0) {
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < 3; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        tot[i][j][e] = it + 1 == per ? acc[i][j][e] : __fadd_rn(tot[i][j][e], acc[i][j][e]);
                        acc[i][j][e] = 0.0f;
                    }
        }
    };
    for (int it = 0; it < KG; it += 2) {
        body(it, bufa, bufb);
        if (it + 1 < KG) body(it + 1, bufb, bufa);
    }
    wait<0>();
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            if (!valid[j]) continue;
            const int col = (ct * DN_SPLIT + warp * 3 + j) * 8 + 2 * t4;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + i * 16 + g8 + h * 8;
                if (row >= M) continue;
                float* dst = PART + static_cast<size_t>(row) * N + col;
                if (col + 1 < N) {
                    if ((N & 1) == 0) *reinterpret_cast<float2*>(dst) = make_float2(tot[i][j][2 * h], tot[i][j][2 * h + 1]);
                    else {
                        dst[0] = tot[i][j][2 * h];
                        dst[1] = tot[i][j][2 * h + 1];
                    }
                } else if (col < N) dst[0] = tot[i][j][2 * h];
            }
        }
}

// up + mix: act times [4 D, K] bf16 weights, gated onto the normed streams, mixed; a CTA 16 MT rows by 32 columns.
template <int MT, int KG>
struct UpShape {
    static constexpr int BM = 16 * MT;
    static constexpr int K = KG * 64;
    static constexpr int A = BM * K * 2;                          // act rows, swizzled by 128-byte groups
    static constexpr int PARTS = BM * 4 * 32 * 4 + BM * 32 * 4;  // each stream's terms, then the mixed rows (on A)
    static constexpr int HS = BM * 4 * 32 * 2;                    // the streams' 32 dimensions of the CTA's rows
    static constexpr int RV = BM * 4 * 4;
    static constexpr int SMEM = (A > PARTS ? A : PARTS) + HS + RV;
};

template <int MT, int KG>
__device__ __forceinline__ void upmix_body(const __nv_bfloat16* __restrict__ ACT, const uint2* __restrict__ W,
                                           const __nv_bfloat16* __restrict__ H, const float* __restrict__ PSS,
                                           const float* __restrict__ SCALE, __nv_bfloat16* __restrict__ MIXED,
                                           float* __restrict__ XSM, int M, int D, int NC, float eps) {
    using S = UpShape<MT, KG>;
    constexpr int BM = S::BM, K = S::K, SS = 4;
    extern __shared__ __align__(128) unsigned char buf[];
    unsigned char* As = buf;
    constexpr int ABYTES = S::A > S::PARTS ? S::A : S::PARTS;
    __nv_bfloat16* Hs = reinterpret_cast<__nv_bfloat16*>(buf + ABYTES);
    float* rinv_s = reinterpret_cast<float*>(buf + ABYTES + S::HS);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, t4 = lane & 3;
    const int dt = D / 32;
    const int m0 = (blockIdx.x / dt) * BM, d0 = (blockIdx.x % dt) * 32;
    const int sw = warp >> 1, q0 = (warp & 1) * 2;

    // this warp's weights: the stream's two n8 tiles, every k16 step, straight from memory
    uint2 b[KG][4][2];
    const uint2* wp[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int c8 = sw * D + d0 + (q0 + j) * 8;
        wp[j] = W + static_cast<size_t>(c8 >> 6) * KG * 1024 + ((c8 >> 3) & 7) * 128 + lane;
    }
#pragma unroll
    for (int g = 0; g < KG; ++g)
#pragma unroll
        for (int kt = 0; kt < 4; ++kt)
#pragma unroll
            for (int j = 0; j < 2; ++j) b[g][kt][j] = __ldg(wp[j] + g * 1024 + kt * 32);

    for (int c = tid; c < BM * (K / 8); c += 256) {
        const int r = c / (K / 8), cc = c % (K / 8), g = cc >> 3, ch = cc & 7, row = min(m0 + r, M - 1);
        cp16z(As + r * (K * 2) + g * 128 + swz<8>(r, ch) * 16, ACT + static_cast<size_t>(row) * K + cc * 8, m0 + r < M);
    }
    for (int c = tid; c < BM * SS * 4; c += 256) {
        const int r = c / (SS * 4), s = (c >> 2) % SS, q = c & 3, row = min(m0 + r, M - 1);
        cp16z(reinterpret_cast<unsigned char*>(Hs) + ((r * SS + s) * 32 + q * 8) * 2,
              H + static_cast<size_t>(row) * (SS * D) + s * D + d0 + q * 8, m0 + r < M);
    }
    commit();
    for (int i = tid; i < BM * SS; i += 256) rinv_s[i] = rinv_of(PSS, min(m0 + i / SS, M - 1), i % SS, NC, SS, D, eps);
    wait<0>();
    __syncthreads();

    float acc[MT][2][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
    for (int g = 0; g < KG; ++g)
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint32_t a[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], As + r * (K * 2) + g * 128 + swz<8>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int i = 0; i < MT; ++i) mma(acc[i][j], a[i], b[g][kt][j].x, b[g][kt][j].y);
        }
    __syncthreads();  // A is dead: its bytes now hold the terms
    float* part = reinterpret_cast<float*>(As);
    float* mixed_s = part + BM * 4 * 32;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int rr = i * 16 + g8 + 8 * (e >> 1), dd = (q0 + j) * 8 + 2 * t4 + (e & 1);
                const float nv = rbf(__fmul_rn(__fmul_rn(bf(Hs[(rr * SS + sw) * 32 + dd]), rinv_s[rr * SS + sw]), SCALE[sw * D + d0 + dd]));
                part[(sw * BM + rr) * 32 + dd] = rbf(__fmul_rn(bsig(rbf(acc[i][j][e])), nv));
            }
    __syncthreads();
    for (int e = tid; e < BM * 32; e += 256) {
        const int rr = e >> 5, dd = e & 31;
        float total = 0.0f;
#pragma unroll
        for (int s = 0; s < SS; ++s) total = __fadd_rn(total, part[(s * BM + rr) * 32 + dd]);
        const float m = rbf(__fdiv_rn(total, static_cast<float>(SS)));
        mixed_s[e] = m;
        if (m0 + rr < M) MIXED[static_cast<size_t>(m0 + rr) * D + d0 + dd] = __float2bfloat16_rn(m);
    }
    __syncthreads();
    if (tid < BM && m0 + tid < M) {
        float sum = 0.0f;
        for (int k = 0; k < 32; ++k) sum = __fadd_rn(sum, mixed_s[tid * 32 + k]);
        XSM[static_cast<size_t>(m0 + tid) * dt + d0 / 32] = sum;
    }
}

}  // namespace hcn

#define HCN_DOWN(NAME, MT)                                                                                                 \
    extern "C" __global__ void __launch_bounds__(hcn::DN_THREADS) NAME(                                                   \
        const __nv_bfloat16* __restrict__ H, const float* __restrict__ PSS, const float* __restrict__ SCALE,              \
        const uint2* __restrict__ W, float* __restrict__ PART, int M, int N, int K, int SK, int D, int NC, int SS,        \
        float eps) {                                                                                                       \
        hcn::down_body<MT, 3>(H, PSS, SCALE, W, PART, M, N, K, SK, D, NC, SS, eps);                                       \
    }

HCN_DOWN(fn_hcn_down2, 2)
HCN_DOWN(fn_hcn_down3, 3)
HCN_DOWN(fn_hcn_down4, 4)

extern "C" __global__ void __launch_bounds__(256) fn_hcn_up3(const __nv_bfloat16* __restrict__ ACT, const uint2* __restrict__ W,
                                                             const __nv_bfloat16* __restrict__ H, const float* __restrict__ PSS,
                                                             const float* __restrict__ SCALE, __nv_bfloat16* __restrict__ MIXED,
                                                             float* __restrict__ XSM, int M, int D, int NC, float eps) {
    hcn::upmix_body<3, 5>(ACT, W, H, PSS, SCALE, MIXED, XSM, M, D, NC, eps);
}
