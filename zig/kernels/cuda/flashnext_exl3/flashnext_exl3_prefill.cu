// Flash Next EXL3 prefill experts: upstream's transposed 64-member prompt tiles (exl3_prompt.cuh, after #976 and

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "../exl3_decode.cuh"
#include "../exl3_prompt.cuh"

namespace tf_exl3pf {
using namespace tf_exl3;
using tf_exl3_prompt::cp_async16;
using tf_exl3_prompt::cp_async_commit;
using tf_exl3_prompt::cp_async_wait;
using tf_exl3_prompt::ldmatrix_x4;
using tf_exl3_prompt::tspill;
using tf_exl3_prompt::ttile;
using tf_exl3_prompt::TPM;
using tf_exl3_prompt::TNW;
using tf_exl3_prompt::TMT;
using tf_exl3_prompt::TNB;
using tf_exl3_prompt::TT;
using tf_exl3_prompt::TPH;
using tf_exl3_prompt::TBK;
using tf_exl3_prompt::TKT;
using tf_exl3_prompt::TSTAGES;
using tf_exl3_prompt::TXS;
using tf_exl3_prompt::GU_ES;
using tf_exl3_prompt::DN_ES;

constexpr int KMAX2 = 10;   // the widest tile a stage holds (5 bits): Flash Next's routed 3-bit and shared 5-bit experts
template <int K2>
constexpr int stage_words() { return 16 * TKT * tile_words<K2>(); }
template <int XR>
constexpr int stage_halves() { return XR * TXS + 2 * stage_words<KMAX2>(); }
constexpr int smem_bytes() {
    const int gu = TSTAGES * stage_halves<2 * TPM>() * 2;
    const int ep = 2 * TPH * GU_ES * 4, dn = TPH * DN_ES * 4;
    return gu > ep ? (gu > dn ? gu : dn) : (ep > dn ? ep : dn);
}

template <int K2>
__device__ __forceinline__ void decode_tile_k(const uint32_t* tile, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t w[lane_words<K2>()];
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int i = 0; i < lane_words<K2>(); ++i) w[i] = tile[(word + i) % tile_words<K2>()];
    decode_lane<K2, CB_MUL1>(w, lane, b0, b1);
}

// acc[i][nb] = warp tile i x members nb * 8 .. + 7 over K; member i's input row is X[xfirst + i] (compact rows).
template <int K2, int MATS, int XR>
__device__ __forceinline__ void pmain(const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
        const uint32_t* __restrict__ T1, int xfirst, int cnt, int K, int NTILES, int cb, half* stage, float (&acc)[TMT][TNB][4]) {
    constexpr int SH = stage_halves<XR>();
    constexpr int TW = tile_words<K2>();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int chunks = K / TBK;
    const int nblocks = (cnt + 15) >> 4;
#pragma unroll
    for (int i = 0; i < TMT; ++i)
#pragma unroll
        for (int j = 0; j < TNB; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
    auto load = [&](int chunk) {
        half* dst = stage + (size_t)(chunk % TSTAGES) * SH;
        const int k0 = chunk * TBK;
        for (int idx = threadIdx.x; idx < XR * (TBK / 8); idx += TT) {
            const int m = idx / (TPM * (TBK / 8)), rem = idx % (TPM * (TBK / 8)), i = rem / (TBK / 8), j = rem % (TBK / 8);
            if (i >= nblocks * 16) continue;
            const bool ok = i < cnt;
            const half* src = (m ? X1 : X0) + (size_t)(xfirst + (ok ? i : 0)) * K + k0 + j * 8;
            cp_async16(dst + ((size_t)m * TPM + i) * TXS + j * 8, src, ok);
        }
        uint32_t* wdst = reinterpret_cast<uint32_t*>(dst + (size_t)XR * TXS);
        for (int idx = threadIdx.x; idx < 16 * TKT * K2; idx += TT) {   // a tile is K2 pieces of 16 bytes
            const int q = idx / (TKT * K2), kk = (idx / K2) % TKT, part = idx % K2;
            int wm, nt;
            ttile<MATS>(q, cb, wm, nt);
            const uint32_t* src = (wm ? T1 : T0) + (((size_t)chunk * TKT + kk) * NTILES + nt) * TW + part * 4;
            cp_async16(wdst + ((size_t)q * TKT + kk) * TW + part * 4, src, true);
        }
    };
    int mat, nt_unused;
    ttile<MATS>(warp * TMT, cb, mat, nt_unused);
#pragma unroll
    for (int c = 0; c < TSTAGES - 1; ++c) {
        if (c < chunks) load(c);
        cp_async_commit();
    }
    for (int c = 0; c < chunks; ++c) {
        cp_async_wait<TSTAGES - 2>();
        __syncthreads();
        if (c + TSTAGES - 1 < chunks) load(c + TSTAGES - 1);
        cp_async_commit();
        const half* base = stage + (size_t)(c % TSTAGES) * SH;
        const half* xs = base + (size_t)(XR == TPM ? 0 : mat) * TPM * TXS;
        const uint32_t* ws = reinterpret_cast<const uint32_t*>(base + (size_t)XR * TXS);
#pragma unroll
        for (int kk = 0; kk < TKT; ++kk) {
            uint32_t a[TMT][4];
#pragma unroll
            for (int i = 0; i < TMT; ++i) {
                uint32_t b0[2], b1[2];
                decode_tile_k<K2>(ws + ((size_t)(warp * TMT + i) * TKT + kk) * TW, lane, b0, b1);
                a[i][0] = b0[0];
                a[i][1] = b1[0];
                a[i][2] = b0[1];
                a[i][3] = b1[1];
            }
#pragma unroll
            for (int jb = 0; jb < TPM / 16; ++jb) {
                if (jb < nblocks) {
                    uint32_t r[4];
                    const int row = jb * 16 + (lane & 7) + ((lane >> 4) & 1) * 8;
                    ldmatrix_x4(r, xs + (size_t)row * TXS + kk * 16 + ((lane >> 3) & 1) * 8);
                    const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
#pragma unroll
                    for (int i = 0; i < TMT; ++i) {
                        mma16816(acc[i][2 * jb], a[i], bl);
                        mma16816(acc[i][2 * jb + 1], a[i], bh);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();   // the stages become the epilogue's sums
}

__device__ __forceinline__ float upTo(float x, float l) { return x > l ? l : x; }
__device__ __forceinline__ float within(float x, float l) { return x > l ? l : (x < -l ? -l : x); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

template <int K2>
__device__ __forceinline__ void gateup(const half* __restrict__ Xg, const half* __restrict__ Xu, const uint32_t* Tg, const uint32_t* Tu,
        int e, int first, int cnt, const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
        half* __restrict__ xd, int K, int N, float limit, int act_mode, unsigned char* smem_raw) {
    const int cb = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[TMT][TNB][4];
    pmain<K2, 2, 2 * TPM>(Xg, Xu, Tg, Tu, first, cnt, K, N >> 4, cb, reinterpret_cast<half*>(smem_raw), acc);
    float* Eg = reinterpret_cast<float*>(smem_raw);
    int mat, nt;
    ttile<2>(warp * TMT, cb, mat, nt);
    const int col0 = (nt - cb * 8) * 16;
    const int n = cb * 128 + 4 * lane;
    for (int hf = 0; hf < 2 && hf * TPH < cnt; ++hf) {
        if (hf) __syncthreads();
        tspill(Eg + (size_t)mat * TPH * GU_ES, GU_ES, col0, acc, hf);
        __syncthreads();
        for (int i = warp; i < TPH && hf * TPH + i < cnt; i += TNW) {
            float gv[4], uv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                gv[j] = Eg[(size_t)i * GU_ES + 4 * lane + j];
                uv[j] = Eg[(size_t)(TPH + i) * GU_ES + 4 * lane + j];
            }
            fwht128(gv, lane);
            fwht128(uv, lane);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float act;
                if (act_mode == 0) {
                    float gg = bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]));
                    float uu = bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]));
                    if (limit > 0) { gg = upTo(gg, limit); uu = within(uu, limit); }
                    act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                } else {   // runSlots' tf_exl3_gateup_epilogue, act_mode 1, operation for operation
                    float gg = gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]);
                    float uu = uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]);
                    if (limit > 0) { gg = upTo(gg, limit); uu = within(uu, limit); }
                    act = gg / (1.f + expf(-gg)) * uu;
                }
                v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
            }
            fwht128(v, lane);
            half* o = xd + (size_t)(first + hf * TPH + i) * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
        }
    }
}

template <int K2>
__device__ __forceinline__ void down(const half* __restrict__ Xd, const uint32_t* Td, int e, int first, int cnt, const int* rows_sh,
        const half* __restrict__ svh_d, float* __restrict__ y, int K, int D, unsigned char* smem_raw) {
    const int cb = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[TMT][TNB][4];
    pmain<K2, 1, TPM>(Xd, Xd, Td, Td, first, cnt, K, D >> 4, cb, reinterpret_cast<half*>(smem_raw), acc);
    float* Ed = reinterpret_cast<float*>(smem_raw);
    for (int hf = 0; hf < 2 && hf * TPH < cnt; ++hf) {
        if (hf) __syncthreads();
        tspill(Ed, DN_ES, warp * TMT * 16, acc, hf);
        __syncthreads();
        for (int i = warp; i < TPH && hf * TPH + i < cnt; i += TNW) {
            const int p = rows_sh[hf * TPH + i];
#pragma unroll
            for (int blk = 0; blk < 2; ++blk) {
                const int n = cb * 256 + blk * 128 + 4 * lane;
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = Ed[(size_t)i * DN_ES + blk * 128 + 4 * lane + j];
                fwht128(v, lane);
                float* o = y + (size_t)p * D + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
            }
        }
    }
}

}  // namespace tf_exl3pf

using namespace tf_exl3pf;

// Program (compact member c, 128-block of D, matrix): xg/xu[c] = fp16((x[row] * suh[e]) @ H) for member c's pair.
extern "C" __global__ void tf_exl3pf_rotate(const __nv_bfloat16* __restrict__ x, int x_stride, const int* __restrict__ pick,
        const int* __restrict__ members, int slots, const half* __restrict__ suh_g, const half* __restrict__ suh_u,
        half* __restrict__ xg, half* __restrict__ xu, int D) {
    const int c = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    const int p = members[c];
    const int e = pick[p];
    const int row = p / slots;
    const int lane = threadIdx.x;
    const half* suh = (mat ? suh_u : suh_g) + (size_t)e * D + blk * 128 + 4 * lane;
    const __nv_bfloat16* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? xu : xg) + (size_t)c * D + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

#define TF_EXL3PF_K2(X) \
    X(4)                \
    X(6)                \
    X(8)                \
    X(10)

// Block (128 width columns of gate and up, item): SwiGLU of the item's members into compact xd rows.
extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3pf_gu(const half* __restrict__ Xg, const half* __restrict__ Xu,
        const uint64_t* __restrict__ Tg, const uint64_t* __restrict__ Tu, const int* __restrict__ k2g, const int* __restrict__ items,
        const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d, half* __restrict__ xd, int K,
        int N, float limit, int act_mode) {
    __shared__ __align__(16) unsigned char smem_raw[smem_bytes()];
    const int item = blockIdx.y;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    const uint32_t* G = reinterpret_cast<const uint32_t*>(Tg[e]);
    const uint32_t* U = reinterpret_cast<const uint32_t*>(Tu[e]);
    switch (k2g[e]) {
#define TF_CASE(K2_) \
    case K2_: gateup<K2_>(Xg, Xu, G, U, e, first, cnt, svh_g, svh_u, suh_d, xd, K, N, limit, act_mode, smem_raw); break;
        TF_EXL3PF_K2(TF_CASE)
#undef TF_CASE
        default: __trap();
    }
}

// Block (256 model-width columns, item): y[pair] = the down epilogue of the item's members.
extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3pf_down(const half* __restrict__ Xd, const uint64_t* __restrict__ Td,
        const int* __restrict__ k2d, const int* __restrict__ items, const int* __restrict__ members, const half* __restrict__ svh_d,
        float* __restrict__ y, int K, int D) {
    __shared__ __align__(16) unsigned char smem_raw[smem_bytes()];
    __shared__ int rows_sh[TPM];
    const int item = blockIdx.y;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (threadIdx.x < TPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const uint32_t* T = reinterpret_cast<const uint32_t*>(Td[e]);
    switch (k2d[e]) {
#define TF_CASE(K2_) \
    case K2_: down<K2_>(Xd, T, e, first, cnt, rows_sh, svh_d, y, K, D, smem_raw); break;
        TF_EXL3PF_K2(TF_CASE)
#undef TF_CASE
        default: __trap();
    }
}

// Expert parallel (cuda_exl3_tp.zig): a rank's deep chunk computed only its own pairs (picks in [LO, HI)); row r's
extern "C" __global__ void __launch_bounds__(256) tf_exl3pf_tp_combine(const float* __restrict__ y, const float* __restrict__ wts,
        const int* __restrict__ pick, float* __restrict__ sum, int slots, int D, int lo, int hi) {
    __shared__ float w_sh[32];
    __shared__ int own_sh[32];
    const int r = blockIdx.x;
    if ((int)threadIdx.x < slots) {
        const int p = pick[r * slots + threadIdx.x];
        own_sh[threadIdx.x] = p >= lo && p < hi;
        w_sh[threadIdx.x] = wts[r * slots + threadIdx.x];
    }
    __syncthreads();
    const float* yr = y + (size_t)r * slots * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < slots; ++k)
            if (own_sh[k]) acc = fmaf(yr[(size_t)k * D + d], w_sh[k], acc);
        sum[(size_t)r * D + d] = acc;
    }
}

// The ranks' summed branch back where the write-back reads it: y[r, 0, :] = sum[r], weights {1, 0, ..., 0}. The other
extern "C" __global__ void __launch_bounds__(256) tf_exl3pf_tp_place(const float* __restrict__ sum, float* __restrict__ y,
        float* __restrict__ wts, int slots, int D) {
    const int r = blockIdx.x;
    float* out = y + (size_t)r * slots * D;
    const float* in = sum + (size_t)r * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) out[d] = in[d];
    if ((int)threadIdx.x < slots) wts[r * slots + threadIdx.x] = threadIdx.x == 0 ? 1.0f : 0.0f;
}
