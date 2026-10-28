// Flash Next EXL3 prefill experts: upstream's transposed 64-member prompt tiles (exl3_prompt.cuh, after #976 and

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "../exl3_decode.cuh"
#include "../exl3_prompt.cuh"

namespace tf_exl3pf {
using namespace tf_exl3;
using tf_exl3_prompt::cp_async16;
using tf_exl3_prompt::cp_commit;
using tf_exl3_prompt::cp_wait;
using tf_exl3_prompt::ldmatrix4;
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
// Members a pass: decode's sums (exl3_experts.cu grouped_body) keep three accumulator sets live; 32 fit the registers.
constexpr int PM = 32;
constexpr int PNB = PM / 8;
constexpr int CHAIN_WARPS = 4;
template <int K2>
constexpr int stage_words() { return 16 * TKT * tile_words<K2>(); }
template <int XR>
constexpr int stage_halves() { return XR * TXS + 2 * stage_words<KMAX2>(); }
constexpr int smem_bytes() {
    const int gu = TSTAGES * stage_halves<2 * PM>() * 2;
    const int ep = 2 * PM * GU_ES * 4, dn = PM * DN_ES * 4;
    return gu > ep ? (gu > dn ? gu : dn) : (ep > dn ? ep : dn);
}

// K splits of exl3_experts.zig's plan: gate/up 4 (else 2, else 1), down 1.
__device__ __forceinline__ int k_splits(int K, bool gateup) {
    if (!gateup) return 1;
    return K % 256 == 0 ? 4 : (K % 128 == 0 ? 2 : 1);
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

// res[i][nb] = warp tile i x members nb * 8 .. + 7 over K, summed as grouped_body does; member i reads X[xfirst + i].
template <int K2, int MATS, int XR>
__device__ __forceinline__ void pmain(const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
        const uint32_t* __restrict__ T1, int xfirst, int cnt, int K, int NTILES, int cb, int chain_chunks, half* stage,
        float (&res)[TMT][PNB][4]) {
    constexpr int SH = stage_halves<XR>();
    constexpr int TW = tile_words<K2>();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int chunks = K / TBK;
    const int nblocks = (cnt + 15) >> 4;
    float acc[TMT][PNB][4], part[TMT][PNB][4];
#pragma unroll
    for (int i = 0; i < TMT; ++i)
#pragma unroll
        for (int j = 0; j < PNB; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = part[i][j][c] = res[i][j][c] = 0.f;
    auto load = [&](int chunk) {
        half* dst = stage + (size_t)(chunk % TSTAGES) * SH;
        const int k0 = chunk * TBK;
        for (int idx = threadIdx.x; idx < XR * (TBK / 8); idx += TT) {
            const int m = idx / (PM * (TBK / 8)), rem = idx % (PM * (TBK / 8)), i = rem / (TBK / 8), j = rem % (TBK / 8);
            if (i >= nblocks * 16) continue;
            const bool ok = i < cnt;
            const half* src = (m ? X1 : X0) + (size_t)(xfirst + (ok ? i : 0)) * K + k0 + j * 8;
            cp_async16(dst + ((size_t)m * PM + i) * TXS + j * 8, src, ok);
        }
        uint32_t* wdst = reinterpret_cast<uint32_t*>(dst + (size_t)XR * TXS);
        for (int idx = threadIdx.x; idx < 16 * TKT * K2; idx += TT) {   // a tile is K2 pieces of 16 bytes
            const int q = idx / (TKT * K2), kk = (idx / K2) % TKT, piece = idx % K2;
            int wm, nt;
            ttile<MATS>(q, cb, wm, nt);
            const uint32_t* src = (wm ? T1 : T0) + (((size_t)chunk * TKT + kk) * NTILES + nt) * TW + piece * 4;
            cp_async16(wdst + ((size_t)q * TKT + kk) * TW + piece * 4, src, true);
        }
    };
    int mat, nt_unused;
    ttile<MATS>(warp * TMT, cb, mat, nt_unused);
#pragma unroll
    for (int c = 0; c < TSTAGES - 1; ++c) {
        if (c < chunks) load(c);
        cp_commit();
    }
    int left = chain_chunks, w = 0;
    for (int c = 0; c < chunks; ++c) {
        cp_wait<TSTAGES - 2>();
        __syncthreads();
        if (c + TSTAGES - 1 < chunks) load(c + TSTAGES - 1);
        cp_commit();
        const half* base = stage + (size_t)(c % TSTAGES) * SH;
        const half* xs = base + (size_t)(XR == PM ? 0 : mat) * PM * TXS;
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
            for (int jb = 0; jb < PM / 16; ++jb) {
                if (jb < nblocks) {
                    uint32_t r[4];
                    const int row = jb * 16 + (lane & 7) + ((lane >> 4) & 1) * 8;
                    ldmatrix4(r, xs + (size_t)row * TXS + kk * 16 + ((lane >> 3) & 1) * 8);
                    const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
#pragma unroll
                    for (int i = 0; i < TMT; ++i) {
                        mma16816(acc[i][2 * jb], a[i], bl);
                        mma16816(acc[i][2 * jb + 1], a[i], bh);
                    }
                }
            }
        }
        if (--left == 0) {   // a chain ends: into its split's warp sum; a split's last into the total (0 + each, as the epilogue)
            left = chain_chunks;
#pragma unroll
            for (int i = 0; i < TMT; ++i)
#pragma unroll
                for (int j = 0; j < PNB; ++j)
#pragma unroll
                    for (int c4 = 0; c4 < 4; ++c4) {
                        part[i][j][c4] = w == 0 ? acc[i][j][c4] : __fadd_rn(part[i][j][c4], acc[i][j][c4]);
                        acc[i][j][c4] = 0.f;
                    }
            if (++w == CHAIN_WARPS) {
                w = 0;
#pragma unroll
                for (int i = 0; i < TMT; ++i)
#pragma unroll
                    for (int j = 0; j < PNB; ++j)
#pragma unroll
                        for (int c4 = 0; c4 < 4; ++c4) res[i][j][c4] = __fadd_rn(res[i][j][c4], part[i][j][c4]);
            }
        }
    }
    cp_wait<0>();
    __syncthreads();   // the stages become the epilogue's sums
}

// res -> E[member][col] fp32, row stride ES; col0: the warp's first column.
__device__ __forceinline__ void spill(float* E, int ES, int col0, const float (&res)[TMT][PNB][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < TMT; ++i)
#pragma unroll
        for (int q = 0; q < PNB; ++q) {
            const int tok = q * 8 + 2 * t, col = col0 + i * 16 + g;
            E[(size_t)tok * ES + col] = res[i][q][0];
            E[(size_t)(tok + 1) * ES + col] = res[i][q][1];
            E[(size_t)tok * ES + col + 8] = res[i][q][2];
            E[(size_t)(tok + 1) * ES + col + 8] = res[i][q][3];
        }
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
    const int chain_chunks = (K / 16 / k_splits(K, true) / CHAIN_WARPS) / TKT;
    int mat, nt;
    ttile<2>(warp * TMT, cb, mat, nt);
    const int col0 = (nt - cb * 8) * 16;
    const int n = cb * 128 + 4 * lane;
    float* Eg = reinterpret_cast<float*>(smem_raw);
    for (int m0 = 0; m0 < cnt; m0 += PM) {
        const int pc = cnt - m0 < PM ? cnt - m0 : PM;
        if (m0) __syncthreads();   // the last pass's sums are read before the stages are refilled
        float res[TMT][PNB][4];
        pmain<K2, 2, 2 * PM>(Xg, Xu, Tg, Tu, first + m0, pc, K, N >> 4, cb, chain_chunks, reinterpret_cast<half*>(smem_raw), res);
        spill(Eg + (size_t)mat * PM * GU_ES, GU_ES, col0, res);
        __syncthreads();
        for (int i = warp; i < pc; i += TNW) {
            float gv[4], uv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                gv[j] = Eg[(size_t)i * GU_ES + 4 * lane + j];
                uv[j] = Eg[(size_t)(PM + i) * GU_ES + 4 * lane + j];
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
            half* o = xd + (size_t)(first + m0 + i) * N + n;
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
    const int chain_chunks = (K / 16 / k_splits(K, false) / CHAIN_WARPS) / TKT;
    float* Ed = reinterpret_cast<float*>(smem_raw);
    for (int m0 = 0; m0 < cnt; m0 += PM) {
        const int pc = cnt - m0 < PM ? cnt - m0 : PM;
        if (m0) __syncthreads();
        float res[TMT][PNB][4];
        pmain<K2, 1, PM>(Xd, Xd, Td, Td, first + m0, pc, K, D >> 4, cb, chain_chunks, reinterpret_cast<half*>(smem_raw), res);
        spill(Ed, DN_ES, warp * TMT * 16, res);
        __syncthreads();
        for (int i = warp; i < pc; i += TNW) {
            const int p = rows_sh[m0 + i];
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

// The canonical expert reduction at any degree: 64 leaves, a balanced tree; a rank sums its maximal complete subtrees.
__device__ __forceinline__ int pf_blocks(int lo, int hi, int* st, int* sz) {
    int n = 0;
    for (int at = lo; at < hi;) {
        int size = 1;
        while (at % (size * 2) == 0 && at + size * 2 <= hi) size *= 2;
        st[n] = at;
        sz[n] = size;
        ++n;
        at += size;
    }
    return n;
}

// Block partials of one rank into out[sub-block][rank * M + j][row][D]: a sub-block's slice is one all-gather's region.
extern "C" __global__ void __launch_bounds__(256) tf_exl3pf_tp_combine(const float* __restrict__ y, const float* __restrict__ wts,
        const int* __restrict__ pick, float* __restrict__ out, int slots, int D, int n, int sub, int world, int rank, int M) {
    __shared__ float w_sh[16];
    __shared__ int nl_sh;
    __shared__ int leaf_sh[16];
    __shared__ unsigned mask_sh[16];
    const int r = blockIdx.x;
    const int llo = 64 * rank / world, lhi = 64 * (rank + 1) / world;
    if ((int)threadIdx.x < slots) w_sh[threadIdx.x] = wts[r * slots + threadIdx.x];
    if (threadIdx.x == 0) {
        int nl = 0;
        for (int k = 0; k < slots; ++k) {
            const int p = pick[r * slots + k];
            if (p < 0) continue;
            const int leaf = p >= 512 ? 63 : p / 8;
            if (leaf < llo || leaf >= lhi) continue;
            int at = 0;
            while (at < nl && leaf_sh[at] < leaf) ++at;
            if (at < nl && leaf_sh[at] == leaf) { mask_sh[at] |= 1u << k; continue; }
            for (int i = nl; i > at; --i) { leaf_sh[i] = leaf_sh[i - 1]; mask_sh[i] = mask_sh[i - 1]; }
            leaf_sh[at] = leaf;
            mask_sh[at] = 1u << k;
            ++nl;
        }
        nl_sh = nl;
    }
    __syncthreads();
    const int nl = nl_sh;
    int bst[16], bsz[16];
    const int nb = pf_blocks(llo, lhi, bst, bsz);
    const int b = r / sub, rr = r - b * sub;
    const int mb = min(sub, n - b * sub);
    const float* yr = y + (size_t)r * slots * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float val[16];
        int st[16];
        for (int i = 0; i < nl; ++i) {
            float acc = 0.0f;
            for (unsigned m = mask_sh[i]; m; m &= m - 1) {
                const int k = __ffs(m) - 1;
                acc = fmaf(yr[(size_t)k * D + d], w_sh[k], acc);
            }
            val[i] = acc;
            st[i] = leaf_sh[i];
        }
        for (int j = 0; j < nb; ++j) {
            float v[16];
            int s[16];
            int cnt = 0;
            for (int i = 0; i < nl; ++i) if (st[i] >= bst[j] && st[i] < bst[j] + bsz[j]) { v[cnt] = val[i]; s[cnt] = st[i]; ++cnt; }
            for (int size = 1; size < bsz[j]; size *= 2) {
                int m = 0;
                for (int i = 0; i < cnt;) {
                    const int start = s[i] & ~(2 * size - 1);
                    if (s[i] % (2 * size) == 0 && i + 1 < cnt && s[i + 1] == s[i] + size) {
                        v[m] = __fadd_rn(v[i], v[i + 1]);
                        i += 2;
                    } else {
                        v[m] = v[i];
                        ++i;
                    }
                    s[m] = start;
                    ++m;
                }
                cnt = m;
            }
            out[(((size_t)b * sub * world * M + (size_t)(rank * M + j) * mb + rr) * D) + d] = cnt ? v[0] : 0.0f;
        }
    }
}

// The tree finished from the gathered partials: equal adjacent aligned blocks merge by fadd_rn until one is left.
extern "C" __global__ void __launch_bounds__(256) tf_exl3pf_tp_finish(const float* __restrict__ gath, float* __restrict__ sum,
        int D, int m, int world, int M) {
    __shared__ int bs_sh[96], bz_sh[96], bq_sh[96];
    __shared__ int nb_sh;
    if (threadIdx.x == 0) {
        int n = 0;
        for (int q = 0; q < world; ++q) {
            int st[16], sz[16];
            const int c = pf_blocks(64 * q / world, 64 * (q + 1) / world, st, sz);
            for (int j = 0; j < c; ++j) { bs_sh[n] = st[j]; bz_sh[n] = sz[j]; bq_sh[n] = q * M + j; ++n; }
        }
        nb_sh = n;
    }
    __syncthreads();
    const int rr = blockIdx.x;
    const int nb = nb_sh;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v[96];
        int s[96], z[96];
        for (int i = 0; i < nb; ++i) { v[i] = gath[((size_t)bq_sh[i] * m + rr) * D + d]; s[i] = bs_sh[i]; z[i] = bz_sh[i]; }
        int cnt = nb;
        while (cnt > 1) {
            int k = 0;
            for (int i = 0; i < cnt;) {
                if (i + 1 < cnt && z[i] == z[i + 1] && s[i] % (z[i] * 2) == 0) {
                    v[k] = __fadd_rn(v[i], v[i + 1]);
                    s[k] = s[i];
                    z[k] = z[i] * 2;
                    i += 2;
                } else {
                    v[k] = v[i];
                    s[k] = s[i];
                    z[k] = z[i];
                    ++i;
                }
                ++k;
            }
            cnt = k;
        }
        sum[(size_t)rr * D + d] = v[0];
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
