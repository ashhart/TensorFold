// Transposed prompt helpers after MiaAI-Lab (GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold 0004, 0009, Apache-2.0).
#pragma once
// Included by exl3.cu after its shared helpers (mcg decode, cp.async, ldmatrix), in the same translation unit.

namespace {

constexpr int TPM = 64, TNW = 8, TMT = 16 / TNW, TNB = TPM / 8, TT = TNW * 32, TPH = TPM / 2;
constexpr int TBK = 32, TKT = TBK / 16, TSTAGES = 3, TXS = TBK + 8;   // a stage: 32 k values a row; +16 B: ldmatrix rows on distinct banks
constexpr int TWORDS = 16 * TKT * 32;                                  // a stage's trellis words (16 tiles x 2 k tiles x 32)
constexpr int GU_ES = 128 + 4, DN_ES = 256 + 4;                        // the epilogue sums' row strides (floats)
template <int XR>
constexpr int tstage_halves() { return XR * TXS + 2 * TWORDS; }
constexpr int tsmem_bytes() {
    const int gu = TSTAGES * tstage_halves<2 * TPM>() * 2;   // gate and up rows apart (no shared rotation)
    const int ep = 2 * TPH * GU_ES * 4, dn = TPH * DN_ES * 4;
    return gu > ep ? (gu > dn ? gu : dn) : (ep > dn ? ep : dn);
}

// Which (matrix, weight tile) warp-tile q (0..15) of column block cb is.
template <int MATS>
__device__ __forceinline__ void ttile(int q, int cb, int& mat, int& nt) {
    mat = MATS == 2 ? q >> 3 : 0;
    nt = MATS == 2 ? cb * 8 + (q & 7) : cb * 16 + q;
}

// Main loop: acc[i][nb] = warp tile i x members nb * 8 .. + 7 over K; XR rows a stage (TPM if X0 == X1, else 2 TPM).
template <int MATS, int XR>
__device__ __forceinline__ void tmain(const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
        const uint32_t* __restrict__ T1, const int* rows_sh, int cnt, int e, int K, int NTILES, int cb, int xshift,
        half* stage, float (&acc)[TMT][TNB][4]) {
    constexpr int SH = tstage_halves<XR>();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int mat, nt_unused;
    ttile<MATS>(warp * TMT, cb, mat, nt_unused);
    const int KT = K >> 4, chunks = K / TBK;
    const int nblocks = (cnt + 15) >> 4;   // 16-member blocks in use
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
            if (i >= nblocks * 16) continue;   // a 16-member block without a member: never read
            const int r = rows_sh[i];
            const half* src = (m ? X1 : X0) + (size_t)(r < 0 ? 0 : (r >> xshift)) * K + k0 + j * 8;
            cp_async16(dst + ((size_t)m * TPM + i) * TXS + j * 8, src, r >= 0);
        }
        uint32_t* wdst = reinterpret_cast<uint32_t*>(dst + (size_t)XR * TXS);
        for (int idx = threadIdx.x; idx < 16 * TKT * 8; idx += TT) {   // 16-byte pieces of the 128-byte tile words
            const int q = idx / (TKT * 8), kk = (idx / 8) % TKT, part = idx % 8;
            int wm, nt;
            ttile<MATS>(q, cb, wm, nt);
            const uint32_t* src = (wm ? T1 : T0) + (((size_t)e * KT + chunk * TKT + kk) * NTILES + nt) * 32 + part * 4;
            cp_async16(wdst + ((size_t)q * TKT + kk) * 32 + part * 4, src, true);
        }
    };
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
                decode_tile_p(ws[((size_t)(warp * TMT + i) * TKT + kk) * 32 + lane], lane, b0, b1);
                a[i][0] = b0[0]; a[i][1] = b1[0]; a[i][2] = b0[1]; a[i][3] = b1[1];
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

// acc (members TPH * hf .. + TPH - 1) -> E[member - TPH * hf][col] fp32, row stride ES; col0: first tile's column.
__device__ __forceinline__ void tspill(float* E, int ES, int col0, const float (&acc)[TMT][TNB][4], int hf) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < TMT; ++i)
#pragma unroll
        for (int q = 0; q < TPM / 16; ++q) {
            const int nb = hf * (TPM / 16) + q;
            const int tok = q * 8 + 2 * t, col = col0 + i * 16 + g;
            E[(size_t)tok * ES + col] = acc[i][nb][0];
            E[(size_t)(tok + 1) * ES + col] = acc[i][nb][1];
            E[(size_t)tok * ES + col + 8] = acc[i][nb][2];
            E[(size_t)(tok + 1) * ES + col + 8] = acc[i][nb][3];
        }
}

template <int XR>
__device__ __forceinline__ void tgateup(const half* __restrict__ Xg, const half* __restrict__ Xu, const uint32_t* __restrict__ Tg,
        const uint32_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
        const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d, half* __restrict__ xd,
        int K, int N, int xshift, float limit, unsigned char* smem_raw, int* rows_sh) {
    const int cb = blockIdx.x, item = blockIdx.y;
    if (item >= counts[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (threadIdx.x < TPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[TMT][TNB][4];
    tmain<2, XR>(Xg, Xu, Tg, Tu, rows_sh, cnt, e, K, N >> 4, cb, xshift, reinterpret_cast<half*>(smem_raw), acc);
    float* Eg = reinterpret_cast<float*>(smem_raw);
    int mat, nt;
    ttile<2>(warp * TMT, cb, mat, nt);
    const int col0 = (nt - cb * 8) * 16;
    const int n = cb * 128 + 4 * lane;
    for (int hf = 0; hf < 2 && hf * TPH < cnt; ++hf) {
        if (hf) __syncthreads();   // the first half's rows are read
        tspill(Eg + (size_t)mat * TPH * GU_ES, GU_ES, col0, acc, hf);
        __syncthreads();
        for (int i = warp; i < TPH && hf * TPH + i < cnt; i += TNW) {
            const int p = rows_sh[hf * TPH + i];
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
                float gg = clamp_hi(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
                float uu = clamp_both(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), limit);
                float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
            }
            fwht128(v, lane);
            half* o = xd + (size_t)p * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
        }
    }
}

}  // namespace
