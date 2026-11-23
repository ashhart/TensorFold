// prompt16.cu's lean tile (text included inside namespace tf_prompt16): prompt_lean and its gate/up SwiGLU form.
#pragma once

template <int MODE, int BM, int BN, int WM, int WN, int STAGES>
struct LeanTile {
    using T = Tile<MODE, BM, BN, WM, WN, STAGES>;
    static constexpr int SMEM = T::SMEM + 2 * STAGES * 8;  // the stages, then their full and empty barriers
};

// prompt_lean's outputs: the bf16 or fp32 product, or bf16(silu(gate) * up) of a gate/up pair (the act kernel's bits)
enum LeanOut : int { LEAN_BF16 = 0, LEAN_F32 = 1, LEAN_SWIGLU = 2 };

// prompt_kernel's bits on a 128 x 256 tile: same fragments and mma chain over (g, kt); persistent, mbarrier ring.
template <int MODE, int BM, int BN, int WM, int WN, int STAGES, int OUT>
__global__ void __launch_bounds__(WM * WN * 32, 1) prompt_lean(
        const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        float scale, void* __restrict__ out, int M, int N, int K, int npad, int ldx, int group,
        const unsigned char* __restrict__ w2, const uint8_t* __restrict__ bs2, float scale2,
        const int* __restrict__ P) {
    using T = Tile<MODE, BM, BN, WM, WN, STAGES>;
    constexpr bool F32 = OUT == LEAN_F32, PAIR = OUT == LEAN_SWIGLU;
    constexpr int TN = PAIR ? BN / 2 : BN;  // output columns a tile: a pair's 128 features (gate's and up's blocks)
    static_assert(!PAIR || (MODE == FP4 && BN == 256 && WN == 8), "the pair tile is the FP4 128 x 256 lean tile");
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS;
    const int tiles = (M + BM - 1) / BM * ((N + TN - 1) / TN);
    constexpr int TILE_BYTES = MODE == FP4 ? 64 * GS / 2 : 64 * GS;
    constexpr int XC = BM * T::CHUNKS, WC = T::W / 16, SC = T::S / 16;
    constexpr int XN = (XC + T::THREADS - 1) / T::THREADS, WNC = (WC + T::THREADS - 1) / T::THREADS;
    constexpr int SN = SC > 0 ? (SC + T::THREADS - 1) / T::THREADS : 1;
    constexpr int PER = SC > 0 ? T::S / (BN / 64) : 16;
    const __nv_bfloat16* xs[XN];
    uint32_t xd[XN];
    bool xin[XN], xon[XN];
    const unsigned char* wsrc[WNC];
    bool win[WNC];
    const uint8_t* ssrc[SN];
    bool sin[SN];
    auto plan = [&](int tile) {  // the copies of tile's groups
        const int2 at = tile_of(tile, M, N, BM, TN, group);
#pragma unroll
        for (int i = 0; i < XN; ++i) {
            const int c = tid + i * T::THREADS, r = min(c / T::CHUNKS, BM - 1), ch = c % T::CHUNKS;
            xs[i] = x + static_cast<size_t>(min(at.x + r, M - 1)) * ldx + ch * 8;
            xd[i] = r * T::ROW + swz<T::CHUNKS>(r, ch) * 16;
            xin[i] = at.x + r < M;
            xon[i] = c < XC;
        }
#pragma unroll
        for (int i = 0; i < WNC; ++i) {
            const int c = tid + i * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            const int tb = PAIR ? (t & 1) : t;  // the pair's blocks 0, 1 are gate's, 2, 3 up's, at the same features
            wsrc[i] = (PAIR && t >= 2 ? w2 : w) + static_cast<size_t>(at.y / 64 + tb) * KG * TILE_BYTES + off * 16;
            win[i] = c < WC && at.y + tb * 64 < npad;
        }
#pragma unroll
        for (int i = 0; i < SN; ++i) {
            const int c = tid + i * T::THREADS, t = c / (PER / 16), off = c % (PER / 16);
            const int tb = PAIR ? (t & 1) : t;
            ssrc[i] = (PAIR && t >= 2 ? bs2 : bs) + static_cast<size_t>(at.y / 64 + tb) * KG * PER + off * 16;
            sin[i] = c < SC && at.y + tb * 64 < npad;
        }
    };
    uint64_t* full = reinterpret_cast<uint64_t*>(buf + STAGES * T::STAGE);
    uint64_t* empty = full + STAGES;
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    // the copy stream: its count of issued stages, and the tile and group it copies next
    uint32_t issued = 0;
    int lt = blockIdx.x, lg = 0;
    auto load = [&]() {
        const int s = issued % STAGES;
        if (issued >= STAGES) mbar_wait(&empty[s], (issued / STAGES - 1) & 1);
        unsigned char* p = stage(s);
#pragma unroll
        for (int i = 0; i < XN; ++i)
            if (xon[i]) cp16z(p + xd[i], xs[i] + lg * GS, xin[i]);
#pragma unroll
        for (int i = 0; i < WNC; ++i)
            if (win[i]) cp16(p + T::X + (tid + i * T::THREADS) * 16, wsrc[i] + static_cast<size_t>(lg) * TILE_BYTES);
        if constexpr (SC > 0) {
#pragma unroll
            for (int i = 0; i < SN; ++i)
                if (sin[i]) cp16(p + T::X + T::W + (tid + i * T::THREADS) * 16, ssrc[i] + static_cast<size_t>(lg) * PER);
        }
        mbar_copies(&full[s]);
        ++issued;
        if (++lg == KG) {
            lg = 0;
            lt += gridDim.x;
            if (lt < tiles) plan(lt);
        }
    };

    if (blockIdx.x >= tiles) return;
    if (tid == 0)
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full[s], T::THREADS);
            mbar_init(&empty[s], T::THREADS / 32);
        }
    __syncthreads();
    plan(lt);
    for (int s = 0; s < STAGES - 1 && lt < tiles; ++s) load();
    uint32_t used = 0;
    for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const int2 at = tile_of(tile, M, N, BM, TN, group);
        const int m0 = at.x, n0 = at.y;
        float acc[T::MT][T::NT][4];
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
        for (int g = 0; g < KG; ++g, ++used) {
            mbar_wait(&full[used % STAGES], (used / STAGES) & 1);
            const unsigned char* p = stage(used % STAGES);
            const unsigned char* pw = p + T::X;
            const uint8_t* ps = p + T::X + T::W;
            uint4 wq[T::NT];
            uint32_t sv[T::NT][4];
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                // a pair's warp: gate subtiles 2wn, 2wn + 1, up's 16 + 2wn, 17 + 2wn (one feature's sums in one lane)
                const int jj = PAIR ? (j >> 1) * 16 + 2 * wn + (j & 1) : wn * T::NT + j;
                const int col = (jj & 7) * 8 + (lane >> 2);
                if constexpr (MODE == FP4) {
                    const uint2 u = reinterpret_cast<const uint2*>(pw)[jj * 32 + lane];
                    wq[j] = make_uint4(u.x, u.y, 0u, 0u);
                    const uint32_t b4 = *reinterpret_cast<const uint32_t*>(ps + (jj >> 3) * 256 + col * 4);
                    // fp8pair is lane-wise: two blocks a call, each half then copied to both (sv as prompt_kernel's)
#if TF_FP_NATIVE
                    const uint32_t s01 = tf_common::e4m3x2(b4), s23 = tf_common::e4m3x2(b4 >> 16);
#else
                    const uint32_t s01 = tf_common::fp8pair_unit(b4, 0x7B807B80u), s23 = tf_common::fp8pair_unit(b4 >> 16, 0x7B807B80u);
#endif
                    sv[j][0] = prmt(s01, 0u, 0x1010u);
                    sv[j][1] = prmt(s01, 0u, 0x3232u);
                    sv[j][2] = prmt(s23, 0u, 0x1010u);
                    sv[j][3] = prmt(s23, 0u, 0x3232u);
                } else {
                    wq[j] = reinterpret_cast<const uint4*>(pw)[jj * 32 + lane];
                }
            }
#pragma unroll
            for (int kt = 0; kt < GS / 16; ++kt) {
                uint32_t a[T::MT][4];
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int ch = kt * 2 + (lane >> 4);
                    ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
                }
#pragma unroll
                for (int j = 0; j < T::NT; ++j) {
                    uint32_t b0, b1;
                    if constexpr (MODE == FP4) {
                        e2m1_pairs(kt < 2 ? wq[j].x : wq[j].y, kt & 1, b0, b1);
                        b0 = mul2(b0, sv[j][kt]);
                        b1 = mul2(b1, sv[j][kt]);
                    } else {
                        const uint32_t word = comp(wq[j], kt);
                        b0 = tf_common::fp8pair_unit(word & 0xFFFFu, 0x7B807B80u);
                        b1 = tf_common::fp8pair_unit(word >> 16, 0x7B807B80u);
                    }
#pragma unroll
                    for (int i = 0; i < T::MT; ++i) mma(acc[i][j], a[i], b0, b1);
                }
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(&empty[used % STAGES]);
            if (lt < tiles) load();
        }
        if constexpr (PAIR) {
            const int rows = P != nullptr ? P[2] : M;  // tf_q35_mlp_act's rows: the device count when there is one
#pragma unroll
            for (int i = 0; i < T::MT; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int col = n0 + wn * 16 + j * 8 + (lane & 3) * 2;
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                        if (row >= rows) continue;
                        // the separate launches' bf16 gate and up, then tf_q35_mlp_act's float steps and rounding
                        __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
#pragma unroll
                        for (int e = 0; e < 2; ++e) {
                            const float g = acc[i][j][2 * h + e] * scale, u = acc[i][j + 2][2 * h + e] * scale2;
                            const float gf = __bfloat162float(__float2bfloat16_rn(g));
                            const float uf = __bfloat162float(__float2bfloat16_rn(u));
                            if (col + e < N) dst[e] = __float2bfloat16_rn(gf / (1.0f + expf(-gf)) * uf);
                        }
                    }
                }
        } else {
#pragma unroll
            for (int i = 0; i < T::MT; ++i)
#pragma unroll
                for (int j = 0; j < T::NT; ++j) {
                    const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                        if (row >= M) continue;
                        const float v0 = acc[i][j][2 * h] * scale, v1 = acc[i][j][2 * h + 1] * scale;
                        if (F32) {
                            float* dst = reinterpret_cast<float*>(out) + static_cast<size_t>(row) * N + col;
                            if (col < N) dst[0] = v0;
                            if (col + 1 < N) dst[1] = v1;
                        } else {
                            __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                            if (col + 1 < N && (N & 1) == 0)
                                *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                            else {
                                if (col < N) dst[0] = __float2bfloat16_rn(v0);
                                if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                            }
                        }
                    }
                }
        }
    }
}
