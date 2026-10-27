// EXL3 linear for 17-128 rows of 4- and 6-bit mul1 layers, a 128-column block a program, in linear_body's bits.

// LinearArgs, then the plan's warps (the K ranges a split holds), packed as the host adds them.
struct WcArgs {
    const void* xh;
    const uint32_t* t;
    long long stride_k;
    long long stride_nb;
    const half* svh;
    const half* bias;
    void* y;
    float* z;
    int* counters;
    int y_dtype;
    int n, sk, m, k;
    int wk;
};

__device__ __forceinline__ void wc_cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void wc_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N_>
__device__ __forceinline__ void wc_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_)); }
__device__ __forceinline__ void wc_ldsm4(uint32_t (&a)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"((uint32_t)__cvta_generic_to_shared(p)));
}

// A lane's words of one tile from shared memory (ldg_lane_words without the read-only cache).
template <int K2>
__device__ __forceinline__ void sh_lane_words(const uint32_t* tile, int lane, uint32_t (&w)[lane_words<K2>()]) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int i = 0; i < lane_words<K2>(); ++i) w[i] = tile[(word + i) % tile_words<K2>()];
}

template <int K2, int RG, int KS, int NS>
struct WcLayout {
    static constexpr int RAW = 8 * tile_words<K2>();          // words of one k step's 8 tiles
    static constexpr int RAW_STAGE = KS * RAW * 4;
    static constexpr int A_STAGE = KS * RG * 32;               // the program's rows, 16 fp16 each a k step
    static constexpr int STAGE = RAW_STAGE + A_STAGE;
    static constexpr int PIPE = NS * STAGE;
    static constexpr int RED = RG * 128 * 4;                   // the finished sums (reuses the ring)
    static constexpr int SMEM = PIPE > RED ? PIPE : RED;
};

template <int K2, int MT, int KS, int NS, bool FOLD>
__device__ __forceinline__ void linear_wc_body(const WcArgs& a) {
    constexpr int CB = 2;                                      // mul1
    constexpr int NW = 8, NTH = NW * 32, RG = 16 * MT;
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    using Lay = WcLayout<K2, RG, KS, NS>;
    extern __shared__ __align__(128) unsigned char smb[];
    __shared__ int last;
    const int M = a.m, K = a.k, N = a.n, SK = a.sk, WK = a.wk;
    const int G = (M + RG - 1) / RG, grp = blockIdx.x % G, nb = blockIdx.x / G, split = blockIdx.y;
    const int NB = N >> 7;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK, nks = per_warp * WK * (FOLD ? SK : 1), kt0 = FOLD ? 0 : split * nks;
    const uint32_t* tiles = a.t + nb * a.stride_nb;
    const half* xh = static_cast<const half*>(a.xh);
    const int col0 = nb * 128, mg = grp * RG, rows = imin(RG, M - mg);
    const int S = nks / KS;

    // stage s: k steps s KS .. s KS + KS - 1 of the split into ring slot s % NS; always one commit group
    auto issue = [&](int s) {
        if (s < S) {
            unsigned char* st = smb + (s % NS) * Lay::STAGE;
            constexpr int RC = Lay::RAW / 4;
            for (int c = threadIdx.x; c < KS * RC; c += NTH) {
                const int ks = c / RC, off = c % RC;
                wc_cp16(st + (ks * Lay::RAW + off * 4) * 4, tiles + (size_t)(kt0 + s * KS + ks) * a.stride_k + off * 4);
            }
            unsigned char* sa = st + Lay::RAW_STAGE;
            for (int c = threadIdx.x; c < KS * RG * 2; c += NTH) {
                const int ks = c / (RG * 2), r = (c >> 1) % RG, ch = c & 1;
                const int row = imin(mg + r, M - 1);
                wc_cp16(sa + (ks * RG + r) * 32 + ((ch ^ ((r >> 2) & 1)) << 4),
                        xh + (size_t)row * K + (kt0 + s * KS + ks) * 16 + ch * 8);
            }
        }
        wc_commit();
    };

    float acc[MT][2][4], tot[MT][2][4], gtot[FOLD ? MT : 1][2][4];
#pragma unroll
    for (int q = 0; q < MT; ++q)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[q][h][c] = tot[q][h][c] = 0.f;

#pragma unroll
    for (int s = 0; s < NS - 1; ++s) issue(s);
    const int lrow = (lane & 7) + 8 * ((lane >> 3) & 1), lch = lane >> 4;   // this lane's ldmatrix row and half
    int r = 0, left = per_warp, sp = 0;                        // the current K range, its k steps to run; split
#pragma unroll 1
    for (int s = 0; s < S; ++s) {
        wc_wait<NS - 2>();                                     // stage s has landed (this thread's copies) ...
        __syncthreads();                                       // ... everyone's; slot s - 1 is free
        issue(s + NS - 1);
        const unsigned char* st = smb + (s % NS) * Lay::STAGE;
        const uint32_t* raw = reinterpret_cast<const uint32_t*>(st);
        const unsigned char* sa = st + Lay::RAW_STAGE;
        uint32_t b[KS][2][2];
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            uint32_t w[LW];
            sh_lane_words<K2>(raw + ks * Lay::RAW + warp * TW, lane, w);
            decode_lane<K2, CB>(w, lane, b[ks][0], b[ks][1]);
        }
#pragma unroll
        for (int ks = 0; ks < KS; ++ks)
#pragma unroll
            for (int q = 0; q < MT; ++q) {
                uint32_t af[4];
                const int rr = 16 * q + lrow;
                wc_ldsm4(af, sa + (ks * RG + rr) * 32 + ((lch ^ ((rr >> 2) & 1)) << 4));
                mma16816(acc[q][0], af, b[ks][0]);
                mma16816(acc[q][1], af, b[ks][1]);
            }
        left -= KS;
        if (left == 0) {                                       // range r done: tot = s0, then tot += s_r
#pragma unroll
            for (int q = 0; q < MT; ++q)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        tot[q][h][c] = r ? tot[q][h][c] + acc[q][h][c] : acc[q][h][c];
                        acc[q][h][c] = 0.f;
                    }
            ++r;
            left = per_warp;
            if constexpr (FOLD) {                              // every split in this program, summed in split order
                if (r == WK) {
#pragma unroll
                    for (int q = 0; q < MT; ++q)
#pragma unroll
                        for (int h = 0; h < 2; ++h)
#pragma unroll
                            for (int c = 0; c < 4; ++c) gtot[q][h][c] = sp ? gtot[q][h][c] + tot[q][h][c] : tot[q][h][c];
                    r = 0;
                    ++sp;
                }
            }
        }
    }
    if constexpr (FOLD) {
#pragma unroll
        for (int q = 0; q < MT; ++q)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) tot[q][h][c] = gtot[q][h][c];
    }
    wc_wait<0>();
    __syncthreads();

    float* red = reinterpret_cast<float*>(smb);
#pragma unroll
    for (int q = 0; q < MT; ++q) {
        const int rr = 16 * q + g;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = warp * 16 + h * 8 + 2 * t;
            *reinterpret_cast<float2*>(red + rr * 128 + col) = make_float2(tot[q][h][0], tot[q][h][1]);
            *reinterpret_cast<float2*>(red + (rr + 8) * 128 + col) = make_float2(tot[q][h][2], tot[q][h][3]);
        }
    }
    __syncthreads();
    if (FOLD || SK == 1) {
        for (int rr = warp; rr < rows; rr += NW) {
            const float4 u = *reinterpret_cast<const float4*>(red + rr * 128 + 4 * lane);
            float v[4] = {u.x, u.y, u.z, u.w};
            finish(v, lane, a.svh, a.bias, col0 + 4 * lane);
            store4(a.y, a.y_dtype, (size_t)(mg + rr) * N + col0 + 4 * lane, v);
        }
        return;
    }
    for (int idx = threadIdx.x; idx < rows * 32; idx += NTH) {
        const int rr = idx >> 5, c = 4 * (idx & 31);
        *reinterpret_cast<float4*>(a.z + ((size_t)split * M + mg + rr) * N + col0 + c) =
            *reinterpret_cast<const float4*>(red + rr * 128 + c);
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(a.counters + grp * NB + nb, 1) == SK - 1;
    __syncthreads();
    if (last) {
        __threadfence();
        for (int rr = warp; rr < rows; rr += NW) {
            const size_t at = ((size_t)mg + rr) * N + col0 + 4 * lane;
            float4 sum = __ldcg(reinterpret_cast<const float4*>(a.z + at));
            for (int q = 1; q < SK; ++q) {
                const float4 w = __ldcg(reinterpret_cast<const float4*>(a.z + (size_t)q * M * N + at));
                sum.x += w.x; sum.y += w.y; sum.z += w.z; sum.w += w.w;
            }
            float v[4] = {sum.x, sum.y, sum.z, sum.w};
            finish(v, lane, a.svh, a.bias, col0 + 4 * lane);
            store4(a.y, a.y_dtype, (size_t)(mg + rr) * N + col0 + 4 * lane, v);
        }
        if (threadIdx.x == 0) a.counters[grp * NB + nb] = 0;
    }
}

// The host picks one by row count and shape (exl3.zig wcPick); grid (NB * ceil(M / RG), FOLD ? 1 : SK), 256 threads.
#define TF_EXL3_WC(K2, MT, KS, F, FN)                                                    \
    extern "C" __global__ void __launch_bounds__(256)                                    \
        tf_exl3_wc_##K2##_##MT##_##KS##_##FN(tf_exl3::WcArgs a) {                        \
        tf_exl3::linear_wc_body<K2, MT, KS, 4, F>(a);                                    \
    }
#define TF_EXL3_WC_K2(K2)                                                                                  \
    TF_EXL3_WC(K2, 2, 2, false, 0) TF_EXL3_WC(K2, 2, 4, false, 0) TF_EXL3_WC(K2, 4, 2, false, 0)            \
    TF_EXL3_WC(K2, 4, 4, false, 0) TF_EXL3_WC(K2, 2, 2, true, 1) TF_EXL3_WC(K2, 2, 4, true, 1)              \
    TF_EXL3_WC(K2, 4, 2, true, 1) TF_EXL3_WC(K2, 4, 4, true, 1)
TF_EXL3_WC_K2(8) TF_EXL3_WC_K2(12)
#undef TF_EXL3_WC_K2
#undef TF_EXL3_WC
