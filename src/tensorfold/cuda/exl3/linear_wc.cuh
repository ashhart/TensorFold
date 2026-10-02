// Verify-sized EXL3 linear for 17-128 rows (its tiling after MLX's 4-bit qmm): one CTA holds a
// 128-column block, one K split and RG = 16 MT rows. Warp w (8 warps) owns column tile w for all RG rows and decodes
// each of its tiles once a CTA straight into its mma B fragments. The split's trellis words and the CTA's fp16 rows
// stream through an NS-stage cp.async ring of KS k steps (KS divides the K range, so a stage never straddles one);
// A by ldmatrix from XOR-swizzled rows. Per row the arithmetic is linear_kernel's: the split's WK K ranges (range r =
// linear_kernel's warp r) each from zero accumulators with the same mma chain in k order, added in range order (tot = s0,
// tot += s1, ...), then the split sums folded through Z in split order: every output bit equals linear_kernel's
// at any row count (rows past M are computed on clamped copies and dropped).
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

template <int K2, int RG, int KS, int NS>
struct WcLayout {
    static constexpr int RAW = 8 * tile_words<K2>();          // words of one k step's 8 tiles
    static constexpr int RAW_STAGE = KS * RAW * 4;
    static constexpr int A_STAGE = KS * RG * 32;               // the CTA's rows, 16 fp16 each a k step
    static constexpr int STAGE = RAW_STAGE + A_STAGE;
    static constexpr int PIPE = NS * STAGE;
    static constexpr int RED = RG * 128 * 4;                   // the finished sums (reuses the ring)
    static constexpr int SMEM = PIPE > RED ? PIPE : RED;
};

template <int K2, int CB, int MT, int KS, int NS, bool FOLD>
__global__ void __launch_bounds__(256) linear_wc_kernel(
    const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,
    const half* __restrict__ svh, const half* __restrict__ bias, void* __restrict__ y, int y_dtype,
    float* __restrict__ Z, int* __restrict__ counters, int M, int K, int N, int SK, int WK) {
    constexpr int NW = 8, NTH = NW * 32, RG = 16 * MT;
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    using Lay = WcLayout<K2, RG, KS, NS>;
    extern __shared__ __align__(128) unsigned char smb[];
    __shared__ int last;
    const int G = (M + RG - 1) / RG, grp = blockIdx.x % G, nb = blockIdx.x / G, split = blockIdx.y;
    const int NB = gridDim.x / G;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK, nks = per_warp * WK * (FOLD ? SK : 1), kt0 = FOLD ? 0 : split * nks;
    const uint32_t* tiles = T + nb * stride_nb;
    const int col0 = nb * 128, mg = grp * RG, rows = min(RG, M - mg);
    const int S = nks / KS;

    // stage s: k steps s KS .. s KS + KS - 1 of the split into ring slot s % NS; always one commit group
    auto issue = [&](int s) {
        if (s < S) {
            unsigned char* st = smb + (s % NS) * Lay::STAGE;
            constexpr int RC = Lay::RAW / 4;
            for (int c = threadIdx.x; c < KS * RC; c += NTH) {
                const int ks = c / RC, off = c % RC;
                wc_cp16(st + (ks * Lay::RAW + off * 4) * 4, tiles + (size_t)(kt0 + s * KS + ks) * stride_k + off * 4);
            }
            unsigned char* sa = st + Lay::RAW_STAGE;
            for (int c = threadIdx.x; c < KS * RG * 2; c += NTH) {
                const int ks = c / (RG * 2), r = (c >> 1) % RG, ch = c & 1;
                const int row = min(mg + r, M - 1);
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
            load_lane_words<K2>(raw + ks * Lay::RAW + warp * TW, lane, w);
            decode_lane<K2, CB>(w, lane, b[ks][0], b[ks][1]);
        }
#pragma unroll
        for (int ks = 0; ks < KS; ++ks)
#pragma unroll
            for (int q = 0; q < MT; ++q) {
                uint32_t a[4];
                const int rr = 16 * q + lrow;
                wc_ldsm4(a, sa + (ks * RG + rr) * 32 + ((lch ^ ((rr >> 2) & 1)) << 4));
                mma16816(acc[q][0], a, b[ks][0]);
                mma16816(acc[q][1], a, b[ks][1]);
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
            if constexpr (FOLD) {                              // FOLD: every split in this CTA, summed in split order
                if (r == WK) {                                 // in registers (the Z fold's gtot = S0, gtot += S1, ...)
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
            finish(v, lane, svh, bias, col0 + 4 * lane);
            store4(y, y_dtype, (size_t)(mg + rr) * N + col0 + 4 * lane, v);
        }
        return;
    }
    for (int idx = threadIdx.x; idx < rows * 32; idx += NTH) {
        const int rr = idx >> 5, c = 4 * (idx & 31);
        *reinterpret_cast<float4*>(Z + ((size_t)split * M + mg + rr) * N + col0 + c) =
            *reinterpret_cast<const float4*>(red + rr * 128 + c);
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(counters + grp * NB + nb, 1) == SK - 1;
    __syncthreads();
    if (last) {
        __threadfence();
        for (int rr = warp; rr < rows; rr += NW) {
            const size_t at = ((size_t)mg + rr) * N + col0 + 4 * lane;
            float4 u[8];                                       // every split's sum in flight, then added in order
#pragma unroll
            for (int q = 0; q < 8; ++q)
                if (q < SK) u[q] = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)q * M * N + at));
            float4 sum = u[0];
#pragma unroll
            for (int q = 1; q < 8; ++q)
                if (q < SK) { sum.x += u[q].x; sum.y += u[q].y; sum.z += u[q].z; sum.w += u[q].w; }
            for (int q = 8; q < SK; ++q) {
                const float4 w = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)q * M * N + at));
                sum.x += w.x; sum.y += w.y; sum.z += w.z; sum.w += w.w;
            }
            float v[4] = {sum.x, sum.y, sum.z, sum.w};
            finish(v, lane, svh, bias, col0 + 4 * lane);
            store4(y, y_dtype, (size_t)(mg + rr) * N + col0 + 4 * lane, v);
        }
        if (threadIdx.x == 0) counters[grp * NB + nb] = 0;
    }
}

template <int K2, int MT, int KS, bool FOLD = false>
void launch_wc(const at::Tensor& xh, const at::Tensor& T, int64_t stride_k, int64_t stride_nb, const at::Tensor& svh,
               const half* bptr, at::Tensor& y, float* zptr, at::Tensor& counters, int M, int K, int N, int SK, int WK) {
    constexpr int NS = 4, RG = 16 * MT;
    auto kernel = linear_wc_kernel<K2, CB_MUL1, MT, KS, NS, FOLD>;
    const int smem = WcLayout<K2, RG, KS, NS>::SMEM;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        configured = true;
    }
    dim3 grid((unsigned)((N / 128) * ((M + RG - 1) / RG)), (unsigned)(FOLD ? 1 : SK));
    kernel<<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(xh.data_ptr()), reinterpret_cast<const uint32_t*>(T.data_ptr()), stride_k,
        stride_nb, reinterpret_cast<const half*>(svh.data_ptr()), bptr, y.data_ptr(), dtype_of(y), zptr,
        counters.data_ptr<int>(), M, K, N, SK, WK);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// mode 7 (the default): 17-128 rows of a 4- or 6-bit mul1 layer by row count and shape (swept on a GB10 over
// the 27B's layers 2+3 and head with the loader's PLANS). Split layers with 32+ column blocks (in_proj_qkv, in_proj_z, out_proj,
// o_proj: SK 5 or 16) run every split in one CTA (FOLD). Rows: 17-32 one 32-row group; 33-40 the mid-M kernels (4-bit,
// unsplit); 41-64 one 64-row group; 65-96 three 32-row groups; 97-128 two 64-row groups (32-row groups for a folded
// layer of 64+ blocks; 6 bits: 2-step stages). False: not taken (the caller runs the mid-M kernels).

template <int K2>
bool dispatch_wc(const at::Tensor& xh, const at::Tensor& T, int64_t stride_k, int64_t stride_nb, const at::Tensor& svh,
                 const half* bptr, at::Tensor& y, float* zptr, at::Tensor& counters, int M, int K, int N, int SK,
                 int WK) {
    const int per_warp = K / 16 / SK / WK;
    if (per_warp % 2 || M <= 16 || M > 128) return false;
    const int NB = N / 128;
    const bool fold = SK > 1 && NB >= 32;
    if (!fold && K2 == 8 && M > 32 && M <= 40) return false;
    const bool four = per_warp % 4 == 0 && !(K2 == 12 && M > 96);
    const bool mt2 = M <= 32 || (M > 64 && M <= 96) || (fold && M > 96 && NB >= 64);
#define TF_WC(MT_, F_)                                                                                           \
    four ? launch_wc<K2, MT_, 4, F_>(xh, T, stride_k, stride_nb, svh, bptr, y, zptr, counters, M, K, N, SK, WK)  \
         : launch_wc<K2, MT_, 2, F_>(xh, T, stride_k, stride_nb, svh, bptr, y, zptr, counters, M, K, N, SK, WK)
    if (fold) {
        if (mt2) TF_WC(2, true);
        else TF_WC(4, true);
    } else {
        if (mt2) TF_WC(2, false);
        else TF_WC(4, false);
    }
#undef TF_WC
    return true;
}
