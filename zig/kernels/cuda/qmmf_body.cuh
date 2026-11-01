// Strided stores are instantiated only by the shared-slot entries.
#ifdef TF_QMMF_K32
    using T = PromptK32Tile;
    static_assert(BK == 32 && MODE == BF16 && !CLUSTER && !FUSE && BN == 128, "unsplit BF16 K32");
#else
    using T = Tile<MODE, BM, BN, WM, WN, STAGES>;
#endif
#ifdef TF_QMMF_STRIDED
    static_assert(BN == 64 || BN == 32, "a block reads one 64-column tile of words and scales, or its half");
#else
    static_assert(BN == 64 || BN == 32 || (MODE == BF16 && BN == 128), "packed column tiles");
#endif
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
#ifdef TF_QMMF_K32
    const int KG = K / GS, steps = K / BK, slice_groups = steps / SK, per = FUSE ? steps : slice_groups;
#else
    const int KG = K / GS, slice_groups = KG / SK, per = FUSE ? KG : slice_groups;
#endif
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y, slice = FUSE ? 0 : blockIdx.z, g0 = slice * per;

#ifndef TF_QMMF_STRIDED
    uint64_t* ready = reinterpret_cast<uint64_t*>(buf + STAGES * T::STAGE);
#if __CUDA_ARCH__ >= 1200
    if constexpr (T::BARRIERS > 0) {
        if (tid == 0) {
#pragma unroll
            for (int s = 0; s < STAGES; ++s)
                asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem(ready + s)) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        __syncthreads();
    }
#endif
#endif
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
#ifdef TF_QMMF_K32
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * BK + ch * 8,
                  m0 + r < M);
#else
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8,
                  m0 + r < M);
#endif
        }
        unsigned char* pw = p + T::X;
        constexpr int TILE_BYTES = MODE == FP4 ? 64 * GS / 2 : MODE == BF16 ? 64 * GS * 2 : 64 * GS;
#if __CUDA_ARCH__ >= 1200 && !defined(TF_QMMF_STRIDED)
        if constexpr (T::BARRIERS > 0) {
            if (tid == 0) {
                const uint32_t bar = smem(ready + s);
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             :: "r"(bar), "r"(T::W) : "memory");
#pragma unroll
                for (int t = 0; t < BN / 64; ++t) {
                    const void* src = w + (static_cast<size_t>(n0 / 64 + t) * KG + g) * TILE_BYTES;
                    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                                 :: "r"(smem(pw + t * TILE_BYTES)), "l"(src), "r"(TILE_BYTES), "r"(bar) : "memory");
                }
            }
        } else
#endif
#ifdef TF_QMMF_K32
        if constexpr (BK == 32) {
            // Gather each n8 tile's two K16 fragments from the unchanged K64 pack.
            constexpr int FRAG_BYTES = 32 * 8, HALF_BYTES = 2 * FRAG_BYTES;
            for (int c = tid; c < T::W / 16; c += T::THREADS) {
                const int t = c / (8 * HALF_BYTES / 16), off = c % (8 * HALF_BYTES / 16);
                const int jj = off / (HALF_BYTES / 16), pair = off % (HALF_BYTES / 16);
                const size_t src = (static_cast<size_t>(n0 / 64 + t) * KG + g / 2) * TILE_BYTES;
                cp16(pw + c * 16, w + src + jj * 2 * HALF_BYTES + (g & 1) * HALF_BYTES + pair * 16);
            }
        } else
#endif
        if constexpr (BN == 32) {
            // half a packed 64-column tile: its n8 tiles 0-3 or 4-7, contiguous in every format's word order
            const int half = (n0 % 64) / 32;
            for (int c = tid; c < T::W / 16; c += T::THREADS)
                cp16(pw + c * 16, w + (static_cast<size_t>(n0 / 64) * KG + g) * TILE_BYTES + half * (TILE_BYTES / 2) + c * 16);
            if constexpr (T::S > 0) {
                unsigned char* ps = pw + T::W;
                for (int c = tid; c < T::S / 16; c += T::THREADS)
                    cp16(ps + c * 16, bs + (static_cast<size_t>(n0 / 64) * KG + g) * (2 * T::S) + half * T::S + c * 16);
            }
        } else {
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            cp16(pw + c * 16, w + (static_cast<size_t>(n0 / 64 + t) * KG + g) * TILE_BYTES + off * 16);
        }
        if constexpr (T::S > 0) {
            unsigned char* ps = pw + T::W;
            for (int c = tid; c < T::S / 16; c += T::THREADS)
                cp16(ps + c * 16, bs + (static_cast<size_t>(n0 / 64) * KG + g) * T::S + c * 16);
        }
        }
    };

    float acc[T::MT][T::NT][4];
    float tot[FUSE ? T::MT : 1][FUSE ? T::NT : 1][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per) load(s, g0 + s);
        commit();
    }
    for (int it = 0; it < per; ++it) {
        if constexpr (FUSE) {
            if (it > 0 && it % slice_groups == 0) {
#pragma unroll
                for (int i = 0; i < T::MT; ++i)
#pragma unroll
                    for (int j = 0; j < T::NT; ++j)
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            tot[i][j][e] = it == slice_groups ? acc[i][j][e] : tot[i][j][e] + acc[i][j][e];
                            acc[i][j][e] = 0.0f;
                        }
            }
        }
        wait<STAGES - 2>();
#if __CUDA_ARCH__ >= 1200 && !defined(TF_QMMF_STRIDED)
        if constexpr (T::BARRIERS > 0) {
            const uint32_t bar = smem(ready + it % STAGES), phase = (it / STAGES) & 1;
            asm volatile("{ .reg .pred done; WAIT: mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;"
                         " @!done bra WAIT; }" :: "r"(bar), "r"(phase) : "memory");
        }
#endif
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < per) load(next % STAGES, g0 + next);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const unsigned char* pw = p + T::X;
        const uint8_t* ps = p + T::X + T::W;
        float d[T::MT][T::NT][4];
        uint4 wq[T::NT];
        uint32_t sq[T::NT][2];
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int jj = wn * T::NT + j;
            const int col = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
            if constexpr (MODE == FP4 || MODE == FP8G) {
                const uint2 v = *reinterpret_cast<const uint2*>(ps + col * 4);
                sq[j][0] = v.x;
                sq[j][1] = v.y;
            } else if constexpr (MODE == MXFP8) {
                const uint32_t v = *reinterpret_cast<const uint32_t*>(ps + col * 2);
                sq[j][0] = v & 0xFFFFu;
                sq[j][1] = v >> 16;
            }
            if constexpr (MODE == FP4) {
                const uint2 u = reinterpret_cast<const uint2*>(pw)[jj * 32 + lane];
                wq[j] = make_uint4(u.x, u.y, 0u, 0u);
            } else if constexpr (MODE != BF16) {
                wq[j] = reinterpret_cast<const uint4*>(pw)[jj * 32 + lane];
            }
        }
#pragma unroll
#ifdef TF_QMMF_K32
        for (int kt = 0; kt < BK / 16; ++kt) {
#else
        for (int kt = 0; kt < GS / 16; ++kt) {
#endif
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
                    const uint32_t word = kt < 2 ? wq[j].x : wq[j].y;
                    b0 = fp4pair(word, (kt & 1) * 8);
                    b1 = fp4pair(word, (kt & 1) * 8 + 4);
                } else if constexpr (MODE == BF16) {
#ifdef TF_QMMF_K32
                    const uint2 v = reinterpret_cast<const uint2*>(pw)[((wn * T::NT + j) * (BK / 16) + kt) * 32 + lane];
#else
                    const uint2 v = reinterpret_cast<const uint2*>(pw)[((wn * T::NT + j) * 4 + kt) * 32 + lane];
#endif
                    b0 = v.x;
                    b1 = v.y;
                } else {
                    const uint32_t word = comp(wq[j], kt);
                    b0 = fp8pair(word & 0xFFFFu);
                    b1 = fp8pair(word >> 16);
                }
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    if constexpr (MODE == FP8 || MODE == BF16) mma(acc[i][j], a[i], b0, b1);
                    else if (MODE == FP4 || (MODE == MXFP8 && (kt & 1) == 0) || (MODE == FP8G && kt == 0))
                        mma0(d[i][j], a[i], b0, b1);
                    else mma(d[i][j], a[i], b0, b1);
                }
            }
            if constexpr (MODE == FP4 || MODE == MXFP8 || MODE == FP8G) {
                if (MODE == FP4 || (MODE == MXFP8 && (kt & 1)) || (MODE == FP8G && kt == GS / 16 - 1)) {
                    const int blk = MODE == FP4 ? kt : kt / 2;
#pragma unroll
                    for (int j = 0; j < T::NT; ++j) {
                        float s0, s1;
                        if constexpr (MODE == FP8G) {
                            s0 = __uint_as_float(sq[j][0]);
                            s1 = __uint_as_float(sq[j][1]);
                        } else if constexpr (MODE == FP4) {
                            s0 = e4m3f(static_cast<uint8_t>(sq[j][0] >> (8 * blk)));
                            s1 = e4m3f(static_cast<uint8_t>(sq[j][1] >> (8 * blk)));
                        } else {
                            s0 = e8m0f(static_cast<uint8_t>(sq[j][0] >> (8 * blk)));
                            s1 = e8m0f(static_cast<uint8_t>(sq[j][1] >> (8 * blk)));
                        }
#pragma unroll
                        for (int i = 0; i < T::MT; ++i)
#pragma unroll
                            for (int e = 0; e < 4; ++e) acc[i][j][e] = __fmaf_rn(d[i][j][e], (e & 1) ? s1 : s0, acc[i][j][e]);
                    }
                }
            }
        }
    }
    wait<0>();
    __syncthreads();
    if constexpr (FUSE) {
        if (SK > 1) {
#pragma unroll
            for (int i = 0; i < T::MT; ++i)
#pragma unroll
                for (int j = 0; j < T::NT; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[i][j][e] = tot[i][j][e] + acc[i][j][e];
        }
    }
    if constexpr (CLUSTER) {
#if __CUDA_ARCH__ < 900
        __trap();
#else
        auto cluster = cooperative_groups::this_cluster();
        float* mine = reinterpret_cast<float*>(buf);
        if (slice != 0) {
#pragma unroll
            for (int i = 0; i < T::MT; ++i)
#pragma unroll
                for (int j = 0; j < T::NT; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) mine[((i * T::NT + j) * 4 + e) * T::THREADS + tid] = acc[i][j][e];
        }
        cluster.sync();
        if (slice == 0) {
            for (int peer = 1; peer < SK; ++peer) {
                const float* theirs = cluster.map_shared_rank(mine, peer);
#pragma unroll
                for (int i = 0; i < T::MT; ++i)
#pragma unroll
                    for (int j = 0; j < T::NT; ++j)
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            acc[i][j][e] = acc[i][j][e] + theirs[((i * T::NT + j) * 4 + e) * T::THREADS + tid];
            }
        }
        cluster.sync();
        if (slice != 0) return;
#endif
    }
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                if (SK > 1 && !CLUSTER && !FUSE) {
                    float* dst = part + (static_cast<size_t>(slice) * M + row) * N + col;
                    if (col < N) dst[0] = acc[i][j][2 * h];
                    if (col + 1 < N) dst[1] = acc[i][j][2 * h + 1];
                    continue;
                }
                const float v0 = acc[i][j][2 * h] * scale, v1 = acc[i][j][2 * h + 1] * scale;
                if (F32) {
#ifdef TF_QMMF_STRIDED
                    float* dst = reinterpret_cast<float*>(out) + static_cast<size_t>(row) * ldy + col;
#else
                    float* dst = reinterpret_cast<float*>(out) + static_cast<size_t>(row) * N + col;
#endif
                    if (col < N) dst[0] = v0;
                    if (col + 1 < N) dst[1] = v1;
                } else {
#ifdef TF_QMMF_STRIDED
                    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * ldy + col;
#else
                    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
#endif
                    if (col + 1 < N && (N & 1) == 0)
                        *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                    else {
                        if (col < N) dst[0] = __float2bfloat16_rn(v0);
                        if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                    }
                }
            }
        }
