// The kernels' body; in scope: q, kc, vc, out, p0, W, H, HK, G, HO, scale, skip, causal, PT, PS and first_row().
    constexpr int C = D / 8, K16 = (D + 15) / 16, NB = 2 * K16;  // the head's 16-byte chunks, 16-dim steps, 8-dim tiles
    constexpr int CS = (NB + 7) / 8 * 8, RB = WARPS / HPC, THREADS = 32 * WARPS;  // CS: a staged key row's chunks
    constexpr bool QREG = D <= 128;               // the queries' fragments stay in registers, else in shared memory
    extern __shared__ uint4 smem[];
    uint4* slots = smem;
    uint4* qs = smem + NS * HALF * CS;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4, mt = lane / 8;
    const int groups = G / HPC, hk = blockIdx.y / groups, hg = blockIdx.y % groups;
    const int row0 = first_row(), last = min(row0 + 16 * RB, W) - 1;
    const int head = hk * G + hg * HPC + warp % HPC, wrow = row0 + (warp / HPC) * 16;
    const int tiles = (p0 + (causal ? last : W - 1)) / BN + 1, items = 4 * tiles, limit = p0 + W;
    auto stage = [&](int item) {
        const __nv_bfloat16* src = item % 4 < 2 ? kc : vc;
        const int key0 = (item / 4) * BN + (item % 2) * HALF;
        uint4* dst = slots + (item % NS) * HALF * CS;
        for (int i = threadIdx.x; i < HALF * NB; i += THREADS) {
            const int key = key0 + i / NB, chunk = i % NB;
            const bool in = key < limit && (NB == C || chunk < C);  // past the filled keys or the head dim: zeros
            const int row = in ? key + (key < p0 ? 0 : skip) : 0;
            // a page's rows are contiguous: PT names each 512-row page, PS elements apart (null PT: one run of rows)
            const int64_t at = PT ? (int64_t)PT[row / 512] * PS + (int64_t)(row % 512) * HK * D : (int64_t)row * HK * D;
            cp16z(dst + sw<8 * CS>(i / NB, chunk), src + at + hk * D + chunk * 8, in);
        }
    };
    uint32_t qa[QREG ? K16 : 1][4];
    if constexpr (QREG) {
        const int r0 = wrow + g, r1 = r0 + 8;
        const uint32_t* q0 = reinterpret_cast<const uint32_t*>(q + ((int64_t)min(r0, W - 1) * H + head) * D);
        const uint32_t* q1 = reinterpret_cast<const uint32_t*>(q + ((int64_t)min(r1, W - 1) * H + head) * D);
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            qa[k][0] = r0 < W ? q0[8 * k + qd] : 0u;
            qa[k][1] = r1 < W ? q1[8 * k + qd] : 0u;
            qa[k][2] = r0 < W && 16 * k + 8 < D ? q0[8 * k + qd + 4] : 0u;
            qa[k][3] = r1 < W && 16 * k + 8 < D ? q1[8 * k + qd + 4] : 0u;
        }
    } else {
        for (int i = threadIdx.x; i < WARPS * 16 * C; i += THREADS) {
            const int w = i / (16 * C), r = (i / C) % 16, row = row0 + (w / HPC) * 16 + r;
            const int hd = hk * G + hg * HPC + w % HPC;
            cp16z(qs + sw<8 * CS>(i / C, i % C), q + ((int64_t)min(row, W - 1) * H + hd) * D + (i % C) * 8, row < W);
        }
        commit();
    }
#pragma unroll
    for (int i = 0; i < NS; ++i) {
        if (i < items) stage(i);
        commit();
    }
    float o[NB][4];
#pragma unroll
    for (int i = 0; i < NB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float sc[8][4];
    uint32_t pa[4][4];
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const int pos0 = causal ? p0 + wrow + g : limit - 1, pos1 = causal ? pos0 + 8 : pos0;
    auto take = [&](int it) -> const uint4* {           // item it's slot, once every thread's copy has landed
        wait_group<NS - 1>();
        __syncthreads();
        return slots + (it % NS) * HALF * CS;
    };
    auto done = [&](int it) {                            // the slot is free: stage the item NS ahead into it
        __syncthreads();
        if (it + NS < items) stage(it + NS);
        commit();
    };
    auto scores = [&](float (&s)[4][4], const uint4* sl) {
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            uint32_t a[4];
            if constexpr (QREG) {
                a[0] = qa[k][0]; a[1] = qa[k][1]; a[2] = qa[k][2]; a[3] = qa[k][3];
            } else {
                ldmatrix4(a, qs + sw<8 * CS>(warp * 16 + lane % 16, 2 * k + lane / 16));
            }
#pragma unroll
            for (int n = 0; n < 4; n += 2) {
                uint32_t b[4];
                ldmatrix4(b, sl + sw<8 * CS>(8 * (n + mt / 2) + lane % 8, 2 * k + mt % 2));
                if (k == 0) {
                    mma0(s[n], a, b[0], b[1]);
                    mma0(s[n + 1], a, b[2], b[3]);
                } else {
                    mma(s[n], a, b[0], b[1]);
                    mma(s[n + 1], a, b[2], b[3]);
                }
            }
        }
    };
    auto values = [&](const uint32_t (&pa0)[4], const uint32_t (&pa1)[4], const uint4* sl) {
#pragma unroll
        for (int i = 0; i < NB; i += 2) {
            uint32_t b[4];
            ldmatrix4t(b, sl + sw<8 * CS>(8 * (mt % 2) + lane % 8, i + mt / 2));
            mma(o[i], pa0, b[0], b[1]);
            mma(o[i + 1], pa0, b[2], b[3]);
        }
#pragma unroll
        for (int i = 0; i < NB; i += 2) {
            uint32_t b[4];
            ldmatrix4t(b, sl + sw<8 * CS>(16 + 8 * (mt % 2) + lane % 8, i + mt / 2));
            mma(o[i], pa1, b[0], b[1]);
            mma(o[i + 1], pa1, b[2], b[3]);
        }
    };
    for (int t = 0; t < tiles; ++t) {
        const int it = 4 * t;
        scores(*reinterpret_cast<float(*)[4][4]>(&sc[0]), take(it));
        done(it);
        scores(*reinterpret_cast<float(*)[4][4]>(&sc[4]), take(it + 1));
        done(it + 1);
        {                                                // _tile's fold of the tile's 64 scores
            const int kb = t * BN + 2 * qd;
            float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int c = kb + 8 * n;
                sc[n][0] = c <= pos0 ? __fmul_rn(sc[n][0], scale) : -INFINITY;
                sc[n][1] = c + 1 <= pos0 ? __fmul_rn(sc[n][1], scale) : -INFINITY;
                sc[n][2] = c <= pos1 ? __fmul_rn(sc[n][2], scale) : -INFINITY;
                sc[n][3] = c + 1 <= pos1 ? __fmul_rn(sc[n][3], scale) : -INFINITY;
                tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
                tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
            }
            for (int x = 1; x <= 2; x *= 2) {
                tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
                tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
            }
            const bool act0 = tm0 != -INFINITY, act1 = tm1 != -INFINITY;
            const float n0 = act0 ? fmaxf(m0, tm0) : m0, n1 = act1 ? fmaxf(m1, tm1) : m1;
            const float a0 = act0 ? (m0 == -INFINITY ? 0.0f : texp(__fsub_rn(m0, n0))) : 1.0f;
            const float a1 = act1 ? (m1 == -INFINITY ? 0.0f : texp(__fsub_rn(m1, n1))) : 1.0f;
            float c0[8], c1[8];
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int c = kb + 8 * n;
                sc[n][0] = act0 && c <= pos0 ? texp(__fsub_rn(sc[n][0], n0)) : 0.0f;
                sc[n][1] = act0 && c + 1 <= pos0 ? texp(__fsub_rn(sc[n][1], n0)) : 0.0f;
                sc[n][2] = act1 && c <= pos1 ? texp(__fsub_rn(sc[n][2], n1)) : 0.0f;
                sc[n][3] = act1 && c + 1 <= pos1 ? texp(__fsub_rn(sc[n][3], n1)) : 0.0f;
                c0[n] = __fadd_rn(sc[n][0], sc[n][1]);
                c1[n] = __fadd_rn(sc[n][2], sc[n][3]);
            }
            l0 = __fmaf_rn(l0, a0, rowsum(c0));
            l1 = __fmaf_rn(l1, a1, rowsum(c1));
            m0 = n0;
            m1 = n1;
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                o[i][0] = __fmul_rn(o[i][0], a0);
                o[i][1] = __fmul_rn(o[i][1], a0);
                o[i][2] = __fmul_rn(o[i][2], a1);
                o[i][3] = __fmul_rn(o[i][3], a1);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                pa[j][0] = pack(sc[2 * j][0], sc[2 * j][1]);
                pa[j][1] = pack(sc[2 * j][2], sc[2 * j][3]);
                pa[j][2] = pack(sc[2 * j + 1][0], sc[2 * j + 1][1]);
                pa[j][3] = pack(sc[2 * j + 1][2], sc[2 * j + 1][3]);
            }
        }
        values(pa[0], pa[1], take(it + 2));
        done(it + 2);
        values(pa[2], pa[3], take(it + 3));
        done(it + 3);
    }
    wait_group<0>();
    const int r0 = wrow + g, r1 = r0 + 8;
#pragma unroll
    for (int i = 0; i < C; ++i) {
        const int d = 8 * i + 2 * qd;
        if (r0 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r0 * HO + head) * D + d) =
                pack(tdiv(o[i][0], l0), tdiv(o[i][1], l0));
        if (r1 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r1 * HO + head) * D + d) =
                pack(tdiv(o[i][2], l1), tdiv(o[i][3], l1));
    }
