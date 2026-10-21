// Ours: qwen35_kv8.cu's tf_q35_attn_partial8 for long streams' rows, each line's arithmetic unchanged.

// RB rows a block (MT m16 tiles of query heads); a stream's rows of one chunk width read each key tile once.
template <int MT>
__device__ __forceinline__ void attn_rows8(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                                      const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                      const int* __restrict__ P, float scale, int total,
                                                                      float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    constexpr int LINES = 16 * MT, WARPS = 2 * MT, THREADS = 32 * WARPS, NT = HD / 8 / WARPS, PER = TK * 16 / THREADS;
    __shared__ __align__(16) uint4 kt[TK * CH];
    __shared__ __align__(16) uint4 vt[TK * CH];
    __shared__ __align__(16) __nv_bfloat16 pt[LINES][TK + 8];
    __shared__ float ksc[TK], vsc[TK], red[2][LINES], alpha[LINES];
    __shared__ int r_k0[32], r_k1[32], r_seg[32], r_ck[32];
    __shared__ Ancestor r_anc[32];
    const int QH = geo.qh, KVH = geo.kvh, G = QH / KVH, RB = LINES / G;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, qd = lane & 3;
    const int c = blockIdx.y, h = blockIdx.z, m0 = blockIdx.x * RB;
    if (c >= MAXCH) return;
    // the group's rows of long streams (a short stream's rows are tf_q35_attn_partial8's)
    if (tid < RB) {
        const int m = m0 + tid;
        r_seg[tid] = -1;
        if (m < total && !past_rows(P, m) && segs[rowseg[m]].len >= GROUP_LEN) {
            const Seg sg = segs[rowseg[m]];
            const Ancestor anc = rowanc[m];
            const int keys = sg.len + rows_seen(anc), ck = chunk_keys(keys);
            r_seg[tid] = rowseg[m];
            r_ck[tid] = ck;
            r_k0[tid] = c * ck;
            r_k1[tid] = min(c * ck + ck, keys);
            r_anc[tid] = anc;
        }
    }
    __syncthreads();
    {
        bool any = false;
        for (int j = 0; j < RB; j++) any |= r_seg[j] >= 0;
        if (!any) return;
    }
    // this warp's m16 tile (lines 16 mt on) and key half for the scores; its query fragments, zero past the rows
    const int mt = warp >> 1, half = warp & 1;
    uint32_t qa[HD / 16][4];
    {
        const int l0 = 16 * mt + g, l1 = l0 + 8;
        const int j0 = l0 / G, j1 = l1 / G;
        const bool in0 = j0 < RB && r_seg[j0] >= 0, in1 = j1 < RB && r_seg[j1] >= 0;
        const uint32_t* q0 = reinterpret_cast<const uint32_t*>(Q + ((size_t)(m0 + j0) * QH + h * G + l0 % G) * HD);
        const uint32_t* q1 = reinterpret_cast<const uint32_t*>(Q + ((size_t)(m0 + j1) * QH + h * G + l1 % G) * HD);
#pragma unroll
        for (int k = 0; k < HD / 16; k++) {
            qa[k][0] = in0 ? q0[8 * k + qd] : 0u;
            qa[k][1] = in1 ? q1[8 * k + qd] : 0u;
            qa[k][2] = in0 ? q0[8 * k + qd + 4] : 0u;
            qa[k][3] = in1 ? q1[8 * k + qd + 4] : 0u;
        }
    }
    for (unsigned done = 0;;) {
        // a class: the group's rows left of one stream and chunk width, so their chunk starts at one logical key
        int lead = -1;
        for (int j = 0; j < RB && lead < 0; j++)
            if (r_seg[j] >= 0 && !(done >> j & 1)) lead = j;
        if (lead < 0) break;
        unsigned cls = 0;
        int kend = 0;
        for (int j = lead; j < RB; j++)
            if (r_seg[j] == r_seg[lead] && r_ck[j] == r_ck[lead]) {
                cls |= 1u << j;
                kend = max(kend, r_k1[j]);
            }
        done |= cls;
        const int k0 = r_k0[lead];
        const Seg sg = segs[r_seg[lead]];
        const unsigned char* K = layer_at(sg, layer, KVH);
        const unsigned char* V = K + (size_t)KVH * sg.cap * HD;
        const float* KS = reinterpret_cast<const float*>(V + (size_t)KVH * sg.cap * HD);
        const float* VS = KS + (size_t)KVH * sg.cap;
        // the physical row of logical key t for row j (-1: past its keys)
        auto rowof = [&](int j, int t) -> int {
            if (t >= r_k1[j]) return -1;
            return h * sg.cap + (t < sg.len ? t : sg.len + seen_row(r_anc[j], t - sg.len));
        };
        const int part = tid & 15;
        uint4 kr[PER], vr[PER];
        float sr = 0.0f;
        // a tile's rows for row j (j < 0: a tile of stream keys only, shared by every row of the class)
        auto fetch = [&](int t0, int j) {
#pragma unroll
            for (int i = 0; i < PER; i++) {
                const int t = t0 + (tid >> 4) + THREADS / 16 * i;
                const int r = j < 0 ? h * sg.cap + t : rowof(j, t);
                kr[i] = vr[i] = make_uint4(0u, 0u, 0u, 0u);
                if (r >= 0 && t < kend) {
                    kr[i] = *reinterpret_cast<const uint4*>(K + (size_t)r * HD + part * 16);
                    vr[i] = *reinterpret_cast<const uint4*>(V + (size_t)r * HD + part * 16);
                }
            }
            const int t = t0 + (tid & 31);
            const int r = j < 0 ? h * sg.cap + t : rowof(j, t);
            sr = 0.0f;
            if (tid < 64 && r >= 0 && t < kend) sr = (tid < 32 ? KS : VS)[r];
        };
        auto put = [&](bool keys) {
#pragma unroll
            for (int i = 0; i < PER; i++) {
                const int u = (tid >> 4) + THREADS / 16 * i;
                if (keys) widen16(kr[i], &kt[swz(u, 2 * part)], &kt[swz(u, 2 * part + 1)]);
                widen16(vr[i], &vt[swz(u, 2 * part)], &vt[swz(u, 2 * part + 1)]);
            }
            if (keys && tid < 32) ksc[tid] = sr;
            else if (tid >= 32 && tid < 64) vsc[tid - 32] = sr;
        };
        // scores of this warp's lines over its 16 keys of the tile in kt
        auto scores = [&](float (&s)[2][4]) {
#pragma unroll
            for (int k = 0; k < HD / 16; k++) {
                uint32_t b[4];
                qmm_frag::ldmatrix4(b, kt + swz(16 * half + 8 * (lane >> 4) + lane % 8, 2 * k + (lane >> 3 & 1)));
                if (k == 0) {
                    qmm_frag::mma0(s[0], qa[k], b[0], b[1]);
                    qmm_frag::mma0(s[1], qa[k], b[2], b[3]);
                } else {
                    qmm_frag::mma(s[0], qa[k], b[0], b[1]);
                    qmm_frag::mma(s[1], qa[k], b[2], b[3]);
                }
            }
        };
        float o[MT][NT][4];
#pragma unroll
        for (int a = 0; a < MT; a++)
#pragma unroll
            for (int n = 0; n < NT; n++) o[a][n][0] = o[a][n][1] = o[a][n][2] = o[a][n][3] = 0.0f;
        // running max and sum of this thread's two lines (g and g + 8 of tile mt)
        const int la = 16 * mt + g, lb = la + 8, ja = la / G, jb = lb / G;
        const bool ina = ja < RB && (cls >> ja & 1), inb = jb < RB && (cls >> jb & 1);
        float mxa = -INFINITY, mxb = -INFINITY, sa = 0.0f, sb = 0.0f;
        const bool shared0 = k0 + TK <= sg.len;
        if (shared0) fetch(k0, -1);
        for (int t0 = k0; t0 < kend; t0 += TK) {
            const bool shared = t0 + TK <= sg.len;
            float s[2][4];
            if (shared) {
                __syncthreads();
                put(true);
                __syncthreads();
                if (t0 + 2 * TK <= sg.len && t0 + TK < kend) fetch(t0 + TK, -1);
                scores(s);
            } else {
                // a tile holding window keys: each row of the class over its own keys, its lines kept
                for (int j = 0; j < RB; j++) {
                    if (!(cls >> j & 1)) continue;
                    __syncthreads();
                    fetch(t0, j);
                    put(true);
                    __syncthreads();
                    float x[2][4];
                    scores(x);
#pragma unroll
                    for (int n = 0; n < 2; n++) {
                        if (ja == j) { s[n][0] = x[n][0]; s[n][1] = x[n][1]; }
                        if (jb == j) { s[n][2] = x[n][2]; s[n][3] = x[n][3]; }
                    }
                }
                if (!ina) for (int n = 0; n < 2; n++) s[n][0] = s[n][1] = 0.0f;
                if (!inb) for (int n = 0; n < 2; n++) s[n][2] = s[n][3] = 0.0f;
            }
            // each key's scale for the two lines: a shared tile's from ksc, a window tile's from each line's own row
            float ka[2][2], kb[2][2];
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int u = 16 * half + 8 * n + 2 * qd + e;
                    if (shared) {
                        ka[n][e] = kb[n][e] = ksc[u];
                    } else {
                        const int ra = ina ? rowof(ja, t0 + u) : -1, rb = inb ? rowof(jb, t0 + u) : -1;
                        ka[n][e] = ra >= 0 ? KS[ra] : 0.0f;
                        kb[n][e] = rb >= 0 ? KS[rb] : 0.0f;
                    }
                }
            // scaled and masked: a key counts for a line while it is below the line's row's k1
            float tma = -INFINITY, tmb = -INFINITY;
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int t = t0 + 16 * half + 8 * n + 2 * qd + e;
                    s[n][e] = ina && t < r_k1[ja] ? s[n][e] * ka[n][e] * scale : -INFINITY;
                    s[n][2 + e] = inb && t < r_k1[jb] ? s[n][2 + e] * kb[n][e] * scale : -INFINITY;
                    tma = fmaxf(tma, s[n][e]);
                    tmb = fmaxf(tmb, s[n][2 + e]);
                }
            for (int x = 1; x <= 2; x *= 2) {
                tma = fmaxf(tma, __shfl_xor_sync(0xffffffffu, tma, x));
                tmb = fmaxf(tmb, __shfl_xor_sync(0xffffffffu, tmb, x));
            }
            if (qd == 0) {
                red[half][la] = tma;
                red[half][lb] = tmb;
            }
            __syncthreads();
            tma = fmaxf(red[0][la], red[1][la]);
            tmb = fmaxf(red[0][lb], red[1][lb]);
            const bool acta = tma != -INFINITY, actb = tmb != -INFINITY;
            const float na = acta ? fmaxf(mxa, tma) : mxa, nb = actb ? fmaxf(mxb, tmb) : mxb;
            const float aa = acta ? (mxa == -INFINITY ? 0.0f : texp(mxa - na)) : 1.0f;
            const float ab = actb ? (mxb == -INFINITY ? 0.0f : texp(mxb - nb)) : 1.0f;
            float pa = 0.0f, pb = 0.0f;
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    s[n][e] = acta && s[n][e] != -INFINITY ? texp(s[n][e] - na) : 0.0f;
                    s[n][2 + e] = actb && s[n][2 + e] != -INFINITY ? texp(s[n][2 + e] - nb) : 0.0f;
                    pa += s[n][e];
                    pb += s[n][2 + e];
                }
            for (int x = 1; x <= 2; x *= 2) {
                pa += __shfl_xor_sync(0xffffffffu, pa, x);
                pb += __shfl_xor_sync(0xffffffffu, pb, x);
            }
            __syncthreads();  // every half's max is read before the sums take red
            if (qd == 0) {
                red[half][la] = pa;
                red[half][lb] = pb;
                if (half == 0) {
                    alpha[la] = aa;
                    alpha[lb] = ab;
                }
            }
            __syncthreads();
            sa = fmaf(sa, aa, red[0][la] + red[1][la]);
            sb = fmaf(sb, ab, red[0][lb] + red[1][lb]);
            mxa = na;
            mxb = nb;
            // the values: rescale every line once, then each tile of values (a window tile's, row by row)
#pragma unroll
            for (int a = 0; a < MT; a++) {
                const float a0 = alpha[16 * a + g], a1 = alpha[16 * a + g + 8];
#pragma unroll
                for (int n = 0; n < NT; n++) {
                    o[a][n][0] *= a0;
                    o[a][n][1] *= a0;
                    o[a][n][2] *= a1;
                    o[a][n][3] *= a1;
                }
            }
            auto values = [&](int only) {
                // pt: this warp's p times each key's value scale, lines of row `only` (-1: all), others zero
#pragma unroll
                for (int n = 0; n < 2; n++)
#pragma unroll
                    for (int e = 0; e < 2; e++) {
                        const int u = 16 * half + 8 * n + 2 * qd + e;
                        const bool wa = only < 0 || ja == only, wb = only < 0 || jb == only;
                        pt[la][u] = __float2bfloat16_rn(wa ? s[n][e] * vsc[u] : 0.0f);
                        pt[lb][u] = __float2bfloat16_rn(wb ? s[n][2 + e] * vsc[u] : 0.0f);
                    }
                __syncthreads();
                const int mq = lane >> 3;
#pragma unroll
                for (int a = 0; a < MT; a++)
#pragma unroll
                    for (int j = 0; j < TK / 16; j++) {
                        uint32_t pa4[4];
                        qmm_frag::ldmatrix4(pa4, &pt[16 * a + lane % 16][16 * j + 8 * (lane / 16)]);
#pragma unroll
                        for (int n = 0; n < NT; n += 2) {
                            uint32_t b[4];
                            ldmatrix4t(b, vt + swz(16 * j + 8 * (mq % 2) + lane % 8, NT * warp + n + mq / 2));
                            qmm_frag::mma(o[a][n], pa4, b[0], b[1]);
                            qmm_frag::mma(o[a][n + 1], pa4, b[2], b[3]);
                        }
                    }
            };
            if (shared) values(-1);
            else
                for (int j = 0; j < RB; j++) {
                    if (!(cls >> j & 1)) continue;
                    __syncthreads();
                    fetch(t0, j);
                    put(false);
                    __syncthreads();
                    values(j);
                }
            if (!shared && t0 + TK < kend && t0 + 2 * TK <= sg.len) fetch(t0 + TK, -1);
        }
        // the class's partials (a row whose chunk is past its keys: empty), stats from the first key half
        if (half == 0 && qd == 0) {
            if (ina) {
                const size_t sl = ((size_t)(m0 + ja) * QH + h * G + la % G) * MAXCH + c;
                PM[sl] = r_k1[ja] > k0 ? mxa : -INFINITY;
                PL[sl] = r_k1[ja] > k0 ? sa : 0.0f;
            }
            if (inb) {
                const size_t sl = ((size_t)(m0 + jb) * QH + h * G + lb % G) * MAXCH + c;
                PM[sl] = r_k1[jb] > k0 ? mxb : -INFINITY;
                PL[sl] = r_k1[jb] > k0 ? sb : 0.0f;
            }
        }
#pragma unroll
        for (int a = 0; a < MT; a++) {
            const int l0 = 16 * a + g, l1 = l0 + 8, j0 = l0 / G, j1 = l1 / G;
#pragma unroll
            for (int n = 0; n < NT; n++) {
                const int d = 8 * (NT * warp + n) + 2 * qd;
                if (j0 < RB && (cls >> j0 & 1) && r_k1[j0] > k0)
                    *reinterpret_cast<float2*>(PO + (((size_t)(m0 + j0) * QH + h * G + l0 % G) * MAXCH + c) * HD + d) = make_float2(o[a][n][0], o[a][n][1]);
                if (j1 < RB && (cls >> j1 & 1) && r_k1[j1] > k0)
                    *reinterpret_cast<float2*>(PO + (((size_t)(m0 + j1) * QH + h * G + l1 % G) * MAXCH + c) * HD + d) = make_float2(o[a][n][2], o[a][n][3]);
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_q35_attn_rows8(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                                      const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                      const int* __restrict__ P, float scale, int total,
                                                                      float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    attn_rows8<4>(geo, Q, segs, rowseg, rowanc, layer, P, scale, total, PM, PL, PO);
}
