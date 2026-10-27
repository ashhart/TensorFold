// Paged shared-window chunk/merge bodies shared by families (Nemotron head 128, LFM2 head 64); SEG_* set the form.

#define SEG_E (SEG_WIDE ? 8 : 4)

extern "C" __global__ void __launch_bounds__(256) SEG_CHUNK(const AttnSegs sg, const __nv_bfloat16* __restrict__ QKV,
                                                                            float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                                                            int NQKV, int H, int HK, int G, int CH, int NCH, float scale) {
    __shared__ __align__(16) float qs[GMAX][SEG_HD];
    __shared__ __align__(16) uint32_t kw[TILE][SEG_KW];
    __shared__ __align__(16) uint32_t vw[TILE][SEG_HD / 2];
    __shared__ float ps[GMAX][TILE];
    const int rq = blockIdx.x, k = seg_of(sg, rq), r = rq - sg.ROW0[k], hk = blockIdx.y, t = threadIdx.x;
    const __nv_bfloat16* __restrict__ KC = sg.KC[k];
    const __nv_bfloat16* __restrict__ VC = sg.VC[k];
    const int* __restrict__ META = sg.META[k];
    const int limit = META[0] + r + 1;
    if (static_cast<int>(blockIdx.z) * CH >= limit) return;
    const int g = t / 16, jj = t % 16;     // softmax and values: head g, lanes jj of its 16
    const int key = t % TILE, quad = t / TILE;  // scores: one key, heads 4 quad .. 4 quad + 3
    for (int i = t; i < GMAX * SEG_HD; i += 256) {
        const int gg = i / SEG_HD, d = i % SEG_HD;
        qs[gg][d] = gg < G ? bf(QKV[static_cast<int64_t>(rq) * NQKV + (hk * G + gg) * SEG_HD + d]) : 0.0f;
    }
    for (int c = blockIdx.z; c * CH < limit; c += gridDim.z) {  // this block's chunks: every gridDim.z-th
        float m = -INFINITY, den = 0.0f, o[SEG_E];
#pragma unroll
        for (int e = 0; e < SEG_E; ++e) o[e] = 0.0f;
        const int64_t cb = chunk_at(sg.PT[k], sg.PS, c, HK * SEG_HD) - static_cast<int64_t>(c) * CH * HK * SEG_HD;
        for (int key0 = c * CH; key0 < (c + 1) * CH && key0 < limit; key0 += TILE) {
            __syncthreads();
            for (int i = t; i < TILE * (SEG_HD / 2); i += 256) {
                const int j = i / (SEG_HD / 2), w = i % (SEG_HD / 2);
                const int64_t at = cb + (static_cast<int64_t>(key0 + j) * HK + hk) * SEG_HD;
                const bool in = key0 + j < limit;
                kw[j][w] = in ? reinterpret_cast<const uint32_t*>(KC + at)[w] : 0u;
                vw[j][w] = in ? reinterpret_cast<const uint32_t*>(VC + at)[w] : 0u;
            }
            __syncthreads();
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
            for (int w = 0; w < SEG_HD / 2; w += 2) {
                const uint2 kk = *reinterpret_cast<const uint2*>(&kw[key][w]);
                const float k0 = lo(kk.x), k1 = hi(kk.x), k2 = lo(kk.y), k3 = hi(kk.y);
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    const float4 q = *reinterpret_cast<const float4*>(&qs[4 * quad + h][2 * w]);
                    acc[h] = __fmaf_rn(q.x, k0, acc[h]);
                    acc[h] = __fmaf_rn(q.y, k1, acc[h]);
                    acc[h] = __fmaf_rn(q.z, k2, acc[h]);
                    acc[h] = __fmaf_rn(q.w, k3, acc[h]);
                }
            }
#pragma unroll
            for (int h = 0; h < 4; ++h) ps[4 * quad + h][key] = key0 + key < limit ? __fmul_rn(acc[h], scale) : -INFINITY;
            __syncthreads();
            float sc[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) sc[u] = ps[g][jj + 16 * u];
            const float tile_m = half_max(fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[2], sc[3])));
            const bool active = tile_m != -INFINITY;
            const float next = active ? fmaxf(m, tile_m) : m;
            const float alpha = active ? (m == -INFINITY ? 0.0f : tf_exp(__fsub_rn(m, next))) : 1.0f;
            float local = 0.0f;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float p = active && key0 + jj + 16 * u < limit ? tf_exp(__fsub_rn(sc[u], next)) : 0.0f;
                ps[g][jj + 16 * u] = p;
                local = __fadd_rn(local, p);
            }
            den = __fmaf_rn(den, alpha, half_sum(local));
            m = next;
            __syncthreads();
            float pv[SEG_E];
#pragma unroll
            for (int e = 0; e < SEG_E; ++e) pv[e] = 0.0f;
            for (int j = 0; j < TILE; ++j) {
                const float p = ps[g][j];
#if SEG_WIDE
                const uint4 v = *reinterpret_cast<const uint4*>(&vw[j][4 * jj]);
                const uint32_t vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
#else
                const uint2 v = *reinterpret_cast<const uint2*>(&vw[j][2 * jj]);
                const uint32_t vv[2] = {v.x, v.y};
#pragma unroll
                for (int e = 0; e < 2; ++e) {
#endif
                    pv[2 * e] = __fmaf_rn(p, lo(vv[e]), pv[2 * e]);
                    pv[2 * e + 1] = __fmaf_rn(p, hi(vv[e]), pv[2 * e + 1]);
                }
            }
#pragma unroll
            for (int e = 0; e < SEG_E; ++e) o[e] = __fmaf_rn(o[e], alpha, pv[e]);
        }
        if (g >= G) continue;
        const int64_t base = (static_cast<int64_t>(rq) * NCH + c) * H + hk * G + g;
        float4* out = reinterpret_cast<float4*>(PO + base * SEG_HD + SEG_E * jj);
        out[0] = make_float4(o[0], o[1], o[2], o[3]);
#if SEG_WIDE
        out[1] = make_float4(o[4], o[5], o[6], o[7]);
#endif
        if (jj == 0) {
            PM[base] = m;
            PL[base] = den;
        }
    }
}


extern "C" __global__ void __launch_bounds__(256) SEG_MERGE(const AttnSegs sg, const float* __restrict__ PO, const float* __restrict__ PM,
                                                                            const float* __restrict__ PL, __nv_bfloat16* __restrict__ OUT, float* __restrict__ XS,
                                                                            int H, int HK, int G, int CH, int NCH) {
    __shared__ float ys[GMAX][SEG_HD];
    const int rq = blockIdx.x, k = seg_of(sg, rq), r = rq - sg.ROW0[k], hk = blockIdx.y, t = threadIdx.x, g = t / 16, jj = t % 16;
    const int* __restrict__ META = sg.META[k];
    const int limit = META[0] + r + 1, nch = (limit + CH - 1) / CH, head = hk * G + g;
    float m = -INFINITY, den = 0.0f, o[SEG_E];
#pragma unroll
    for (int e = 0; e < SEG_E; ++e) o[e] = 0.0f;
    if (g < G) {
        for (int c = 0; c < nch; ++c) {
            const int64_t base = (static_cast<int64_t>(rq) * NCH + c) * H + head;
            const float cm = PM[base], cl = PL[base];
            const bool active = cl > 0.0f;
            const float next = active ? fmaxf(m, cm) : m;
            const float a = active ? (m == -INFINITY ? 0.0f : tf_exp(__fsub_rn(m, next))) : 1.0f;
            const float b = active ? tf_exp(__fsub_rn(cm, next)) : 0.0f;
            const float4* po = reinterpret_cast<const float4*>(PO + base * SEG_HD + SEG_E * jj);
#if SEG_WIDE
            const float4 p0 = po[0], p1 = po[1];
            const float co[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#else
            const float4 p0 = po[0];
            const float co[4] = {p0.x, p0.y, p0.z, p0.w};
#endif
#pragma unroll
            for (int e = 0; e < SEG_E; ++e) o[e] = __fmaf_rn(o[e], a, __fmul_rn(co[e], b));
            den = __fmaf_rn(den, a, __fmul_rn(cl, b));
            m = next;
        }
#pragma unroll
        for (int e = 0; e < SEG_E; ++e) {
            const __nv_bfloat16 y = __float2bfloat16_rn(__fdiv_rn(o[e], den));
            OUT[(static_cast<int64_t>(rq) * H + head) * SEG_HD + SEG_E * jj + e] = y;
            ys[g][SEG_E * jj + e] = bf(y);
        }
    }
    __syncthreads();
    if (t >= GMAX * (SEG_HD / 64)) return;
    const int gg = t / (SEG_HD / 64), gi = t % (SEG_HD / 64);
    if (gg >= G) return;
    float s = 0.0f;
    for (int j = 0; j < 64; ++j) s = __fadd_rn(s, ys[gg][gi * 64 + j]);
    XS[static_cast<int64_t>(rq) * (H * SEG_HD / 64) + (hk * G + gg) * (SEG_HD / 64) + gi] = s;
}

#undef SEG_E
#undef SEG_HD
#undef SEG_KW
#undef SEG_WIDE
#undef SEG_CHUNK
#undef SEG_MERGE
