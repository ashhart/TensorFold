// Ours: decode attention on tensor cores for the Qwen3.5 hybrids, over bf16 (qwen35.cu) or e4m3 (qwen35_kv8.cu) caches.

#pragma once
#include <cuda_fp8.h>

#include "qmm_frag.cuh"
#include "qwen35.cuh"
#include "qwen35_page_run.h"

namespace tf_q35_tc {

using namespace tf_q35;

constexpr int TK = 32;      // keys a tile
constexpr int CH = HD / 8;  // a bf16 row's 16-byte chunks
constexpr float LOG2E = 1.4426950408889634f;
constexpr int GROUP_LEN = 8192;  // a stream this long shares each key tile across a group of its rows

// A tile row's chunk c, swizzled so ldmatrix's eight rows fall on distinct banks.
__device__ __forceinline__ int swz(int r, int c) { return r * CH + (c ^ (r & 7)); }

__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(qmm_frag::smem(p)));
}

__device__ __forceinline__ float texp(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(__fmul_rn(x, LOG2E)));
    return y;
}

// Four e4m3 codes as two bf16 pairs; every code is a bf16 value, so the widening is exact.
__device__ __forceinline__ uint2 widen4(uint32_t w) {
    const __nv_fp8x2_storage_t* p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&w);
    const float2 a = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(p[0], __NV_E4M3)));
    const float2 b = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(p[1], __NV_E4M3)));
    __nv_bfloat162 x = __floats2bfloat162_rn(a.x, a.y), y = __floats2bfloat162_rn(b.x, b.y);
    return make_uint2(*reinterpret_cast<uint32_t*>(&x), *reinterpret_cast<uint32_t*>(&y));
}

// Sixteen codes as two widened chunks of a tile row.
__device__ __forceinline__ void widen16(uint4 w, uint4* lo, uint4* hi) {
    const uint2 a = widen4(w.x), b = widen4(w.y), c = widen4(w.z), d = widen4(w.w);
    *lo = make_uint4(a.x, a.y, b.x, b.y);
    *hi = make_uint4(c.x, c.y, d.x, d.y);
}


// Shared memory a kernel's tiles live in: fp8's widened key and value tiles and their scales, or bf16's two buffers.
struct TileStore {
    uint4 *kt, *vt;
    float *ksc, *vsc;
};

// e4m3 rows (qwen35_kv8.cu's layout): a tile's codes and scales loaded into registers a tile ahead, widened on put.
#include "qwen35_attn_tiles.cuh"
#include "qwen35_attn_whole.cuh"

// A short stream's rows, a block a (row, chunk, key head) of 128 threads; a 32-key tile a softmax step.
template <class Tiles>
__device__ __forceinline__ void attn_one(TileStore store, Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                         const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                         const int* __restrict__ P, float scale,
                                         float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    __shared__ __align__(16) uint4 qs[16 * CH];
    __shared__ float st[16][TK + 1];                       // a tile's scores
    __shared__ __align__(16) __nv_bfloat16 pt[16][TK + 8];  // its weights times the values' scales (80-byte rows)
    __shared__ float alpha[16];
    const int QH = geo.qh, KVH = geo.kvh, G = QH / KVH;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, qd = lane & 3;
    const size_t m = blockIdx.x;
    const int c = blockIdx.y, h = blockIdx.z;
    if (c >= MAXCH || past_rows(P, (int)m)) return;
    const Seg sg = segs[rowseg[m]];
    if (sg.len >= GROUP_LEN) return;  // a long stream's rows: attn_rows
    const Ancestor anc = rowanc[m];
    // chunks cut at logical positions (a window key's: the stream's length plus its ancestors), as a step's are
    const int keys = sg.len + rows_seen(anc), ck = chunk_keys(keys);
    const int k0 = c * ck, k1 = min(k0 + ck, keys);
    const size_t base = (m * QH + (size_t)h * G) * MAXCH + c;  // line i's slot: base + i * MAXCH
    if (k0 >= k1) {
        if (tid < G) {
            PM[base + tid * MAXCH] = -INFINITY;
            PL[base + tid * MAXCH] = 0.0f;
        }
        return;
    }
    Tiles tiles = [&] {
        if constexpr (Tiles::paged_layout) return Tiles(store, sg, layer, KVH, h);
        else return Tiles(store, sg, layer, KVH);
    }();
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int i = tid + 128 * j, line = i / CH, chunk = i % CH;
        uint4 w = make_uint4(0u, 0u, 0u, 0u);
        if (line < G) w = *reinterpret_cast<const uint4*>(Q + (m * QH + (size_t)h * G + line) * HD + chunk * 8);
        qs[swz(line, chunk)] = w;
    }
    // logical key t's cache position (-1: past the chunk), its tile loaded a tile ahead
    auto rowof = [&](int t) -> int {
        if constexpr (Tiles::paged_layout) return t < k1 ? (t < sg.len ? t : sg.len + seen_row(anc, t - sg.len)) : -1;
        else return t < k1 ? h * sg.cap + (t < sg.len ? t : sg.len + seen_row(anc, t - sg.len)) : -1;
    };
    __syncthreads();
    uint32_t qa[HD / 16][4];
#pragma unroll
    for (int k = 0; k < HD / 16; k++) qmm_frag::ldmatrix4(qa[k], qs + swz(lane % 16, 2 * k + lane / 16));
    float o[8][4];
#pragma unroll
    for (int n = 0; n < 8; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float mx = -INFINITY, l = 0.0f;  // the line tid / 8's running max and sum
    const int line = tid >> 3, part8 = tid & 7;
    tiles.fetch(k0, rowof);
    for (int t0 = k0; t0 < k1; t0 += TK) {
        __syncthreads();
        tiles.put(true);
        __syncthreads();
        if (t0 + TK < k1) tiles.fetch(t0 + TK, rowof);
        const uint4* kt = tiles.keys();
        const uint4* vt = tiles.values();
        {  // scores: warp w the tile's keys 8 w on, every line
            float s[4];
#pragma unroll
            for (int k = 0; k < HD / 16; k++) {
                uint32_t b[2];
                qmm_frag::ldmatrix2(b, kt + swz(8 * warp + lane % 8, 2 * k + (lane >> 3 & 1)));
                if (k == 0) qmm_frag::mma0(s, qa[k], b[0], b[1]);
                else qmm_frag::mma(s, qa[k], b[0], b[1]);
            }
            const int u = 8 * warp + 2 * qd;
            const bool in0 = t0 + u < k1, in1 = t0 + u + 1 < k1;
            st[g][u] = in0 ? s[0] * tiles.kscale(u) * scale : -INFINITY;
            st[g][u + 1] = in1 ? s[1] * tiles.kscale(u + 1) * scale : -INFINITY;
            st[g + 8][u] = in0 ? s[2] * tiles.kscale(u) * scale : -INFINITY;
            st[g + 8][u + 1] = in1 ? s[3] * tiles.kscale(u + 1) * scale : -INFINITY;
        }
        __syncthreads();
        {  // attn_rows's order: keys 16 hf + 2 qq + {0, 1, 8, 9}, the qq butterfly, the halves
            const int hf = part8 >> 2, qq = part8 & 3;
            float sv[4], tm = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                sv[j] = st[line][16 * hf + 2 * qq + (j & 1) + 8 * (j >> 1)];
                tm = fmaxf(tm, sv[j]);
            }
#pragma unroll
            for (int x = 1; x < 8; x *= 2) tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, x));
            const bool act = tm != -INFINITY;
            const float nm = act ? fmaxf(mx, tm) : mx;
            const float a = act ? (mx == -INFINITY ? 0.0f : texp(mx - nm)) : 1.0f;
            float ps = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int u = 16 * hf + 2 * qq + (j & 1) + 8 * (j >> 1);
                const float p = act && sv[j] != -INFINITY ? texp(sv[j] - nm) : 0.0f;
                pt[line][u] = __float2bfloat16_rn(p * tiles.vscale(u));
                ps += p;
            }
#pragma unroll
            for (int x = 1; x < 8; x *= 2) ps += __shfl_xor_sync(0xffffffffu, ps, x);
            l = fmaf(l, a, ps);
            mx = nm;
            if (part8 == 0) alpha[line] = a;
        }
        __syncthreads();
        {  // values: warp w the dims 64 w on, every line
            const float a0 = alpha[g], a1 = alpha[g + 8];
#pragma unroll
            for (int n = 0; n < 8; n++) {
                o[n][0] *= a0;
                o[n][1] *= a0;
                o[n][2] *= a1;
                o[n][3] *= a1;
            }
            const int mt = lane >> 3;
#pragma unroll
            for (int j = 0; j < TK / 16; j++) {
                uint32_t pa[4];
                qmm_frag::ldmatrix4(pa, &pt[lane % 16][16 * j + 8 * (lane / 16)]);
#pragma unroll
                for (int n = 0; n < 8; n += 2) {
                    uint32_t b[4];
                    ldmatrix4t(b, vt + swz(16 * j + 8 * (mt % 2) + lane % 8, 8 * warp + n + mt / 2));
                    qmm_frag::mma(o[n], pa, b[0], b[1]);
                    qmm_frag::mma(o[n + 1], pa, b[2], b[3]);
                }
            }
        }
    }
    if (part8 == 0 && line < G) {
        PM[base + line * MAXCH] = mx;
        PL[base + line * MAXCH] = l;
    }
#pragma unroll
    for (int n = 0; n < 8; n++) {
        const int d = 8 * (8 * warp + n) + 2 * qd;
        if (g < G) *reinterpret_cast<float2*>(PO + (base + g * MAXCH) * HD + d) = make_float2(o[n][0], o[n][1]);
        if (g + 8 < G) *reinterpret_cast<float2*>(PO + (base + (g + 8) * MAXCH) * HD + d) = make_float2(o[n][2], o[n][3]);
    }
}

// RB rows a block (MT m16 tiles of query heads); a stream's rows of one chunk width read each key tile once.
template <int MT, class Tiles>
__device__ __forceinline__ void attn_rows(TileStore store, Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                                      const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                      const int* __restrict__ P, float scale, int total,
                                                                      float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    constexpr int LINES = 16 * MT, WARPS = 2 * MT, NT = HD / 8 / WARPS;
    __shared__ __align__(16) __nv_bfloat16 pt[LINES][TK + 8];
    __shared__ float red[2][LINES], alpha[LINES];
    __shared__ int r_k0[32], r_k1[32], r_seg[32], r_ck[32];
    __shared__ Ancestor r_anc[32];
    const int QH = geo.qh, KVH = geo.kvh, G = QH / KVH, RB = LINES / G;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, qd = lane & 3;
    const int c = blockIdx.y, h = blockIdx.z, m0 = blockIdx.x * RB;
    if (c >= MAXCH) return;
    // the group's rows of long streams (a short stream's rows are attn_one's)
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
        Tiles tiles = [&] {
            if constexpr (Tiles::paged_layout) return Tiles(store, sg, layer, KVH, h);
            else return Tiles(store, sg, layer, KVH);
        }();
        // the cache position of logical key t for row j (-1: past its keys)
        auto rowof = [&](int j, int t) -> int {
            if (t >= r_k1[j]) return -1;
            if constexpr (Tiles::paged_layout) return t < sg.len ? t : sg.len + seen_row(r_anc[j], t - sg.len);
            else return h * sg.cap + (t < sg.len ? t : sg.len + seen_row(r_anc[j], t - sg.len));
        };
        // a tile's rows for row j (j < 0: a tile of stream keys only, shared by every row of the class)
        auto fetch = [&](int t0, int j) {
            tiles.fetch(t0, [&](int t) {
                if constexpr (Tiles::paged_layout) return t >= kend ? -1 : j < 0 ? t : rowof(j, t);
                else return t >= kend ? -1 : j < 0 ? h * sg.cap + t : rowof(j, t);
            });
        };
        // scores of this warp's lines over its 16 keys of the tile staged last
        auto scores = [&](float (&s)[2][4]) {
            const uint4* kt = tiles.keys();
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
                tiles.put(true);
                __syncthreads();
                if (t0 + 2 * TK <= sg.len && t0 + TK < kend) fetch(t0 + TK, -1);
                scores(s);
            } else {
                // a tile holding window keys: each row of the class over its own keys, its lines kept
                for (int j = 0; j < RB; j++) {
                    if (!(cls >> j & 1)) continue;
                    __syncthreads();
                    fetch(t0, j);
                    tiles.put(true);
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
            // each key's scale for the two lines: a shared tile's staged one, a window tile's from each line's own row
            float ka[2][2], kb[2][2];
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int u = 16 * half + 8 * n + 2 * qd + e;
                    if (shared) {
                        ka[n][e] = kb[n][e] = tiles.kscale(u);
                    } else {
                        const int ra = ina ? rowof(ja, t0 + u) : -1, rb = inb ? rowof(jb, t0 + u) : -1;
                        ka[n][e] = ra >= 0 ? tiles.key_scale(ra) : 0.0f;
                        kb[n][e] = rb >= 0 ? tiles.key_scale(rb) : 0.0f;
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
                        pt[la][u] = __float2bfloat16_rn(wa ? s[n][e] * tiles.vscale(u) : 0.0f);
                        pt[lb][u] = __float2bfloat16_rn(wb ? s[n][2 + e] * tiles.vscale(u) : 0.0f);
                    }
                __syncthreads();
                const int mq = lane >> 3;
                const uint4* vt = tiles.values();
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
                    tiles.put(false);
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

}  // namespace tf_q35_tc
