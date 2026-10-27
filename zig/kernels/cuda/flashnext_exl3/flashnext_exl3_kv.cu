// Flash Next attention over an int8 KV cache (ours, K10 prototype): upstream's flashnext_attn.cu prep / chunks / merge

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

#include "../glue_math.cuh"
#include "../qmm_frag.cuh"

using namespace tf_glue;

namespace fn_kv8 {

constexpr int TK = 32;   // keys a tile holds
constexpr int HD = 256;  // head width
constexpr int QR = 16;   // query heads a KV head's tile holds
constexpr int GR = 32;   // quant group
constexpr int NG = HD / GR;

__device__ __forceinline__ float block_sum(float v, float* scratch) {
    v = warp_sum(v);
    const int warps = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < warps; ++w) total = __fadd_rn(total, scratch[w]);
    __syncthreads();
    return total;
}

__device__ __forceinline__ float inv_rms(float ss, int width, float eps) {
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(width)), eps)));
}

__device__ __forceinline__ float rope(float xn, float xpn, int d, int half, int pos, const float* __restrict__ inv) {
    if (d >= 2 * half) return xn;
    const int i = d < half ? d : d - half;
    float s, c;
    sincosf(__fmul_rn(static_cast<float>(pos), inv[i]), &s, &c);
    return d < half ? __fsub_rn(__fmul_rn(xn, c), __fmul_rn(xpn, s)) : __fadd_rn(__fmul_rn(xpn, s), __fmul_rn(xn, c));
}

__device__ __forceinline__ int partner(int d, int half) {
    return d < half ? d + half : (d < 2 * half ? d - half : d);
}

// The warp's 32 lanes hold one group (lane = index in the group): its normalized Sylvester Hadamard transform.
__device__ __forceinline__ float wht32(float x) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const float y = __shfl_xor_sync(0xffffffffu, x, m);
        x = (lane & m) ? __fsub_rn(y, x) : __fadd_rn(x, y);
    }
    return __fmul_rn(x, 0.17677669529663688f);  // 1 / sqrt(32)
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

// One lane's value of a group into its code; lane 0 writes the group's fp16 scale.
__device__ __forceinline__ void quant8(float x, int8_t* code, __half* scale) {
    const float peak = warp_max(fabsf(x));
    const __half sh = __float2half_rn(fmaxf(peak, 6.103515625e-05f));
    const float s = __half2float(sh);
    const float q = fminf(fmaxf(floorf(__fmul_rn(__fdiv_rn(x, s), 128.0f)) + 128.0f, 0.0f), 255.0f);
    *code = static_cast<int8_t>(static_cast<int>(q) - 128);
    if ((threadIdx.x & 31) == 0) *scale = sh;
}

__device__ __forceinline__ float dequant8(int8_t c, float s) {
    return __fmul_rn(__fadd_rn(static_cast<float>(c), 0.5f), __fmul_rn(s, 0.0078125f));
}

}  // namespace fn_kv8

using namespace fn_kv8;

// fn_attn_prep (text positions) with the K and V rows stored as int8 codes and group scales; queries rotated if ROT.
extern "C" __global__ void __launch_bounds__(256) fn_kv8_prep(const __nv_bfloat16* __restrict__ P, const int* __restrict__ POS0,
                                                              const float* __restrict__ QW, const float* __restrict__ KW,
                                                              const float* __restrict__ IW, const float* __restrict__ INV,
                                                              __nv_bfloat16* __restrict__ Q, int8_t* __restrict__ K8,
                                                              __half* __restrict__ KS, int8_t* __restrict__ V8,
                                                              __half* __restrict__ VS, __nv_bfloat16* __restrict__ IQ,
                                                              __nv_bfloat16* __restrict__ IKC, float eps, int PW, int NQ, int NKV,
                                                              int NI, int IHD, int HALF, int ROT) {
    __shared__ float scratch[32];
    const int r = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const int pos = POS0[0] + r;
    const long long row = static_cast<long long>(r) * PW;
    if (head >= NQ + NKV + NI) {
        if (d < IHD) IKC[static_cast<long long>(pos) * IHD + d] = P[row + NQ * 2 * HD + 2 * NKV * HD + NI * IHD + d];
        return;
    }
    const bool index = head >= NQ + NKV;
    const int width = index ? IHD : HD;
    const bool live = d < width;
    const long long src = row + (head < NQ ? head * 2 * HD : (head < NQ + NKV ? NQ * 2 * HD + (head - NQ) * HD : NQ * 2 * HD + 2 * NKV * HD + (head - NQ - NKV) * IHD));
    const float* w = head < NQ ? QW : (head < NQ + NKV ? KW : IW);
    const float x = live ? bf(P[src + d]) : 0.0f;
    const float rinv = inv_rms(block_sum(__fmul_rn(x, x), scratch), width, eps);
    if (!live) return;
    const int p = partner(d, HALF);
    const float xn = rbf(__fmul_rn(__fmul_rn(x, rinv), w[d]));
    const float xpn = rbf(__fmul_rn(__fmul_rn(bf(P[src + p]), rinv), w[p]));
    const float roped = rope(xn, xpn, d, HALF, pos, INV);
    if (index) {
        IQ[(static_cast<long long>(r) * NI + head - NQ - NKV) * IHD + d] = __float2bfloat16_rn(roped);
        return;
    }
    // the cache keeps upstream's bf16 rounding point, then the rotation and the code (a query: rotated, then rounded)
    if (head < NQ) {
        const float qv = ROT ? wht32(roped) : roped;
        Q[(static_cast<long long>(r) * NQ + head) * HD + d] = __float2bfloat16_rn(qv);
        return;
    }
    const long long slot = static_cast<long long>(pos) * NKV + head - NQ;
    const float kv = rbf(roped);
    const float vv = bf(P[row + NQ * 2 * HD + NKV * HD + (head - NQ) * HD + d]);
    quant8(ROT ? wht32(kv) : kv, K8 + slot * HD + d, KS + slot * NG + d / GR);
    quant8(ROT ? wht32(vv) : vv, V8 + slot * HD + d, VS + slot * NG + d / GR);
}

// Upstream's attn_chunks with each tile's keys and values read as int8 codes + scales (cp.async into a raw buffer, NB
template <int NB>
__device__ __forceinline__ void kv8_chunks(const __nv_bfloat16* __restrict__ Q, const int8_t* __restrict__ K8,
                                           const __half* __restrict__ KS, const int8_t* __restrict__ V8,
                                           const __half* __restrict__ VS, const int* __restrict__ POS0,
                                           float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                           const int* __restrict__ IDS, const int* __restrict__ NKR, const int* __restrict__ SPR,
                                           int H, int HK, int G, int CH, int NCH, float SCALE, int IDW, int QSA) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    typedef __nv_bfloat16 Row[HD + 8];
    Row* qs = reinterpret_cast<Row*>(smem_raw);                  // [QR]
    Row* tiles_kv = qs + QR;                                     // [2][TK] keys then values (bf16, dequantised)
    __nv_bfloat16 (*ps)[TK + 8] = reinterpret_cast<__nv_bfloat16 (*)[TK + 8]>(tiles_kv + 2 * TK);  // [QR]
    int8_t (*raw)[TK][HD] = reinterpret_cast<int8_t (*)[TK][HD]>(ps + QR);  // [NB * 2][TK][HD]
    __half (*rsc)[TK][NG] = reinterpret_cast<__half (*)[TK][NG]>(raw + NB * 2);  // [NB * 2][TK][NG]
    __shared__ float red_m[4][QR], red_l[4][QR];
    __shared__ int key_ids[NB][TK];
    const int r = blockIdx.x, hk = blockIdx.y, c = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int n = POS0[0] + r + 1;
    bool sparse = false;
    if (QSA) {
        sparse = SPR[r] != 0;
        if (sparse) n = NKR[r];
    }
    const int start = c * CH;
    if (start >= n) return;
    for (int e = threadIdx.x; e < QR * HD / 8; e += blockDim.x) {
        const int i = e / (HD / 8), v = e % (HD / 8);
        qmm_frag::cp16z(&qs[i][8 * v], Q + (static_cast<long long>(r) * H + hk * G + min(i, G - 1)) * HD + 8 * v, i < G);
    }
    qmm_frag::commit();
    auto load = [&](int t0, int buf) {
        if (threadIdx.x < TK) {
            const int ki = start + t0 + threadIdx.x;
            key_ids[buf][threadIdx.x] = ki < n ? (sparse ? IDS[static_cast<long long>(r) * IDW + ki] : ki) : -1;
        }
        __syncthreads();
        for (int pass = 0; pass < 2; ++pass) {
            const int8_t* src = pass == 0 ? K8 : V8;
            const __half* ssrc = pass == 0 ? KS : VS;
            for (int e = threadIdx.x; e < TK * HD / 16; e += blockDim.x) {
                const int k = e / (HD / 16), v = e % (HD / 16), key = key_ids[buf][k];
                qmm_frag::cp16z(&raw[buf * 2 + pass][k][16 * v], src + (static_cast<long long>(max(key, 0)) * HK + hk) * HD + 16 * v, key >= 0);
            }
            if (threadIdx.x < TK) {
                const int key = key_ids[buf][threadIdx.x];
                qmm_frag::cp16z(&rsc[buf * 2 + pass][threadIdx.x][0], ssrc + (static_cast<long long>(max(key, 0)) * HK + hk) * NG, key >= 0);
            }
        }
        qmm_frag::commit();
    };
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    float o[8][4] = {};
    const int tiles = min(n - start, CH);
    load(0, 0);
    for (int t0 = 0, it = 0; t0 < tiles; t0 += TK, ++it) {
        const int buf = NB == 2 ? it & 1 : 0;
        const bool ahead = NB == 2 && t0 + TK < tiles;
        if (ahead) load(t0 + TK, buf ^ 1);
        if (ahead) qmm_frag::wait<1>(); else qmm_frag::wait<0>();
        __syncthreads();
        // dequantise the raw tile into the bf16 pair: a thread 16 codes of one key (half a group) at a time
        for (int e = threadIdx.x; e < 2 * TK * HD / 16; e += blockDim.x) {
            const int pass = e / (TK * HD / 16), rem = e % (TK * HD / 16), k = rem / (HD / 16), v = rem % (HD / 16);
            const float s = __half2float(rsc[buf * 2 + pass][k][v / 2]);
            const int4 codes = *reinterpret_cast<const int4*>(&raw[buf * 2 + pass][k][16 * v]);
            const int8_t* cb = reinterpret_cast<const int8_t*>(&codes);
            __align__(16) __nv_bfloat16 outv[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) outv[j] = __float2bfloat16_rn(dequant8(cb[j], s));
            int4* dst = reinterpret_cast<int4*>(&tiles_kv[pass * TK + k][16 * v]);
            dst[0] = reinterpret_cast<const int4*>(outv)[0];
            dst[1] = reinterpret_cast<const int4*>(outv)[1];
        }
        __syncthreads();
        Row* kt = tiles_kv;
        Row* vt = tiles_kv + TK;
        const int* keys = key_ids[buf];
        constexpr int JT = TK / 32;
        float s[JT][4] = {};
#pragma unroll 4
        for (int ks = 0; ks < HD / 16; ks += 2) {
            uint32_t a0[4], a1[4];
            qmm_frag::ldmatrix4(a0, &qs[lane % 16][ks * 16 + (lane / 16) * 8]);
            qmm_frag::ldmatrix4(a1, &qs[lane % 16][(ks + 1) * 16 + (lane / 16) * 8]);
#pragma unroll
            for (int jt = 0; jt < JT; ++jt) {
                uint32_t b[4];
                qmm_frag::ldmatrix4(b, &kt[warp * 8 * JT + jt * 8 + lane % 8][ks * 16 + (lane / 8) * 8]);
                qmm_frag::mma(s[jt], a0, b[0], b[1]);
                qmm_frag::mma(s[jt], a1, b[2], b[3]);
            }
        }
        float tile_m[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int jt = 0; jt < JT; ++jt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const bool valid = keys[warp * 8 * JT + jt * 8 + 2 * t + (e & 1)] >= 0;
                s[jt][e] = valid ? __fmul_rn(s[jt][e], SCALE) : -INFINITY;
                tile_m[e >> 1] = fmaxf(tile_m[e >> 1], s[jt][e]);
            }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            tile_m[h] = fmaxf(tile_m[h], __shfl_xor_sync(0xffffffffu, tile_m[h], 1));
            tile_m[h] = fmaxf(tile_m[h], __shfl_xor_sync(0xffffffffu, tile_m[h], 2));
            if (t == 0) red_m[warp][g + 8 * h] = tile_m[h];
        }
        __syncthreads();
        float next_m[2], alpha[2], rows[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = g + 8 * h;
            const float tm = fmaxf(fmaxf(red_m[0][row], red_m[1][row]), fmaxf(red_m[2][row], red_m[3][row]));
            const bool active = tm != -INFINITY;
            next_m[h] = active ? fmaxf(m[h], tm) : m[h];
            alpha[h] = active ? (m[h] == -INFINITY ? 0.0f : tf_exp(__fsub_rn(m[h], next_m[h]))) : 1.0f;
            rows[h] = 0.0f;
        }
#pragma unroll
        for (int jt = 0; jt < JT; ++jt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int h = e >> 1;
                const float p = s[jt][e] == -INFINITY ? 0.0f : tf_exp(__fsub_rn(s[jt][e], next_m[h]));
                rows[h] = __fadd_rn(rows[h], p);
                ps[g + 8 * h][warp * 8 * JT + jt * 8 + 2 * t + (e & 1)] = __float2bfloat16_rn(p);
            }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            rows[h] = __fadd_rn(rows[h], __shfl_xor_sync(0xffffffffu, rows[h], 1));
            rows[h] = __fadd_rn(rows[h], __shfl_xor_sync(0xffffffffu, rows[h], 2));
            if (t == 0) red_l[warp][g + 8 * h] = rows[h];
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = g + 8 * h;
            const float sum = __fadd_rn(__fadd_rn(__fadd_rn(red_l[0][row], red_l[1][row]), red_l[2][row]), red_l[3][row]);
            l[h] = __fmaf_rn(l[h], alpha[h], sum);
            m[h] = next_m[h];
        }
#pragma unroll
        for (int q = 0; q < 8; ++q)
#pragma unroll
            for (int e = 0; e < 4; ++e) o[q][e] = __fmul_rn(o[q][e], alpha[e >> 1]);
        for (int ks = 0; ks < TK / 16; ++ks) {
            uint32_t a[4];
            qmm_frag::ldmatrix4(a, &ps[lane % 16][ks * 16 + (lane / 16) * 8]);
#pragma unroll
            for (int q = 0; q < 8; q += 2) {
                uint32_t b[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
                             : "r"(qmm_frag::smem(&vt[ks * 16 + lane % 16][warp * 64 + q * 8 + (lane / 16) * 8])));
                qmm_frag::mma(o[q], a, b[0], b[1]);
                qmm_frag::mma(o[q + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();
        if (NB == 1 && t0 + TK < tiles) load(t0 + TK, 0);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int i = g + 8 * h;
        if (i >= G) continue;
        const long long base = (static_cast<long long>(r) * NCH + c) * H + hk * G + i;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            float2 v = make_float2(o[q][2 * h], o[q][2 * h + 1]);
            *reinterpret_cast<float2*>(PO + base * HD + warp * 64 + q * 8 + 2 * t) = v;
        }
        if (warp == 0 && t == 0) {
            PM[base] = m[h];
            PL[base] = l[h];
        }
    }
}

extern "C" __global__ void __launch_bounds__(128) fn_kv8_chunks(const __nv_bfloat16* __restrict__ Q, const int8_t* __restrict__ K8,
                                                                const __half* __restrict__ KS, const int8_t* __restrict__ V8,
                                                                const __half* __restrict__ VS, const int* __restrict__ POS0,
                                                                float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                                                const int* __restrict__ IDS, const int* __restrict__ NKR,
                                                                const int* __restrict__ SPR, int H, int HK, int G, int CH, int NCH,
                                                                float SCALE, int IDW, int QSA) {
    kv8_chunks<1>(Q, K8, KS, V8, VS, POS0, PO, PM, PL, IDS, NKR, SPR, H, HK, G, CH, NCH, SCALE, IDW, QSA);
}

extern "C" __global__ void __launch_bounds__(128) fn_kv8_chunks_ahead(const __nv_bfloat16* __restrict__ Q, const int8_t* __restrict__ K8,
                                                                      const __half* __restrict__ KS, const int8_t* __restrict__ V8,
                                                                      const __half* __restrict__ VS, const int* __restrict__ POS0,
                                                                      float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                                                      const int* __restrict__ IDS, const int* __restrict__ NKR,
                                                                      const int* __restrict__ SPR, int H, int HK, int G, int CH, int NCH,
                                                                      float SCALE, int IDW, int QSA) {
    kv8_chunks<2>(Q, K8, KS, V8, VS, POS0, PO, PM, PL, IDS, NKR, SPR, H, HK, G, CH, NCH, SCALE, IDW, QSA);
}

// fn_attn_merge, then (ROT) the rotation back in fp32 before the one bf16 rounding.
extern "C" __global__ void __launch_bounds__(256) fn_kv8_merge(const float* __restrict__ PO, const float* __restrict__ PM,
                                                               const float* __restrict__ PL, const int* __restrict__ POS0,
                                                               __nv_bfloat16* __restrict__ OUT, const int* __restrict__ NKR,
                                                               const int* __restrict__ SPR, int H, int HK, int G, int CH, int NCH,
                                                               int QSA, int ROT) {
    const int r = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    int n = POS0[0] + r + 1;
    if (QSA && SPR[r] != 0) n = NKR[r];
    const int chunks = (n + CH - 1) / CH;
    float m = -INFINITY, l = 0.0f, o = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        const long long base = (static_cast<long long>(r) * NCH + c) * H + head;
        const float cm = PM[base], cl = PL[base], co = PO[base * HD + d];
        const bool active = cl > 0.0f;
        const float next = active ? fmaxf(m, cm) : m;
        const float a = active ? (m == -INFINITY ? 0.0f : tf_exp(__fsub_rn(m, next))) : 1.0f;
        const float b = active ? tf_exp(__fsub_rn(cm, next)) : 0.0f;
        o = __fmaf_rn(co, b, __fmul_rn(o, a));
        l = __fmaf_rn(cl, b, __fmul_rn(l, a));
        m = next;
    }
    float y = __fdiv_rn(o, l);
    if (ROT) y = wht32(y);
    OUT[(static_cast<long long>(r) * H + head) * HD + d] = __float2bfloat16_rn(y);
}

// A bf16 cache's rows [n][HK][256] into int8 codes + scales (an existing cache, or a test's input); ROT as above.
extern "C" __global__ void __launch_bounds__(256) fn_kv8_pack(const __nv_bfloat16* __restrict__ X, int8_t* __restrict__ C,
                                                              __half* __restrict__ S, long long rows, int ROT) {
    const long long row = blockIdx.x;  // one (pos, head) row of 256 a block
    if (row >= rows) return;
    const float x = bf(X[row * HD + threadIdx.x]);
    quant8(ROT ? wht32(x) : x, C + row * HD + threadIdx.x, S + row * NG + threadIdx.x / GR);
}

// Gather (the explicit-staging alternative to reading a host cache in place):
extern "C" __global__ void __launch_bounds__(256) fn_kv_gather(const uint4* __restrict__ src, uint4* __restrict__ dst,
                                                               const int* __restrict__ ids, int n, int row_vecs) {
    const int k = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (k >= n) return;
    const int id = ids[k];
    if (id < 0) return;
    for (int v = threadIdx.x & 31; v < row_vecs; v += 32) dst[static_cast<long long>(k) * row_vecs + v] = src[static_cast<long long>(id) * row_vecs + v];
}

// Test support (tf-cuda-test exl3-kv full): a bf16 cache's rows [rows][256] replaced in place by what an int8 (BITS 8)
extern "C" __global__ void __launch_bounds__(256) fn_kv_roundtrip(__nv_bfloat16* __restrict__ X, long long rows, int ROT, int BITS) {
    const long long row = blockIdx.x;
    if (row >= rows) return;
    float x = bf(X[row * HD + threadIdx.x]);
    if (ROT) x = wht32(x);
    const float peak = warp_max(fabsf(x));
    const float s = __half2float(__float2half_rn(fmaxf(peak, 6.103515625e-05f)));
    // L = 2^(BITS-1) levels a side: BITS 8 is fn_kv8's grid ((c + 0.5) s / 128), BITS 4 exl3_cache.zig's int4 grid
    const float lv = static_cast<float>(1 << (BITS - 1));
    const float q = fminf(fmaxf(floorf(__fmul_rn(__fdiv_rn(x, s), lv)) + lv, 0.0f), 2.0f * lv - 1.0f);
    float y = BITS == 8 ? dequant8(static_cast<int8_t>(static_cast<int>(q) - 128), s) : __fmul_rn(__fmul_rn(__fadd_rn(__fsub_rn(q, lv), 0.5f), s), __fdiv_rn(1.0f, lv));
    if (ROT) y = wht32(y);
    X[row * HD + threadIdx.x] = __float2bfloat16_rn(y);
}
