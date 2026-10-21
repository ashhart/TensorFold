// Ours: --kv-cache fp8 for Qwen3.5: key and value rows as e4m3, an fp32 scale (amax / 448) each, as qwen27 keeps them.

#include <cuda_fp8.h>

#include "qmm_frag.cuh"
#include "qwen35.cuh"

using namespace tf_q35;

// A layer's block in Seg.kv: kvh * cap rows of HD bytes for keys then values, then kvh * cap fp32 scales for each.
__device__ __forceinline__ unsigned char* layer_at(const Seg& sg, int layer, int kvh) {
    return reinterpret_cast<unsigned char*>(sg.kv) + (size_t)layer * 2 * kvh * sg.cap * (HD + 4);
}

#include "kv_fp8.cuh"

// A key and value as e4m3 rows and their scales in their cache slot; every thread of the block calls it together.
struct StoreFp8 {
    int layer, kvh;
    float* red;
    __device__ __forceinline__ void operator()(const Seg& sg, int h, int slot, int d, __nv_bfloat16 k, __nv_bfloat16 v) const {
        unsigned char* keys = layer_at(sg, layer, kvh);
        unsigned char* values = keys + (size_t)kvh * sg.cap * HD;
        float* kscale = reinterpret_cast<float*>(values + (size_t)kvh * sg.cap * HD);
        float* vscale = kscale + (size_t)kvh * sg.cap;
        const size_t row = (size_t)h * sg.cap + slot;
        tf_kv_fp8::quant_row<HD>(bf(k), d, keys + row * HD, kscale + row, red);
        tf_kv_fp8::quant_row<HD>(bf(v), d, values + row * HD, vscale + row, red);
    }
};

// tf_q35_rope's rows and t, h, w positions (rope_rows) with the keys and values written as e4m3 rows and their scales.
extern "C" __global__ void __launch_bounds__(256) tf_q35_rope8(Geo geo, __nv_bfloat16* __restrict__ Qn, const __nv_bfloat16* __restrict__ Kn,
                                                                 const __nv_bfloat16* __restrict__ Vr, const Seg* __restrict__ segs,
                                                                 const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                 const int* __restrict__ P, const int* __restrict__ M) {
    __shared__ float red[HD / 32];
    rope_rows(geo, Qn, Kn, Vr, segs, rowseg, rowanc, P, M, StoreFp8{layer, geo.kvh, red});
}

namespace {

constexpr int TK = 32;      // keys a tile
constexpr int CH = HD / 8;  // a widened row's 16-byte chunks
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

}  // namespace

// Attention over e4m3 rows on tensor cores, a block a (row, chunk, key head); a 32-key tile a softmax step.
extern "C" __global__ void __launch_bounds__(128) tf_q35_attn_partial8(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                                         const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                         const int* __restrict__ P, float scale,
                                                                         float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    __shared__ __align__(16) uint4 qs[16 * CH];
    __shared__ __align__(16) uint4 kt[TK * CH];
    __shared__ __align__(16) uint4 vt[TK * CH];
    __shared__ float st[16][TK + 1];                       // a tile's scores
    __shared__ __align__(16) __nv_bfloat16 pt[16][TK + 8];  // its weights times the values' scales (80-byte rows)
    __shared__ float ksc[TK], vsc[TK], alpha[16];
    const int QH = geo.qh, KVH = geo.kvh, G = QH / KVH;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, qd = lane & 3;
    const size_t m = blockIdx.x;
    const int c = blockIdx.y, h = blockIdx.z;
    if (c >= MAXCH || past_rows(P, (int)m)) return;
    const Seg sg = segs[rowseg[m]];
    if (sg.len >= GROUP_LEN) return;  // a long stream's rows: tf_q35_attn_rows8
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
    const unsigned char* K = layer_at(sg, layer, KVH);
    const unsigned char* V = K + (size_t)KVH * sg.cap * HD;
    const float* KS = reinterpret_cast<const float*>(V + (size_t)KVH * sg.cap * HD);
    const float* VS = KS + (size_t)KVH * sg.cap;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int i = tid + 128 * j, line = i / CH, chunk = i % CH;
        uint4 w = make_uint4(0u, 0u, 0u, 0u);
        if (line < G) w = *reinterpret_cast<const uint4*>(Q + (m * QH + (size_t)h * G + line) * HD + chunk * 8);
        qs[swz(line, chunk)] = w;
    }
    // a thread's four key rows of a tile (keys tid / 16 + 8 j, codes 16 (tid % 16) on), loaded a tile ahead
    const int part = tid & 15;
    uint4 kr[4], vr[4];
    float sr = 0.0f;
    auto fetch = [&](int t0) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int t = t0 + (tid >> 4) + 8 * j;
            kr[j] = vr[j] = make_uint4(0u, 0u, 0u, 0u);
            if (t < k1) {
                const int pos = t < sg.len ? t : sg.len + seen_row(anc, t - sg.len);
                const size_t row = (size_t)h * sg.cap + pos;
                kr[j] = *reinterpret_cast<const uint4*>(K + row * HD + part * 16);
                vr[j] = *reinterpret_cast<const uint4*>(V + row * HD + part * 16);
            }
        }
        const int t = t0 + (tid & 31);
        sr = 0.0f;
        if (tid < 64 && t < k1) {
            const int pos = t < sg.len ? t : sg.len + seen_row(anc, t - sg.len);
            sr = (tid < 32 ? KS : VS)[(size_t)h * sg.cap + pos];
        }
    };
    auto put = [&]() {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int u = (tid >> 4) + 8 * j;
            widen16(kr[j], &kt[swz(u, 2 * part)], &kt[swz(u, 2 * part + 1)]);
            widen16(vr[j], &vt[swz(u, 2 * part)], &vt[swz(u, 2 * part + 1)]);
        }
        if (tid < 32) ksc[tid] = sr;
        else if (tid < 64) vsc[tid - 32] = sr;
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
    fetch(k0);
    for (int t0 = k0; t0 < k1; t0 += TK) {
        __syncthreads();
        put();
        __syncthreads();
        if (t0 + TK < k1) fetch(t0 + TK);
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
            st[g][u] = in0 ? s[0] * ksc[u] * scale : -INFINITY;
            st[g][u + 1] = in1 ? s[1] * ksc[u + 1] * scale : -INFINITY;
            st[g + 8][u] = in0 ? s[2] * ksc[u] * scale : -INFINITY;
            st[g + 8][u + 1] = in1 ? s[3] * ksc[u + 1] * scale : -INFINITY;
        }
        __syncthreads();
        {  // tf_q35_attn_rows8's order: keys 16 hf + 2 qq + {0, 1, 8, 9}, the qq butterfly, the halves
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
                pt[line][u] = __float2bfloat16_rn(p * vsc[u]);
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

#include "qwen35_kv8_rows.cuh"

// tf_q35k_compact over e4m3 rows: the path's keys, values and their scales move down to the slots after the cache.
extern "C" __global__ void __launch_bounds__(256) tf_q35k_compact8(Geo geo, Seg sg, Path path) {
    const int KVH = geo.kvh;
    const int layer = blockIdx.x, h = blockIdx.y;
    unsigned char* keys = layer_at(sg, layer, KVH);
    unsigned char* values = keys + (size_t)KVH * sg.cap * HD;
    float* kscale = reinterpret_cast<float*>(values + (size_t)KVH * sg.cap * HD);
    float* vscale = kscale + (size_t)KVH * sg.cap;
    const size_t base = (size_t)h * sg.cap;
    const int d = threadIdx.x;
    for (int j = 0; j < path.n; j++) {
        const int from = sg.len + path.rows[j], to = sg.len + j;
        if (from != to) {
            keys[(base + to) * HD + d] = keys[(base + from) * HD + d];
            values[(base + to) * HD + d] = values[(base + from) * HD + d];
            if (d == 0) {
                kscale[base + to] = kscale[base + from];
                vscale[base + to] = vscale[base + from];
            }
        }
        __syncthreads();
    }
}
