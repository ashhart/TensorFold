// Muse Spark's decode attention, head_dim 64, GQA 8:1: QK prep and cache write, chunked rows, merge and gate.

#include <cstdint>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

constexpr int HD = 64;    // head dim
constexpr int G = 8;      // query heads a KV head
constexpr int TILE = 32;  // keys a tile
constexpr int KP = 33;    // K tile row stride in 32-bit words (64 halves + 2): conflict-free row-per-lane reads

}  // namespace

// Grid (rows, H + HK), a warp a head (q heads, then KV heads); cache row pos % LEN of [slots, LEN, HK, 64].
extern "C" __global__ void muse_qk_prep(const __half* __restrict__ q, const __half* __restrict__ k,
                                        const __half* __restrict__ v, const float* __restrict__ cos_t,
                                        const float* __restrict__ sin_t, int rope, const int32_t* __restrict__ pos,
                                        const int32_t* __restrict__ slot, __half* __restrict__ qo,
                                        __half* __restrict__ kc, __half* __restrict__ vc, int LEN, int H, int HK,
                                        float eps) {
    const int r = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
    const bool is_q = h < H;
    const int kh = h - H;
    const __half* src = is_q ? q + ((int64_t)r * H + h) * HD : k + ((int64_t)r * HK + kh) * HD;
    const float x0 = __half2float(src[lane]), x1 = __half2float(src[lane + 32]);
    const float rs = rsqrtf(warp_sum(x0 * x0 + x1 * x1) / (float)HD + eps);
    float a = x0 * rs, b = x1 * rs;
    const int p = pos[r];
    if (rope) {
        const float c = cos_t[(int64_t)p * 32 + lane], s = sin_t[(int64_t)p * 32 + lane];
        const float a2 = a * c - b * s, b2 = b * c + a * s;
        a = a2;
        b = b2;
    }
    if (is_q) {
        __half* o = qo + ((int64_t)r * H + h) * HD;
        o[lane] = __float2half_rn(a);
        o[lane + 32] = __float2half_rn(b);
        return;
    }
    const int64_t at = (((int64_t)slot[r] * LEN + p % LEN) * HK + kh) * HD;
    kc[at + lane] = __float2half_rn(a);
    kc[at + lane + 32] = __float2half_rn(b);
    const __half* vs = v + ((int64_t)r * HK + kh) * HD;
    vc[at + lane] = vs[lane];
    vc[at + lane + 32] = vs[lane + 32];
}

// A full layer's chunk keys for a row at p: 256 up to a 16K bucket, then bucket / 64 rounded up to 256.
__device__ __forceinline__ int muse_chunk_keys(int p, int full) {
    int b = 4096;
    while (b < p + 1) b <<= 1;
    b = min(full, b);
    return max(256, ((b + 63) / 64 + 255) / 256 * 256);
}

// Grid (rows, HK, NCH): keys [max(0, pos - win), pos] in chunks of ck (CK <= 0: each row's own width).
extern "C" __global__ void __launch_bounds__(256)
    muse_attn_rows(const __half* __restrict__ qo, const __half* __restrict__ kc, const __half* __restrict__ vc,
                   const int32_t* __restrict__ pos, const int32_t* __restrict__ slot, float* __restrict__ po,
                   float* __restrict__ pm, float* __restrict__ pl, int LEN, int NCH, int CK, int win, int H, int HK,
                   float scale) {
    __shared__ uint32_t ks[TILE * KP];
    __shared__ __half2 vs[TILE * 32];
    __shared__ float qs[G][HD];
    const int r = blockIdx.x, kvh = blockIdx.y, c = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = kvh * G + warp;
    const int p = pos[r];
    const int lo = win > 0 ? max(0, p - win) : 0;
    const int ck = CK > 0 ? CK : muse_chunk_keys(p, -CK);
    const int start = lo + c * ck;
    const int end = min(start + ck, p + 1);
    const int64_t out = ((int64_t)r * H + h) * NCH + c;
    if (start >= end) {
        if (lane == 0) {
            pm[out] = -CUDART_INF_F;
            pl[out] = 0.f;
        }
        po[out * HD + lane] = 0.f;
        po[out * HD + lane + 32] = 0.f;
        return;
    }
    {
        const __half* qr = qo + ((int64_t)r * H + h) * HD;
        qs[warp][lane] = __half2float(qr[lane]) * scale;
        qs[warp][lane + 32] = __half2float(qr[lane + 32]) * scale;
    }
    const int64_t base = (int64_t)slot[r] * LEN;
    float m = -CUDART_INF_F, l = 0.f, o0 = 0.f, o1 = 0.f;
    for (int t0 = start; t0 < end; t0 += TILE) {
        __syncthreads();
        {
            const int row = threadIdx.x >> 3, part = threadIdx.x & 7;  // a key row is 8 x 16 bytes
            if (t0 + row < end) {
                const int64_t at = ((base + (t0 + row) % LEN) * HK + kvh) * HD + part * 8;
                const uint4 kk = *(const uint4*)(kc + at), vv = *(const uint4*)(vc + at);
                uint32_t* kd = ks + row * KP + part * 4;
                kd[0] = kk.x;
                kd[1] = kk.y;
                kd[2] = kk.z;
                kd[3] = kk.w;
                *(uint4*)(vs + row * 32 + part * 4) = vv;
            }
        }
        __syncthreads();
        const int nk = min(TILE, end - t0);
        float s = -CUDART_INF_F;
        if (lane < nk) {
            float acc = 0.f;
            const uint32_t* kr = ks + lane * KP;
#pragma unroll 8
            for (int w = 0; w < 32; ++w) {
                const uint32_t u = kr[w];
                const float2 kf = __half22float2(*(const __half2*)&u);
                acc += qs[warp][2 * w] * kf.x + qs[warp][2 * w + 1] * kf.y;
            }
            s = acc;
        }
        const float mt = warp_max(s);
        const float mn = fmaxf(m, mt);
        const float corr = expf(m - mn);  // m = -inf on the first tile: 0
        const float pj = lane < nk ? expf(s - mn) : 0.f;
        l = l * corr + warp_sum(pj);
        o0 *= corr;
        o1 *= corr;
        for (int j = 0; j < nk; ++j) {
            const float pb = __shfl_sync(0xffffffffu, pj, j);
            const float2 vf = __half22float2(vs[j * 32 + lane]);
            o0 += pb * vf.x;
            o1 += pb * vf.y;
        }
        m = mn;
    }
    if (lane == 0) {
        pm[out] = m;
        pl[out] = l;
    }
    po[out * HD + 2 * lane] = o0;
    po[out * HD + 2 * lane + 1] = o1;
}

// Grid (rows, H / 8), a warp a head: merged chunks, o_norm, then o * sigmoid(g) (half on sliding layers).
extern "C" __global__ void muse_attn_merge(const float* __restrict__ po, const float* __restrict__ pm,
                                           const float* __restrict__ pl, int NCH, const __half* __restrict__ g,
                                           __half* __restrict__ out, int H, float eps, int gate_mode) {
    const int r = blockIdx.x, h = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int64_t b = ((int64_t)r * H + h) * NCH;
    float M = -CUDART_INF_F;
    for (int c = 0; c < NCH; ++c) M = fmaxf(M, pm[b + c]);
    float L = 0.f, o0 = 0.f, o1 = 0.f;
    for (int c = 0; c < NCH; ++c) {
        const float l = pl[b + c];
        if (l == 0.f) continue;
        const float f = expf(pm[b + c] - M);
        L += l * f;
        o0 += po[(b + c) * HD + 2 * lane] * f;
        o1 += po[(b + c) * HD + 2 * lane + 1] * f;
    }
    const float a0 = __half2float(__float2half_rn(o0 / L)), a1 = __half2float(__float2half_rn(o1 / L));
    const float rs = rsqrtf(warp_sum(a0 * a0 + a1 * a1) / (float)HD + eps);
    const float n0 = __half2float(__float2half_rn(a0 * rs)), n1 = __half2float(__float2half_rn(a1 * rs));
    const int64_t at = ((int64_t)r * H + h) * HD + 2 * lane;
    float s0 = 1.f / (1.f + expf(-__half2float(g[at]))), s1 = 1.f / (1.f + expf(-__half2float(g[at + 1])));
    if (gate_mode == 0) {
        s0 = __half2float(__float2half_rn(s0));
        s1 = __half2float(__float2half_rn(s1));
    }
    out[at] = __float2half_rn(n0 * s0);
    out[at + 1] = __float2half_rn(n1 * s1);
}
