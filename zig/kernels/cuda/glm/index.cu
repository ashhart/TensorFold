// GLM-5.3-Flash on CUDA: the DSA indexer, ours. A row writes its index key (LayerNorm) and gate into its stream's
// rings, the row that completes a pool of 4 writes the pooled key, each row scores its visible pools, keeps its 512
// best (the lower pool on ties) and lists their tokens ascending. Rows name their own rings, pools and positions, so
// one launch serves any mix of streams, and a row's choice never depends on the rows beside it.

#include <cuda_bf16.h>
#include <stdint.h>

#include "qmm_frag.cuh"

namespace {

using namespace qmm_frag;

constexpr int D = 128, IH = 32, POOL = 4, TOPK = 512, SEL_W = TOPK * POOL + POOL - 1;

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// a score as a uint32 whose unsigned order is the scores' order (-0 counted as +0)
__device__ __forceinline__ uint32_t order_key(float s) {
    const int b = __float_as_int(s + 0.0f);
    return (uint32_t)(b ^ ((b >> 31) | (int)0x80000000));
}

}  // namespace

// row r: LayerNorm(k_raw) -> bf16 at its position of its key ring ``ik[r]``, the fp32 gate -> bf16 into ``ig[r]``
// (ring pointers already offset so that row p of the positions is at p * 128). 128 threads.
extern "C" __global__ void glm_index_write(const __nv_bfloat16* __restrict__ kr, long long k_stride, const float* __restrict__ gr,
        const __nv_bfloat16* __restrict__ lnw, const __nv_bfloat16* __restrict__ lnb, const unsigned long long* __restrict__ ik,
        const unsigned long long* __restrict__ ig, const int* __restrict__ pos, float eps) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    __shared__ float red[4];
    const int r = blockIdx.x, d = threadIdx.x, lane = d & 31, warp = d >> 5;
    const float x = __bfloat162float(kr[r * k_stride + d]);
    auto sum = [&](float v) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        __syncthreads();
        if (lane == 0) red[warp] = v;
        __syncthreads();
        return ((red[0] + red[1]) + red[2]) + red[3];
    };
    const float mean = sum(x) / D;
    const float xc = x - mean;
    const float var = sum(xc * xc) / D;
    const float y = xc / sqrtf(var + eps) * __bfloat162float(lnw[d]) + __bfloat162float(lnb[d]);
    const long long p = pos[r];
    reinterpret_cast<__nv_bfloat16*>(ik[r])[p * D + d] = __float2bfloat16_rn(y);
    reinterpret_cast<__nv_bfloat16*>(ig[r])[p * D + d] = __float2bfloat16_rn(gr[(long long)r * D + d]);
}

// the row that completes a pool (position % 4 == 3) writes its pooled key: the 4 keys weighted by the softmax of
// gate + ape, each product rounded to bf16, summed in order -> pk[r] row pos / 4. 128 threads. Run after
// glm_index_write over the same rows (a pool's earlier keys may be the window's or earlier ones still in the ring).
extern "C" __global__ void glm_pool_keys(const unsigned long long* __restrict__ ik, const unsigned long long* __restrict__ ig,
        const __nv_bfloat16* __restrict__ ape, const unsigned long long* __restrict__ pk, const int* __restrict__ pos) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    const int r = blockIdx.x, d = threadIdx.x;
    const long long q = pos[r];
    if ((q & 3) != 3) return;
    const long long p = q >> 2;
    const __nv_bfloat16* K = reinterpret_cast<const __nv_bfloat16*>(ik[r]);
    const __nv_bfloat16* G = reinterpret_cast<const __nv_bfloat16*>(ig[r]);
    float l[4], k[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        l[i] = __bfloat162float(G[(4 * p + i) * D + d]) + __bfloat162float(ape[i * D + d]);
        k[i] = __bfloat162float(K[(4 * p + i) * D + d]);
    }
    const float m = fmaxf(fmaxf(l[0], l[1]), fmaxf(l[2], l[3]));
    float e[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) e[i] = expf(l[i] - m);
    const float s = ((e[0] + e[1]) + e[2]) + e[3];
    float t[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) t[i] = bf(bf(e[i] / s) * k[i]);
    reinterpret_cast<__nv_bfloat16*>(pk[r])[p * D + d] = __float2bfloat16_rn(((t[0] + t[1]) + t[2]) + t[3]);
}

// each row's pool scores: s_p = sum_h w_h relu(scale * qi_h . pk_p) for its visible pools (p < (pos + 1) / 4), -inf
// for the rest of the np columns. qi [R, 32, 128], w the head weights (bf16, row stride w_stride). grid (R, np / 64),
// 128 threads: a block of 64 pools on tensor cores, the heads summed in a fixed order.
extern "C" __global__ void __launch_bounds__(128) glm_index_scores(const __nv_bfloat16* __restrict__ qi, const __nv_bfloat16* __restrict__ w,
        long long w_stride, const unsigned long long* __restrict__ pk, float* __restrict__ out, const int* __restrict__ pos, int np,
        float scale, float wscale) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    __shared__ __align__(128) unsigned char sq[IH * D * 2];
    __shared__ __align__(128) unsigned char sk[64 * D * 2];
    __shared__ float hw[IH];
    const int r = blockIdx.x, p0 = blockIdx.y * 64, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int npool = (pos[r] + 1) >> 2;
    float* dst = out + (long long)r * np;
    if (p0 >= npool) {
        if (tid < 64 && p0 + tid < np) dst[p0 + tid] = -INFINITY;
        return;
    }
    const __nv_bfloat16* P = reinterpret_cast<const __nv_bfloat16*>(pk[r]);
    // rows are 256 bytes, 16 chunks; chunk c of row x at c ^ (x & 7) within its group of 8
    auto sw = [](int x, int c) { return (c & ~7) | ((c ^ x) & 7); };
    for (int c = tid; c < IH * 16; c += 128) {
        const int h = c >> 4, ch = c & 15;
        cp16(sq + h * 256 + sw(h, ch) * 16, qi + ((long long)r * IH + h) * D + ch * 8);
    }
    for (int c = tid; c < 64 * 16; c += 128) {
        const int pp = c >> 4, ch = c & 15;
        const bool ok = p0 + pp < npool;
        cp16z(sk + pp * 256 + sw(pp, ch) * 16, P + (long long)(ok ? p0 + pp : 0) * D + ch * 8, ok);
    }
    commit();
    if (tid < IH) hw[tid] = __bfloat162float(w[r * w_stride + tid]) * wscale;
    wait<0>();
    __syncthreads();
    // warp: pools 16 warp .. +16 (two n8 tiles), all 32 heads (two m16 tiles)
    float d[2][2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) d[i][j][e] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
        uint32_t a[2][4];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int ar = i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
            ldmatrix4(a[i], sq + ar * 256 + sw(ar, ks * 2 + (lane >> 4)) * 16);
        }
        uint32_t b[4];
        const int br = warp * 16 + (lane & 7) + ((lane >> 4) << 3);
        ldmatrix4(b, sk + br * 256 + sw(br, ks * 2 + ((lane >> 3) & 1)) * 16);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            mma(d[i][0], a[i], b[0], b[1]);
            mma(d[i][1], a[i], b[2], b[3]);
        }
    }
    const int g = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            // heads g, g + 8, g + 16, g + 24 of this lane, then the 8 lanes of the column in a xor tree
            float s = (hw[g] * fmaxf(d[0][j][u] * scale, 0.0f) + hw[g + 8] * fmaxf(d[0][j][2 + u] * scale, 0.0f)) +
                      (hw[g + 16] * fmaxf(d[1][j][u] * scale, 0.0f) + hw[g + 24] * fmaxf(d[1][j][2 + u] * scale, 0.0f));
#pragma unroll
            for (int off = 4; off < 32; off <<= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
            const int p = p0 + warp * 16 + j * 8 + 2 * tq + u;
            if (g == 0 && p < np) dst[p] = p < npool ? s : -INFINITY;
        }
}

// each row's 512 best pools of its visible ones (the lower pool on ties), ascending (int64): a radix select (8 bits a
// pass) finds the 512th best score, then one pass in pool order writes the pools above it and the lowest ties.
// scores [R, np]. 1024 threads.
extern "C" __global__ void __launch_bounds__(1024) glm_index_select(const float* __restrict__ sc, long long* __restrict__ out, int np,
                                                                  const int* __restrict__ pos) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    __shared__ int hist[256];
    __shared__ int warp_tot[32];
    __shared__ uint32_t s_digit;
    __shared__ int s_need;
    const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float* row = sc + (long long)r * np;
    const int limit = min(np, max(TOPK, (pos[r] + 1) >> 2));
    uint32_t prefix = 0, fixed = 0;
    int need = TOPK;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        for (int i = tid; i < limit; i += 1024) {
            const uint32_t u = order_key(row[i]);
            if ((u & fixed) == prefix) atomicAdd(&hist[(u >> shift) & 0xFF], 1);
        }
        __syncthreads();
        if (tid == 0) {   // the highest digit with at least ``need`` pools at or above it
            int above = 0, digit = 0;
            for (int b = 255; b >= 0; --b) {
                if (above + hist[b] >= need) {
                    digit = b;
                    break;
                }
                above += hist[b];
            }
            s_digit = (uint32_t)digit;
            s_need = need - above;
        }
        __syncthreads();
        prefix |= s_digit << shift;
        fixed |= 0xFFu << shift;
        need = s_need;
        __syncthreads();
    }
    // pools above the threshold, and the first ``need`` equal to it, in pool order
    int written = 0, equal_seen = 0;
    for (int c = 0; c < limit; c += 1024) {
        const int i = c + tid;
        const uint32_t u = i < limit ? order_key(row[i]) : 0u;
        const int eq = (i < limit && u == prefix) ? 1 : 0;
        // the block's exclusive prefix of eq, then of take
        auto scan = [&](int v, int& total) {
            int x = v;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, x, off);
                if (lane >= off) x += y;
            }
            __syncthreads();
            if (lane == 31) warp_tot[warp] = x;
            __syncthreads();
            int before = 0;
            for (int w = 0; w < warp; ++w) before += warp_tot[w];
            total = 0;
            for (int w = 0; w < 32; ++w) total += warp_tot[w];
            return before + x - v;
        };
        int eq_total;
        const int eq_before = scan(eq, eq_total);
        const int take = (i < limit && (u > prefix || (eq && eq_before + equal_seen < need))) ? 1 : 0;
        int take_total;
        const int at = scan(take, take_total);
        if (take) out[(long long)r * TOPK + written + at] = i;
        written += take_total;
        equal_seen += eq_total;
    }
}

// each row's attended tokens (its 512 pools' tokens ascending, then the incomplete last pool's visible tokens, -1 past
// the row) and their count: 2048 plus those when the row has more than 512 complete pools, else 0 (a dense row).
extern "C" __global__ void glm_index_tokens(const long long* __restrict__ pools, const int* __restrict__ pos,
                                            int* __restrict__ tokens, int* __restrict__ counts) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    const int r = blockIdx.x;
    const long long q = pos[r];
    const long long npool = (q + 1) / 4;
    int* o = tokens + (long long)r * SEL_W;
    for (int i = threadIdx.x; i < TOPK * POOL; i += blockDim.x) o[i] = (int)(pools[(long long)r * TOPK + i / 4] * 4 + i % 4);
    if (threadIdx.x < 3) {
        const long long t = npool * 4 + threadIdx.x;
        o[TOPK * POOL + threadIdx.x] = t <= q ? (int)t : -1;
    }
    if (threadIdx.x == 0) {
        int tail = 0;
        for (int j = 0; j < 3; ++j) tail += (npool * 4 + j) <= q;
        counts[r] = npool > TOPK ? TOPK * POOL + tail : 0;
    }
}
