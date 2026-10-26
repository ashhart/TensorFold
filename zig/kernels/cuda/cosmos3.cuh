// Cosmos3-Edge's CUDA glue at cosmos3.metal's rounding points; K keeps them, F plants a fault (contract controls).
#pragma once

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace tf_cos3 {

constexpr int D = 2048;   // text hidden
constexpr int QH = 16;    // query heads
constexpr int KVH = 8;    // key and value heads
constexpr int HD = 128;   // head dim
constexpr int SEC_H = 20; // rotary pairs on the h axis, then w; the rest t (Qwen3-VL's interleave)
constexpr int SEC_W = 20;
constexpr int TR = 8;     // query rows (a warp each) of one head an attention block
constexpr int KS = HD + 2; // a key row's stride in halves: 65 words, odd, so lane j's row starts in its own bank

// Rounding points (K bits; the served kernels keep them all) and the attention kernels' planted faults (F).
constexpr int SQ = 1;      // dequantize: scale * q rounded to bf16 before the bias
constexpr int ADD = 2;     // norm: h + r rounded to bf16
constexpr int INNER = 4;   // norm: h * rsqrt rounded to bf16 before the weight
constexpr int AXES = 8;    // M-RoPE: pairs on the h and w axes (else every pair on t)
constexpr int SUMS = 16;   // relu^2: group sums of the bf16 outputs (else of the fp32 squares)
constexpr int NAN_KEPT = 32; // relu^2: a NaN input stays NaN (else fmaxf drops it to 0)
constexpr int ALL = SQ | ADD | INNER | AXES | SUMS | NAN_KEPT;
constexpr int LATE = 1;    // prompt attention: a row also sees the next row's key
constexpr int STALE = 2;   // prompt attention: earlier blocks not rescaled to a new maximum
constexpr int MERGE = 1;   // decode attention: warps' partials merged without exp(m_w - M)
constexpr int SHORT = 2;   // decode attention: the row's own (last) key left out

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 tb(float x) { return __float2bfloat16_rn(x); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// MLX's dequantize in bf16: scale * q rounded, then plus bias rounded.
template <int K>
__device__ __forceinline__ __nv_bfloat16 dequant(float scale, uint32_t q, float bias) {
    const float sq = __fmul_rn(scale, static_cast<float>(q));
    return tb(__fadd_rn((K & SQ) ? bf(tb(sq)) : sq, bias));
}

// 64 values in order, one fp32 add each: the projections' group sum (any order is fp32-close; this one is fixed).
__device__ __forceinline__ float group64(const float* v) {
    float s = 0.0f;
#pragma unroll 8
    for (int i = 0; i < 64; ++i) s = __fadd_rn(s, v[i]);
    return s;
}

// The axis of rotary pair j (0 t, 1 h, 2 w).
template <int K>
__device__ __forceinline__ int axis(int j) {
    if (!(K & AXES)) return 0;
    if (j % 3 == 1 && j < 3 * SEC_H) return 1;
    if (j % 3 == 2 && j < 3 * SEC_W) return 2;
    return 0;
}

// One head of one row rotated (rotate_half pairs j and j + HD/2) at its three positions; lane l: pairs l and l + 32.
template <int K>
__device__ __forceinline__ void rotate(const __nv_bfloat16* src, const int* pos, float log2_theta, int lane, __nv_bfloat16* dst) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int j = lane + 32 * i;
        const float inv_freq = exp2f(__fmul_rn(-(static_cast<float>(2 * j) / static_cast<float>(HD)), log2_theta));
        const float angle = __fmul_rn(static_cast<float>(pos[axis<K>(j)]), inv_freq);
        const float c = cosf(angle), s = sinf(angle);
        const float a = bf(src[j]), b = bf(src[j + HD / 2]);
        dst[j] = tb(__fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s)));
        dst[j + HD / 2] = tb(__fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s)));
    }
}

// out[r, c] = the 8-bit table row ids[r] dequantized: a block of 256 columns a row.
template <int K>
__device__ __forceinline__ void embed(const int* __restrict__ ids, const uint8_t* __restrict__ table, const __nv_bfloat16* __restrict__ scales,
                                      const __nv_bfloat16* __restrict__ biases, __nv_bfloat16* __restrict__ out) {
    const int r = blockIdx.x, c = blockIdx.y * 256 + threadIdx.x;
    const int64_t id = ids[r];
    const int64_t g = id * (D / 64) + c / 64;
    out[static_cast<int64_t>(r) * D + c] = dequant<K>(bf(scales[g]), table[id * D + c], bf(biases[g]));
}

// Image rows over their placeholders: row r takes rows[map[r]] when map[r] >= 0 (F: a fault's row 0 for every row).
template <int F>
__device__ __forceinline__ void place(__nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ rows, const int* __restrict__ map) {
    const int r = blockIdx.x, c = blockIdx.y * 256 + threadIdx.x;
    const int src = F ? max(map[r], 0) : map[r];
    if (src >= 0) h[static_cast<int64_t>(r) * D + c] = rows[static_cast<int64_t>(src) * D + c];
}

// h += r (bf16) if add, then x = w * bf16(h * rsqrt(mean(h^2) + eps)); xs gets x's 64-value group sums as stored.
template <int K>
__device__ __forceinline__ void norm(__nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ r, const __nv_bfloat16* __restrict__ w,
                                     float eps, int add, __nv_bfloat16* __restrict__ x, float* __restrict__ xs) {
    __shared__ float part[8];
    __shared__ float row[D];
    const int64_t base = static_cast<int64_t>(blockIdx.x) * D;
    float v[8];
    float sq = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int i = threadIdx.x + 256 * k;
        __nv_bfloat16 hv = h[base + i];
        if (add) {
            const float sum = __fadd_rn(bf(hv), bf(r[base + i]));
            hv = tb(sum);
            h[base + i] = hv;
            v[k] = (K & ADD) ? bf(hv) : sum;
        } else v[k] = bf(hv);
        sq = __fmaf_rn(v[k], v[k], sq);
    }
    sq = warp_sum(sq);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    float all = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) all = __fadd_rn(all, part[i]);
    const float inv = rsqrtf(__fadd_rn(__fdiv_rn(all, static_cast<float>(D)), eps));
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int i = threadIdx.x + 256 * k;
        const float nv = __fmul_rn(v[k], inv);
        const __nv_bfloat16 o = tb(__fmul_rn(bf(w[i]), (K & INNER) ? bf(tb(nv)) : nv));
        x[base + i] = o;
        row[i] = bf(o);
    }
    __syncthreads();
    if (threadIdx.x < D / 64) xs[static_cast<int64_t>(blockIdx.x) * (D / 64) + threadIdx.x] = group64(row + 64 * threadIdx.x);
}

// A warp a (head, row): rotate queries; rotate keys, copy values into the caches at slots[r], positions pos[r].
template <int K>
__device__ __forceinline__ void qk(const __nv_bfloat16* __restrict__ qkv, const int* __restrict__ pos, const int* __restrict__ slots, int cap,
                                   float log2_theta, __nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ keys,
                                   __nv_bfloat16* __restrict__ values) {
    const int head = blockIdx.x, r = blockIdx.y, lane = threadIdx.x;
    const int at[3] = {pos[3 * r], pos[3 * r + 1], pos[3 * r + 2]};
    const __nv_bfloat16* row = qkv + static_cast<int64_t>(r) * (QH + 2 * KVH) * HD;
    if (head < QH) {
        rotate<K>(row + head * HD, at, log2_theta, lane, queries + (static_cast<int64_t>(r) * QH + head) * HD);
        return;
    }
    const int kh = head - QH;
    const int64_t dst = (static_cast<int64_t>(kh) * cap + slots[r]) * HD;
    rotate<K>(row + (QH + kh) * HD, at, log2_theta, lane, keys + dst);
    const __nv_bfloat16* v = row + (QH + KVH + kh) * HD;
    for (int d = lane; d < HD; d += 32) values[dst + d] = v[d];
}

// x = bf16(relu(x)^2) in place (NaN kept), a block a row's 256 values; xs gets the result's 64-value group sums.
template <int K>
__device__ __forceinline__ void relu2(__nv_bfloat16* __restrict__ x, int width, float* __restrict__ xs) {
    __shared__ float vals[256];
    const int64_t i = static_cast<int64_t>(blockIdx.x) * width + blockIdx.y * 256 + threadIdx.x;
    const float xv = bf(x[i]);
    const float v = ((K & NAN_KEPT) && isnan(xv)) ? xv : fmaxf(xv, 0.0f);  // fmaxf alone drops a NaN; torch's relu keeps it
    const float sq = __fmul_rn(v, v);
    const __nv_bfloat16 o = tb(sq);
    x[i] = o;
    vals[threadIdx.x] = (K & SUMS) ? bf(o) : sq;
    __syncthreads();
    if (threadIdx.x < 4) xs[static_cast<int64_t>(blockIdx.x) * (width / 64) + blockIdx.y * 4 + threadIdx.x] = group64(vals + 64 * threadIdx.x);
}

// Causal GQA over a stream's cache, TR rows (a warp each) of a head a block; xs: out's 64-value sums (two a head).
template <int F>
__device__ __forceinline__ void attn(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ keys, const __nv_bfloat16* __restrict__ values,
                                     __nv_bfloat16* __restrict__ out, const int* __restrict__ slots, int rows, int cap, float scale, float* __restrict__ xs) {
    __shared__ __nv_bfloat16 ks[32 * KS];
    __shared__ __nv_bfloat16 vs[32 * HD];
    __shared__ float qs[TR][HD];
    const int head = blockIdx.x, kh = head / (QH / KVH), r0 = blockIdx.y * TR;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = r0 + warp;
    const bool live = r < rows;
    const int rr = live ? r : rows - 1;
    for (int d = lane; d < HD; d += 32) qs[warp][d] = bf(q[(static_cast<int64_t>(rr) * QH + head) * HD + d]);
    const int first = slots[0];                                      // the segment's first row's cache slot: its keys before it
    const int n = first + rr + 1 + ((F & LATE) && rr + 1 < rows);    // keys this row sees
    const int top = first + min(r0 + TR, rows);                      // the most keys a row of this block sees
    const __nv_bfloat16* kb = keys + static_cast<int64_t>(kh) * cap * HD;
    const __nv_bfloat16* vb = values + static_cast<int64_t>(kh) * cap * HD;
    float m = -INFINITY, l = 0.0f;
    float acc[HD / 32];
#pragma unroll
    for (int i = 0; i < HD / 32; ++i) acc[i] = 0.0f;
    for (int j0 = 0; j0 < top; j0 += 32) {
        const int count = min(32, top - j0);
        __syncthreads();
        for (int e = threadIdx.x; e < 32 * HD; e += 32 * TR) {
            const int kj = e / HD, d = e % HD;
            const bool in = kj < count;
            ks[kj * KS + d] = in ? kb[static_cast<int64_t>(j0 + kj) * HD + d] : tb(0.0f);
            vs[kj * HD + d] = in ? vb[static_cast<int64_t>(j0 + kj) * HD + d] : tb(0.0f);
        }
        __syncthreads();
        const int j = j0 + lane;
        float s = -INFINITY;
        if (j < n && lane < count) {
            float dot = 0.0f;
#pragma unroll 8
            for (int d = 0; d < HD; ++d) dot = __fmaf_rn(qs[warp][d], bf(ks[lane * KS + d]), dot);
            s = __fmul_rn(dot, scale);
        }
        const float mb = warp_max(s);
        if (mb == -INFINITY) continue;                                  // every key of this block is past this row
        const float mn = fmaxf(m, mb);
        const float pj = s == -INFINITY ? 0.0f : expf(__fsub_rn(s, mn));
        const float alpha = (F & STALE) ? 1.0f : m == -INFINITY ? 0.0f : expf(__fsub_rn(m, mn));
        l = __fadd_rn(__fmul_rn(l, alpha), warp_sum(pj));
#pragma unroll
        for (int i = 0; i < HD / 32; ++i) acc[i] = __fmul_rn(acc[i], alpha);
        const int upto = min(count, n - j0);
        for (int t = 0; t < upto; ++t) {
            const float pt = __shfl_sync(0xffffffffu, pj, t);
#pragma unroll
            for (int i = 0; i < HD / 32; ++i) acc[i] = __fmaf_rn(pt, bf(vs[t * HD + lane + 32 * i]), acc[i]);
        }
        m = mn;
    }
    if (!live) return;
    float* row = qs[warp];  // this row's query is read: its slot takes the output for the group sums
#pragma unroll
    for (int i = 0; i < HD / 32; ++i) {
        const __nv_bfloat16 v = tb(__fdiv_rn(acc[i], l));
        out[(static_cast<int64_t>(r) * QH + head) * HD + lane + 32 * i] = v;
        row[lane + 32 * i] = bf(v);
    }
    __syncwarp();
    if (lane < HD / 64) xs[static_cast<int64_t>(r) * (QH * HD / 64) + (HD / 64) * head + lane] = group64(row + 64 * lane);
}

// Decode attention, a block a head: 8 warps take every 8th 32-key block with online softmaxes; warp 0 folds them.
template <int F>
__device__ __forceinline__ void attn1(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ keys, const __nv_bfloat16* __restrict__ values,
                                      __nv_bfloat16* __restrict__ out, const int* __restrict__ slots, int cap, float scale, float* __restrict__ xs) {
    __shared__ float ms[8], ls[8];
    __shared__ float os[8][HD];
    const int head = blockIdx.x, kh = head / (QH / KVH);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float qv[4];
    {
        const uint2 w = *reinterpret_cast<const uint2*>(q + head * HD + 4 * lane);
        const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&w);
#pragma unroll
        for (int t = 0; t < 4; ++t) qv[t] = bf(e[t]);
    }
    const int n = slots[0] + ((F & SHORT) ? 0 : 1);  // keys the row sees: every cached one and its own
    const __nv_bfloat16* kb = keys + static_cast<int64_t>(kh) * cap * HD + 4 * lane;
    const __nv_bfloat16* vb = values + static_cast<int64_t>(kh) * cap * HD + 4 * lane;
    float m = -INFINITY, l = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int j0 = 32 * warp; j0 < n; j0 += 32 * 8) {
        const int count = min(32, n - j0);
        float mine = -INFINITY;  // lane t's: key j0 + t's score
        // all 32 keys load (past the last: the last again, score dropped), so the loads go out together
#pragma unroll
        for (int t = 0; t < 32; ++t) {
            const uint2 w = *reinterpret_cast<const uint2*>(kb + static_cast<int64_t>(min(j0 + t, n - 1)) * HD);
            const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&w);
            float part = __fmul_rn(qv[0], bf(e[0]));
#pragma unroll
            for (int d = 1; d < 4; ++d) part = __fmaf_rn(qv[d], bf(e[d]), part);
            const float sc = __fmul_rn(warp_sum(part), scale);
            if (lane == t && t < count) mine = sc;
        }
        const float mn = fmaxf(m, warp_max(mine));
        const float pj = mine == -INFINITY ? 0.0f : expf(__fsub_rn(mine, mn));
        const float alpha = m == -INFINITY ? 0.0f : expf(__fsub_rn(m, mn));
        l = __fadd_rn(__fmul_rn(l, alpha), warp_sum(pj));
#pragma unroll
        for (int d = 0; d < 4; ++d) acc[d] = __fmul_rn(acc[d], alpha);
#pragma unroll
        for (int t = 0; t < 32; ++t) {
            const float pt = __shfl_sync(0xffffffffu, pj, t);  // 0 past the last key
            const uint2 w = *reinterpret_cast<const uint2*>(vb + static_cast<int64_t>(min(j0 + t, n - 1)) * HD);
            const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&w);
#pragma unroll
            for (int d = 0; d < 4; ++d) acc[d] = __fmaf_rn(pt, bf(e[d]), acc[d]);
        }
        m = mn;
    }
    if (lane == 0) {
        ms[warp] = m;
        ls[warp] = l;
    }
#pragma unroll
    for (int d = 0; d < 4; ++d) os[warp][4 * lane + d] = acc[d];
    __syncthreads();
    if (warp != 0) return;
    float mall = -INFINITY;
#pragma unroll
    for (int w = 0; w < 8; ++w) mall = fmaxf(mall, ms[w]);
    float lall = 0.0f, f[8];
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        f[w] = ms[w] == -INFINITY ? 0.0f : (F & MERGE) ? 1.0f : expf(__fsub_rn(ms[w], mall));
        lall = __fadd_rn(lall, __fmul_rn(ls[w], f[w]));
    }
    __syncwarp();
    float* row = os[0];  // warp 0's partials are read into registers first: its slot takes the output
    float o[4];
#pragma unroll
    for (int d = 0; d < 4; ++d) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) v = __fmaf_rn(os[w][4 * lane + d], f[w], v);
        o[d] = v;
    }
    __syncwarp();
#pragma unroll
    for (int d = 0; d < 4; ++d) {
        const __nv_bfloat16 v = tb(__fdiv_rn(o[d], lall));
        out[head * HD + 4 * lane + d] = v;
        row[4 * lane + d] = bf(v);
    }
    __syncwarp();
    if (lane < HD / 64) xs[(HD / 64) * head + lane] = group64(row + 64 * lane);
}

// out [n, k] bf16 = MLX 8-bit codes [n, k] (four a word) with (n, k / 64) scales and biases, dequantized.
template <int K>
__device__ __forceinline__ void widen8(const uint8_t* __restrict__ codes, const __nv_bfloat16* __restrict__ scales,
                                       const __nv_bfloat16* __restrict__ biases, __nv_bfloat16* __restrict__ out, int k, int64_t count) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= count) return;
    const int64_t row = i / k;
    const int64_t g = row * (k / 64) + (i % k) / 64;
    out[i] = dequant<K>(bf(scales[g]), codes[i], bf(biases[g]));
}

}  // namespace tf_cos3
