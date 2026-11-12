// The DFlash2 drafter's kernels (cuda_draft*.zig's design): tap gathers, key ring, dynamic conv, block attention.

#include "ops/common.hpp"

namespace {

constexpr int kHead = 128;   // the drafter's head width (q, k, v)
constexpr int kChunk = 256;  // keys a partial block scores
constexpr int kChunks = 9;   // partials a row: a 2048-row window and a block of up to 256 rows
constexpr int kWaves = 4;    // waves a partial block, each a slice of its chunk
constexpr int kSlice = kChunk / kWaves;
constexpr int kPer = kSlice / 32;  // keys a lane scores
constexpr int kDpl = kHead / 32;   // head dims a lane accumulates

__device__ __forceinline__ float wave_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x += __shfl_xor(x, mask, 32);
    return x;
}

__device__ __forceinline__ float wave_max(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x = fmaxf(x, __shfl_xor(x, mask, 32));
    return x;
}

// q (fp32, in LDS) dot one 16-bit key of `kind`, eight values a 16-byte load.
__device__ __forceinline__ float dot16(const float* q, const uint16_t* k, int kind) {
    float dot = 0.0f;
    for (int c = 0; c < kHead / 8; ++c) {
        const uint4 w = reinterpret_cast<const uint4*>(k)[c];
        const uint16_t* h = reinterpret_cast<const uint16_t*>(&w);
#pragma unroll
        for (int i = 0; i < 8; ++i) dot = fmaf(q[8 * c + i], load(h, kind, i), dot);
    }
    return dot;
}

// q (fp32, in LDS) dot one fp32 key, four values a 16-byte load.
__device__ __forceinline__ float dot32(const float* q, const float* k) {
    float dot = 0.0f;
    for (int c = 0; c < kHead / 4; ++c) {
        const float4 w = reinterpret_cast<const float4*>(k)[c];
        dot = fmaf(q[4 * c + 0], w.x, dot);
        dot = fmaf(q[4 * c + 1], w.y, dot);
        dot = fmaf(q[4 * c + 2], w.z, dot);
        dot = fmaf(q[4 * c + 3], w.w, dot);
    }
    return dot;
}

}  // namespace

// Rows of a window to gather, by value (a tree's kept path); n 0: rows 0, 1, 2, ... in order.
struct DraftRows {
    int n;
    int rows[64];
};

// A stream's slice of a frame: its drafter ring, and the context's first and next positions.
struct DraftSegment {
    unsigned long long ring;
    long long begin, position;
};

extern "C" {

// Tap ring rows start + i (start + rows.rows[i]) of R rows, `words` 32-bit words each, into OUT rows 0..n. Grid (n).
__global__ void __launch_bounds__(256)
tf_draft_gather(const unsigned* ring, int R, long long start, DraftRows rows, int words, unsigned* out) {
    const long long row = start + (rows.n > 0 ? rows.rows[blockIdx.x] : static_cast<int>(blockIdx.x));
    const unsigned* src = ring + (row % R) * words;
    unsigned* dst = out + static_cast<long long>(blockIdx.x) * words;
    for (int i = threadIdx.x; i < words; i += blockDim.x) dst[i] = src[i];
}

// fp32 keys (stored as `kind`) and values of positions pos0 + i into a layer's rings at pos % W. Grid (rows).
__global__ void __launch_bounds__(256)
tf_draft_ring_put(const float* K, const uint16_t* V, int kind, void* RK, uint16_t* RV, int width, long long pos0, int W) {
    const long long i = blockIdx.x;
    const long long slot = (pos0 + i) % W;
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        store(RK, kind, slot * width + c, K[i * width + c]);
        RV[slot * width + c] = V[i * width + c];
    }
}

// y[t, c] = sum of (BASE[b, l, c] + DYN[t, b, l, c / group]) * x[t - l, c] over its block's lags, + RES, CUDA's steps.
__global__ void __launch_bounds__(256)
tf_draft_dyn_conv(const void* X, const void* DYN, const float* BASE, const void* RES, void* OUT, int kind, int D,
                  int group, int branch, int block_rows) {
    const int t = blockIdx.x;
    const int groups = D / group;
    for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < D; c += gridDim.y * blockDim.x) {
        const int g = c / group;
        float total = 0.0f;
#pragma unroll
        for (int l = 0; l < 2; l++) {
            if (t % block_rows - l < 0) continue;
            const float x = load(X, kind, static_cast<long long>(t - l) * D + c);
            const float base = BASE[(branch * 2 + l) * D + c];
            const float dyn = load(DYN, kind, static_cast<long long>(t) * (4 * groups) + (branch * 2 + l) * groups + g);
            total = rounded(total + rounded(base * x, kind), kind);
            total = rounded(total + rounded(dyn * x, kind), kind);
        }
        if (RES != nullptr) total = rounded(total + load(RES, kind, static_cast<long long>(t) * D + c), kind);
        store(OUT, kind, static_cast<long long>(t) * D + c, total);
    }
}

// Each stream's block rows but its first (the pending token), packed in stream order. Grid (rows out).
__global__ void __launch_bounds__(256) tf_draft_mask_rows(const uint16_t* X, uint16_t* OUT, int D, int block_rows) {
    const long long row = blockIdx.x;
    const long long source = row / (block_rows - 1) * block_rows + 1 + row % (block_rows - 1);
    for (int c = threadIdx.x; c < D; c += blockDim.x) OUT[row * D + c] = X[source * D + c];
}

// 256 of row m's keys for head h (ring window, then block rows mask[m] keeps) into PM, PL, PO. Grid (rows, heads, 9).
__global__ void __launch_bounds__(128)
tf_draft_attn_partial(const float* Q, const float* KB, const uint16_t* VB, const unsigned* mask,
                      const DraftSegment* segments, int kind, int block_rows, int heads, int kv_heads, int W, int window,
                      int layer, float* PM, float* PL, float* PO) {
    __shared__ float qs[kHead];
    __shared__ float ps[kChunk];
    __shared__ float wm[kWaves], wl[kWaves];
    __shared__ float wo[kWaves][kHead];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int m = blockIdx.x, h = blockIdx.y, chunk = blockIdx.z;
    const int stream = m / block_rows, local = m % block_rows;
    const DraftSegment sg = segments[stream];
    const int kvh = h / (heads / kv_heads), width = kv_heads * kHead;
    const uint16_t* RK = reinterpret_cast<const uint16_t*>(sg.ring) + static_cast<long long>(layer) * 2 * W * width;
    const uint16_t* RV = RK + static_cast<long long>(W) * width;
    long long first = max(sg.begin, sg.position + local - window + 1);
    if (first > sg.position) first = sg.position;
    const int ring_keys = static_cast<int>(sg.position - first), keys = ring_keys + block_rows;
    const long long slot = (static_cast<long long>(m) * heads + h) * kChunks + chunk;
    if (chunk * kChunk >= keys) {  // past the row's keys: an empty partial, which the merge skips
        if (threadIdx.x == 0) {
            PM[slot] = -INFINITY;
            PL[slot] = 0.0f;
        }
        return;
    }
    const unsigned seen = mask != nullptr ? mask[m] : 0u;
    const int start = chunk * kChunk + wave * kSlice, n = max(min(keys - start, kSlice), 0);  // this wave's keys
    // key `index` is seen: a ring key, or a block row the mask keeps
    auto visible = [&](int index) {
        return index < ring_keys || mask == nullptr || ((seen >> (index - ring_keys)) & 1u) != 0;
    };
    auto ring_row = [&](int index) { return static_cast<long long>((first + index) % W) * width + kvh * kHead; };
    auto block_row = [&](int index) {
        return (static_cast<long long>(stream * block_rows + index - ring_keys) * kv_heads + kvh) * kHead;
    };
    for (int i = threadIdx.x; i < kHead; i += 32 * kWaves) qs[i] = Q[(static_cast<long long>(m) * heads + h) * kHead + i];
    __syncthreads();
    float mx = -INFINITY, score[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int e = lane + 32 * j, index = start + e;
        score[j] = -INFINITY;
        if (e >= n || !visible(index)) continue;
        const float dot = index < ring_keys ? dot16(qs, RK + ring_row(index), kind) : dot32(qs, KB + block_row(index));
        score[j] = dot * rsqrtf(static_cast<float>(kHead));
        mx = fmaxf(mx, score[j]);
    }
    mx = wave_max(mx);
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int e = lane + 32 * j;
        const float p = e < n && visible(start + e) ? expf(score[j] - mx) : 0.0f;
        ps[wave * kSlice + e] = p;
        sum += p;
    }
    sum = wave_sum(sum);
    __syncthreads();
    float o[kDpl];
#pragma unroll
    for (int i = 0; i < kDpl; ++i) o[i] = 0.0f;
    for (int e = 0; e < n; ++e) {
        const int index = start + e;
        if (!visible(index)) continue;
        const uint16_t* v = index < ring_keys ? RV + ring_row(index) : VB + block_row(index);
        const uint2 w = *reinterpret_cast<const uint2*>(v + lane * kDpl);
        const uint16_t* vh = reinterpret_cast<const uint16_t*>(&w);
        const float p = ps[wave * kSlice + e];
#pragma unroll
        for (int i = 0; i < kDpl; ++i) o[i] = fmaf(p, load(vh, kind, i), o[i]);
    }
    if (lane == 0) {
        wm[wave] = n > 0 ? mx : -INFINITY;
        wl[wave] = n > 0 ? sum : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < kDpl; ++i) wo[wave][lane * kDpl + i] = o[i];
    __syncthreads();
    if (wave != 0) return;
    float top = -INFINITY;
#pragma unroll
    for (int w = 0; w < kWaves; ++w)
        if (wl[w] > 0.0f) top = fmaxf(top, wm[w]);
    float total = 0.0f, out[kDpl];
#pragma unroll
    for (int i = 0; i < kDpl; ++i) out[i] = 0.0f;
#pragma unroll
    for (int w = 0; w < kWaves; ++w) {
        if (wl[w] == 0.0f) continue;
        const float scale = expf(wm[w] - top);
        total = fmaf(wl[w], scale, total);
#pragma unroll
        for (int i = 0; i < kDpl; ++i) out[i] = fmaf(wo[w][lane * kDpl + i], scale, out[i]);
    }
    if (lane == 0) {
        PM[slot] = total > 0.0f ? top : -INFINITY;
        PL[slot] = total;
    }
#pragma unroll
    for (int i = 0; i < kDpl; ++i) PO[slot * kHead + lane * kDpl + i] = out[i];
}

// Row m's nine partials for head h merged and normalized into OUT (rows, heads, 128) of `kind`. Grid (rows, heads).
__global__ void __launch_bounds__(32)
tf_draft_attn_merge(const float* PM, const float* PL, const float* PO, int heads, int kind, void* OUT) {
    const int lane = threadIdx.x, m = blockIdx.x, h = blockIdx.y;
    const long long base = (static_cast<long long>(m) * heads + h) * kChunks;
    float mx = -INFINITY, sum = 0.0f, out[kDpl];
#pragma unroll
    for (int i = 0; i < kDpl; ++i) out[i] = 0.0f;
    for (int chunk = 0; chunk < kChunks; ++chunk) {
        const long long slot = base + chunk;
        if (PL[slot] == 0.0f) continue;
        const float next = fmaxf(mx, PM[slot]), a = expf(mx - next), b = expf(PM[slot] - next);
        sum = sum * a + PL[slot] * b;
#pragma unroll
        for (int i = 0; i < kDpl; ++i) out[i] = fmaf(b, PO[slot * kHead + lane * kDpl + i], out[i] * a);
        mx = next;
    }
    const long long at = (static_cast<long long>(m) * heads + h) * kHead + lane * kDpl;
#pragma unroll
    for (int i = 0; i < kDpl; ++i) store(OUT, kind, at + i, out[i] / sum);
}

}  // extern "C"
