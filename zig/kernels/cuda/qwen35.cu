// Qwen3.5's hybrids (2B, 27B) on CUDA: the Metal kernels' arithmetic and rounding; a segment is one stream's window.

#include "qwen35.cuh"
#include "qwen35_attn_tc.cuh"

using namespace tf_q35;

// P (device counts, one graph for every shape): chunks, width, rows, windows, recorded; rowpar/rowanc give the tree.
__device__ __forceinline__ bool past_windows(const int* P, int s) { return P != nullptr && s >= P[3]; }
__device__ __forceinline__ bool recorded(const int* P) { return P != nullptr && P[4] != 0; }

// One 64-entry group's sum, its entries added in order (x 16-byte aligned).
__device__ __forceinline__ float group_sum(const __nv_bfloat16* x) {
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 64; j += 8) {
        const uint4 w = *reinterpret_cast<const uint4*>(x + j);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&w);
#pragma unroll
        for (int k = 0; k < 8; k++) s += bf(b[k]);
    }
    return s;
}

// MLX affine 4-bit token rows: out = bf16(s * q + b), D wide.
extern "C" __global__ void __launch_bounds__(256) tf_q35_embed(const int* __restrict__ ids, const uint8_t* __restrict__ W,
                                                                 const __nv_bfloat16* __restrict__ S, const __nv_bfloat16* __restrict__ B,
                                                                 __nv_bfloat16* __restrict__ out, int D, const int* __restrict__ P) {
    if (past_rows(P, blockIdx.x)) return;
    const size_t id = ids[blockIdx.x];
    for (int b = threadIdx.x; b < D / 2; b += blockDim.x) {
        const uint8_t v = W[id * (D / 2) + b];
        const size_t g = id * (D / 64) + (2 * b) / 64;
        const float s = bf(S[g]), z = bf(B[g]);
        out[(size_t)blockIdx.x * D + 2 * b] = __float2bfloat16_rn(s * (float)(v & 15) + z);
        out[(size_t)blockIdx.x * D + 2 * b + 1] = __float2bfloat16_rn(s * (float)(v >> 4) + z);
    }
}

// MLX affine 8-bit token rows (a mixed checkpoint's, or widened to 8 at load): out = bf16(s * q + b), D wide.
extern "C" __global__ void __launch_bounds__(256) tf_q35_embed8(const int* __restrict__ ids, const uint8_t* __restrict__ W,
                                                                  const __nv_bfloat16* __restrict__ S, const __nv_bfloat16* __restrict__ B,
                                                                  __nv_bfloat16* __restrict__ out, int D, const int* __restrict__ P) {
    if (past_rows(P, blockIdx.x)) return;
    const size_t id = ids[blockIdx.x];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const size_t g = id * (D / 64) + d / 64;
        out[(size_t)blockIdx.x * D + d] = __float2bfloat16_rn(bf(S[g]) * (float)W[id * D + d] + bf(B[g]));
    }
}

// Image rows into a prompt chunk's embeddings: row m of H (D wide) takes feature row src[m] where src[m] >= 0.
extern "C" __global__ void __launch_bounds__(256) tf_q35_splice(__nv_bfloat16* __restrict__ H, const __nv_bfloat16* __restrict__ F,
                                                                  const int* __restrict__ src, int D, const int* __restrict__ P) {
    if (past_rows(P, blockIdx.x)) return;
    const int s = src[blockIdx.x];
    if (s < 0) return;
    for (int i = threadIdx.x; i < D; i += blockDim.x) H[(size_t)blockIdx.x * D + i] = F[(size_t)s * D + i];
}

// A norm weight as the kernel scales by it: as stored, or (SHIFT) HF's raw w as 1 + w in FP32, as torch computes it.
template <bool SHIFT>
__device__ __forceinline__ float norm_weight(__nv_bfloat16 w) {
    return SHIFT ? 1.0f + bf(w) : bf(w);
}

template <bool SHIFT>
__device__ __forceinline__ void norm_rows(__nv_bfloat16* __restrict__ H, const __nv_bfloat16* __restrict__ R, const __nv_bfloat16* __restrict__ W,
                                          __nv_bfloat16* __restrict__ X, float eps, int D, const int* __restrict__ P, float* __restrict__ XS) {
    constexpr int MAXE = 32;
    __shared__ float red[8];
    __shared__ __align__(16) __nv_bfloat16 xrow[8192];
    if (past_rows(P, blockIdx.x)) return;
    const size_t row = blockIdx.x;
    const int t = threadIdx.x;
    const int E = D / 256;
    if ((E & 3) == 0) {
        // h + r and w * (h * inv) move 8 entries a lane through xrow; the squares keep a thread's E entries in order
        uint4 wv[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int d = (t + 256 * j) * 8;
            if (d < D) wv[j] = *reinterpret_cast<const uint4*>(W + d);
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int d = (t + 256 * j) * 8;
            if (d >= D) break;
            const size_t at = row * D + d;
            uint4 hw = *reinterpret_cast<const uint4*>(H + at);
            if (R != nullptr) {
                const uint4 rw = *reinterpret_cast<const uint4*>(R + at);
                __nv_bfloat16* hb = reinterpret_cast<__nv_bfloat16*>(&hw);
                const __nv_bfloat16* rb = reinterpret_cast<const __nv_bfloat16*>(&rw);
#pragma unroll
                for (int k = 0; k < 8; k++) hb[k] = __float2bfloat16_rn(bf(hb[k]) + bf(rb[k]));
                *reinterpret_cast<uint4*>(H + at) = hw;
            }
            *reinterpret_cast<uint4*>(xrow + d) = hw;
        }
        __syncthreads();
        float sq = 0.0f;
#pragma unroll
        for (int i = 0; i < MAXE; i++) {
            if (i >= E) break;
            const float h = bf(xrow[t * E + i]);
            sq = fmaf(h, h, sq);
        }
        sq = warp_sum(sq);
        if ((t & 31) == 0) red[t >> 5] = sq;
        __syncthreads();
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) sum += red[i];
        const float scale = rsqrtf(sum / (float)D + eps);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int d = (t + 256 * j) * 8;
            if (d >= D) break;
            const uint4 hw = *reinterpret_cast<const uint4*>(xrow + d);
            const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&hw);
            const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv[j]);
            uint4 xw;
            __nv_bfloat16* xb = reinterpret_cast<__nv_bfloat16*>(&xw);
#pragma unroll
            for (int k = 0; k < 8; k++) xb[k] = __float2bfloat16_rn(norm_weight<SHIFT>(wb[k]) * (bf(hb[k]) * scale));
            *reinterpret_cast<uint4*>(X + row * D + d) = xw;
            *reinterpret_cast<uint4*>(xrow + d) = xw;
        }
        if (XS == nullptr) return;
        __syncthreads();
        for (int g = t; g < D / 64; g += blockDim.x) XS[row * (D / 64) + g] = group_sum(xrow + g * 64);
        return;
    }
    // E not a multiple of 4: a thread's entries one at a time
    float hv[MAXE];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < MAXE; i++) {
        if (i >= E) break;
        const size_t at = row * D + t * E + i;
        float h = bf(H[at]);
        if (R != nullptr) {
            const __nv_bfloat16 hb = __float2bfloat16_rn(h + bf(R[at]));
            H[at] = hb;
            h = bf(hb);
        }
        hv[i] = h;
        ss = fmaf(h, h, ss);
    }
    ss = warp_sum(ss);
    if ((t & 31) == 0) red[t >> 5] = ss;
    __syncthreads();
    float total = 0.0f;
    for (int i = 0; i < 8; i++) total += red[i];
    const float inv = rsqrtf(total / (float)D + eps);
#pragma unroll
    for (int i = 0; i < MAXE; i++) {
        if (i >= E) break;
        const int d = t * E + i;
        const __nv_bfloat16 x = __float2bfloat16_rn(norm_weight<SHIFT>(W[d]) * (hv[i] * inv));
        X[row * D + d] = x;
        xrow[d] = x;
    }
    if (XS == nullptr) return;
    __syncthreads();
    for (int g = t; g < D / 64; g += blockDim.x) XS[row * (D / 64) + g] = group_sum(xrow + g * 64);
}

// h = bf16(h + r) (r may be null), x = bf16(w * (h * inv)), XS (may be null) x's group sums; D 256k up to 8192.
extern "C" __global__ void __launch_bounds__(256) tf_q35_norm(__nv_bfloat16* __restrict__ H, const __nv_bfloat16* __restrict__ R,
                                                               const __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ X, float eps, int D,
                                                               const int* __restrict__ P, float* __restrict__ XS) {
    norm_rows<false>(H, R, W, X, eps, D, P, XS);
}

// tf_q35_norm over HF's raw weights: x = bf16((1 + w) * (h * inv)) with 1 + w in FP32 (a BF16 checkpoint's norms).
extern "C" __global__ void __launch_bounds__(256) tf_q35_norm_shift(__nv_bfloat16* __restrict__ H, const __nv_bfloat16* __restrict__ R,
                                                                     const __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ X, float eps,
                                                                     int D, const int* __restrict__ P, float* __restrict__ XS) {
    norm_rows<true>(H, R, W, X, eps, D, P, XS);
}

// 64-group sums of bf16 rows D wide (the decode projections' input sums). Grid (rows, groups / 64), a thread a group.
extern "C" __global__ void __launch_bounds__(64) tf_q35_sums(const __nv_bfloat16* __restrict__ X, float* __restrict__ XS, int D, const int* __restrict__ P) {
    if (past_rows(P, blockIdx.x)) return;
    const size_t row = blockIdx.x;
    const int g = blockIdx.y * blockDim.x + threadIdx.x;
    if (g < D / 64) XS[row * (D / 64) + g] = group_sum(X + row * D + g * 64);
}

// gdn_pre, conv_commit and gdn_chain: main's kernels, then their _table copies (qwen35_gdn.cuh, TF_Q35_K)
#define TF_Q35_TABLE 0
#include "qwen35_gdn.cuh"
#undef TF_Q35_TABLE
#define TF_Q35_TABLE 1
#include "qwen35_gdn.cuh"
#undef TF_Q35_TABLE

// out = bf16(silu(z) * bf16(w * rmsnorm(y))) per linear value head; rows nv * DK wide. Grid (1, nv, rows), a warp each.
extern "C" __global__ void __launch_bounds__(32) tf_q35_gdn_post(Geo geo, const __nv_bfloat16* __restrict__ Y, const __nv_bfloat16* __restrict__ Z,
                                                                   const __nv_bfloat16* __restrict__ NW, float eps, __nv_bfloat16* __restrict__ OUT,
                                                                   const int* __restrict__ P, float* __restrict__ XS) {
    __shared__ __align__(16) __nv_bfloat16 done[DK];  // the head's outputs, for XS's group sums (XS may be null)
    const int lane = threadIdx.x, hv = blockIdx.y;
    if (past_rows(P, blockIdx.z)) return;
    const size_t m = blockIdx.z;
    const size_t W = (size_t)geo.nv * DK;
    constexpr int PER = DK / 32;
    float yv[PER], ss = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; j++) {
        yv[j] = bf(Y[m * W + hv * DK + lane * PER + j]);
        ss += yv[j] * yv[j];
    }
    ss = warp_sum(ss);
    const float inv = rsqrtf(ss / (float)DK + eps);
#pragma unroll
    for (int j = 0; j < PER; j++) {
        const int d = lane * PER + j;
        const float x = rbf(bf(NW[d]) * (yv[j] * inv));
        const float zf = bf(Z[m * W + hv * DK + d]);
        const __nv_bfloat16 o = __float2bfloat16_rn(zf / (1.0f + expf(-zf)) * x);
        OUT[m * W + hv * DK + d] = o;
        done[d] = o;
    }
    if (XS == nullptr) return;
    __syncwarp();
    if (lane < DK / 64) XS[m * (W / 64) + hv * (DK / 64) + lane] = group_sum(done + lane * 64);
}

// Per-head RMSNorm of attention rows, out packed (rows, heads, HD); grid (rows * heads), 64 threads of 4 values.
template <bool SHIFT>
__device__ __forceinline__ void head_rows(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ OUT,
                                          int heads, int stride, int hstride, float eps, const int* __restrict__ P) {
    __shared__ float sums[2];
    const int head = blockIdx.x % heads;
    if (past_rows(P, blockIdx.x / heads)) return;
    const size_t m = blockIdx.x / heads;
    const int t = threadIdx.x;
    const __nv_bfloat16* x = X + m * stride + head * hstride;
    float v[4], acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = bf(x[4 * t + i]);
        acc += v[i] * v[i];
    }
    acc = warp_sum(acc);
    if ((t & 31) == 0) sums[t >> 5] = acc;
    __syncthreads();
    const float inv = rsqrtf((sums[0] + sums[1]) / (float)HD + eps);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int d = 4 * t + i;
        OUT[(m * heads + head) * HD + d] = __float2bfloat16_rn(norm_weight<SHIFT>(W[d]) * (v[i] * inv));
    }
}

extern "C" __global__ void __launch_bounds__(64) tf_q35_head_norm(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                    __nv_bfloat16* __restrict__ OUT, int heads, int stride, int hstride, float eps,
                                                                    const int* __restrict__ P) {
    head_rows<false>(X, W, OUT, heads, stride, hstride, eps, P);
}

// tf_q35_head_norm over HF's raw q and k norm weights, 1 + w in FP32.
extern "C" __global__ void __launch_bounds__(64) tf_q35_head_norm_shift(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                          __nv_bfloat16* __restrict__ OUT, int heads, int stride, int hstride,
                                                                          float eps, const int* __restrict__ P) {
    head_rows<true>(X, W, OUT, heads, stride, hstride, eps, P);
}

// A key and value as bf16 rows in their cache slot (qwen35.cu's layout: a layer's keys then values, kvh blocks of cap).
struct StoreBf16 {
    int layer, kvh;
    __device__ __forceinline__ void operator()(const Seg& sg, int h, int slot, int d, __nv_bfloat16 k, __nv_bfloat16 v) const {
        __nv_bfloat16* kv = reinterpret_cast<__nv_bfloat16*>(sg.kv);
        kv[kv_row(sg, kvh, layer, 0, h, slot) + d] = k;
        kv[kv_row(sg, kvh, layer, 1, h, slot) + d] = v;
    }
};

// Queries rotated in place, keys rotated and values copied into the row's cache slot; grid (rows), HD threads.
extern "C" __global__ void __launch_bounds__(256) tf_q35_rope(Geo geo, __nv_bfloat16* __restrict__ Qn, const __nv_bfloat16* __restrict__ Kn,
                                                                const __nv_bfloat16* __restrict__ Vr, const Seg* __restrict__ segs,
                                                                const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                const int* __restrict__ P, const int* __restrict__ M, const float* __restrict__ Y) {
    rope_rows(geo, Qn, Kn, Vr, segs, rowseg, rowanc, P, M, Y, StoreBf16{layer, geo.kvh});
}

// Causal attention in chunks of chunk_keys(the row's keys), a warp a query head; partials by (row, head, chunk).
#include "qwen35_attn.cuh"

// A row's chunks' partials into each (row, query head)'s output, rows qh * HD wide. Grid (qh, rows), a warp each.
extern "C" __global__ void __launch_bounds__(32) tf_q35_attn_merge(Geo geo, const float* __restrict__ PM, const float* __restrict__ PL,
                                                                     const float* __restrict__ PO, const Seg* __restrict__ segs,
                                                                     const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc,
                                                                     const int* __restrict__ P, __nv_bfloat16* __restrict__ OUT) {
    constexpr int DPL = HD / 32;
    const int QH = geo.qh;
    const int lane = threadIdx.x, qh = blockIdx.x;
    if (past_rows(P, blockIdx.y)) return;
    const size_t m = blockIdx.y;
    const int keys = segs[rowseg[m]].len + rows_seen(rowanc[m]);
    const int ck = chunk_keys(keys), nch = (keys + ck - 1) / ck;
    const size_t base = (m * QH + qh) * MAXCH;
    float mx = -INFINITY;
    for (int c = 0; c < nch; c++)
        if (PL[base + c] > 0.0f) mx = fmaxf(mx, PM[base + c]);
    float lsum = 0.0f, acc[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) acc[i] = 0.0f;
    for (int c = 0; c < nch; c++) {
        if (!(PL[base + c] > 0.0f)) continue;
        const float e = __expf(PM[base + c] - mx);
        lsum = fmaf(PL[base + c], e, lsum);
#pragma unroll
        for (int i = 0; i < DPL; i++) acc[i] = fmaf(PO[(base + c) * HD + lane * DPL + i], e, acc[i]);
    }
#pragma unroll
    for (int i = 0; i < DPL; i++) OUT[(m * QH + qh) * HD + lane * DPL + i] = __float2bfloat16_rn(acc[i] / lsum);
}

// A 256-thread block's outputs' 64-entry group sums into XS (null: none); rows are whole groups, so a block's are too.
__device__ __forceinline__ void block_sums(__nv_bfloat16* done, size_t total, float* XS) {
    if (XS == nullptr) return;
    __syncthreads();
    const size_t first = (size_t)blockIdx.x * 256 + threadIdx.x * 64;
    if (threadIdx.x < 4 && first < total) XS[first / 64] = group_sum(done + threadIdx.x * 64);
}

// out = bf16(y * bf16(sigmoid(gate))), the gate the second half of each query head's q_proj rows; rows qh * HD wide.
extern "C" __global__ void tf_q35_attn_gate(Geo geo, const __nv_bfloat16* __restrict__ Y, const __nv_bfloat16* __restrict__ QRAW,
                                              __nv_bfloat16* __restrict__ OUT, int rows, const int* __restrict__ P, float* __restrict__ XS) {
    __shared__ __align__(16) __nv_bfloat16 done[256];
    const size_t W = (size_t)geo.qh * HD;
    if (P != nullptr) rows = P[2];
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)rows * W) {
        const size_t r = i / W;
        const int d = (int)(i % W), head = d / HD;
        const float gate = bf(QRAW[r * 2 * W + head * 2 * HD + HD + d % HD]);
        const float s = rbf(1.0f / (1.0f + expf(-gate)));
        OUT[i] = done[threadIdx.x] = __float2bfloat16_rn(bf(Y[i]) * s);
    }
    block_sums(done, (size_t)rows * W, XS);
}

#include "qwen35_rows.cuh"
#define TF_Q35_TABLE 0
#include "qwen35_slots.cuh"
#undef TF_Q35_TABLE
#define TF_Q35_TABLE 1
#include "qwen35_slots.cuh"
#undef TF_Q35_TABLE
#include "qwen35_argmax.cuh"


extern "C" __global__ void __launch_bounds__(128) tf_q35_attn_partial(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                                        const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                        const int* __restrict__ P, float scale,
                                                                        float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    extern __shared__ __align__(16) uint4 attn_tiles[];
    tf_q35_tc::attn_one<tf_q35_tc::Bf16Tiles<128>>({attn_tiles, nullptr, nullptr, nullptr}, geo, Q, segs, rowseg, rowanc, layer, P, scale, PM, PL, PO);
}

extern "C" __global__ void __launch_bounds__(256) tf_q35_attn_rows(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                                     const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                     const int* __restrict__ P, float scale, int total,
                                                                     float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    extern __shared__ __align__(16) uint4 attn_tiles[];
    tf_q35_tc::attn_rows<4, tf_q35_tc::Bf16Tiles<256>>({attn_tiles, nullptr, nullptr, nullptr}, geo, Q, segs, rowseg, rowanc, layer, P, scale, total, PM, PL, PO);
}

extern "C" __global__ void __launch_bounds__(128) tf_q35_attn_partial_whole(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                                        const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                        const int* __restrict__ P, float scale,
                                                                        float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    extern __shared__ __align__(16) uint4 attn_tiles[];
    tf_q35_tc::attn_one<tf_q35_tc::WholeBf16Tiles<128>>({attn_tiles, nullptr, nullptr, nullptr}, geo, Q, segs, rowseg, rowanc, layer, P, scale, PM, PL, PO);
}

extern "C" __global__ void __launch_bounds__(256) tf_q35_attn_rows_whole(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                                     const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                     const int* __restrict__ P, float scale, int total,
                                                                     float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    extern __shared__ __align__(16) uint4 attn_tiles[];
    tf_q35_tc::attn_rows<4, tf_q35_tc::WholeBf16Tiles<256>>({attn_tiles, nullptr, nullptr, nullptr}, geo, Q, segs, rowseg, rowanc, layer, P, scale, total, PM, PL, PO);
}
