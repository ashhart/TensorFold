// Production Hy4-flash row, argmax, TP sharding, BF16 KV write, RoPE, and ViT helper kernels.
#include "rows.cuh"
#include "../common/reduce.cuh"

using namespace tf_hy4_rows;

extern "C" __global__ void __launch_bounds__(WARPS * 32)
    hy4_rows_wide(const __nv_bfloat16* x, long long x_stride, const __nv_bfloat16* w, float* out, long long out_stride,
                  int rows, int n, int k, int repeat) {
    tile<16, 4, 0>(x, x_stride, w, out, out_stride, rows, n, k, repeat);
}

extern "C" __global__ void __launch_bounds__(WARPS * 32)
    hy4_rows_narrow(const __nv_bfloat16* x, long long x_stride, const __nv_bfloat16* w, float* out,
                    long long out_stride, int rows, int n, int k, int repeat) {
    tile<4, 2, 0>(x, x_stride, w, out, out_stride, rows, n, k, repeat);
}

// Row-wise fp32 argmax `{float_bits(max_v), global_id}` (lowest `global_id` tie-break; non-finite sets NaN).
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_argmax(const float* __restrict__ logits, int vocab_local, int vocab_lo, int rows,
                    unsigned int* __restrict__ out) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float* row = logits + static_cast<size_t>(r) * vocab_local;
    float best_v = -INFINITY;
    int best_id = 0x7fffffff;
    int bad = 0;
    auto consider = [&](float v, int id) {
        if (!isfinite(v)) bad = 1;
        else if (v > best_v || (v == best_v && id < best_id)) {
            best_v = v;
            best_id = id;
        }
    };
    if ((vocab_local & 3) == 0 && (reinterpret_cast<uintptr_t>(row) & 15u) == 0) {
        const int n4 = vocab_local >> 2;
        const float4* row4 = reinterpret_cast<const float4*>(row);
        for (int i = tid; i < n4; i += 256) {
            const float4 v = row4[i];
            const int base = i << 2;
            consider(v.x, base);
            consider(v.y, base + 1);
            consider(v.z, base + 2);
            consider(v.w, base + 3);
        }
    } else {
        for (int i = tid; i < vocab_local; i += 256) consider(row[i], i);
    }
    bad = __any_sync(0xFFFFFFFFu, bad);
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFu, best_v, m);
        const int oid = __shfl_xor_sync(0xFFFFFFFFu, best_id, m);
        if (ov > best_v || (ov == best_v && oid < best_id)) {
            best_v = ov;
            best_id = oid;
        }
    }
    __shared__ float sv[8];
    __shared__ int sid[8];
    __shared__ int sbad[8];
    if (lane == 0) {
        sv[warp] = best_v;
        sid[warp] = best_id;
        sbad[warp] = bad;
    }
    __syncthreads();
    if (tid == 0) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        int any_bad = 0;
        for (int w = 0; w < 8; ++w) {
            any_bad |= sbad[w];
            if (sv[w] > bv || (sv[w] == bv && sid[w] < bi)) {
                bv = sv[w];
                bi = sid[w];
            }
        }
        out[2 * r] = any_bad ? 0x7fc00000u : __float_as_uint(bv);
        out[2 * r + 1] = static_cast<unsigned int>(vocab_lo + bi);
    }
}

// Extract leading `dst_u4` uint4s per row from strided `src` [rows, src_u4] into contiguous `dst` [rows, dst_u4].
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_slice_bf16(const uint4* __restrict__ src, int src_u4, uint4* __restrict__ dst, int dst_u4, int rows) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(rows) * dst_u4;
    if (idx >= total) return;
    const int r = static_cast<int>(idx / dst_u4);
    const int c = static_cast<int>(idx % dst_u4);
    dst[idx] = src[static_cast<long long>(r) * src_u4 + c];
}

// Interleave world=2 allGather output `src` [2, rows, half_u4] into `dst` [rows, 2 * half_u4].
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_interleave2_bf16(const uint4* __restrict__ src, uint4* __restrict__ dst, int half_u4, int rows) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long plane = static_cast<long long>(rows) * half_u4;
    if (idx >= plane * 2) return;
    const int r = static_cast<int>(idx / (2 * half_u4));
    const int c = static_cast<int>(idx % (2 * half_u4));
    const int rk = c >= half_u4 ? 1 : 0;
    const int lc = rk ? (c - half_u4) : c;
    dst[idx] = src[static_cast<long long>(rk) * plane + static_cast<long long>(r) * half_u4 + lc];
}

// Extract `dst_f4` float4s per row at `col_off_f4` from strided `src` `[rows, src_f4]` into `dst` `[rows, dst_f4]`.
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_slice_f32(const float4* __restrict__ src, int src_f4, int col_off_f4, float4* __restrict__ dst,
                       int dst_f4, int rows) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(rows) * dst_f4;
    if (idx >= total) return;
    const int r = static_cast<int>(idx / dst_f4);
    const int c = static_cast<int>(idx % dst_f4);
    dst[idx] = src[static_cast<long long>(r) * src_f4 + col_off_f4 + c];
}

// Fused world=2 `latent_up` combine reduction (`[0, half)`) + shared SwiGLU (`[half, +sw)`) into bf16 `out`.
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_comb_swiglu_bf16(const float* __restrict__ comb_f32, int comb_stride, int my_col,
                              const float* __restrict__ peer_f32, const __nv_bfloat16* __restrict__ gu, int gu_stride,
                              int gate_off, int up_off, __nv_bfloat16* __restrict__ out, int half, int sw, int rows,
                              int rank) {
    const int width = half + sw;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(rows) * width;
    if (idx >= total) return;
    const int r = static_cast<int>(idx / width);
    const int col = static_cast<int>(idx % width);
    if (col < half) {
        const float mine = comb_f32[static_cast<long long>(r) * comb_stride + my_col + col];
        const float peer = peer_f32[static_cast<long long>(r) * half + col];
        const float sum = (rank == 0) ? __fadd_rn(mine, peer) : __fadd_rn(peer, mine);
        out[idx] = __float2bfloat16_rn(sum);
    } else {
        const int c = col - half;
        const float g = __bfloat162float(gu[static_cast<long long>(r) * gu_stride + gate_off + c]);
        const float u = __bfloat162float(gu[static_cast<long long>(r) * gu_stride + up_off + c]);
        const float sig = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g)));
        out[idx] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(g, sig), u));
    }
}

// Fused batched `kv_a_norm` (latent=512) + `k_rope` (64 cols) for single-stream or `caches[r]` with ring wrap.
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_kv_write(const __nv_bfloat16* __restrict__ qa, int qa_stride, int q_lora,
                      const __nv_bfloat16* __restrict__ kv_norm_w, const float* __restrict__ freqs,
                      const int* __restrict__ pos, const unsigned long long* __restrict__ caches,
                      __nv_bfloat16* __restrict__ single_cache, int ring, float eps) {
    __shared__ float red[4];
    const int r = blockIdx.x;
    const int p = pos[r];
    if (p < 0) return;
    const int t = threadIdx.x;
    const int lane = t & 31, wid = t >> 5;
    const size_t slot = (ring > 0) ? static_cast<size_t>(p % ring) : static_cast<size_t>(p);
    __nv_bfloat16* dst = (caches != nullptr ? reinterpret_cast<__nv_bfloat16*>(caches[r]) : single_cache) + slot * 576;
    const __nv_bfloat16* src = qa + static_cast<size_t>(r) * qa_stride + q_lora;

    if (t < 128) {
        float sq = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float v = __bfloat162float(src[t + i * 128]);
            sq = __fmaf_rn(v, v, sq);
        }
        sq = tf_common::warp_sum(sq);
        if (lane == 0) red[wid] = sq;
    }
    __syncthreads();
    const float sum = tf_common::warp_sum(lane < 4 ? red[lane] : 0.0f);
    const float inv = rsqrtf(sum / 512.0f + eps);
    const float n0 = __bfloat162float(__float2bfloat16_rn(__bfloat162float(src[t]) * inv));
    const float n1 = __bfloat162float(__float2bfloat16_rn(__bfloat162float(src[t + 256]) * inv));
    dst[t] = __float2bfloat16_rn(n0 * __bfloat162float(kv_norm_w[t]));
    dst[t + 256] = __float2bfloat16_rn(n1 * __bfloat162float(kv_norm_w[t + 256]));

    if (t < 32) {
        const float x0 = __bfloat162float(src[512 + 2 * t]);
        const float x1 = __bfloat162float(src[512 + 2 * t + 1]);
        float sv, cv;
        sincosf(static_cast<float>(p) * freqs[t], &sv, &cv);
        const float c = __bfloat162float(__float2bfloat16_rn(cv));
        const float s = __bfloat162float(__float2bfloat16_rn(sv));
        dst[512 + 2 * t] = __float2bfloat16_rn(x0 * c - x1 * s);
        dst[512 + 2 * t + 1] = __float2bfloat16_rn(x1 * c + x0 * s);
    }
}

// Batched per-row GPT-J RoPE (64 cols, reading `pos[r]`); grid `(rows, heads, 1)`, block `32`.
extern "C" __global__ void __launch_bounds__(32)
    hy4_rows_q_rope(const __nv_bfloat16* __restrict__ x, long long xs, long long xh, __nv_bfloat16* __restrict__ out,
                    long long os, long long oh, const int* __restrict__ pos, const float* __restrict__ inv_freq) {
    const long long r = blockIdx.x, h = blockIdx.y;
    const int i = threadIdx.x;
    const __nv_bfloat16* src = x + r * xs + h * xh + 2 * i;
    const float x0 = __bfloat162float(src[0]), x1 = __bfloat162float(src[1]);
    float sv, cv;
    sincosf(static_cast<float>(pos[r]) * inv_freq[i], &sv, &cv);
    const float c = __bfloat162float(__float2bfloat16_rn(cv));
    const float s = __bfloat162float(__float2bfloat16_rn(sv));
    __nv_bfloat16* dst = out + r * os + h * oh + 2 * i;
    dst[0] = __float2bfloat16_rn(x0 * c - x1 * s);
    dst[1] = __float2bfloat16_rn(x1 * c + x0 * s);
}

// Convert fp32 logits [total] -> bf16 [total] (RNE); sets *nonfinite = 1 on NaN/Inf if non-null.
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_f32_to_bf16(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, int total,
                         unsigned int* __restrict__ nonfinite) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= total) return;
    if (i + 4 <= total) {
        const float4 v = *reinterpret_cast<const float4*>(src + i);
        if (nonfinite != nullptr && (!isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z) || !isfinite(v.w))) {
            *nonfinite = 1u;
        }
        __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y);
        __nv_bfloat162 b = __floats2bfloat162_rn(v.z, v.w);
        uint2 out;
        out.x = *reinterpret_cast<unsigned int*>(&a);
        out.y = *reinterpret_cast<unsigned int*>(&b);
        *reinterpret_cast<uint2*>(dst + i) = out;
    } else {
        for (int j = i; j < total; ++j) {
            const float x = src[j];
            if (nonfinite != nullptr && !isfinite(x)) *nonfinite = 1u;
            dst[j] = __float2bfloat16_rn(x);
        }
    }
}

// Interleave allGather output [world, rows, local_u4] -> [rows, world * local_u4] bf16; flags Inf/NaN.
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_interleave_bf16(const uint4* __restrict__ src, uint4* __restrict__ dst, int local_u4, int rows, int world,
                             unsigned int* __restrict__ nonfinite) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long plane = static_cast<long long>(rows) * local_u4;
    const long long total = plane * world;
    if (idx >= total) return;
    const int row_u4 = world * local_u4;
    const int r = static_cast<int>(idx / row_u4);
    const int c = static_cast<int>(idx % row_u4);
    const int rk = c / local_u4;
    const int lc = c - rk * local_u4;
    const uint4 v = src[static_cast<long long>(rk) * plane + static_cast<long long>(r) * local_u4 + lc];
    if (nonfinite != nullptr) {
        const unsigned int w[4] = {v.x, v.y, v.z, v.w};
        bool bad = false;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (((w[k] & 0x7f80u) == 0x7f80u) || (((w[k] >> 16) & 0x7f80u) == 0x7f80u)) bad = true;
        }
        if (bad) *nonfinite = 1u;
    }
    dst[idx] = v;
}

__device__ __forceinline__ int mrope_axis(int pair) { return (pair < 30 && (pair % 3) != 0) ? (pair % 3) : 0; }

__device__ __forceinline__ void kv_write_bf16_body(const __nv_bfloat16* __restrict__ qa, int qa_stride, int q_lora,
                                                   const __nv_bfloat16* __restrict__ kv_norm_w,
                                                   const float* __restrict__ freqs, int r, int sp,
                                                   const int* __restrict__ rot_pos, const int* __restrict__ pos3,
                                                   int ld3, __nv_bfloat16* base, size_t slot, float eps, float* red) {
    const int t = threadIdx.x;
    const int lane = t & 31, wid = t >> 5;
    const __nv_bfloat16* src = qa + static_cast<size_t>(r) * qa_stride + q_lora;
    if (t < 128) {
        float sq = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float v = __bfloat162float(src[t + i * 128]);
            sq = __fmaf_rn(v, v, sq);
        }
        sq = tf_common::warp_sum(sq);
        if (lane == 0) red[wid] = sq;
    }
    __syncthreads();
    const float sum = tf_common::warp_sum(lane < 4 ? red[lane] : 0.0f);
    const float inv = rsqrtf(sum / 512.0f + eps);
    __nv_bfloat16* dst = base + slot * 576;
    const float norm = __bfloat162float(__float2bfloat16_rn(__bfloat162float(src[t]) * inv));
    dst[t] = __float2bfloat16_rn(norm * __bfloat162float(kv_norm_w[t]));

    if (t < 32) {
        const int rp = pos3 != nullptr ? pos3[static_cast<size_t>(mrope_axis(t)) * ld3 + r]
                                       : (rot_pos != nullptr ? rot_pos[r] : sp);
        const float x0 = __bfloat162float(src[512 + 2 * t]);
        const float x1 = __bfloat162float(src[512 + 2 * t + 1]);
        float sv, cv;
        sincosf(static_cast<float>(rp) * freqs[t], &sv, &cv);
        const float c = __bfloat162float(__float2bfloat16_rn(cv));
        const float s = __bfloat162float(__float2bfloat16_rn(sv));
        dst[512 + 2 * t] = __float2bfloat16_rn(x0 * c - x1 * s);
        dst[512 + 2 * t + 1] = __float2bfloat16_rn(x1 * c + x0 * s);
    }
}

// Extended BF16 `kv_a_norm` + `k_rope` write supporting 3D M-RoPE (`pos3` / `rot_pos`).
extern "C" __global__ void __launch_bounds__(512)
    hy4_rows_kv_write_ex(const __nv_bfloat16* __restrict__ qa, int qa_stride, int q_lora,
                         const __nv_bfloat16* __restrict__ kv_norm_w, const float* __restrict__ freqs,
                         const int* __restrict__ slot_pos, const int* __restrict__ rot_pos,
                         const int* __restrict__ pos3, int ld3, const unsigned long long* __restrict__ caches,
                         __nv_bfloat16* __restrict__ single_cache, int ring, float eps) {
    __shared__ float red[4];
    const int r = blockIdx.x;
    const int sp = slot_pos[r];
    if (sp < 0) return;
    const size_t slot = (ring > 0) ? static_cast<size_t>(sp % ring) : static_cast<size_t>(sp);
    __nv_bfloat16* base = caches != nullptr ? reinterpret_cast<__nv_bfloat16*>(caches[r]) : single_cache;
    kv_write_bf16_body(qa, qa_stride, q_lora, kv_norm_w, freqs, r, sp, rot_pos, pos3, ld3, base, slot, eps, red);
}

// Paged BF16 `kv_a_norm` + `k_rope` write (`256`-token pages via `page_table[s * pt_stride + (sp >> 8)]`).
extern "C" __global__ void __launch_bounds__(512)
    hy4_rows_kv_write_paged(const __nv_bfloat16* __restrict__ qa, int qa_stride, int q_lora,
                            const __nv_bfloat16* __restrict__ kv_norm_w, const float* __restrict__ freqs,
                            const int* __restrict__ slot_pos, const int* __restrict__ rot_pos,
                            const int* __restrict__ pos3, int ld3, const unsigned int* __restrict__ page_table,
                            const int* __restrict__ row_stream, int pt_stride, __nv_bfloat16* __restrict__ arena_base,
                            float eps) {
    __shared__ float red[4];
    const int r = blockIdx.x;
    const int sp = slot_pos[r];
    if (sp < 0) return;
    const int s = row_stream != nullptr ? row_stream[r] : 0;
    const unsigned up = static_cast<unsigned>(sp);
    const size_t pg = page_table[static_cast<size_t>(s) * pt_stride + (up >> 8)];
    const size_t slot = pg * 256u + (up & 255u);
    kv_write_bf16_body(qa, qa_stride, q_lora, kv_norm_w, freqs, r, sp, rot_pos, pos3, ld3, arena_base, slot, eps, red);
}

// Batched per-row 3D interleaved M-RoPE (`mrope_section = [12, 10, 10]`) on query heads (`pos3` [3, ld3]).
extern "C" __global__ void __launch_bounds__(32)
    hy4_rows_q_rope_mrope(const __nv_bfloat16* __restrict__ x, long long xs, long long xh,
                          __nv_bfloat16* __restrict__ out, long long os, long long oh, const int* __restrict__ pos,
                          const int* __restrict__ pos3, int ld3, const float* __restrict__ inv_freq) {
    const long long r = blockIdx.x, h = blockIdx.y;
    const int i = threadIdx.x;
    const int rp = pos3 != nullptr ? pos3[static_cast<size_t>(mrope_axis(i)) * ld3 + r] : pos[r];
    const __nv_bfloat16* src = x + r * xs + h * xh + 2 * i;
    const float x0 = __bfloat162float(src[0]), x1 = __bfloat162float(src[1]);
    float sv, cv;
    sincosf(static_cast<float>(rp) * inv_freq[i], &sv, &cv);
    const float c = __bfloat162float(__float2bfloat16_rn(cv));
    const float s = __bfloat162float(__float2bfloat16_rn(sv));
    __nv_bfloat16* dst = out + r * os + h * oh + 2 * i;
    dst[0] = __float2bfloat16_rn(x0 * c - x1 * s);
    dst[1] = __float2bfloat16_rn(x1 * c + x0 * s);
}

// Splices encoded ViT image features `feat[src_idx[r]]` across all 4 iHC residual streams `streams[r, 0..3]`.
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_splice_streams(const uint4* __restrict__ feat, const int* __restrict__ src_idx,
                            uint4* __restrict__ streams, int hidden_u4, int rows) {
    const int r = blockIdx.x;
    const int c = static_cast<int>(blockIdx.y) * 256 + static_cast<int>(threadIdx.x);
    if (r >= rows || c >= hidden_u4) return;
    const int s = src_idx[r];
    if (s < 0) return;
    const uint4 v = feat[static_cast<size_t>(s) * hidden_u4 + c];
    const size_t base = static_cast<size_t>(r) * 4 * hidden_u4 + c;
    streams[base] = v;
    streams[base + hidden_u4] = v;
    streams[base + 2 * hidden_u4] = v;
    streams[base + 3 * hidden_u4] = v;
}

// ViT QKV bias + transpose `[rows, 3, heads, hd]` fp32 -> `q, k, v [heads, rows, hd]` bf16 (no 2D RoPE).
extern "C" __global__ void __launch_bounds__(128)
    hy4_rows_vit_qkv(const float* __restrict__ acc, const __nv_bfloat16* __restrict__ bias,
                     __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v,
                     int rows, int heads, int hd) {
    const long long r = blockIdx.x;
    const int h = blockIdx.y;
    const int D = heads * hd;
    const float* a = acc + r * 3 * D;
    const long long o = (static_cast<long long>(h) * rows + r) * hd;
    for (int j = threadIdx.x; j < hd; j += blockDim.x) {
        const int c = h * hd + j;
        q[o + j] = __float2bfloat16_rn(a[c] + __bfloat162float(bias[c]));
        k[o + j] = __float2bfloat16_rn(a[D + c] + __bfloat162float(bias[D + c]));
        v[o + j] = __float2bfloat16_rn(a[2 * D + c] + __bfloat162float(bias[2 * D + c]));
    }
}

// Gathers raster `[gh, gw, d_u4]` into 2x2 merged blocks `[gh/2 * gw/2, 2, 2, d_u4]` for `perceive.proj.0`.
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_vit_merge_gather(const uint4* __restrict__ src, uint4* __restrict__ dst, int gh, int gw, int d_u4) {
    const int mh = gh >> 1, mw = gw >> 1;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(mh) * mw * 4 * d_u4;
    if (idx >= total) return;
    const int c = static_cast<int>(idx % d_u4);
    const int sub = static_cast<int>((idx / d_u4) & 3);
    const int blk = static_cast<int>(idx / (4 * d_u4));
    const int oh = blk / mw, ow = blk % mw;
    const int r = (oh * 2 + (sub >> 1)) * gw + (ow * 2 + (sub & 1));
    dst[idx] = src[static_cast<long long>(r) * d_u4 + c];
}

// Appends `image_newline` `[out_u4]` after each merged row `[mh, mw, out_u4]` -> `[mh, mw + 1, out_u4]`.
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_vit_append_newline(const uint4* __restrict__ merged, const uint4* __restrict__ newline,
                                uint4* __restrict__ out, int mh, int mw, int out_u4) {
    const int out_w = mw + 1;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(mh) * out_w * out_u4;
    if (idx >= total) return;
    const int c = static_cast<int>(idx % out_u4);
    const int tok = static_cast<int>(idx / out_u4);
    const int oh = tok / out_w, ow = tok % out_w;
    out[idx] = (ow < mw) ? merged[static_cast<long long>(oh * mw + ow) * out_u4 + c] : newline[c];
}
