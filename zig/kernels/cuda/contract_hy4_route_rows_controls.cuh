// Planted contract controls for Hy4-flash router, row-dot, argmax, and sampling kernels.
#pragma once

#include "hy4_flash/route.cuh"
#include "hy4_flash/rows.cuh"
#include "hy4_flash/dfly2_glue.cuh"

extern "C" __global__ void __launch_bounds__(128, 1)
    tf_ctl_hy4_route_high_ties(const float* __restrict__ LOG, const float* __restrict__ BIAS, int R, float scale,
                               int* __restrict__ IDS, float* __restrict__ W) {
    tf_hy4_route::route_m4<tf_hy4_route::kHighTies>(LOG, BIAS, R, scale, IDS, W);
}

extern "C" __global__ void __launch_bounds__(128, 1)
    tf_ctl_hy4_route_distinct(const float* __restrict__ LOG, const float* __restrict__ BIAS, int R, float scale,
                              int* __restrict__ IDS, float* __restrict__ W) {
    tf_hy4_route::route_m4<tf_hy4_route::kDistinct>(LOG, BIAS, R, scale, IDS, W);
}

#define HY4_ROWS_CTL(NAME, RT, C, CONTROL)                                                                             \
    extern "C" __global__ void __launch_bounds__(tf_hy4_rows::WARPS * 32)                                              \
        NAME(const __nv_bfloat16* x, long long x_stride, const __nv_bfloat16* w, float* out, long long out_stride,     \
             int rows, int n, int k, int repeat) {                                                                     \
        tf_hy4_rows::tile<RT, C, CONTROL>(x, x_stride, w, out, out_stride, rows, n, k, repeat);                        \
    }
HY4_ROWS_CTL(hy4_rows_wide_bad_nan, 16, 4, 1)
HY4_ROWS_CTL(hy4_rows_wide_bad_repeat, 16, 4, 2)
HY4_ROWS_CTL(hy4_rows_wide_bad_tail, 16, 4, 3)
HY4_ROWS_CTL(hy4_rows_wide_bad_sums, 16, 4, 4)
HY4_ROWS_CTL(hy4_rows_narrow_bad_nan, 4, 2, 1)
HY4_ROWS_CTL(hy4_rows_narrow_bad_repeat, 4, 2, 2)
HY4_ROWS_CTL(hy4_rows_narrow_bad_tail, 4, 2, 3)
HY4_ROWS_CTL(hy4_rows_narrow_bad_sums, 4, 2, 4)
#undef HY4_ROWS_CTL

namespace tf_ctl_hy4_argmax {
template <int MODE>
__device__ __forceinline__ void body(const float* __restrict__ logits, int vocab_local, int vocab_lo, int rows,
                                     unsigned int* __restrict__ out) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float* row = logits + static_cast<size_t>(r) * vocab_local;
    float best_v = -INFINITY;
    int best_id = MODE == 2 ? -1 : 0x7fffffff;
    int bad = 0;
    for (int i = tid; i < vocab_local; i += 256) {
        const float v = row[i];
        if (!isfinite(v)) {
            if constexpr (MODE != 1) bad = 1;
        } else if (v > best_v || (v == best_v && (MODE == 2 ? i > best_id : i < best_id))) {
            best_v = v;
            best_id = i;
        }
    }
    bad = __any_sync(0xFFFFFFFFu, bad);
    for (int m = 16; m > 0; m >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFu, best_v, m);
        const int oid = __shfl_xor_sync(0xFFFFFFFFu, best_id, m);
        if (ov > best_v || (ov == best_v && (MODE == 2 ? oid > best_id : oid < best_id))) {
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
        int bi = MODE == 2 ? -1 : 0x7fffffff;
        int any_bad = 0;
        for (int w = 0; w < 8; ++w) {
            any_bad |= sbad[w];
            if (sv[w] > bv || (sv[w] == bv && (MODE == 2 ? sid[w] > bi : sid[w] < bi))) {
                bv = sv[w];
                bi = sid[w];
            }
        }
        out[2 * r] = any_bad ? 0x7fc00000u : __float_as_uint(bv);
        out[2 * r + 1] = static_cast<unsigned int>(vocab_lo + bi);
    }
}
}  // namespace tf_ctl_hy4_argmax

extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_hy4_argmax_bad_nan(const float* __restrict__ logits, int vocab_local, int vocab_lo, int rows,
                              unsigned int* __restrict__ out) {
    tf_ctl_hy4_argmax::body<1>(logits, vocab_local, vocab_lo, rows, out);
}

extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_hy4_argmax_bad_tie(const float* __restrict__ logits, int vocab_local, int vocab_lo, int rows,
                              unsigned int* __restrict__ out) {
    tf_ctl_hy4_argmax::body<2>(logits, vocab_local, vocab_lo, rows, out);
}

extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_hy4_f32_to_bf16_bad_nan(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, int total,
                                   unsigned int* __restrict__ nonfinite) {
    (void)nonfinite;
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= total) return;
    if (i + 4 <= total) {
        const float4 v = *reinterpret_cast<const float4*>(src + i);
        __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y);
        __nv_bfloat162 b = __floats2bfloat162_rn(v.z, v.w);
        uint2 out;
        out.x = *reinterpret_cast<unsigned int*>(&a);
        out.y = *reinterpret_cast<unsigned int*>(&b);
        *reinterpret_cast<uint2*>(dst + i) = out;
    } else {
        for (int j = i; j < total; ++j) dst[j] = __float2bfloat16_rn(src[j]);
    }
}

extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_hy4_interleave_bad_nan(const uint4* __restrict__ src, uint4* __restrict__ dst, int local_u4, int rows,
                                  int world, unsigned int* __restrict__ nonfinite) {
    (void)nonfinite;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long plane = static_cast<long long>(rows) * local_u4;
    if (idx >= plane * world) return;
    const int row_u4 = world * local_u4;
    const int r = static_cast<int>(idx / row_u4);
    const int c = static_cast<int>(idx % row_u4);
    const int rk = c / local_u4;
    const int lc = c - rk * local_u4;
    dst[idx] = src[static_cast<long long>(rk) * plane + static_cast<long long>(r) * local_u4 + lc];
}

extern "C" __global__ void __launch_bounds__(128)
    tf_ctl_hy4_dfly_extend_topk_stale_causal(const Hy4DflyTopkArgs a) {
    const int b = blockIdx.x / a.num_layers, l = blockIdx.x % a.num_layers;
    const bool act = a.active[b] != 0;
    const int anchor = a.anchors[b];

    if (l == 0 && blockIdx.y == 0 && static_cast<int>(threadIdx.x) < a.qw) {
        const int q = threadIdx.x;
        const long long row = static_cast<long long>(b) * a.qw + q;
        const long long pos = static_cast<long long>(anchor) + q;
        long long slot = a.pad_slot;
        if (act && pos >= 0 && pos < a.max_model_len) {
            if (a.block_table != nullptr) {
                const long long blk = pos / a.block_size;
                if (blk < a.blocks_per_req) {
                    const int id = a.block_table[static_cast<long long>(b) * a.bt_stride + blk];
                    if (id >= 0) slot = static_cast<long long>(id) * a.block_size + pos % a.block_size;
                }
            } else if (pos < a.ctx) {
                const long long st = a.req_stream != nullptr ? a.req_stream[b] : b;
                if (st >= 0) slot = st * a.ctx + pos;
            }
        }
        if (a.slot_out != nullptr) a.slot_out[row] = slot;
        if (a.row_pos_out != nullptr) a.row_pos_out[row] = act ? static_cast<int>(pos) : q;
    }

    const int c4 = blockIdx.y * 128 + threadIdx.x;
    if (c4 >= ((a.target_topk + a.local_window) >> 2)) return;
    const int col0 = c4 * 4, TT = a.target_topk, LW = a.local_window;
    const int ws = anchor - LW;

    int hv[4] = {-1, -1, -1, -1};
    int cj[4] = {-1, -1, -1, -1};
    if (act) {
        if (col0 < TT) {
            const int srow = a.source_rows[b];
            if (srow >= 0 && srow < a.src_rows_cap) {
                const long long base = static_cast<long long>(a.layer_slot[l]) * a.src_slot_stride +
                                       static_cast<long long>(srow) * a.src_row_stride;
                int h[4];
                if (a.mode == HY4D_SRC_TOK) {
                    const int4 t = __ldg(reinterpret_cast<const int4*>(a.src + base + col0));
                    h[0] = t.x; h[1] = t.y; h[2] = t.z; h[3] = t.w;
                } else if (anchor - 1 < a.dense_below) {
#pragma unroll
                    for (int k = 0; k < 4; ++k) h[k] = col0 + k;
                } else {
                    const int g = __ldg(a.src + base + (col0 >> 2));
                    const bool ok = g >= 0 && g <= (INT_MAX - 3) / 4;
#pragma unroll
                    for (int k = 0; k < 4; ++k) h[k] = ok ? 4 * g + k : -1;
                }
#pragma unroll
                for (int k = 0; k < 4; ++k) hv[k] = (h[k] >= 0 && h[k] < ws) ? h[k] : -1;
            }
        } else {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int c = col0 + k - TT;
                if (c < LW) {
                    hv[k] = max(ws + c, -1);
                } else if (c - LW < a.qw) {
                    cj[k] = c - LW;
                }
            }
        }
    }
    int* o = a.out + static_cast<long long>(l) * a.out_layer_stride +
             static_cast<long long>(b) * a.qw * a.k_pad + col0;
    for (int q = 0; q < a.qw; ++q) {
        int4 w;
        w.x = cj[0] >= 0 ? (cj[0] <= q ? anchor + cj[0] : -1) : hv[0];
        w.y = cj[1] >= 0 ? (cj[1] <= q ? anchor + cj[1] : -1) : hv[1];
        w.z = cj[2] >= 0 ? (cj[2] <= q ? anchor + cj[2] : -1) : hv[2];
        w.w = cj[3] >= 0 ? (cj[3] <= q ? anchor + cj[3] : -1) : hv[3];
        *reinterpret_cast<int4*>(o + static_cast<long long>(q) * a.k_pad) = w;
    }
}
