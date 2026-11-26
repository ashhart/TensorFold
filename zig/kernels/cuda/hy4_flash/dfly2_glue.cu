// DFly2 5-layer speculative drafter glue kernels (`s1` CTA row norms, top-k expand, argmax, gather).

#include "dfly2_glue.cuh"
#include "../common/reduce.cuh"
#include "../common/vec.cuh"

#include <limits.h>

namespace hy4d {

typedef __nv_bfloat16 bf16;

constexpr int H = HY4D_H;    // 3072
constexpr int V8 = H / 8;    // 384 uint4 a row
constexpr int NW = V8 / 32;  // 12 warps a row

__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

using tf_common::pack8;
using tf_common::unpack8;

__device__ __forceinline__ void f4x2(const float4& a, const float4& b, float (&f)[8]) {
    f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}

// The thread's part of a row's sum of squares: its 8 columns in order.
__device__ __forceinline__ float ss8(const float (&x)[8]) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s = __fmaf_rn(x[i], x[i], s);
    return s;
}

__device__ __forceinline__ float rinv_of(float ss, float eps) {
    return __frsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(H)), eps));
}

// bf16(w * bf16(x * rinv)) for 8 columns.
__device__ __forceinline__ uint4 norm8(const float (&x)[8], const uint4& w, float ri) {
    float wf[8], o[8];
    unpack8(w, wf);
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = __fmul_rn(wf[i], rbf(__fmul_rn(x[i], ri)));
    return pack8(o);
}

// Row sum of `v` across 12 warps: warp butterfly then 12 warp sums in warp order (`part` is `[NW]` scratch).
__device__ __forceinline__ float row_sum(float v, float* part) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = tf_common::warp_sum(v);
    if (lane == 0) part[w] = v;
    __syncthreads();
    const float mine = lane < NW ? part[lane] : 0.f;
    float t = 0.f;
#pragma unroll
    for (int g = 0; g < NW; ++g) t = __fadd_rn(t, __shfl_sync(0xffffffffu, mine, g));
    return t;
}

__device__ __forceinline__ uint4 load_row(const Hy4DflyRows& s, int r, int v) {
    long long i = r;
    if (s.ids != nullptr) {
        const int id = s.ids[static_cast<long long>(r) * s.ids_stride];
        if (id < 0 || id >= s.nrows) return make_uint4(0u, 0u, 0u, 0u);
        i = id;
    }
    return __ldg(reinterpret_cast<const uint4*>(s.base + i * s.ld) + v);
}

__device__ __forceinline__ uint4 ldw(const bf16* w, int v) { return __ldg(reinterpret_cast<const uint4*>(w) + v); }

}  // namespace hy4d

using namespace hy4d;

// (1) Dual RMSNorm + concat: grid `(1, 2, R)`, 384 threads (capped at 40 regs via `minBlocksPerMultiprocessor=4`).
extern "C" __global__ void __launch_bounds__(V8, 4)
    hy4_dfly_dual_rmsnorm_cat_s1(Hy4DflyRows a, const bf16* __restrict__ wa, Hy4DflyRows b,
                                 const bf16* __restrict__ wb, float eps, bf16* __restrict__ out, long long ld_out) {
    __shared__ float part[NW];
    const int which = blockIdx.y, r = blockIdx.z;
    const int v = static_cast<int>(threadIdx.x);
    const uint4 x = load_row(which ? b : a, r, v);
    const uint4 wv = ldw(which ? wb : wa, v);
    float f[8];
    unpack8(x, f);
    const float ss = row_sum(ss8(f), part);
    const float ri = rinv_of(ss, eps);
    reinterpret_cast<uint4*>(out + static_cast<long long>(r) * ld_out + which * H)[v] = norm8(f, wv, ri);
}

// (2) `mtp_fusion` gated add over the query block + layer-0 `input_layernorm`: grid `(1, N)`, 384 threads.
extern "C" __global__ void __launch_bounds__(V8, 4)
    hy4_dfly_fusion_gated_add_s1(Hy4DflyRows e, const float* __restrict__ eh, long long eh_ld, long long eh_ws,
                                 int eh_world, const bf16* __restrict__ gate, const bf16* __restrict__ wn, float eps,
                                 int qw, bf16* __restrict__ res_out, long long ld_res, bf16* __restrict__ norm_out,
                                 long long ld_norm) {
    __shared__ float part[NW];
    const int r = blockIdx.y;
    const int v = static_cast<int>(threadIdx.x);
    float h[8];
    unpack8(load_row(e, r, v), h);
    if (r % qw == 0) {
        const float4* p = reinterpret_cast<const float4*>(eh + static_cast<long long>(r / qw) * eh_ld) + 2 * v;
        float f[8];
        f4x2(__ldg(p), __ldg(p + 1), f);
        const uint4 gv = ldw(gate, v);
        for (int w = 1; w < eh_world; ++w) {
            p += eh_ws / 4;
            const float4 q0 = __ldg(p), q1 = __ldg(p + 1);
            f[0] = __fadd_rn(f[0], q0.x); f[1] = __fadd_rn(f[1], q0.y); f[2] = __fadd_rn(f[2], q0.z);
            f[3] = __fadd_rn(f[3], q0.w); f[4] = __fadd_rn(f[4], q1.x); f[5] = __fadd_rn(f[5], q1.y);
            f[6] = __fadd_rn(f[6], q1.z); f[7] = __fadd_rn(f[7], q1.w);
        }
        float g[8];
        unpack8(gv, g);
#pragma unroll
        for (int i = 0; i < 8; ++i) h[i] = rbf(__fadd_rn(h[i], rbf(__fmul_rn(g[i], rbf(f[i])))));
    }
    reinterpret_cast<uint4*>(res_out + static_cast<long long>(r) * ld_res)[v] = pack8(h);
    const uint4 wv = ldw(wn, v);
    const float ss = row_sum(ss8(h), part);
    const float ri = rinv_of(ss, eps);
    reinterpret_cast<uint4*>(norm_out + static_cast<long long>(r) * ld_norm)[v] = norm8(h, wv, ri);
}

// (3) Sublayer boundary: grid `(1, R)`, 384 threads; `res` may alias `ro` in place.
extern "C" __global__ void __launch_bounds__(V8, 4)
    hy4_dfly_sublayer_boundary_s1(const float* __restrict__ P, long long pld, long long pws, int world,
                                  const bf16* __restrict__ sw, const bf16* res, long long ld_ri,
                                  const bf16* __restrict__ nw, bf16* ro, long long ld_ro, bf16* __restrict__ no,
                                  long long ld_no, float eps) {
    __shared__ float part[2][NW];
    const int r = blockIdx.y;
    const int v = static_cast<int>(threadIdx.x);
    const float4* p = reinterpret_cast<const float4*>(P + static_cast<long long>(r) * pld) + 2 * v;
    float s[8];
    f4x2(__ldg(p), __ldg(p + 1), s);
    const uint4 rv = reinterpret_cast<const uint4*>(res + static_cast<long long>(r) * ld_ri)[v];
    const bool has_sw = sw != nullptr, has_nw = nw != nullptr;
    uint4 wv = make_uint4(0u, 0u, 0u, 0u);
    if (has_sw) wv = ldw(sw, v);
    for (int w = 1; w < world; ++w) {
        p += pws / 4;
        const float4 q0 = __ldg(p), q1 = __ldg(p + 1);
        s[0] = __fadd_rn(s[0], q0.x); s[1] = __fadd_rn(s[1], q0.y); s[2] = __fadd_rn(s[2], q0.z);
        s[3] = __fadd_rn(s[3], q0.w); s[4] = __fadd_rn(s[4], q1.x); s[5] = __fadd_rn(s[5], q1.y);
        s[6] = __fadd_rn(s[6], q1.z); s[7] = __fadd_rn(s[7], q1.w);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = rbf(s[i]);
    if (has_sw) {
        const float ss = row_sum(ss8(s), part[0]);
        const float ri = rinv_of(ss, eps);
        float wf[8];
        unpack8(wv, wf);
#pragma unroll
        for (int i = 0; i < 8; ++i) s[i] = rbf(__fmul_rn(wf[i], rbf(__fmul_rn(s[i], ri))));
    }
    if (has_nw) wv = ldw(nw, v);
    float h[8];
    unpack8(rv, h);
#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = rbf(__fadd_rn(h[i], s[i]));
    if (ro != nullptr) reinterpret_cast<uint4*>(ro + static_cast<long long>(r) * ld_ro)[v] = pack8(h);
    if (has_nw) {
        const float ss = row_sum(ss8(h), part[1]);
        const float ri = rinv_of(ss, eps);
        reinterpret_cast<uint4*>(no + static_cast<long long>(r) * ld_no)[v] = norm8(h, wv, ri);
    }
}

// (4) `extend_topk`: grid `(B * num_layers, ceil(k_pad / 512))`, 128 threads (one `int4` across `qw` rows per thread).
extern "C" __global__ void __launch_bounds__(128) hy4_dfly_extend_topk_k(const Hy4DflyTopkArgs a) {
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
    if (c4 >= (a.k_pad >> 2)) return;
    const int col0 = c4 * 4, TT = a.target_topk, LW = a.local_window;
    const int ws = anchor - LW;  // history keeps h < ws

    int hv[4] = {-1, -1, -1, -1};  // q-independent value (history or window); causal columns use cj
    int cj[4] = {-1, -1, -1, -1};  // causal block index j for causal columns
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

// (5) Cross-rank argmax reduction over `argmax_full[world][B][2]`, writing `prev_ids[b]` and `draft_out` on GPU.
extern "C" __global__ void __launch_bounds__(64) hy4_dfly_pick_argmax(
    const unsigned int* __restrict__ argmax_full, int world, int B, int step, int stride,
    int* __restrict__ prev_ids, unsigned int* __restrict__ draft_out) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    float best_v = -INFINITY;
    unsigned int best_id = 0xffffffffu;
    for (int rk = 0; rk < world; ++rk) {
        const int idx = (rk * B + b) * 2;
        const float v = __uint_as_float(argmax_full[idx]);
        const unsigned int id = argmax_full[idx + 1];
        if (!isfinite(v)) {
            best_v = v;
            best_id = 0xffffffffu;
            break;
        }
        if (v > best_v || (v == best_v && id < best_id)) {
            best_v = v;
            best_id = id;
        }
    }
    prev_ids[b] = static_cast<int>(best_id);
    draft_out[b * stride + step] = best_id;
}

// (6) Gather B target_hidden rows (each bf16 [3072]) from per-request device pointers into contiguous [B, 3072].
extern "C" __global__ void __launch_bounds__(384) hy4_dfly_gather_hidden(
    const unsigned long long* __restrict__ src_ptrs, bf16* __restrict__ dst, int B) {
    const int b = blockIdx.x;
    const int t = threadIdx.x;
    if (b >= B || t >= V8) return;
    const uint4* src = reinterpret_cast<const uint4*>(src_ptrs[b]);
    reinterpret_cast<uint4*>(dst + static_cast<size_t>(b) * H)[t] = src[t];
}
