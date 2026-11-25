// DFly2 5-layer speculative drafter glue kernels (`s1` CTA and `sm_90+` cluster variants; see `hy4_dfly2_glue.cuh`).

#include "dfly2_glue.cuh"
#include "../common/vec.cuh"

#include <cooperative_groups.h>
#include <limits.h>

namespace cg = cooperative_groups;

namespace hy4d {

typedef __nv_bfloat16 bf16;

constexpr int H = HY4D_H;     // 3072
constexpr int V8 = H / 8;     // 384 uint4 a row
constexpr int NW = V8 / 32;   // 12 warps a row

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

// bf16(w * bf16(x * rinv)) for 8 columns
__device__ __forceinline__ uint4 norm8(const float (&x)[8], const uint4& w, float ri) {
    float wf[8], o[8];
    unpack8(w, wf);
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = __fmul_rn(wf[i], rbf(__fmul_rn(x[i], ri)));
    return pack8(o);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
__device__ __forceinline__ void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
#else
__device__ __forceinline__ void cluster_arrive() {}
__device__ __forceinline__ void cluster_wait() {}
#endif

template <int S>
struct Row {
    static constexpr int TPB = V8 / S;  // threads a CTA
    static constexpr int WPC = NW / S;  // warps a CTA
    static_assert(V8 % S == 0 && NW % S == 0, "split must divide the 12 warps of a row");

    __device__ static __forceinline__ int rank() {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
        if constexpr (S == 1) return 0;
        else return static_cast<int>(cg::this_cluster().block_rank());
#else
        return 0;
#endif
    }
    __device__ static __forceinline__ int col() { return rank() * TPB + static_cast<int>(threadIdx.x); }

    // Row sum of `v`: warp butterfly then 12 warp sums in warp order (`part` is `[NW]` scratch).
    __device__ static __forceinline__ float sum(float v, float* part) {
        const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (lane == 0) part[rank() * WPC + w] = v;
        if constexpr (S == 1) {
            __syncthreads();
        } else {
            cluster_arrive();
            cluster_wait();
        }
        float mine = 0.f;
        if (lane < NW) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
            if constexpr (S == 1) {
                mine = part[lane];
            } else {
                mine = cg::this_cluster().map_shared_rank(part, static_cast<unsigned>(lane / WPC))[lane];
            }
#else
            mine = part[lane];
#endif
        }
        float t = 0.f;
#pragma unroll
        for (int g = 0; g < NW; ++g) t = __fadd_rn(t, __shfl_sync(0xffffffffu, mine, g));
        return t;
    }
    // After the last sum: let peers go once they are done with our scratch.
    __device__ static __forceinline__ void done_reading() {
        if constexpr (S > 1) cluster_arrive();
    }
    __device__ static __forceinline__ void before_exit() {
        if constexpr (S > 1) cluster_wait();
    }
};

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

// (1) Dual RMSNorm + concat: grid `(S, 2, R)`.
template <int S>
__device__ __forceinline__ void dual_body(const Hy4DflyRows& a, const bf16* __restrict__ wa, const Hy4DflyRows& b,
                                          const bf16* __restrict__ wb, float eps, bf16* __restrict__ out,
                                          long long ld_out) {
    __shared__ float part[NW];
    const int which = blockIdx.y, r = blockIdx.z;
    const int v = Row<S>::col();
    const uint4 x = load_row(which ? b : a, r, v);
    const uint4 wv = ldw(which ? wb : wa, v);
    float f[8];
    unpack8(x, f);
    const float ss = Row<S>::sum(ss8(f), part);
    Row<S>::done_reading();
    const float ri = rinv_of(ss, eps);
    reinterpret_cast<uint4*>(out + static_cast<long long>(r) * ld_out + which * H)[v] = norm8(f, wv, ri);
    Row<S>::before_exit();
}

// (2) `mtp_fusion` gated add over the query block + layer-0 `input_layernorm`: grid `(S, N)`.
template <int S>
__device__ __forceinline__ void fusion_body(const Hy4DflyRows& e, const float* __restrict__ eh, long long eh_ld,
                                            long long eh_ws, int eh_world, const bf16* __restrict__ gate,
                                            const bf16* __restrict__ wn, float eps, int qw, bf16* __restrict__ res_out,
                                            long long ld_res, bf16* __restrict__ norm_out, long long ld_norm) {
    __shared__ float part[NW];
    const int r = blockIdx.y;
    const int v = Row<S>::col();
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
    const uint4 wv = ldw(wn, v);  // in flight across the reduction
    const float ss = Row<S>::sum(ss8(h), part);
    Row<S>::done_reading();
    const float ri = rinv_of(ss, eps);
    reinterpret_cast<uint4*>(norm_out + static_cast<long long>(r) * ld_norm)[v] = norm8(h, wv, ri);
    Row<S>::before_exit();
}

// (3) Sublayer boundary: grid `(S, R)`; `res` may alias `ro` in place.
template <int S>
__device__ __forceinline__ void boundary_body(const float* __restrict__ P, long long pld, long long pws, int world,
                                              const bf16* __restrict__ sw, const bf16* res, long long ld_ri,
                                              const bf16* __restrict__ nw, bf16* ro, long long ld_ro,
                                              bf16* __restrict__ no, long long ld_no, float eps) {
    __shared__ float part[2][NW];
    const int r = blockIdx.y;
    const int v = Row<S>::col();
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
        const float ss = Row<S>::sum(ss8(s), part[0]);
        if (!has_nw) Row<S>::done_reading();
        const float ri = rinv_of(ss, eps);
        float wf[8];
        unpack8(wv, wf);
#pragma unroll
        for (int i = 0; i < 8; ++i) s[i] = rbf(__fmul_rn(wf[i], rbf(__fmul_rn(s[i], ri))));
    }
    if (has_nw) wv = ldw(nw, v);  // in flight across the add and the reduction
    float h[8];
    unpack8(rv, h);
#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = rbf(__fadd_rn(h[i], s[i]));
    if (ro != nullptr) reinterpret_cast<uint4*>(ro + static_cast<long long>(r) * ld_ro)[v] = pack8(h);
    if (has_nw) {
        const float ss = Row<S>::sum(ss8(h), part[1]);
        Row<S>::done_reading();
        const float ri = rinv_of(ss, eps);
        reinterpret_cast<uint4*>(no + static_cast<long long>(r) * ld_no)[v] = norm8(h, wv, ri);
    }
    if (has_sw || has_nw) Row<S>::before_exit();
}

}  // namespace hy4d

using namespace hy4d;

// One symbol per split (`384 / S` threads; `S <= 3` caps at 40 regs, `S = 4, 6` at 48 regs).
#define HY4D_MINB(S) ((S) <= 3 ? 1536 / (V8 / (S)) : 65536 / (48 * (V8 / (S))))
#define HY4D_ROW_KERNELS(S)                                                                                            \
    extern "C" __global__ void __launch_bounds__(V8 / S, HY4D_MINB(S))             \
        hy4_dfly_dual_rmsnorm_cat_s##S(Hy4DflyRows a, const bf16* __restrict__ wa, Hy4DflyRows b,                     \
                                       const bf16* __restrict__ wb, float eps, bf16* __restrict__ out,               \
                                       long long ld_out) {                                                            \
        dual_body<S>(a, wa, b, wb, eps, out, ld_out);                                                                 \
    }                                                                                                                 \
    extern "C" __global__ void __launch_bounds__(V8 / S, HY4D_MINB(S))             \
        hy4_dfly_fusion_gated_add_s##S(Hy4DflyRows e, const float* __restrict__ eh, long long eh_ld,                 \
                                       long long eh_ws, int eh_world, const bf16* __restrict__ gate,                 \
                                       const bf16* __restrict__ wn, float eps, int qw, bf16* __restrict__ res_out,   \
                                       long long ld_res, bf16* __restrict__ norm_out, long long ld_norm) {           \
        fusion_body<S>(e, eh, eh_ld, eh_ws, eh_world, gate, wn, eps, qw, res_out, ld_res, norm_out, ld_norm);         \
    }                                                                                                                 \
    extern "C" __global__ void __launch_bounds__(V8 / S, HY4D_MINB(S))             \
        hy4_dfly_sublayer_boundary_s##S(const float* __restrict__ P, long long pld, long long pws, int world,         \
                                        const bf16* __restrict__ sw, const bf16* res, long long ld_ri,               \
                                        const bf16* __restrict__ nw, bf16* ro, long long ld_ro,                      \
                                        bf16* __restrict__ no, long long ld_no, float eps) {                          \
        boundary_body<S>(P, pld, pws, world, sw, res, ld_ri, nw, ro, ld_ro, no, ld_no, eps);                          \
    }

extern "C" __global__ void __launch_bounds__(V8, HY4D_MINB(1))
    hy4_dfly_dual_rmsnorm_cat_s1(Hy4DflyRows a, const bf16* __restrict__ wa, Hy4DflyRows b,
                                 const bf16* __restrict__ wb, float eps, bf16* __restrict__ out, long long ld_out) {
    dual_body<1>(a, wa, b, wb, eps, out, ld_out);
}
extern "C" __global__ void __launch_bounds__(V8, HY4D_MINB(1))
    hy4_dfly_fusion_gated_add_s1(Hy4DflyRows e, const float* __restrict__ eh, long long eh_ld, long long eh_ws,
                                 int eh_world, const bf16* __restrict__ gate, const bf16* __restrict__ wn, float eps,
                                 int qw, bf16* __restrict__ res_out, long long ld_res, bf16* __restrict__ norm_out,
                                 long long ld_norm) {
    fusion_body<1>(e, eh, eh_ld, eh_ws, eh_world, gate, wn, eps, qw, res_out, ld_res, norm_out, ld_norm);
}
extern "C" __global__ void __launch_bounds__(V8, HY4D_MINB(1))
    hy4_dfly_sublayer_boundary_s1(const float* __restrict__ P, long long pld, long long pws, int world,
                                  const bf16* __restrict__ sw, const bf16* res, long long ld_ri,
                                  const bf16* __restrict__ nw, bf16* ro, long long ld_ro, bf16* __restrict__ no,
                                  long long ld_no, float eps) {
    boundary_body<1>(P, pld, pws, world, sw, res, ld_ri, nw, ro, ld_ro, no, ld_no, eps);
}
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
HY4D_ROW_KERNELS(2)
HY4D_ROW_KERNELS(3)
HY4D_ROW_KERNELS(4)
HY4D_ROW_KERNELS(6)
#endif

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
                    if (id > 0) slot = static_cast<long long>(id) * a.block_size + pos % a.block_size;
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

// (5) Guarded cache-row scatter: 72 `uint4` per row (64 latent + 8 rope).
extern "C" __global__ void __launch_bounds__(256) hy4_dfly_kv_overlay_write_k(
    const bf16* __restrict__ kv_c, long long ld_c, const bf16* __restrict__ k_pe, long long ld_pe,
    const long long* __restrict__ slots, int N, bf16* __restrict__ cache, long long cache_ld, long long num_slots) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(N) * 72) return;
    const int row = static_cast<int>(i / 72), part = static_cast<int>(i % 72);
    const long long slot = slots[row];
    if (slot < 0 || slot >= num_slots) return;
    const uint4 v = part < 64 ? __ldg(reinterpret_cast<const uint4*>(kv_c + row * ld_c) + part)
                              : __ldg(reinterpret_cast<const uint4*>(k_pe + row * ld_pe) + (part - 64));
    reinterpret_cast<uint4*>(cache + slot * cache_ld)[part] = v;
}

// (6) Cross-rank argmax reduction over `argmax_full[world][B][2]`, writing `prev_ids[b]` and `draft_out` on GPU.
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

// (7) Gather B target_hidden rows (each bf16 [3072]) from per-request device pointers into contiguous [B, 3072].
extern "C" __global__ void __launch_bounds__(384) hy4_dfly_gather_hidden(
    const unsigned long long* __restrict__ src_ptrs, bf16* __restrict__ dst, int B) {
    const int b = blockIdx.x;
    const int t = threadIdx.x;
    if (b >= B || t >= V8) return;
    const uint4* src = reinterpret_cast<const uint4*>(src_ptrs[b]);
    reinterpret_cast<uint4*>(dst + static_cast<size_t>(b) * H)[t] = src[t];
}
