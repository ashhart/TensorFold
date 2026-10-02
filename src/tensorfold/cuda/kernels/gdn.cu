// Trees, chains and replays run one fp32 step, each op rounded alone (--fmad=false), so windows never change bits.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int DK = 128;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int m = 16; m; m >>= 1) x += __shfl_xor_sync(0xffffffffu, x, m);
    return x;
}

// N warp sums on warp_sum's own pairs and order (same bits), shipping half a level; lane L gets value L >> (5 - log2 N)
template <int N>
__device__ __forceinline__ float spread_sum(float (&v)[N], int lane) {
#pragma unroll
    for (int level = 0; level < 5; ++level) {
        const int m = 16 >> level, h = N >> (level + 1);
        if (h >= 1) {
            const bool up = lane & m;
#pragma unroll
            for (int i = 0; i < h; ++i) {
                const float give = up ? v[i] : v[h + i], keep = up ? v[h + i] : v[i];
                v[i] = keep + __shfl_xor_sync(0xffffffffu, give, m);
            }
        } else {
            v[0] += __shfl_xor_sync(0xffffffffu, v[0], m);
        }
    }
    return v[0];
}

template <int N>
constexpr int log2c() { return N <= 1 ? 0 : 1 + log2c<N / 2>(); }

// bf16 to fp32 is exact: the low half shifted up, the high half masked (one op an element)
__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

__device__ __forceinline__ void load4(const __nv_bfloat16* p, float (&x)[4]) {
    const uint2 w = *reinterpret_cast<const uint2*>(p);
    x[0] = bf16_lo(w.x); x[1] = bf16_hi(w.x); x[2] = bf16_lo(w.y); x[3] = bf16_hi(w.y);
}

__device__ __forceinline__ void load4(const float* p, float (&x)[4]) {
    const float4 w = *reinterpret_cast<const float4*>(p);
    x[0] = w.x; x[1] = w.y; x[2] = w.z; x[3] = w.w;
}

// One warp's input pointers at node 0 (its lane's q, k, its first value row, its head's gates) and strides a node
template <typename QK>
struct Feed {
    const QK* q;
    const QK* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    int qk, vs, gs;
};

// A node's inputs for one warp: its key head's q and k (this lane's 4 elements), R values, the gates.
template <typename QK, int R>
struct Inputs {
    float q[4], k[4], v[R], g, beta;

    __device__ __forceinline__ void load(const QK* qp, const QK* kp, const __nv_bfloat16* vp, const float* gp,
                                         const float* bp, int node, int key_head, int head, int value0, int hk,
                                         int hv, int dv, int lane, bool with_q) {
        const size_t key = (static_cast<size_t>(node) * hk + key_head) * DK + lane * 4;
        if (with_q) load4(qp + key, q);
        load4(kp + key, k);
        const size_t val = (static_cast<size_t>(node) * hv + head) * dv + value0;
#pragma unroll
        for (int r = 0; r < R; ++r) v[r] = value0 + r < dv ? __bfloat162float(vp[val + r]) : 0.0f;
        g = gp[node * hv + head];
        beta = bp[node * hv + head];
    }

    // The same values through a Feed's lane pointers; ``vec``: the R values in one aligned load.
    __device__ __forceinline__ void fetch(const Feed<QK>& f, int node, bool with_q, bool vec, int value0, int dv) {
        const long long key = static_cast<long long>(node) * f.qk;
        if (with_q) load4(f.q + key, q);
        load4(f.k + key, k);
        const __nv_bfloat16* vp = f.v + static_cast<long long>(node) * f.vs;
        if (vec) {
#pragma unroll
            for (int i = 0; i < R; i += 8 < R ? 8 : R) {
                if constexpr (R == 2) {
                    const uint32_t w = *reinterpret_cast<const uint32_t*>(vp);
                    v[0] = bf16_lo(w); v[1] = bf16_hi(w);
                } else if constexpr (R == 4) {
                    const uint2 w = *reinterpret_cast<const uint2*>(vp);
                    v[0] = bf16_lo(w.x); v[1] = bf16_hi(w.x); v[2] = bf16_lo(w.y); v[3] = bf16_hi(w.y);
                } else {
                    const uint4 w = *reinterpret_cast<const uint4*>(vp + i);
                    v[i] = bf16_lo(w.x); v[i + 1] = bf16_hi(w.x); v[i + 2] = bf16_lo(w.y); v[i + 3] = bf16_hi(w.y);
                    v[i + 4] = bf16_lo(w.z); v[i + 5] = bf16_hi(w.z); v[i + 6] = bf16_lo(w.w); v[i + 7] = bf16_hi(w.w);
                }
            }
        } else {
#pragma unroll
            for (int r = 0; r < R; ++r) v[r] = value0 + r < dv ? __bfloat162float(vp[r]) : 0.0f;
        }
        g = f.g[node * f.gs];
        beta = f.beta[node * f.gs];
    }
};

// One step for one value row: ``s`` holds the parent's state on entry and the node's on exit.
__device__ __forceinline__ void step(float (&s)[4], const float (&k)[4], float v, float g, float beta) {
    float mem = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        s[i] = s[i] * g;
        mem = mem + s[i] * k[i];
    }
    const float delta = (v - warp_sum(mem)) * beta;
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i] = s[i] + k[i] * delta;
}

// The previous round's accepted rows not yet folded into each stream's committed state (k null: none).
template <typename QK>
struct Pending {
    const QK* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const int* rows;
    int row_stride;
    const int* counts;
    int count_stride;
};

// Entries (node, source, dest): source -1 committed, -2 the node just run, else a slot; dest -1 none, else a slot.
template <typename QK, int SLOTS, int R, int WARPS, bool CHAIN>
__global__ void __launch_bounds__(32 * WARPS) tree_kernel(
        const QK* __restrict__ q, const QK* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const float* __restrict__ g, const float* __restrict__ beta, const float* __restrict__ state,
        const long long* __restrict__ table, const int* __restrict__ starts, const int* __restrict__ plan,
        int nodes, __nv_bfloat16* __restrict__ y, int hk, int hv, int dv, Pending<QK> pending, float* final_state,
        const long long* __restrict__ final_table, bool vec) {
    extern __shared__ float4 smem[];
    int* order = reinterpret_cast<int*>(smem + WARPS * SLOTS * R * 32);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head = blockIdx.y, stream = blockIdx.z;
    const int value0 = (blockIdx.x * WARPS + warp) * R;
    const int key_head = head / (hv / hk);
    const int begin = starts ? starts[stream] : 0, end = starts ? starts[stream + 1] : nodes;
    if (!CHAIN) {
        for (int i = threadIdx.x; i < 3 * (end - begin); i += 32 * WARPS) order[i] = plan[3 * begin + i];
        __syncthreads();
    }
    if (value0 >= dv) return;
    const float* s0 = table ? reinterpret_cast<const float*>(table[stream]) : state;
    float cur[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = value0 + r < dv ? value0 + r : value0;
        load4(s0 + (static_cast<size_t>(head) * dv + row) * DK + lane * 4, cur[r]);
    }
    if (pending.k != nullptr) {
        // the last commit's rows as replay steps them; each thread writes back only the elements it loaded
        const int n = pending.counts[stream * pending.count_stride];
        const int* rows = pending.rows + stream * pending.row_stride;
        Inputs<QK, R> nx;
        if (n > 0) nx.load(nullptr, pending.k, pending.v, pending.g, pending.beta, rows[0], key_head, head, value0, hk, hv, dv,
                           lane, false);
        for (int j = 0; j < n; ++j) {
            const Inputs<QK, R> in = nx;
            if (j + 1 < n)
                nx.load(nullptr, pending.k, pending.v, pending.g, pending.beta, rows[j + 1], key_head, head, value0, hk, hv, dv,
                        lane, false);
#pragma unroll
            for (int r = 0; r < R; ++r) step(cur[r], in.k, in.v[r], in.g, in.beta);
        }
        if (n > 0) {
            float* committed = const_cast<float*>(s0);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (value0 + r >= dv) break;
                *reinterpret_cast<float4*>(committed + (static_cast<size_t>(head) * dv + value0 + r) * DK + lane * 4) =
                    make_float4(cur[r][0], cur[r][1], cur[r][2], cur[r][3]);
            }
        }
    }
    float4* slots = smem + warp * SLOTS * R * 32;
    const int count = end - begin;
    const Feed<QK> f{q + key_head * DK + lane * 4, k + key_head * DK + lane * 4, v + head * dv + value0, g + head,
                     beta + head, hk * DK, hv * dv, hv};
    // a node's dots and the node before's read-outs ride one spread sum: lane u << SH holds value u (dots, read-outs)
    constexpr int SH = 5 - log2c<2 * R>();
    const int yrow = value0 + (lane >> SH) - R;
    const bool ylane = (lane & ((1 << SH) - 1)) == 0 && (lane >> SH) >= R && yrow < dv;
    float pend[R];
#pragma unroll
    for (int r = 0; r < R; ++r) pend[r] = 0.0f;
    int pend_node = -1;
    // one node: ``in`` holds its inputs, ``nx`` takes the next node's (two buffers in turn, so nothing is copied)
    auto visit = [&](const Inputs<QK, R>& in, Inputs<QK, R>& nx, int e) {
        const int node = CHAIN ? begin + e : order[3 * e];
        const int source = CHAIN ? (e == 0 ? -1 : -2) : order[3 * e + 1], dest = CHAIN ? -1 : order[3 * e + 2];
        if (e + 1 < count) nx.fetch(f, CHAIN ? begin + e + 1 : order[3 * e + 3], true, vec, value0, dv);
        float s[R][4], mem[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (source >= 0 && SLOTS > 0) {
                const float4 t = slots[(source * R + r) * 32 + lane];
                s[r][0] = t.x; s[r][1] = t.y; s[r][2] = t.z; s[r][3] = t.w;
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) s[r][i] = cur[r][i];
            }
            mem[r] = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[r][i] = s[r][i] * in.g;
                mem[r] = mem[r] + s[r][i] * in.k[i];
            }
        }
        float both[2 * R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            both[r] = mem[r];
            both[R + r] = pend[r];
        }
        const float tot = spread_sum<2 * R>(both, lane);
#pragma unroll
        for (int r = 0; r < R; ++r) mem[r] = __shfl_sync(0xffffffffu, tot, r << SH);
        if (pend_node >= 0 && ylane)
            y[static_cast<long long>(pend_node) * f.vs + head * dv + yrow] = __float2bfloat16_rn(tot);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float delta = (in.v[r] - mem[r]) * in.beta;
            pend[r] = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[r][i] = s[r][i] + in.k[i] * delta;
                pend[r] = pend[r] + s[r][i] * in.q[i];
            }
            if (SLOTS > 0 && dest >= 0)
                slots[(dest * R + r) * 32 + lane] = make_float4(s[r][0], s[r][1], s[r][2], s[r][3]);
#pragma unroll
            for (int i = 0; i < 4; ++i) cur[r][i] = s[r][i];
        }
        pend_node = node;
    };
    Inputs<QK, R> a, b;
    if (count > 0) a.fetch(f, CHAIN ? begin : order[0], true, vec, value0, dv);
#pragma unroll 1
    for (int e = 0; e < count; e += 2) {
        visit(a, b, e);
        if (e + 1 < count) visit(b, a, e + 1);
    }
    if (pend_node >= 0) {
        constexpr int SR = 5 - log2c<R>();
        const float tot = spread_sum<R>(pend, lane);
        if ((lane & ((1 << SR) - 1)) == 0 && value0 + (lane >> SR) < dv)
            y[(static_cast<size_t>(pend_node) * hv + head) * dv + value0 + (lane >> SR)] = __float2bfloat16_rn(tot);
    }
    float* last = final_table ? reinterpret_cast<float*>(final_table[stream]) : final_state;
    if (CHAIN && last != nullptr && count > 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (value0 + r >= dv) break;
            *reinterpret_cast<float4*>(last + (static_cast<size_t>(head) * dv + value0 + r) * DK + lane * 4) =
                make_float4(cur[r][0], cur[r][1], cur[r][2], cur[r][3]);
        }
    }
}

// Commit: each (stream, layer) replays its rows into out[stream][layer], or over its state when out is null.
template <typename QK, int R, int WARPS>
__global__ void __launch_bounds__(32 * WARPS) replay_kernel(
        const long long* __restrict__ table, int layers, const int* __restrict__ rows, int row_stride,
        const int* __restrict__ counts, int count_stride, float* __restrict__ out, int hk, int hv, int dv) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head = blockIdx.y, stream = blockIdx.z / layers, layer = blockIdx.z % layers;
    const int value0 = (blockIdx.x * WARPS + warp) * R;
    if (value0 >= dv) return;
    const int key_head = head / (hv / hk);
    const auto* k = reinterpret_cast<const QK*>(table[4 * layer]);
    const auto* v = reinterpret_cast<const __nv_bfloat16*>(table[4 * layer + 1]);
    const auto* g = reinterpret_cast<const float*>(table[4 * layer + 2]);
    const auto* beta = reinterpret_cast<const float*>(table[4 * layer + 3]);
    const auto* s0 = reinterpret_cast<const float*>(table[4 * layers + stream * layers + layer]);
    // in place (no out): each thread stores exactly the elements it loaded, so blocks never overlap
    float* dst = out ? out + (static_cast<size_t>(stream) * layers + layer) * hv * dv * DK : const_cast<float*>(s0);
    float s[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = value0 + r < dv ? value0 + r : value0;
        load4(s0 + (static_cast<size_t>(head) * dv + row) * DK + lane * 4, s[r]);
    }
    const int* path = rows + stream * row_stride;
    const int count = counts[stream * count_stride];
    Inputs<QK, R> next;
    if (count > 0) next.load(nullptr, k, v, g, beta, path[0], key_head, head, value0, hk, hv, dv, lane, false);
    for (int j = 0; j < count; ++j) {
        const Inputs<QK, R> in = next;
        if (j + 1 < count)
            next.load(nullptr, k, v, g, beta, path[j + 1], key_head, head, value0, hk, hv, dv, lane, false);
#pragma unroll
        for (int r = 0; r < R; ++r) step(s[r], in.k, in.v[r], in.g, in.beta);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (value0 + r >= dv) break;
        *reinterpret_cast<float4*>(dst + (static_cast<size_t>(head) * dv + value0 + r) * DK + lane * 4) =
            make_float4(s[r][0], s[r][1], s[r][2], s[r][3]);
    }
}

template <typename QK, int SLOTS, int R, int WARPS, bool CHAIN>
void launch_tree(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                 const at::Tensor& beta, const float* state, const long long* table, const int* starts,
                 const at::Tensor& plan, int nodes, int streams, int max_rows, at::Tensor& y, Pending<QK> pend,
                 float* final_state, const long long* final_table) {
    const int hk = q.size(1), hv = v.size(1), dv = v.size(2);
    const dim3 grid((dv + R * WARPS - 1) / (R * WARPS), hv, streams);
    const size_t shared = sizeof(float4) * WARPS * SLOTS * R * 32 + (CHAIN ? 0 : sizeof(int) * 3 * max_rows);
    auto kernel = tree_kernel<QK, SLOTS, R, WARPS, CHAIN>;
    if (shared > 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    // a warp's R values in one load when every row group starts aligned
    const bool vec = dv % R == 0 && reinterpret_cast<uintptr_t>(v.data_ptr()) % (2 * (R < 8 ? R : 8)) == 0;
    kernel<<<grid, 32 * WARPS, shared, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const QK*>(q.data_ptr()), reinterpret_cast<const QK*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), g.data_ptr<float>(), beta.data_ptr<float>(),
        state, table, starts, plan.data_ptr<int>(), nodes,
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), hk, hv, dv, pend, final_state, final_table, vec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename QK>
Pending<QK> pending_of(const at::Tensor& pk, const at::Tensor& pv, const at::Tensor& pg, const at::Tensor& pb,
                       const at::Tensor& rows, const at::Tensor& counts) {
    if (!pk.defined()) return Pending<QK>{nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr, 0};
    return Pending<QK>{reinterpret_cast<const QK*>(pk.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(pv.data_ptr()),
                       pg.data_ptr<float>(), pb.data_ptr<float>(), rows.data_ptr<int>(), static_cast<int>(rows.stride(0)),
                       counts.data_ptr<int>(), static_cast<int>(counts.stride(0))};
}

template <typename QK>
void dispatch_tree(int slots, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                   const at::Tensor& beta, const float* state, const long long* table, const int* starts,
                   const at::Tensor& plan, int nodes, int streams, int max_rows, at::Tensor& y, Pending<QK> pend,
                   float* final_state, const long long* final_table) {
    // no slots: every window is a chain in row order, no plan to read; more slots, fewer rows a warp to cap memory
#define TREE(S, R, W, C) launch_tree<QK, S, R, W, C>(q, k, v, g, beta, state, table, starts, plan, nodes, streams, \
                                                     max_rows, y, pend, final_state, final_table)
#define TREE_(S, R, W) TREE(S, R, W, false)
    // many-SM sm_120: several streams take 8 rows a warp (fewer instructions a row), one stream 4 (warps for every SM)
    const auto* prop = at::cuda::getCurrentDeviceProperties();
    const bool wide = streams >= 2 && prop->major == 12 && prop->minor == 0 && prop->multiProcessorCount >= 96;
    if (slots == 0) TREE(0, 8, 4, true);
    else if (wide && slots <= 1) TREE_(1, 8, 4);
    else if (wide && slots <= 2) TREE_(2, 8, 2);
    else if (slots <= 2) TREE_(2, 4, 4);
    else if (slots <= 4) TREE_(4, 2, 4);
    else if (slots <= 8) TREE_(8, 2, 4);
    else if (slots <= 16) TREE_(16, 2, 2);
    else TREE_(32, 2, 1);
#undef TREE_
#undef TREE
}

} // namespace

void gdn_tree_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                   const at::Tensor& beta, const at::Tensor& state, const at::Tensor& table, const at::Tensor& starts,
                   const at::Tensor& plan, int slots, int streams, int max_rows, at::Tensor& y, const at::Tensor& pk,
                   const at::Tensor& pv, const at::Tensor& pg, const at::Tensor& pb, const at::Tensor& prows,
                   const at::Tensor& pcounts, const at::Tensor& fstate, const at::Tensor& ftable) {
    const float* s = state.defined() ? state.data_ptr<float>() : nullptr;
    const long long* t = table.defined() ? reinterpret_cast<const long long*>(table.data_ptr<int64_t>()) : nullptr;
    const int* st = starts.defined() ? starts.data_ptr<int>() : nullptr;
    const int nodes = q.size(0);
    float* fs = fstate.defined() ? fstate.data_ptr<float>() : nullptr;
    const long long* ft = ftable.defined() ? reinterpret_cast<const long long*>(ftable.data_ptr<int64_t>()) : nullptr;
    if (q.scalar_type() == at::kFloat)
        dispatch_tree<float>(slots, q, k, v, g, beta, s, t, st, plan, nodes, streams, max_rows, y,
                             pending_of<float>(pk, pv, pg, pb, prows, pcounts), fs, ft);
    else
        dispatch_tree<__nv_bfloat16>(slots, q, k, v, g, beta, s, t, st, plan, nodes, streams, max_rows, y,
                                     pending_of<__nv_bfloat16>(pk, pv, pg, pb, prows, pcounts), fs, ft);
}

void gdn_replay_cuda(const at::Tensor& table, int layers, int streams, const at::Tensor& rows,
                     const at::Tensor& counts, const at::Tensor& out, int hk, int hv, int dv, bool fp32_keys) {
    constexpr int R = 8, WARPS = 4;
    const dim3 grid((dv + R * WARPS - 1) / (R * WARPS), hv, layers * streams);
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* t = reinterpret_cast<const long long*>(table.data_ptr<int64_t>());
    const int rs = static_cast<int>(rows.stride(0)), cs = static_cast<int>(counts.stride(0));
    float* dst = out.defined() ? out.data_ptr<float>() : nullptr;
    if (fp32_keys)
        replay_kernel<float, R, WARPS><<<grid, 32 * WARPS, 0, stream>>>(
            t, layers, rows.data_ptr<int>(), rs, counts.data_ptr<int>(), cs, dst, hk, hv, dv);
    else
        replay_kernel<__nv_bfloat16, R, WARPS><<<grid, 32 * WARPS, 0, stream>>>(
            t, layers, rows.data_ptr<int>(), rs, counts.data_ptr<int>(), cs, dst, hk, hv, dv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
