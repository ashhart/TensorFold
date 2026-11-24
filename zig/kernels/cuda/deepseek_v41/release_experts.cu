// DeepSeek V4.1 release experts (ours): grouping, MXFP4 gate/up and down, the shared SwiGLU round trip, the combine.
#include "release_moe_tile.cuh"

// One CTA: count each slot's pairs, scan, then place pairs in (slot, row, k) order a warp at a time (deterministic).
extern "C" __global__ void __launch_bounds__(1024) tf_dsv41_rel_group(RelGroup p) {
    extern __shared__ int cursor[];
    const int pairs = p.m * p.topk, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    auto slot = [&](int i) {
        const int id = p.ids[i];
        return (id >= 0 && id < p.nids) ? p.slot_of[id] : -1;
    };
    for (int s = threadIdx.x; s < p.nslots; s += 1024) cursor[s] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < pairs; i += 1024) {
        const int s = slot(i);
        if (s >= 0) atomicAdd(&cursor[s], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int at = 0, used = 0;
        for (int s = 0; s < p.nslots; ++s) {
            const int c = cursor[s];
            p.offs[s] = at;
            cursor[s] = at;
            if (c > 0) p.active[used++] = s;
            at += c;
        }
        p.offs[p.nslots] = at;
        *p.nactive = used;
    }
    __syncthreads();
    for (int base = 0; base < pairs; base += 1024) {
        const int i = base + threadIdx.x;
        const int s = i < pairs ? slot(i) : -2;
        for (int w = 0; w < 32; ++w) {
            if (warp == w) {
                const unsigned same = __match_any_sync(0xffffffffu, s);
                const int leader = __ffs(same) - 1;
                const int rank = __popc(same & ((1u << lane) - 1u));
                int start = 0;
                if (lane == leader && s >= 0) {
                    start = cursor[s];
                    cursor[s] = start + __popc(same);
                }
                start = __shfl_sync(0xffffffffu, start, leader);
                if (s >= 0) {
                    p.pair_tok[start + rank] = i;
                    p.pair_of[i] = start + rank;
                } else if (s == -1) {
                    p.pair_of[i] = -1;
                }
            }
            __syncthreads();
        }
    }
}

// Staged CTA tiles over (active slot, columns): 16 pairs by 32 columns (decode) or 64 by 64 (prompt); the same steps.
extern "C" __global__ void __launch_bounds__(32) tf_dsv41_rel_gateup(RelGateUp p) {
    extern __shared__ __align__(16) uint8_t smem[];
    rel::gateup_ctas<16, 32, 4>(p, smem, threadIdx.x, blockIdx.x, gridDim.x);
}

extern "C" __global__ void __launch_bounds__(128) tf_dsv41_rel_gateup_prompt(RelGateUp p) {
    extern __shared__ __align__(16) uint8_t smem[];
    rel::gateup_ctas<64, 64, 4>(p, smem, threadIdx.x, blockIdx.x, gridDim.x);
}

extern "C" __global__ void __launch_bounds__(32) tf_dsv41_rel_down(RelDown p) {
    extern __shared__ __align__(16) uint8_t smem[];
    rel::down_ctas<16, 32, 4>(p, smem, threadIdx.x, blockIdx.x, gridDim.x);
}

extern "C" __global__ void __launch_bounds__(128) tf_dsv41_rel_down_prompt(RelDown p) {
    extern __shared__ __align__(16) uint8_t smem[];
    rel::down_ctas<64, 64, 4>(p, smem, threadIdx.x, blockIdx.x, gridDim.x);
}

// Warp tiles over (active slot, 32 columns) straight from global memory: the stage-free form, bit-equal.
extern "C" __global__ void __launch_bounds__(128) tf_dsv41_rel_gateup_warp(RelGateUp p) {
    const rel::Lane l = rel::lane_of(threadIdx.x & 31);
    rel::slot_tiles(p.offs, p.active, p.nactive, p.n, blockIdx.x * 4 + (threadIdx.x >> 5), gridDim.x * 4,
                    [&](int slot, int first, int npairs, int col0) { rel::gateup_tile(p, slot, first, npairs, col0, l); });
}

extern "C" __global__ void __launch_bounds__(128) tf_dsv41_rel_down_warp(RelDown p) {
    const rel::Lane l = rel::lane_of(threadIdx.x & 31);
    rel::slot_tiles(p.offs, p.active, p.nactive, p.n, blockIdx.x * 4 + (threadIdx.x >> 5), gridDim.x * 4,
                    [&](int slot, int first, int npairs, int col0) { rel::down_tile(p, slot, first, npairs, col0, l); });
}

// part [pairs, n] bf16 grouped partials, plus the shared expert's [m, n] (optional), into y at a row stride.
struct RelCombine {
    const __nv_bfloat16* part;
    const int* pair_of;
    const __nv_bfloat16* shared;
    long long shared_ld;
    void* y;
    long long y_ld;
    long long y_col;
    int m;
    int n;
    int topk;
    int f32;
};

extern "C" __global__ void __launch_bounds__(256) tf_dsv41_rel_combine(RelCombine p) {
    const long long mn = (long long)p.m * p.n;
    for (long long i = (long long)blockIdx.x * 256 + threadIdx.x; i < mn; i += (long long)gridDim.x * 256) {
        const long long row = i / p.n;
        const int col = (int)(i % p.n);
        const float r = rel::combine_row(p.part, p.pair_of, p.shared, p.shared_ld, p.n, p.topk, row, col);
        const long long at = row * p.y_ld + p.y_col + col;
        if (p.f32) {
            static_cast<float*>(p.y)[at] = r;
        } else {
            static_cast<__nv_bfloat16*>(p.y)[at] = __float2bfloat16_rn(r);
        }
    }
}

// The shared expert's glue: bf16 gate and up rows (views) through the routed epilogue without a route weight.
struct RelSwiglu {
    const __nv_bfloat16* gate;
    long long gate_ld;
    const __nv_bfloat16* up;
    long long up_ld;
    uint8_t* codes;
    uint8_t* scales;
    __nv_bfloat16* v;  // optional bf16 v (the contract's)
    int m;
    int n;
};

extern "C" __global__ void __launch_bounds__(256) tf_dsv41_rel_swiglu_actq(RelSwiglu p) {
    const int lane = threadIdx.x & 31, per = p.n >> 5;
    for (long long blk = (long long)blockIdx.x * 8 + (threadIdx.x >> 5); blk < (long long)p.m * per;
         blk += (long long)gridDim.x * 8) {
        const long long row = blk / per;
        const int col = (int)(blk % per) * 32 + lane;
        const float v = rel::swiglu(__bfloat162float(p.gate[row * p.gate_ld + col]),
                                    __bfloat162float(p.up[row * p.up_ld + col]), 1.0f, false);
        const long long at = row * p.n + (col - lane);
        if (p.v) p.v[at + lane] = __float2bfloat16_rn(v);
        rel::actq32(v, lane, p.codes + at, p.scales + row * per + (col >> 5), nullptr);
    }
}
