#pragma once

#include <hip/hip_runtime.h>

#include "decode/pages.hpp"
#include "ops/common.hpp"

// A lane round's plan in device memory: no stream state is a launch argument, so one launch or graph serves any.
struct PlanArgs {
    const int* pos;                   // per row: its position
    const int* slot;                  // per row: its slot
    const int* first;                 // per slot: its first row
    const int* count;                 // per slot: its rows
    const unsigned long long* desc;   // per slot: the address of its descriptor
    const unsigned long long* snaps;  // per layer: the round's conv and DeltaNet snapshots, a row each
    unsigned pages;                   // words of a descriptor before its page table
    unsigned pool;                    // pages a head of a layer's pool holds
};

// Slot descriptor: [0] cached positions, [1] last kept final row, two cache addresses a layer, then the page table.
__device__ __forceinline__ const unsigned long long* plan_desc(const PlanArgs& p, int slot) {
    return reinterpret_cast<const unsigned long long*>(p.desc[slot]);
}

__device__ __forceinline__ int plan_first(int layer) { return 2 + 2 * layer; }

__device__ __forceinline__ int plan_second(int layer) { return 3 + 2 * layer; }

__device__ __forceinline__ const unsigned* plan_pages(const PlanArgs& p, const unsigned long long* desc) {
    return reinterpret_cast<const unsigned*>(desc + p.pages);
}

// The d = 256 long-span score's pieces (plan.hip's tf_plan_score256_*, plan_rows.hip's kernels).
namespace {

// x from lane ^ M of its 16-lane row (M < 16): a DPP row_xmask move, all lanes active.
template <int M>
__device__ __forceinline__ float row_xor(float x) {
    return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x160 + M, 0xf, 0xf, false));
}

// wave_sum's folds, own value then partner's at xor 16, 8, 4, 2, 1, without LDS: every lane ends with the sum.
__device__ __forceinline__ float wave_sum_dpp(float x) {
    const int w = __builtin_bit_cast(int, x);
    x += __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(w, w, 0x76543210u, 0xfedcba98u, false, false));
    x += row_xor<8>(x);
    x += row_xor<4>(x);
    x += row_xor<2>(x);
    return x + row_xor<1>(x);
}

// Query heads a block takes together: they share a KV head, so its keys and values are read once for all of them.
constexpr int kMaxGroup = 16;

// Keys a wave scores on long spans (plan.hip's kScoreKeysShort on short ones): their loads go out together.
constexpr int kScoreKeys = 4;
// Steps of its waves' keys a long-span block walks at d = 256 (launches/plan.zig's score_steps).
constexpr int kScoreSteps = 8;
// load() of a 16-bit cache element as a global load: a flat load also counts on lgkmcnt, which LDS reads wait on.
__device__ __forceinline__ float cache_load(const void* p, int kind, long long i) {
    const uint16_t bits = ((const __attribute__((address_space(1))) uint16_t*)p)[i];
    return kind == 1 ? __half2float(__ushort_as_half(bits)) : bf16_to_float(bits);
}

// A wave's KEYS keys from key0 at d = 256 into k, lane l taking elements l + 32 i; keys past `visible` are not loaded.
template <int KEYS>
__device__ __forceinline__ void score_fetch(float (&k)[KEYS][8], const void* keys, int kind, const unsigned* pages,
                                            unsigned pool, int kv, int key0, int visible, int lane) {
#pragma unroll
    for (int j = 0; j < KEYS; ++j) {
        if (key0 + j >= visible) break;
        const long long base = page_at(pages, pool, kv, key0 + j, 256);
#pragma unroll
        for (int i = 0; i < 8; ++i) k[j][i] = cache_load(keys, kind, base + lane + i * 32);
    }
}

// Lane 0's KEYS scores of a head from `at`: one 16-byte store when `whole` (aligned rows, span a multiple of KEYS).
template <int KEYS>
__device__ __forceinline__ void score_store(float* at, const float (&sc)[KEYS], bool whole, int key0, int span) {
    if (whole && KEYS == 4) {
        *reinterpret_cast<float4*>(at) = make_float4(sc[0], sc[1 % KEYS], sc[2 % KEYS], sc[3 % KEYS]);
    } else {
        for (int j = 0; j < KEYS; ++j) if (key0 + j < span) at[j] = sc[j];
    }
}

// plan_score_body on long spans at d = 256 (tf_plan_score256_*): no element tests, kScoreSteps steps a block.
template <int KIND, int QKIND>
__device__ void plan_score256_body(const void* __restrict__ q, float* __restrict__ scores, int span, int heads,
                                   int kv_heads, int d, float scale, PlanArgs plan, int layer) {
    constexpr int KEYS = kScoreKeys, STEPS = kScoreSteps;
    extern __shared__ float qs[];  // the block's query heads, group * d
    const int batch = blockIdx.z, group = heads / gridDim.y, head0 = blockIdx.y * group;
    const long long q0 = (static_cast<long long>(batch) * heads + head0) * d;
    for (int i = threadIdx.x; i < group * d; i += blockDim.x) qs[i] = load(q, QKIND, q0 + i);  // kind 0: q fp32
    __syncthreads();
    const int visible = min(plan.pos[batch] + 1, span);
    const int lane = threadIdx.x & 31;
    const unsigned long long* desc = plan_desc(plan, plan.slot[batch]);
    float* __restrict__ rows = scores + (static_cast<long long>(batch) * heads + head0) * span;
    const void* keys = reinterpret_cast<const void*>(desc[plan_first(layer)]);
    const unsigned* pages = plan_pages(plan, desc);
    const unsigned pool = plan.pool;
    const int kv = head0 / (heads / kv_heads);
    const int stride = (blockDim.x >> 5) * KEYS;  // keys a step of the block
    const int first = blockIdx.x * STEPS * stride + (threadIdx.x >> 5) * KEYS;
    const bool whole = (span % KEYS) == 0 && (reinterpret_cast<unsigned long long>(rows) & 15) == 0;
    float kreg[KEYS][8] = {};
    score_fetch<KEYS>(kreg, keys, KIND, pages, pool, kv, first, visible, lane);
#pragma unroll 1
    for (int step = 0; step < STEPS; ++step) {
        const int key0 = first + step * stride;
        if (key0 >= span) break;  // the same for the whole wave
        float next[KEYS][8] = {};
        score_fetch<KEYS>(next, keys, KIND, pages, pool, kv, step + 1 < STEPS ? key0 + stride : span, visible, lane);
        if (key0 >= visible) {  // the wave's keys are all past the row's: -INFINITY wherever the span has them
            float none[KEYS];
            for (int j = 0; j < KEYS; ++j) none[j] = -INFINITY;
            for (int g = 0; lane == 0 && g < group; ++g) {
                score_store<KEYS>(rows + key0 + static_cast<long long>(g) * span, none, whole, key0, span);
            }
        } else {
#pragma unroll
            for (int g = 0; g < kMaxGroup; ++g) {
                if (g < group) {  // the same for the whole block, so every lane reaches the folds
                    float qv[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) qv[i] = qs[g * 256 + lane + i * 32];
                    float dot[KEYS];
#pragma unroll
                    for (int j = 0; j < KEYS; ++j) {
                        float partial = 0.f;
#pragma unroll
                        for (int i = 0; i < 8; ++i) partial += qv[i] * kreg[j][i];
                        dot[j] = wave_sum_dpp(partial);
                    }
                    if (lane == 0) {
                        float sc[KEYS];
#pragma unroll
                        for (int j = 0; j < KEYS; ++j) sc[j] = key0 + j < visible ? dot[j] * scale : -INFINITY;
                        score_store<KEYS>(rows + key0 + static_cast<long long>(g) * span, sc, whole, key0, span);
                    }
                }
            }
        }
        __builtin_memcpy(kreg, next, sizeof(kreg));
    }
}

}  // namespace
