// A round's multi-row slots scored and weighed from one read of their keys and values, the others alone (d 256).

#include "ops/common.hpp"
#include "decode/plan.hpp"

namespace {

// Rows of one slot a block scores (launches/plan.zig's score_rows): their queries share the block's LDS.
constexpr int kRows = 2;

// A row's place in its slot's list when that slot lists more than one row, else -1: the row is scored alone.
__device__ __forceinline__ int plan_shared_at(const PlanArgs& plan, int batch) {
    const int slot = plan.slot[batch], at = batch - plan.first[slot], count = plan.count[slot];
    return count > 1 && at >= 0 && at < count ? at : -1;
}

// plan_score256_body (plan.hpp) for a row that scores alone: its bits and its block's LDS are tf_plan_score256_*'s.
template <int KIND>
__device__ void plan_lone_score256_body(const float* __restrict__ q, float* __restrict__ scores, int span, int heads,
                                        int kv_heads, int d, float scale, PlanArgs plan, int layer) {
    if (plan_shared_at(plan, blockIdx.z) >= 0) return;  // tf_plan_rows_score256_* scores it
    plan_score256_body<KIND, 0>(q, scores, span, heads, kv_heads, d, scale, plan, layer);
}

// plan_score256_body on a chunk of a multi-row slot's rows: the keys read once, each row's dots in the tree's order.
template <int KIND>
__device__ void plan_rows_score256_body(const float* __restrict__ q, float* __restrict__ scores, int span, int heads,
                                        int kv_heads, int d, float scale, PlanArgs plan, int layer) {
    constexpr int KEYS = kScoreKeys, STEPS = kScoreSteps;
    extern __shared__ float qs[];  // the chunk's rows' query heads of the block, rows * group * d
    const int batch = blockIdx.z, slot = plan.slot[batch], at = plan_shared_at(plan, batch);
    if (at < 0 || at % kRows != 0) return;  // a row scored alone, or the chunk's first row's block scores it
    const int rows = min(min(kRows, plan.count[slot] - at), static_cast<int>(gridDim.z) - batch);
    const int group = heads / gridDim.y, head0 = blockIdx.y * group;
    for (int i = threadIdx.x; i < rows * group * 256; i += blockDim.x) {
        const int r = i / (group * 256);
        qs[i] = q[(static_cast<long long>(batch + r) * heads + head0) * 256 + (i - r * group * 256)];
    }
    __syncthreads();
    int most = 0;
    for (int r = 0; r < rows; ++r) most = max(most, min(plan.pos[batch + r] + 1, span));
    const int lane = threadIdx.x & 31;
    const unsigned long long* desc = plan_desc(plan, slot);
    const void* keys = reinterpret_cast<const void*>(desc[plan_first(layer)]);
    const unsigned* pages = plan_pages(plan, desc);
    const unsigned pool = plan.pool;
    const int kv = head0 / (heads / kv_heads);
    const int stride = (blockDim.x >> 5) * KEYS;  // keys a step of the block
    const int first = blockIdx.x * STEPS * stride + (threadIdx.x >> 5) * KEYS;
    float kreg[KEYS][8] = {};
    score_fetch<KEYS>(kreg, keys, KIND, pages, pool, kv, first, most, lane);
#pragma unroll 1
    for (int step = 0; step < STEPS; ++step) {
        const int key0 = first + step * stride;
        if (key0 >= span) break;  // the same for the whole wave
        float next[KEYS][8] = {};
        score_fetch<KEYS>(next, keys, KIND, pages, pool, kv, step + 1 < STEPS ? key0 + stride : span, most, lane);
#pragma unroll 1
        for (int r = 0; r < rows; ++r) {
            const int visible = min(plan.pos[batch + r] + 1, span);
            float* __restrict__ out = scores + (static_cast<long long>(batch + r) * heads + head0) * span;
            const bool whole = (span % KEYS) == 0 && (reinterpret_cast<unsigned long long>(out) & 15) == 0;
            if (key0 >= visible) {  // the wave's keys are all past the row's: -INFINITY wherever the span has them
                float none[KEYS];
                for (int j = 0; j < KEYS; ++j) none[j] = -INFINITY;
                for (int g = 0; lane == 0 && g < group; ++g) {
                    score_store<KEYS>(out + key0 + static_cast<long long>(g) * span, none, whole, key0, span);
                }
            } else {
#pragma unroll
                for (int g = 0; g < kMaxGroup; ++g) {
                    if (g < group) {  // the same for the whole block, so every lane reaches the folds
                        float qv[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) qv[i] = qs[(r * group + g) * 256 + lane + i * 32];
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
                            score_store<KEYS>(out + key0 + static_cast<long long>(g) * span, sc, whole, key0, span);
                        }
                    }
                }
            }
        }
        __builtin_memcpy(kreg, next, sizeof(kreg));
    }
}

// Rows of one slot (launches/plan.zig's apply_rows) and query heads at most a values block takes: 16 KiB of weights.
constexpr int kApplyRows = 4, kApplyHeads = 8;

// plan_apply_body (plan.hpp) for a row that applies alone: its bits and its block's LDS are tf_plan_apply_*'s.
template <int KIND>
__device__ void plan_lone_apply_body(const float* __restrict__ scores, const float* __restrict__ stats,
                                     float* __restrict__ partials, int span, int heads, int kv_heads, int d,
                                     PlanArgs plan, int layer) {
    if (plan_shared_at(plan, blockIdx.z) >= 0) return;  // tf_plan_rows_apply_* applies it
    plan_apply_body<KIND>(scores, stats, partials, span, heads, kv_heads, d, plan, layer);
}

// N values from tile key `at` on into each row's and head's sum, a key at a time; MASK: only keys before a row's stop.
template <int N, bool MASK>
__device__ __forceinline__ void rows_add(float (&acc)[kApplyRows][kApplyHeads], const float* weights,
                                         const float (&v)[N], int at, const int (&stop)[kApplyRows], int key, int rows,
                                         int group) {
#pragma unroll
    for (int r = 0; r < kApplyRows; ++r) {
        if (r < rows) {
#pragma unroll
            for (int u = 0; u < N; ++u) {
                if (!MASK || key + u < stop[r]) {  // a key past the row's own is left out, as the tree leaves it
#pragma unroll
                    for (int g = 0; g < kApplyHeads; ++g) {
                        if (g < group) acc[r][g] += weights[(r * kApplyHeads + g) * 128 + at + u] * v[u];
                    }
                }
            }
        }
    }
}

// plan_apply_body on a chunk of a multi-row slot's rows: the values read once, each row's sums in the tree's order.
template <int KIND>
__device__ __forceinline__ void plan_rows_apply_body(const float* __restrict__ scores, const float* __restrict__ stats,
                                                     float* __restrict__ partials, int span, int heads, int kv_heads,
                                                     int d, PlanArgs plan, int layer) {
    __shared__ float weights[kApplyRows * kApplyHeads * 128];  // the chunk's rows' weights, (row, head, tile key)
    const int batch = blockIdx.z, slot = plan.slot[batch], at = plan_shared_at(plan, batch);
    if (at < 0 || at % kApplyRows != 0) return;  // a row applied alone, or the chunk's first row's block applies it
    const int rows = min(min(kApplyRows, plan.count[slot] - at), static_cast<int>(gridDim.z) - batch);
    const int tile = blockIdx.x, group = heads / gridDim.y, head0 = blockIdx.y * group;
    const long long first = static_cast<long long>(batch) * heads + head0;
    const int lane = threadIdx.x;
    const int key0 = tile * 128;
    // a row's keys of the tile end at its stop, key0 when the tile is past them: the tree adds none and writes zero
    int stop[kApplyRows];
    int least = key0 + 128, most = key0;
#pragma unroll
    for (int r = 0; r < kApplyRows; ++r) {
        stop[r] = r < rows ? max(key0, min(key0 + 128, min(plan.pos[batch + r] + 1, span))) : key0;
        if (r < rows) least = min(least, stop[r]);
        most = max(most, stop[r]);
    }
    if (most == key0) {  // a tile past every row's keys adds zero
        if (lane < d) {
            for (int r = 0; r < rows; ++r) {
                for (int g = 0; g < group; ++g) {
                    partials[((first + r * heads + g) * gridDim.x + tile) * d + lane] = 0.f;
                }
            }
        }
        return;
    }
    for (int i = threadIdx.x; i < rows * group * 128; i += blockDim.x) {
        const int r = i / (group * 128), g = i / 128 % group, key = key0 + i % 128;
        const long long h = first + r * heads + g;
        if (key < min(plan.pos[batch + r] + 1, span)) {
            const float w = expf(scores[h * span + key] - stats[h * 2]) / stats[h * 2 + 1];
            weights[(r * kApplyHeads + g) * 128 + i % 128] = w;
        }
    }
    __syncthreads();
    if (lane >= d) return;
    const unsigned long long* desc = plan_desc(plan, slot);
    const void* values = reinterpret_cast<const void*>(desc[plan_second(layer)]);
    const unsigned* table = plan_pages(plan, desc);
    const int kv = head0 / (heads / kv_heads);
    float acc[kApplyRows][kApplyHeads] = {};
    // plan_apply_body's walk, a page at a time and eight values ahead, up to the chunk's furthest key
    for (int key = key0; key < most;) {
        const int edge = min((key / kPageTokens + 1) * kPageTokens, most);
        const long long slot0 = page_row(table[key / kPageTokens], plan.pool, kv, 0, d) + lane;
        for (; key + 8 <= edge; key += 8) {
            float ahead[8];
#pragma unroll
            for (int u = 0; u < 8; ++u)
                ahead[u] = cache_load(values, KIND, slot0 + static_cast<long long>((key + u) % kPageTokens) * d);
            if (key + 8 <= least) {  // every row of the chunk adds all eight
                rows_add<8, false>(acc, weights, ahead, key - key0, stop, key, rows, group);
            } else {
                rows_add<8, true>(acc, weights, ahead, key - key0, stop, key, rows, group);
            }
        }
        for (; key < edge; ++key) {
            const float value[1] = {cache_load(values, KIND, slot0 + static_cast<long long>(key % kPageTokens) * d)};
            rows_add<1, true>(acc, weights, value, key - key0, stop, key, rows, group);
        }
    }
#pragma unroll
    for (int r = 0; r < kApplyRows; ++r) {
        if (r < rows) {
#pragma unroll
            for (int g = 0; g < kApplyHeads; ++g) {
                if (g < group) partials[((first + r * heads + g) * gridDim.x + tile) * d + lane] = acc[r][g];
            }
        }
    }
}

}  // namespace

extern "C" {

// tf_plan_score256_*'s arguments and grid: _lone_ scores the rows that score alone, _rows_ the multi-row slots' chunks.
#define TF_PLAN_ROWS_SCORE256(Name, Body, KIND) __global__ void __launch_bounds__(128) Name(const float* q,        \
    float* scores, int span, int heads, int kv_heads, int d, float scale, PlanArgs plan, int layer) {              \
    Body<KIND>(q, scores, span, heads, kv_heads, d, scale, plan, layer); }
TF_PLAN_ROWS_SCORE256(tf_plan_lone_score256_f16, plan_lone_score256_body, 1)
TF_PLAN_ROWS_SCORE256(tf_plan_lone_score256_bf16, plan_lone_score256_body, 2)
TF_PLAN_ROWS_SCORE256(tf_plan_rows_score256_f16, plan_rows_score256_body, 1)
TF_PLAN_ROWS_SCORE256(tf_plan_rows_score256_bf16, plan_rows_score256_body, 2)

// tf_plan_apply_*'s arguments and grid (at most kApplyHeads heads a block for _rows_): _lone_ the rows applied alone.
#define TF_PLAN_ROWS_APPLY(Name, Body, KIND) __global__ void __launch_bounds__(256) Name(const float* scores,         \
    const float* stats, float* partials, int span, int heads, int kv_heads, int d, PlanArgs plan, int layer) {     \
    Body<KIND>(scores, stats, partials, span, heads, kv_heads, d, plan, layer); }
TF_PLAN_ROWS_APPLY(tf_plan_lone_apply_f16, plan_lone_apply_body, 1)
TF_PLAN_ROWS_APPLY(tf_plan_lone_apply_bf16, plan_lone_apply_body, 2)
TF_PLAN_ROWS_APPLY(tf_plan_rows_apply_f16, plan_rows_apply_body, 1)
TF_PLAN_ROWS_APPLY(tf_plan_rows_apply_bf16, plan_rows_apply_body, 2)

}  // extern "C"
