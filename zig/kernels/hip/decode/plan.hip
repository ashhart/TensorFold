// A lane round's attention walk, cache writes and keep, read from the round's plan on the device (plan.hpp).

#include "ops/common.hpp"
#include "decode/plan.hpp"

namespace {

__device__ __forceinline__ float wave_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x += __shfl_xor(x, mask, 32);
    return __shfl(x, 0, 32);
}

// Row r's keys (already rotated, fp32) and values (the activation kind) into its slot's pages at its position.
__device__ void plan_kv_write_body(const float* keys, const void* values, int kind, int rows, int kv_heads, int d,
                                     PlanArgs plan, int layer) {
    const long long i = gid();
    if (i >= static_cast<long long>(rows) * kv_heads * d) return;
    const int j = static_cast<int>(i % d);
    const int h = static_cast<int>((i / d) % kv_heads);
    const int r = static_cast<int>(i / (static_cast<long long>(d) * kv_heads));
    const unsigned long long* desc = plan_desc(plan, plan.slot[r]);
    const long long to = page_at(plan_pages(plan, desc), plan.pool, h, plan.pos[r], d) + j;
    store(reinterpret_cast<void*>(desc[plan_first(layer)]), kind, to, keys[i]);
    static_cast<uint16_t*>(reinterpret_cast<void*>(desc[plan_second(layer)]))[to] = static_cast<const uint16_t*>(values)[i];
}

// Query heads a block takes together: they share a KV head, so its keys and values are read once for all of them.
constexpr int kMaxGroup = 16;

// score_keys over a plan's rows: a warp scores a key for each of the block's heads (one KV head's), in its order.
template <int KIND>
__device__ void plan_score_body(const float* __restrict__ q, float* __restrict__ scores, int span, int heads,
                                  int kv_heads, int d, float scale, PlanArgs plan, int layer) {
    extern __shared__ float qs[];  // the block's query heads, group * d
    const int batch = blockIdx.z;
    const int group = heads / gridDim.y;
    const int head0 = blockIdx.y * group;
    const float* qrows = q + (static_cast<long long>(batch) * heads + head0) * d;
    for (int i = threadIdx.x; i < group * d; i += blockDim.x) qs[i] = qrows[i];
    __syncthreads();
    const int visible = min(plan.pos[batch] + 1, span);
    const int lane = threadIdx.x & 31;
    const int key = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    float kreg[8] = {};
    if (key < visible) {
        const unsigned long long* desc = plan_desc(plan, plan.slot[batch]);
        const void* keys = reinterpret_cast<const void*>(desc[plan_first(layer)]);
        const long long base = page_at(plan_pages(plan, desc), plan.pool, head0 / (heads / kv_heads), key, d);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int idx = lane + i * 32;
            if (idx < d) kreg[i] = load(keys, KIND, base + idx);
        }
    }
#pragma unroll
    for (int g = 0; g < kMaxGroup; ++g) {
        if (g < group) {  // the same for the whole block, so every lane reaches wave_sum
            float partial = 0.f;
            if (key < visible) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int idx = lane + i * 32;
                    if (idx < d) partial += qs[g * d + idx] * kreg[i];
                }
            }
            const float dot = wave_sum(partial);
            if (lane == 0 && key < span) {
                const float score = key < visible ? dot * scale : -INFINITY;
                scores[(static_cast<long long>(batch) * heads + head0 + g) * span + key] = score;
            }
        }
    }
}

// apply_values over a plan's rows: a 128-key tile of the block's heads, weights worked once a block, in its order.
template <int KIND>
__device__ void plan_apply_body(const float* __restrict__ scores, const float* __restrict__ stats,
                                  float* __restrict__ partials, int span, int heads, int kv_heads, int d,
                                  PlanArgs plan, int layer) {
    __shared__ float weights[kMaxGroup * 128];
    const int batch = blockIdx.z;
    const int visible = min(plan.pos[batch] + 1, span);
    const int tile = blockIdx.x;
    const int group = heads / gridDim.y;
    const int head0 = blockIdx.y * group;
    const long long first = static_cast<long long>(batch) * heads + head0;
    const int lane = threadIdx.x;
    const int key0 = tile * 128;
    if (key0 >= visible) {  // a tile past the keys in use adds zero
        if (lane < d) {
            for (int g = 0; g < group; ++g) partials[((first + g) * gridDim.x + tile) * d + lane] = 0.f;
        }
        return;
    }
    const int stop = min(key0 + 128, visible);
    for (int i = threadIdx.x; i < group * 128; i += blockDim.x) {
        const long long h = first + i / 128;
        const int key = key0 + i % 128;
        if (key < stop) weights[i] = expf(scores[h * span + key] - stats[h * 2]) / stats[h * 2 + 1];
    }
    __syncthreads();
    if (lane >= d) return;
    const unsigned long long* desc = plan_desc(plan, plan.slot[batch]);
    const void* values = reinterpret_cast<const void*>(desc[plan_second(layer)]);
    const unsigned* table = plan_pages(plan, desc);
    const int kv = head0 / (heads / kv_heads);
    float acc[kMaxGroup] = {};
    // a page at a time: its table entry is read once, and the keys are summed in order
    for (int key = key0; key < stop;) {
        const int edge = min((key / kPageTokens + 1) * kPageTokens, stop);
        const long long slot0 = page_row(table[key / kPageTokens], plan.pool, kv, 0, d) + lane;
        for (; key < edge; ++key) {
            const float value = load(values, KIND, slot0 + static_cast<long long>(key % kPageTokens) * d);
#pragma unroll
            for (int g = 0; g < kMaxGroup; ++g) {
                if (g < group) acc[g] += weights[g * 128 + key - key0] * value;
            }
        }
    }
#pragma unroll
    for (int g = 0; g < kMaxGroup; ++g) {
        if (g < group) partials[((first + g) * gridDim.x + tile) * d + lane] = acc[g];
    }
}

// The words of a row copied by a block row of threads, four at a time.
__device__ __forceinline__ void copy_words(unsigned int* to, const unsigned int* from, int words) {
    const int stride = gridDim.x * blockDim.x;
    for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4; i < words; i += stride * 4) {
        *reinterpret_cast<uint4*>(to + i) = *reinterpret_cast<const uint4*>(from + i);
    }
}

}  // namespace

extern "C" {

__global__ void __launch_bounds__(256)
tf_plan_kv_write(const float* keys, const void* values, int kind, int rows, int kv_heads, int d, PlanArgs plan, int layer) {
    plan_kv_write_body(keys, values, kind, rows, kv_heads, d, plan, layer);
}

__global__ void __launch_bounds__(128)
tf_plan_score_f16(const float* q, float* scores, int span, int heads, int kv_heads, int d, float scale, PlanArgs plan, int layer) {
    plan_score_body<1>(q, scores, span, heads, kv_heads, d, scale, plan, layer);
}

__global__ void __launch_bounds__(128)
tf_plan_score_bf16(const float* q, float* scores, int span, int heads, int kv_heads, int d, float scale, PlanArgs plan, int layer) {
    plan_score_body<2>(q, scores, span, heads, kv_heads, d, scale, plan, layer);
}

__global__ void __launch_bounds__(256)
tf_plan_apply_f16(const float* scores, const float* stats, float* partials, int span, int heads, int kv_heads, int d, PlanArgs plan, int layer) {
    plan_apply_body<1>(scores, stats, partials, span, heads, kv_heads, d, plan, layer);
}

__global__ void __launch_bounds__(256)
tf_plan_apply_bf16(const float* scores, const float* stats, float* partials, int span, int heads, int kv_heads, int d, PlanArgs plan, int layer) {
    plan_apply_body<2>(scores, stats, partials, span, heads, kv_heads, d, plan, layer);
}

// A prompt pass's rotated, rounded keys and values (16-bit) into positions pos0 .. pos0 + len of a stream's pages.
__global__ void __launch_bounds__(256)
tf_page_write(const uint16_t* src, uint16_t* pool, const unsigned* table, int len, int kv_heads, int d, long long s_head,
              long long s_row, int pos0, int count) {
    const long long i = gid();
    if (i >= static_cast<long long>(len) * kv_heads * d) return;
    const int j = static_cast<int>(i % d);
    const int h = static_cast<int>((i / d) % kv_heads);
    const int r = static_cast<int>(i / (static_cast<long long>(d) * kv_heads));
    pool[page_at(table, count, h, pos0 + r, d) + j] = src[h * s_head + r * s_row + j];
}

// The first `len` positions of a stream's pages as (kv_heads, len, d), the flat layout other attention kernels read.
__global__ void __launch_bounds__(256)
tf_page_gather(const uint16_t* pool, const unsigned* table, uint16_t* dst, int len, int kv_heads, int d, int count) {
    const long long i = gid();
    if (i >= static_cast<long long>(len) * kv_heads * d) return;
    const int j = static_cast<int>(i % d);
    const int r = static_cast<int>((i / d) % len);
    const int h = static_cast<int>(i / (static_cast<long long>(d) * len));
    dst[i] = pool[page_at(table, count, h, r, d) + j];
}

// Row r of `dst` is `words` words from the address srcs[r]: a launch whose rows come from anywhere keeps its arguments.
__global__ void __launch_bounds__(256) tf_plan_gather(const unsigned long long* srcs, unsigned int* dst, int words) {
    const unsigned int* from = reinterpret_cast<const unsigned int*>(srcs[blockIdx.y]);
    unsigned int* to = dst + static_cast<long long>(blockIdx.y) * words;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += gridDim.x * blockDim.x) to[i] = from[i];
}

// Each slot's conv window and last final row after its first keep[slot] rows (negative: none); states: tf_gdn_replay.
__global__ void __launch_bounds__(256)
tf_plan_keep(PlanArgs plan, const int* keep, const unsigned int* hidden, int hidden_words, int channels, int taps, int layers) {
    const int slot = blockIdx.z;
    const int n = keep[slot];
    if (n < 0) return;
    const unsigned long long* desc = plan_desc(plan, slot);
    if (blockIdx.y == layers) {
        const int row = plan.first[slot] + n - 1;
        if (desc[1] != 0) copy_words(reinterpret_cast<unsigned int*>(desc[1]), hidden + static_cast<long long>(row) * hidden_words, hidden_words);
        return;
    }
    const int layer = blockIdx.y;
    if (plan.count[slot] <= 1 || plan.snaps[2 * layer] == 0) return;
    float* window = reinterpret_cast<float*>(desc[plan_first(layer)]);
    const float* samples = reinterpret_cast<const float*>(plan.snaps[2 * layer]) + static_cast<long long>(plan.first[slot]) * channels;
    // tap t takes position n - taps + t of the round's first window followed by its rows: ascending taps read ahead
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < channels; c += gridDim.x * blockDim.x) {
        for (int t = 0; t < taps; ++t) {
            const int at = n - taps + t;
            window[static_cast<long long>(t) * channels + c] = at < 0 ? window[static_cast<long long>(taps + at) * channels + c] : samples[static_cast<long long>(at) * channels + c];
        }
    }
}

}  // extern "C"
