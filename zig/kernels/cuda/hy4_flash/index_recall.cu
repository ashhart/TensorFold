// Score only earlier completed groups. When they all fit, selection reads no scores.
#include <cuda_bf16.h>
#include <stdint.h>
#include "../common/reduce.cuh"
#include "../glm_index_pools.cuh"

struct Hy4RecallShape { uint32_t rows, score_groups, group_budget; };
struct Hy4RecallRow { uint32_t group_base, token; };

extern "C" __global__ void __launch_bounds__(128) tf_hy4_group_scores(
    const __nv_bfloat16 *query, const __nv_bfloat16 *keys, const __nv_bfloat16 *head_weights,
    const Hy4RecallRow *rows, float *scores, Hy4RecallShape a, float head_scale) {
    const uint32_t r = blockIdx.y, lane = threadIdx.x % 32;
    const uint32_t group = blockIdx.x * 4 + threadIdx.x / 32;
    const Hy4RecallRow row = rows[r];
    const uint32_t closed = row.token / 4;
    if (closed < a.group_budget || closed > a.score_groups || group >= closed) return;
    const __nv_bfloat16 *key = keys + (uint64_t(row.group_base) + group) * 128;
    float cached[4];
    for (uint32_t d = 0; d < 4; ++d) cached[d] = __bfloat162float(key[lane + d * 32]);
    float score = 0.0f;
    for (uint32_t head = 0; head < 16; ++head) {
        const __nv_bfloat16 *q = query + (uint64_t(r) * 16 + head) * 128;
        float dot = 0.0f;
        for (uint32_t d = 0; d < 4; ++d) dot = __fmaf_rn(__bfloat162float(q[lane + d * 32]), cached[d], dot);
        dot = tf_common::warp_sum(dot);
        const float weight = __bfloat162float(head_weights[uint64_t(r) * 16 + head]) * head_scale;
        score = __fmaf_rn(weight, fmaxf(dot, 0.0f), score);
    }
    if (lane == 0) scores[uint64_t(r) * a.score_groups + group] = score;
}

extern "C" __global__ void __launch_bounds__(256) tf_hy4_recall_groups(
    const float *scores, const Hy4RecallRow *rows, int32_t *tokens, int32_t *counts, Hy4RecallShape a) {
    __shared__ glm_index::Smem scratch;
    __shared__ int64_t selected[511];
    const uint32_t r = blockIdx.x, position = rows[r].token, closed = position / 4;
    if (closed > a.score_groups) {
        if (threadIdx.x == 0) counts[r] = 0;
        return;
    }
    const uint32_t keep = min(closed, a.group_budget - 1);
    if (closed > keep) {
        int phase = 0;
        const glm_index::Row candidates { scores + uint64_t(r) * a.score_groups, int(closed) };
        glm_index::select(scratch, candidates, int(keep), selected, phase);
    } else {
        for (uint32_t i = threadIdx.x; i < keep; i += 256) selected[i] = i;
    }
    __syncthreads();
    int32_t *out = tokens + uint64_t(r) * a.group_budget * 4;
    const uint32_t count = keep * 4 + position % 4 + 1;
    for (uint32_t i = threadIdx.x; i < count; i += 256) {
        const uint32_t id = i / 4 < keep ? uint32_t(selected[i / 4]) : closed;
        out[i] = int32_t(id * 4 + i % 4);
    }
    if (threadIdx.x == 0) counts[r] = int32_t(count);
}
