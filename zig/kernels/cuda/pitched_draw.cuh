// Only the pitched entry owns this buffer; both ID mappings retain the existing sampling functions.
#pragma once

namespace tf_sample {

static_assert(sizeof(Shared) == 9752, "pitched draw shared layout changed");

template <bool MAP>
__device__ void draw_pitched(const __nv_bfloat16* L, unsigned V, const Rule* rule, const int* meta, int offset, unsigned* TOK,
                             const long long* IDS, float* PROB, unsigned pitch, Shared& s) {
    const unsigned row = blockIdx.x;
    const Rule rl = *rule;
    const unsigned position = static_cast<unsigned>(meta[0] + static_cast<int>(row) + 1 + offset);
    const __nv_bfloat16* L_row = L + static_cast<size_t>(row) * pitch;
    if (rl.top_k == 0u || rl.top_k > C) {
        sample_full<MAP>(L_row, V, rl, position, IDS, TOK + row, PROB ? PROB + row : nullptr, s);
    } else {
        gpu_sample<MAP>(L_row, V, rl, position, IDS, TOK + row, PROB ? PROB + row : nullptr, s);
    }
}

}  // namespace tf_sample
