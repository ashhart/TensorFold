#pragma once

// Whole target rows with request-owned membership; original tf_draw retains its ABI and null-mask path.
extern "C" __global__ void __launch_bounds__(1024) tf_draw_repetition(
    const __nv_bfloat16* L, unsigned V, const tf_sample::Rule* rule, const int* meta, int offset,
    unsigned* TOK, float* PROB, const unsigned* BITS, unsigned W, const float* FACTORS) {
    __shared__ tf_sample::Shared s;
    const unsigned row = blockIdx.x;
    const tf_sample::Rule rl = *rule;
    const unsigned position = static_cast<unsigned>(meta[0] + static_cast<int>(row) + 1 + offset);
    const __nv_bfloat16* logits = L + static_cast<size_t>(row) * V;
    const unsigned* seen = BITS + static_cast<size_t>(row) * W;
    const float factor = FACTORS[row];
    if (rl.top_k == 0u || rl.top_k > tf_sample::C) {
        tf_sample::sample_full<false>(logits, V, rl, position, nullptr, TOK + row, PROB ? PROB + row : nullptr,
                                     s, seen, factor);
    } else {
        tf_sample::gpu_sample<false>(logits, V, rl, position, nullptr, TOK + row, PROB ? PROB + row : nullptr,
                                    s, seen, factor);
    }
}

// Same lowest-token-ID tie policy as GLM's original rank-gathered greedy path.
extern "C" __global__ void __launch_bounds__(1024) tf_greedy_repetition(
    const __nv_bfloat16* L, unsigned V, const unsigned* BITS, unsigned W, const float* FACTORS, unsigned* TOK) {
    __shared__ tf_sample::Shared s;
    const unsigned row = blockIdx.x, t = threadIdx.x, lane = t & 31u, sg = t >> 5;
    const tf_sample::Row r{L + static_cast<size_t>(row) * V, V, 1.0f, t, lane, sg,
                           BITS + static_cast<size_t>(row) * W, FACTORS[row]};
    const float best = tf_sample::row_max(r, s);
    unsigned id = 0xffffffffu;
    for (unsigned i = t; i < V; i += tf_sample::TG) if (r.v(i) == best) id = min(id, i);
    id = tf_sample::warp_min_u(id);
    if (lane == 0) s.ush[sg] = id;
    __syncthreads();
    if (t == 0) {
        for (unsigned g = 0; g < tf_sample::NSG; ++g) id = min(id, s.ush[g]);
        TOK[row] = (best == -INFINITY || id == 0xffffffffu) ? 0u : id;
    }
}
