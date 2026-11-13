#pragma once

namespace tf_sample {

// Active repetition rows keep their request-owned membership and transform out of the identity path.
struct RepetitionRow {
    const __nv_bfloat16* L;
    unsigned V;
    float inv_t;
    unsigned t, lane, sg;
    const unsigned* seen;
    float factor;
    __device__ __forceinline__ float v(unsigned i) const {
        const bool present = ((seen[i / 32] >> (i % 32)) & 1u) != 0;
        return repetition_value(L[i], present, factor) * inv_t;
    }
};

template <bool MAP>
__device__ void gpu_sample_repetition(const __nv_bfloat16* L, unsigned V, const Rule& rl, unsigned position,
                                      const long long* IDS, unsigned* TOK, float* PROB, Shared& s,
                                      const unsigned* seen, float factor) {
    const unsigned t = threadIdx.x, lane = t & 31u, sg = t >> 5;
    const RepetitionRow r{L, V, rl.inv_t, t, lane, sg, seen, factor};
    const float m = row_max(r, s);
    float z = 0.0f;
    const unsigned n_cand = first_block(r, m, rl.top_p, rl.near, rl.top_k, z, s);
    pick<MAP>(rl, m, z, n_cand, position, IDS, TOK, PROB, s);
}

template <bool MAP>
__device__ void sample_full_repetition(const __nv_bfloat16* L, unsigned V, const Rule& rl, unsigned position,
                                       const long long* IDS, unsigned* TOK, float* PROB, Shared& s,
                                       const unsigned* seen, float factor) {
    const unsigned t = threadIdx.x, lane = t & 31u, sg = t >> 5;
    const RepetitionRow r{L, V, rl.inv_t, t, lane, sg, seen, factor};
    const unsigned kc = rl.top_k;
    const float m = row_max(r, s);
    const float lo = fmaxf(m + rl.min_log, m - NEVER);
    const bool whole = kc == 0u || kc >= V;
    Mark top = end_mark();
    if (!whole) {
        const Top u = top_keys(r, kc, s);
        top = Mark{u.key, u.idcut};
    }
    const Mark cut = (rl.top_p > 0.0f && rl.top_p < 1.0f) ? nucleus(r, m, lo, rl.top_p, rl.near, kc, whole, top, s) : end_mark();
    race<MAP>(r, lo, top, cut, rl.seed, position, IDS, TOK, s);
    if (!PROB) return;
    __syncthreads();
    const float pick = val(s.st[3]);
    const float z = mass(r, m, 1.0f, -INFINITY, Mark{0xFFFFFFFFu, 0xFFFFFFFFu}, top, end_mark(), false, s);
    if (r.t == 0) *PROB = expf(pick - m) / z;
}

}  // namespace tf_sample

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
        tf_sample::sample_full_repetition<false>(logits, V, rl, position, nullptr, TOK + row,
                                                PROB ? PROB + row : nullptr, s, seen, factor);
    } else {
        tf_sample::gpu_sample_repetition<false>(logits, V, rl, position, nullptr, TOK + row,
                                               PROB ? PROB + row : nullptr, s, seen, factor);
    }
}

// Same lowest-token-ID tie policy as GLM's original rank-gathered greedy path.
extern "C" __global__ void __launch_bounds__(1024) tf_greedy_repetition(
    const __nv_bfloat16* L, unsigned V, const unsigned* BITS, unsigned W, const float* FACTORS, unsigned* TOK) {
    __shared__ tf_sample::Shared s;
    const unsigned row = blockIdx.x, t = threadIdx.x, lane = t & 31u, sg = t >> 5;
    const tf_sample::RepetitionRow r{L + static_cast<size_t>(row) * V, V, 1.0f, t, lane, sg,
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
