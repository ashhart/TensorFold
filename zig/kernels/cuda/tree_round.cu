// Verified tree rounds for every CUDA family (host reference: core/tree_round.zig, the Metal kernels' contract).

// Deterministic BF16 picks, each tree's verified path and emitted tokens, and the commit steps that read the result.

#include <stdint.h>

#include "round_ids.cuh"

namespace tf_tree_round {

constexpr int MAX_ROWS = 16;
constexpr uint32_t INVALID = 0xffffffffu;

struct Pick {
    uint32_t token, nonfinite;
};

struct Result {
    uint32_t status, stop, nonfinite, consumed_count, emitted_count, matched_count;
    uint32_t bonus_emitted, pending_valid, pending_token, reserved;
    uint32_t path[MAX_ROWS], tokens[MAX_ROWS];
};

// One tree of a round: its rows first .. first + rows of tokens, parents (tree-local, -1 at the root) and picks.
struct Tree {
    uint32_t first, rows, budget, eos_first, eos_count;
};

__device__ __forceinline__ bool valid(const Result& r, uint32_t rows) {
    if (r.status || r.nonfinite || r.reserved || !r.consumed_count || r.consumed_count > rows ||
        r.consumed_count != r.emitted_count || r.path[0] != 0)
        return false;
    for (uint32_t i = 0; i < r.consumed_count; ++i)
        if (r.path[i] >= rows || (i && r.path[i] <= r.path[i - 1])) return false;
    return true;
}

}  // namespace tf_tree_round

using namespace tf_tree_round;

// picks[row]: a row's first largest finite bf16 logit, or INVALID and its nonfinite kinds (1 NaN, 2 +inf, 4 -inf).
extern "C" __global__ void __launch_bounds__(256) tf_round_bf16_argmax(const uint16_t* __restrict__ logits,
                                                                       Pick* __restrict__ picks, uint32_t vocab,
                                                                       uint32_t stride) {
    __shared__ float values[256];
    __shared__ uint32_t ids[256], masks[256];
    const uint32_t row = blockIdx.x, lane = threadIdx.x;
    float best = -INFINITY;
    uint32_t id = INVALID, mask = 0;
    for (uint64_t token = lane; token < vocab; token += 256) {
        const float value = __uint_as_float(static_cast<uint32_t>(logits[static_cast<uint64_t>(row) * stride + token]) << 16);
        if (!isfinite(value)) mask |= isnan(value) ? 1u : value > 0 ? 2u : 4u;
        else if (id == INVALID || value > best || (value == best && token < id)) {
            best = value;
            id = static_cast<uint32_t>(token);
        }
    }
    values[lane] = best;
    ids[lane] = id;
    masks[lane] = mask;
    __syncthreads();
    for (uint32_t step = 128; step > 0; step >>= 1) {
        if (lane < step) {
            const float other = values[lane + step];
            const uint32_t other_id = ids[lane + step];
            if (other > values[lane] || (other == values[lane] && other_id < ids[lane])) {
                values[lane] = other;
                ids[lane] = other_id;
            }
            masks[lane] |= masks[lane + step];
        }
        __syncthreads();
    }
    if (lane == 0) picks[row] = {masks[0] != 0 ? INVALID : ids[0], masks[0]};
}

// out[t]: tree t's verified round (core/tree_round.oracle), one thread a tree.
extern "C" __global__ void tf_tree_round_match(const uint32_t* __restrict__ tokens, const int32_t* __restrict__ parents,
                                               const Pick* __restrict__ picks, const uint32_t* __restrict__ eos,
                                               const Tree* __restrict__ trees, Result* __restrict__ results,
                                               uint32_t vocab, uint32_t count) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const Tree tr = trees[t];
    Result& out = results[t];
    const uint32_t* tok = tokens + tr.first;
    const int32_t* par = parents + tr.first;
    const Pick* pk = picks + tr.first;
    const uint32_t* stop_ids = eos + tr.eos_first;
    out.status = 0;
    out.stop = 0;
    out.nonfinite = 0;
    out.consumed_count = 0;
    out.emitted_count = 0;
    out.matched_count = 0;
    out.bonus_emitted = 0;
    out.pending_valid = 0;
    out.pending_token = 0;
    out.reserved = 0;
    for (int i = 0; i < MAX_ROWS; ++i) out.path[i] = 0, out.tokens[i] = 0;
    if (tr.rows == 0 || tr.rows > MAX_ROWS || tr.eos_count > MAX_ROWS) {
        out.status = 2;
        out.stop = 3;
        return;
    }
    for (uint32_t row = 0; row < tr.rows; ++row)
        if (tok[row] >= vocab || (row == 0 ? par[row] != -1 : par[row] < 0 || par[row] >= static_cast<int32_t>(row))) {
            out.status = 2;
            out.stop = 3;
            return;
        }
    for (uint32_t i = 0; i < tr.eos_count; ++i)
        if (stop_ids[i] >= vocab) {
            out.status = 2;
            out.stop = 3;
            return;
        }
    bool bad_pick = false;
    for (uint32_t row = 0; row < tr.rows; ++row) {
        out.nonfinite |= pk[row].nonfinite & 7u;
        bad_pick = bad_pick || (pk[row].nonfinite & ~7u) != 0 || (pk[row].nonfinite == 0 && pk[row].token >= vocab);
    }
    if (bad_pick || out.nonfinite != 0) {
        out.status = bad_pick ? 3u : 1u;
        out.stop = 3;
        return;
    }
    if (tr.budget == 0) {
        out.stop = 1;
        return;
    }
    out.path[out.consumed_count++] = 0;
    uint32_t node = 0;
    for (;;) {
        const uint32_t token = pk[node].token;
        uint32_t child = INVALID;
        for (uint32_t row = node + 1; row < tr.rows; ++row)
            if (par[row] == static_cast<int32_t>(node) && tok[row] == token) {
                child = row;
                break;
            }
        out.tokens[out.emitted_count++] = token;
        if (child == INVALID) out.bonus_emitted = 1;
        else ++out.matched_count;
        bool hit_eos = false;
        for (uint32_t i = 0; i < tr.eos_count; ++i) hit_eos = hit_eos || stop_ids[i] == token;
        if (hit_eos) {
            out.stop = 2;
            return;
        }
        if (out.emitted_count == tr.budget) {
            out.stop = 1;
            return;
        }
        if (child == INVALID) {
            out.pending_valid = 1;
            out.pending_token = token;
            return;
        }
        out.path[out.consumed_count++] = child;
        node = child;
    }
}

// keep[t * 16 + i]: tree t's i-th kept row (first + path[i]) or -1 past its consumed count; counts[t]: kept.
extern "C" __global__ void tf_round_keep_rows(const Result* __restrict__ results, const Tree* __restrict__ trees,
                                              int32_t* __restrict__ keep, uint32_t* __restrict__ counts, uint32_t count) {
    const uint32_t t = blockIdx.x, i = threadIdx.x;
    if (t >= count || i >= MAX_ROWS) return;
    const Result& r = results[t];
    const uint32_t n = valid(r, trees[t].rows) ? r.consumed_count : 0;
    keep[t * MAX_ROWS + i] = i < n ? static_cast<int32_t>(trees[t].first + r.path[i]) : -1;
    if (i == 0) counts[t] = n;
}

// out[t, i, :] = rows[first_t + path_t[i], :] (width bf16 a row) for each tree's kept rows: taps, hidden rows.
extern "C" __global__ void tf_round_gather_rows(const Result* __restrict__ results, const Tree* __restrict__ trees,
                                                const uint16_t* __restrict__ rows, uint16_t* __restrict__ out,
                                                uint32_t width, uint32_t count) {
    const uint32_t t = blockIdx.y, i = blockIdx.z, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count || c >= width) return;
    const Result& r = results[t];
    if (!valid(r, trees[t].rows) || i >= r.consumed_count) return;
    out[(static_cast<uint64_t>(t) * MAX_ROWS + i) * width + c] = rows[static_cast<uint64_t>(trees[t].first + r.path[i]) * width + c];
}

// head[t, :] = logits[first_t + the tree's last kept row, :]: the row the next round's first pick reads.
extern "C" __global__ void tf_round_keep_head(const Result* __restrict__ results, const Tree* __restrict__ trees,
                                              const uint16_t* __restrict__ logits, uint16_t* __restrict__ head,
                                              uint32_t width, uint32_t stride, uint32_t count) {
    const uint32_t t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count || c >= width) return;
    const Result& r = results[t];
    if (!valid(r, trees[t].rows)) return;
    head[static_cast<uint64_t>(t) * width + c] = logits[static_cast<uint64_t>(trees[t].first + r.path[r.consumed_count - 1]) * stride + c];
}

// picks [rows] as int32 token ids for the next forward (a draft chain's next step): no host round trip between steps.
extern "C" __global__ void __launch_bounds__(256) tf_round_pick_ids(const uint32_t* picks, int* ids, unsigned rows,
                                                                     unsigned vocab, int fallback) {
    tf_round_ids::pick_ids(picks, ids, rows, vocab, fallback, tf_round_ids::ALL);
}
