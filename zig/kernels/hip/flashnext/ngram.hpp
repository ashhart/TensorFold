#pragma once

#include "../ops/common.hpp"

// The n-gram table rows of a window: each head hashes the last n tokens of the row's segment into its prime table.

namespace {

// The tokens before a window the hashes may read: the n-gram size less one, at most.
constexpr int kNgramHistory = 2;

// meta (int64): history (kNgramHistory, oldest first), eos, multipliers (n), table sizes (heads), offsets (heads).
extern "C" __global__ void tf_fn_ngram_ids(const int* tokens, const long long* meta, unsigned* ids, int rows,
                                           int heads, int per, int n) {
    const long long i = gid();
    if (i >= static_cast<long long>(rows) * heads || n - 1 > kNgramHistory) return;
    const int row = static_cast<int>(i / heads), head = static_cast<int>(i % heads);
    const long long eos = meta[kNgramHistory];
    const long long* mult = meta + kNgramHistory + 1;
    const long long* sizes = mult + n;
    const long long* offsets = sizes + heads;
    const int at = kNgramHistory + row;
    auto token = [&](int k) -> long long {
        return k < kNgramHistory ? meta[k] : static_cast<long long>(tokens[k - kNgramHistory]);
    };
    int in_segment = 0;  // tokens of the row's segment before it, counted up to n - 1
    while (in_segment < n - 1 && token(at - 1 - in_segment) != eos) ++in_segment;
    const int span = 2 + head / per;  // head h hashes 2 + h / per tokens, those before the segment read as eos
    unsigned long long mixed = 0;
    for (int q = 0; q < span; ++q) {
        const long long t = in_segment >= q ? token(at - q) : eos;
        const unsigned long long term = static_cast<unsigned long long>(t) * static_cast<unsigned long long>(mult[q]);
        mixed = q == 0 ? term : (mixed ^ term);
    }
    long long mod = static_cast<long long>(mixed) % sizes[head];
    if (mod < 0) mod += sizes[head];
    ids[i] = static_cast<unsigned>(mod + offsets[head]);
}

}  // namespace
