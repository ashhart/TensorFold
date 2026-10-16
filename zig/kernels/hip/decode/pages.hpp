#pragma once

// Paged KV: a pool of `count` 64-position pages a head, named by a stream's table; a head's pages sit side by side.
constexpr int kPageTokens = 64;

// The element offset in a pool of `count` pages a head of head `h`, slot `slot` of page `page`.
__device__ __forceinline__ long long page_row(unsigned page, unsigned count, int h, int slot, int d) {
    return ((static_cast<long long>(h) * count + page) * kPageTokens + slot) * d;
}

// The element offset in a pool of head `h` at stream position `pos`.
__device__ __forceinline__ long long page_at(const unsigned* table, unsigned count, int h, int pos, int d) {
    return page_row(table[pos / kPageTokens], count, h, pos % kPageTokens, d);
}
