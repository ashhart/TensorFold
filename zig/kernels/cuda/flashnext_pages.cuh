// Flash Next page addresses in bf16 elements.
struct Pg {
    const int* pt;  // the stream's page table, an int a page
    long long ps;   // elements (bf16) a page holds: every attention slot's keys, values, indexer and pooled keys
    int ptok;       // positions a page holds, a multiple of four: no pooled block straddles two
};

// Row p's element in a paged cache of `row`-element rows, `per` rows a page: its page's start, then its own row.
__device__ __forceinline__ long long page_row(const Pg& pg, int p, int per, int row) {
    return static_cast<long long>(pg.pt[p / per]) * pg.ps + static_cast<long long>(p % per) * row;
}
