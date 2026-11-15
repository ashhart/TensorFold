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

// Pages of a host tier: K/V rows by frame (pt, ps as in Pg), indexer keys and pooled rows by logical page (ipt, ips).
struct PgP {
    const int* pt;
    long long ps;
    int ptok;
    long long ips;  // elements (bf16) a page holds of the index plane
    const int* ipt; // the index plane's table, an int a page
};

__device__ __forceinline__ long long page_row(const PgP& pg, int p, int per, int row) {
    return static_cast<long long>(pg.pt[p / per]) * pg.ps + static_cast<long long>(p % per) * row;
}

// Row p of an indexer-key or pooled cache: its page's start in the index plane (a Pg has one plane), then its own row.
__device__ __forceinline__ long long ix_row(const Pg& pg, int p, int per, int row) {
    return page_row(pg, p, per, row);
}

__device__ __forceinline__ long long ix_row(const PgP& pg, int p, int per, int row) {
    return static_cast<long long>(pg.ipt[p / per]) * pg.ips + static_cast<long long>(p % per) * row;
}
