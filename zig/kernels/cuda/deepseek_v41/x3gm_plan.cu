// x3gm's pass tables on the device in one launch: same integers as the torch ops of x3gm.plan.

// order = stable argsort(pick); cnt/off/npe/pend per expert; pass t -> expert e, poff, pcnt; npass = pend[E - 1].

// One CTA (E < 512): histogram, scan, stable chunked scatter (per-warp counts + match_any); integers only, so exact.

#include <cstdint>

namespace dsv41_x3gm_plan {

constexpr int NT = 512, NW = NT / 32, EMAX = 512;   // E <= 511: the routers take <= 512 experts (GE_MAX)

__global__ void __launch_bounds__(NT) plan_kernel(const int* __restrict__ pick, int P, int E, int bm,
                                                  int* __restrict__ order, int* __restrict__ pe,
                                                  int* __restrict__ poff, int* __restrict__ pcnt,
                                                  int* __restrict__ npass) {
    __shared__ int cnt[EMAX + 1];        // counts of values 0..E, then (scatter) the moving bases
    __shared__ int pend[EMAX];           // inclusive scan of passes an expert
    __shared__ int wc[NW][EMAX + 1];     // a chunk's per-warp counts of each value (scatter)
    __shared__ int scan_tmp[NT];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, V = E + 1;

    for (int v = tid; v < V; v += NT) cnt[v] = 0;
    __syncthreads();
    for (int i = tid; i < P; i += NT) atomicAdd(&cnt[pick[i]], 1);    // counts: order-free
    __syncthreads();

    // exclusive scans: bases of values 0..E and pend over the first E; V <= NT * 2, two elements a thread
    auto block_scan = [&](int x) {       // inclusive scan of one value a thread
        scan_tmp[tid] = x;
        __syncthreads();
        for (int o = 1; o < NT; o <<= 1) {
            const int y = tid >= o ? scan_tmp[tid - o] : 0;
            __syncthreads();
            scan_tmp[tid] += y;
            __syncthreads();
        }
        const int r = scan_tmp[tid];
        __syncthreads();
        return r;
    };
    int c0 = 2 * tid < V ? cnt[2 * tid] : 0, c1 = 2 * tid + 1 < V ? cnt[2 * tid + 1] : 0;
    const int b_end = block_scan(c0 + c1);                    // inclusive over pairs of values
    const int n0 = 2 * tid < E ? (c0 + bm - 1) / bm : 0, n1 = 2 * tid + 1 < E ? (c1 + bm - 1) / bm : 0;
    const int p_end = block_scan(n0 + n1);
    __syncthreads();
    if (2 * tid < V) cnt[2 * tid] = b_end - c0 - c1;           // base of value 2 tid
    if (2 * tid + 1 < V) cnt[2 * tid + 1] = b_end - c1;
    if (2 * tid < E) pend[2 * tid] = p_end - n1;
    if (2 * tid + 1 < E) pend[2 * tid + 1] = p_end;
    __syncthreads();

    // pass tables (counts recomputed from the bases: base[v + 1] - base[v], base[E + 1] = P)
    const int tmax = P / bm + E + 1;
    const int npe_last = E > 0 ? pend[E - 1] : 0;
    for (int t = tid; t < tmax; t += NT) {
        int lo = 0, hi = E;                                    // searchsorted(pend, t, right): first pend > t
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (pend[mid] <= t) lo = mid + 1; else hi = mid;
        }
        const int e = lo < E - 1 ? lo : E - 1;
        const int ce = (e + 1 < V ? cnt[e + 1] : P) - cnt[e];
        const int npe_e = (ce + bm - 1) / bm;
        const long long j = (long long)t - (pend[e] - npe_e);
        long long o = (long long)cnt[e] + j * bm;
        const long long cap = P - 1 > 0 ? P - 1 : 0;
        if (o > cap) o = cap;
        long long c = (long long)ce - j * bm;
        c = c < 0 ? 0 : (c > bm ? bm : c);
        pe[t] = e;
        poff[t] = (int)o;
        pcnt[t] = (int)c;
    }
    if (tid == 0) npass[0] = npe_last;
    __syncthreads();

    // stable scatter, NT pairs a chunk in index order
    for (int base = 0; base < P; base += NT) {
        for (int i = tid; i < NW * V; i += NT) (&wc[0][0])[(i / V) * (EMAX + 1) + i % V] = 0;
        __syncthreads();
        const int i = base + tid;
        const bool live = i < P;
        const int v = live ? pick[i] : -1;
        const unsigned same = __match_any_sync(0xffffffffu, v);
        const int before = __popc(same & ((1u << lane) - 1u));  // equal values in earlier lanes of the warp
        if (live && before == 0) wc[warp][v] = __popc(same);     // the warp's count of v, by its first lane
        __syncthreads();
        if (live) {
            int slot = cnt[v] + before;
            for (int w = 0; w < warp; ++w) slot += wc[w][v];
            order[slot] = i;
        }
        __syncthreads();
        for (int u = tid; u < V; u += NT) {                       // the bases move past this chunk's pairs
            int s = 0;
            for (int w = 0; w < NW; ++w) s += wc[w][u];
            cnt[u] += s;
        }
        __syncthreads();
    }
}

}  // namespace dsv41_x3gm_plan
