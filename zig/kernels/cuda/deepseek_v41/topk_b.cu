

// Row-bounded decode top-k (TF_DSV41_INDEX_BOUND): topk_kernel with each row cut at its visible end, same selection.

// Bound B: mode 0 min(nk, max(nvis, k)), mode 2 min(nk, cdiv(nvis, bs)), mode 3 nk; past-B entries rank below all kept.

// Every barrier and cluster sync is kept (a CTA with no entries still takes part); other lines are topk_kernel's.

#include "topk_cuda.cu"

namespace dsv41_topk {

__global__ void __launch_bounds__(NT, 1) topk_b_kernel(Args a) {
    cg::cluster_group cl = cg::this_cluster();
    const Job J = blockIdx.z == 0 ? a.job[0] : a.job[1];        // (no dynamic indexing: no local memory)
    const int CL = static_cast<int>(cl.num_blocks()), rank = static_cast<int>(cl.block_rank());
    const int r = blockIdx.y, tid = threadIdx.x, lane = tid & 31;
    extern __shared__ uint32_t img[];                            // SL = 512 x ept images (0: missing)
    __shared__ int hist[2][256];
    __shared__ int tot[256];
    __shared__ int ws[2 * NW];
    __shared__ int ex[2];                                        // (above, equal) of this CTA, read by the cluster
    __shared__ int dec[4];
    const int q = a.pos64 ? static_cast<int>(static_cast<const long long*>(a.pos)[r])
                          : static_cast<const int*>(a.pos)[0] + r;
    const int nvis = (q + 1) / a.ratio;
    // the row's bound (above): entries [B, nk) are never read
    const int B = J.mode == 0 ? min(J.nk, max(nvis, J.k))
                : J.mode == 2 ? min(J.nk, max(0, (nvis + a.bs - 1) / a.bs))
                              : J.nk;
    const int SL = NT * J.ept, lo = rank * SL, n = max(0, min(SL, B - lo));
    const int nr = (n + NT - 1) / NT * NT;                       // the warp-collective loops' end (<= SL)
    if (rank == 0 && tid == 0 && J.cnt) J.cnt[r] = 0;            // before the first cluster barrier (release)
    const float* srow = J.s + static_cast<long long>(r) * J.ss;
    const int* crow = J.cand ? J.cand + static_cast<long long>(r) * J.cs : nullptr;
    for (int i = tid; i < nr; i += NT) {                         // images, coalesced
        const int e = lo + i;
        uint32_t u = 0;
        if (i < n) {
            if (J.mode == 2) {                                   // block e: max over its visible members, newest +inf
                float m = 0.f;                                   // tl.max (maxnumf) over the members, masked -inf
                for (int j = 0; j < a.bs; ++j) {
                    const int p = e * a.bs + j;
                    const float v = p < nvis ? srow[p] : -INFINITY;
                    m = j ? fmaxf(m, v) : v;
                }
                if (e == (nvis - 1) / a.bs) m = INFINITY;
                if (e * a.bs < nvis) u = image(m);
            } else if (J.mode == 3) {
                if (crow[e / a.bs] >= 0) u = image(srow[e]);
            } else {
                u = image(srow[e]);
            }
        }
        img[i] = u;
    }
    __syncthreads();
    uint32_t prefix = 0, pmask = 0;
    int above = 0, quota_mode = 0;                               // 0: every match selected; 1: equal images capped
    for (int p = 0; p < 4; ++p) {
        int* h = hist[p & 1];
        if (tid < 256) h[tid] = 0;
        __syncthreads();
        const int shift = 24 - 8 * p;
        for (int i0 = 0; i0 < nr; i0 += NT) {
            const uint32_t v = img[i0 + tid];
            const bool m = v != 0 && (v & pmask) == prefix;
            const unsigned act = __ballot_sync(FULL, m);
            if (m) {
                const int d = (v >> shift) & 255;
                const unsigned peers = __match_any_sync(act, d);
                if (lane == __ffs(peers) - 1) atomicAdd(&h[d], __popc(peers));
            }
        }
        cl.sync();
        if (tid < 256) {
            int v[MAXCL];
#pragma unroll
            for (int x = 0; x < MAXCL; ++x) v[x] = x < CL ? cl.map_shared_rank(h, x)[tid] : 0;
            int sum = 0;
#pragma unroll
            for (int x = 0; x < MAXCL; ++x) sum += v[x];
            tot[tid] = sum;
        }
        __syncthreads();
        if (tid < 32) {                                          // suffix sums over 256 digits, 8 a lane; the digit
            int loc[8], run = 0;
            for (int x = 7; x >= 0; --x) { run += tot[lane * 8 + x]; loc[x] = run; }
            int suf = run;                                       // inclusive suffix over lanes >= this one
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_down_sync(FULL, suf, o);
                if (lane + o < 32) suf += y;
            }
            const int after = suf - run;                         // lanes above this one
            int best = -1, bge = 0;
            for (int x = 7; x >= 0; --x)
                if (best < 0 && above + after + loc[x] >= J.k) { best = lane * 8 + x; bge = after + loc[x]; }
            const unsigned has = __ballot_sync(FULL, best >= 0);
            const int src = has ? 31 - __clz(has) : 0;
            const int dg = __shfl_sync(FULL, best, src), ge = __shfl_sync(FULL, bge, src);
            const int all = __shfl_sync(FULL, suf, 0);
            if (lane == 0) { dec[0] = dg; dec[1] = ge; dec[2] = all; }
        }
        __syncthreads();
        const int dg = dec[0], ge = dec[1], all = dec[2];
        if (p == 0 && all <= J.k) break;                         // every valid entry (prefix / mask stay 0)
        const int hd = tot[dg];
        above += ge - hd;
        prefix |= static_cast<uint32_t>(dg) << shift;
        pmask |= 255u << shift;
        if (above + hd == J.k) break;                            // the prefix's entries complete the top K
        if (p == 3) quota_mode = 1;                              // ties at the K-th image: the first ones win
    }
    // classify: 2 above (selected), 1 equal (selected up to the quota, in scan order), 0 out
    const int e0 = tid * J.ept;
    const int xe = max(0, min(J.ept, n - e0));                   // this thread's entries below the bound
    int gt = 0, eq = 0;
    for (int x = 0; x < xe; ++x) {
        const uint32_t v = img[e0 + x];
        if (v == 0) continue;
        const uint32_t mv = v & pmask;
        gt += mv > prefix;
        eq += mv == prefix;
    }
    int tgt, teq;
    const int pre_eq = block_scan(eq, ws, &teq);
    block_scan(gt, ws, &tgt);
    if (tid == 0) { ex[0] = tgt; ex[1] = teq; }
    cl.sync();
    int base = 0, eq_before = 0, total = 0, quota = teq;
    {
        const int need = J.k - above;                            // equal images to take (quota mode)
        for (int x = 0; x < CL; ++x) {
            const int* rx = cl.map_shared_rank(ex, x);
            const int xg = rx[0], xe2 = rx[1];
            const int xq = quota_mode ? max(0, min(xe2, need - eq_before)) : xe2;
            if (x < rank) base += xg + xq;
            if (x == rank) quota = xq;
            eq_before += xe2;
            total += xg + xq;
        }
    }
    cl.sync();                                                   // no DSMEM access after this: CTAs may exit
    int sel = 0;
    {
        int er = pre_eq;
        for (int x = 0; x < xe; ++x) {
            const uint32_t v = img[e0 + x];
            if (v == 0) continue;
            const uint32_t mv = v & pmask;
            if (mv > prefix) ++sel;
            else if (mv == prefix) { sel += er < quota; ++er; }
        }
    }
    int tsel;
    const int off = base + block_scan(sel, ws, &tsel);
    int* orow = J.out + static_cast<long long>(r) * J.os;
    int vis = 0;
    {
        int er = pre_eq, o = off;
        for (int x = 0; x < xe; ++x) {
            const uint32_t v = img[e0 + x];
            if (v == 0) continue;
            const uint32_t mv = v & pmask;
            bool s = mv > prefix;
            if (!s && mv == prefix) { s = er < quota; ++er; }
            if (!s) continue;
            const int e = lo + e0 + x;
            const int pp = J.mode == 3 ? crow[e / a.bs] * a.bs + e % a.bs : e;
            orow[o++] = pp;
            vis += pp < nvis;
        }
    }
    if (J.cnt) {
        int tv;
        block_scan(vis, ws, &tv);
        if (tid == 0 && tv) atomicAdd(J.cnt + r, tv);
    }
    if (rank == 0)
        for (int i = total + tid; i < J.k; i += NT) orow[i] = -1;
}

}  // namespace dsv41_topk
