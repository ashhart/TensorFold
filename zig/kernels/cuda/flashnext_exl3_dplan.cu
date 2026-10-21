// The Flash Next family's EXL3 device-side pool plan (cuda_exl3_dplan.zig): one layer's offload round planned on the
// device, so a decode step needs no host wait between the router and the experts and runs as ONE captured graph.
//
// The rules are the host planner's (exl3_round.zig `plan` over exl3_offload.zig's Vram, one pass, no prefetch, no
// double-buffer protection): the round's routed picks raise the layer's lifetime counts; held experts are touched
// (stamp = now); misses, hottest first (count desc, appearances desc, id asc), each take the first empty line, else the
// coldest resident line not used by this round (count asc, stamp asc, higher line). A miss's three table entries point
// into its line, the evicted expert's are zeroed, and its copy (host bank -> line) joins the work list that
// fn_exl3_dcopy runs next. Every line must fit: the caller only plans rounds of at most `lines` distinct experts.
//
// State, per layer (u32 words at `st`): line_id[L] (expert or -1), stamp[L], line_of[512] (line or -1), counts[512],
// then now, rounds, hit_slots, miss_slots, copies, evictions, overflow, in-place reads (more misses than lines
// free of the round's experts: read from the mapped bank in place for that round).

#include <stdint.h>

#define DP_EXPERTS 512
#define DP_SHARED 512
#define DP_MAXU 256

struct DpMisc {
    unsigned now, rounds, hit_slots, miss_slots, copies, evictions, overflow, inplace;
};

__device__ __forceinline__ bool dp_colder(unsigned ca, unsigned sa, int ia, unsigned cb, unsigned sb, int ib) {
    if (ca != cb) return ca < cb;
    if (sa != sb) return sa < sb;
    return ia > ib;
}

// One block of 256 threads. picks [n] int32 (-1 skipped, 512 the shared expert, masked to the rank's ids at tp=2).
// tab: three int64 [E] pointer tables then (k2tab) three int32 [E] width tables (the pool's, exl3_pool.Tables).
// src: per routed expert three host-bank device addresses, lens: their byte counts, k2: their half-bit widths.
// work: per copy three {src, dst, len} triples (int64), work_n: the copy count.
extern "C" __global__ void __launch_bounds__(256) fn_exl3_dplan(
    const int* __restrict__ picks, int n, unsigned* __restrict__ st, int lines, long long* __restrict__ tab0,
    long long* __restrict__ tab1, long long* __restrict__ tab2, int* __restrict__ k2t0, int* __restrict__ k2t1,
    int* __restrict__ k2t2, const long long* __restrict__ src, const long long* __restrict__ lens,
    const int* __restrict__ k2, unsigned long long line_base, unsigned long long line_bytes, unsigned long long off0,
    unsigned long long off1, unsigned long long off2, long long* __restrict__ work, int* __restrict__ work_n,
    int* __restrict__ hit_picks, int* __restrict__ miss_picks, int* __restrict__ ovf, int* __restrict__ ovf_n) {
    int* line_id = reinterpret_cast<int*>(st);
    unsigned* stamp = st + lines;
    int* line_of = reinterpret_cast<int*>(st + 2 * lines);
    unsigned* counts = st + 2 * lines + DP_EXPERTS;
    DpMisc* misc = reinterpret_cast<DpMisc*>(st + 2 * lines + 2 * DP_EXPERTS);

    __shared__ unsigned short uniq[DP_MAXU];
    __shared__ unsigned short app[DP_MAXU];
    __shared__ unsigned short miss[DP_MAXU];
    __shared__ unsigned keep[DP_EXPERTS / 32];
    __shared__ int n_uniq, n_miss, n_copy, n_ovf;
    __shared__ unsigned now;
    __shared__ unsigned rkey_c[256], rkey_s[256];
    __shared__ int rkey_i[256];
    const int t = threadIdx.x;

    if (t < DP_EXPERTS / 32) keep[t] = 0;
    __syncthreads();
    if (t == 0) {
        now = ++misc->now;
        misc->rounds += 1;
        int nu = 0;
        bool over = false;
        for (int i = 0; i < n; ++i) {
            const int p = picks[i];
            if (p < 0 || p >= DP_SHARED) continue;
            // observe: the lifetime count (saturating) and this round's appearances
            if (counts[p] != 0xffffffffu) counts[p] += 1;
            if (line_of[p] >= 0) misc->hit_slots += 1; else misc->miss_slots += 1;
            int j = 0;
            while (j < nu && uniq[j] != p) ++j;
            if (j == nu) {
                if (nu == DP_MAXU) { over = true; continue; }
                uniq[nu] = static_cast<unsigned short>(p);
                app[nu] = 0;
                keep[p >> 5] |= 1u << (p & 31);
                ++nu;
            }
            app[j] += 1;
        }
        // held experts are touched; misses keep first-appearance order, then sort hottest first (insertion sort)
        int nm = 0;
        for (int j = 0; j < nu; ++j) {
            const int id = uniq[j];
            const int ln = line_of[id];
            if (ln >= 0) {
                stamp[ln] = now;
            } else {
                miss[nm] = static_cast<unsigned short>(j);  // index into uniq/app
                ++nm;
            }
        }
        for (int a = 1; a < nm; ++a) {
            const unsigned short x = miss[a];
            int b = a - 1;
            while (b >= 0) {
                const unsigned short y = miss[b];
                const unsigned cx = counts[uniq[x]], cy = counts[uniq[y]];
                const bool hotter = cx != cy ? cx > cy : (app[x] != app[y] ? app[x] > app[y] : uniq[x] < uniq[y]);
                if (!hotter) break;
                miss[b + 1] = y;
                --b;
            }
            miss[b + 1] = x;
        }
        if (over) misc->overflow += 1;  // more distinct experts than DP_MAXU: the caller's row bound prevents it
        n_uniq = nu;
        n_miss = over ? 0 : nm;
        n_copy = 0;
        n_ovf = 0;
    }
    __syncthreads();
    // the split (optional): picks held when the round started (and the shared expert) vs the misses this round copies;
    // the experts of the first set run while the copies of the second land
    if (hit_picks != nullptr) {
        for (int i = t; i < n; i += blockDim.x) {
            const int p = picks[i];
            const bool held = p >= DP_SHARED || (p >= 0 && line_of[p] >= 0);
            hit_picks[i] = p < 0 ? -1 : (held ? p : -1);
            miss_picks[i] = p < 0 ? -1 : (held ? -1 : p);
        }
        __syncthreads();
    }
    const unsigned long long offs[3] = {off0, off1, off2};
    long long* tabs[3] = {tab0, tab1, tab2};
    int* k2s[3] = {k2t0, k2t1, k2t2};
    for (int m = 0; m < n_miss; ++m) {
        // victim: the first empty line, else the coldest resident not in this round (count, stamp, higher line)
        unsigned bc = 0xffffffffu, bs = 0xffffffffu;
        int bi = -1;
        bool bempty = false;
        for (int i = t; i < lines; i += blockDim.x) {
            const int id = line_id[i];
            if (id < 0) {
                if (!bempty || i < bi) { bempty = true; bi = i; }
                continue;
            }
            if (bempty) continue;
            if (keep[id >> 5] & (1u << (id & 31))) continue;
            const unsigned c = counts[id], s = stamp[i];
            if (bi < 0 || dp_colder(c, s, i, bc, bs, bi)) { bc = c; bs = s; bi = i; }
        }
        // empty lines win (encoded as count 0 with a flag in the stamp key's top bit), lowest index among them
        rkey_c[t] = bempty ? 0u : bc;
        rkey_s[t] = bempty ? 0u : bs;
        rkey_i[t] = bempty ? -2 - bi : bi;  // -2 - i < -1 marks an empty line i
        __syncthreads();
        for (int w = blockDim.x / 2; w > 0; w >>= 1) {
            if (t < w) {
                const int ia = rkey_i[t], ib = rkey_i[t + w];
                bool take_b;
                if (ib == -1) take_b = false;
                else if (ia == -1) take_b = true;
                else if (ia <= -2 || ib <= -2) {
                    // an empty line beats a resident one; between empties the lower line
                    if (ia <= -2 && ib <= -2) take_b = (-2 - ib) < (-2 - ia);
                    else take_b = ib <= -2;
                } else take_b = dp_colder(rkey_c[t + w], rkey_s[t + w], ib, rkey_c[t], rkey_s[t], ia);
                if (take_b) {
                    rkey_c[t] = rkey_c[t + w];
                    rkey_s[t] = rkey_s[t + w];
                    rkey_i[t] = ib;
                }
            }
            __syncthreads();
        }
        if (t == 0) {
            const int r = rkey_i[0];
            const int v = r <= -2 ? -2 - r : r;
            const int id = uniq[miss[m]];
            if (v >= 0) {
                const int old = line_id[v];
                if (old >= 0) {
                    line_of[old] = -1;
                    for (int w = 0; w < 3; ++w) {
                        tabs[w][old] = 0;
                        k2s[w][old] = 0;
                    }
                    misc->evictions += 1;
                }
                line_id[v] = id;
                line_of[id] = v;
                stamp[v] = now;
                const unsigned long long base = line_base + static_cast<unsigned long long>(v) * line_bytes;
                long long* wk = work + static_cast<long long>(n_copy) * 9;
                for (int w = 0; w < 3; ++w) {
                    tabs[w][id] = static_cast<long long>(base + offs[w]);
                    k2s[w][id] = k2[id * 3 + w];
                    wk[w * 3 + 0] = src[id * 3 + w];
                    wk[w * 3 + 1] = static_cast<long long>(base + offs[w]);
                    wk[w * 3 + 2] = lens[id * 3 + w];
                }
                n_copy += 1;
                misc->copies += 1;
            } else {
                // no line is free of this round's experts: the expert is read in place from the mapped host bank
                // this round (same bytes, so the same results; slower), and its entries are cleared after the round
                for (int w = 0; w < 3; ++w) {
                    tabs[w][id] = src[id * 3 + w];
                    k2s[w][id] = k2[id * 3 + w];
                }
                ovf[n_ovf] = id;
                n_ovf += 1;
                misc->inplace += 1;
            }
        }
        __syncthreads();
    }
    if (t == 0) {
        *work_n = n_copy;
        *ovf_n = n_ovf;
    }
}

// After the round's experts: the in-place (host-bank) experts' table entries cleared again (they hold no line).
extern "C" __global__ void fn_exl3_drestore(const int* __restrict__ ovf, const int* __restrict__ ovf_n,
                                           long long* __restrict__ tab0, long long* __restrict__ tab1,
                                           long long* __restrict__ tab2, int* __restrict__ k2t0, int* __restrict__ k2t1,
                                           int* __restrict__ k2t2) {
    const int n = *ovf_n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int id = ovf[i];
        tab0[id] = 0;
        tab1[id] = 0;
        tab2[id] = 0;
        k2t0[id] = 0;
        k2t1[id] = 0;
        k2t2[id] = 0;
    }
}

// The planned copies: each segment {src (host bank, mapped), dst (a line), len (bytes, a multiple of 16)} read as
// 16-byte words by the whole grid. Reads of pinned host memory over PCIe: many independent loads in flight.
extern "C" __global__ void __launch_bounds__(256) fn_exl3_dcopy(const long long* __restrict__ work,
                                                                const int* __restrict__ work_n) {
    const int segs = *work_n * 3;
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (int g = 0; g < segs; ++g) {
        const uint4* __restrict__ s = reinterpret_cast<const uint4*>(work[g * 3 + 0]);
        uint4* __restrict__ d = reinterpret_cast<uint4*>(work[g * 3 + 1]);
        const long long words = work[g * 3 + 2] / 16;
        long long i = tid;
        for (; i + 3 * stride < words; i += 4 * stride) {
            const uint4 a = s[i], b = s[i + stride], c = s[i + 2 * stride], e = s[i + 3 * stride];
            d[i] = a;
            d[i + stride] = b;
            d[i + 2 * stride] = c;
            d[i + 3 * stride] = e;
        }
        for (; i < words; i += stride) d[i] = s[i];
    }
}

// The draft chain on the device (cuda_exl3_chain.zig): fn_row_best's record [best, log-sum-exp, column bits, -] of a
// draft row picks the draft vocabulary's id at that column as the next chained step's input token.
extern "C" __global__ void fn_exl3_draft_token(const float* __restrict__ best, const unsigned* __restrict__ ids,
                                               int* __restrict__ token) {
    if (threadIdx.x == 0 && blockIdx.x == 0) token[0] = static_cast<int>(ids[__float_as_int(best[2])]);
}

// The chain's step decision on the device: the next token as fn_exl3_draft_token, then upstream's stop rule
// (Ops.draft: a pick below `confidence` ends the chain) folded into `alive`, which also drives the conditional
// node around the next head step (cudaGraphSetConditional) and is kept per step for the host.
extern "C" __global__ void fn_exl3_draft_next(const float* __restrict__ best, const unsigned* __restrict__ ids,
                                              int* __restrict__ token, int* __restrict__ alive,
                                              int* __restrict__ alive_out, cudaGraphConditionalHandle handle,
                                              double confidence) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    token[0] = static_cast<int>(ids[__float_as_int(best[2])]);
    const double prob = exp(static_cast<double>(best[0]) - static_cast<double>(best[1]));
    const int low = confidence > 0.0 && prob < confidence;
    const int a = alive[0] && !low;
    alive[0] = a;
    alive_out[0] = a;
    cudaGraphSetConditional(handle, static_cast<unsigned>(a));
}
