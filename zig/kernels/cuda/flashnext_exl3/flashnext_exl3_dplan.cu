// The Flash Next family's EXL3 device-side pool plan (cuda_exl3_dplan.zig): one layer's offload round planned on the

#include <stdint.h>

#define DP_EXPERTS 512
#define DP_SHARED 512
#define DP_MAXU 256
#define DP_RECENT_WINDOW 256
#define DP_RECENT_WORDS_PER_ROW (DP_EXPERTS / 32)
#define DP_RECENT_WORDS (DP_RECENT_WINDOW * DP_RECENT_WORDS_PER_ROW)

struct DpMisc {
    unsigned now, rounds, hit_slots, miss_slots, copies, evictions, overflow, inplace, recent_cursor, recent_filled;
};

struct DpTraceHeader {
    unsigned magic, version, pool_index, layer, lines, max_rows, seed_words, record_words;
    unsigned capacity, recorded, dropped_lo, dropped_hi, discontinuity, invalid_launch, checkpoints, dropped_checkpoints;
};

static_assert(sizeof(DpTraceHeader) == 64, "trace header ABI");
static_assert(sizeof(DpMisc) == 40, "planner misc ABI");

static constexpr unsigned DP_TRACE_WORDS = 1024 * 1024 / 4;

__device__ __forceinline__ bool dp_trace_valid(DpTraceHeader* header, int lines) {
    const unsigned long long state_words = 2ull * static_cast<unsigned>(lines) + 3 * DP_EXPERTS + DP_RECENT_WORDS + 10;
    const unsigned long long record_words = 15ull + 11ull * header->max_rows;
    const unsigned long long base = 16 + 2 * state_words;
    const unsigned long long checkpoint_words = 4 + 2 * state_words;
    if (lines <= 0 || header->magic != 0x44505431 || header->version != 4 ||
        header->lines != static_cast<unsigned>(lines) || header->max_rows == 0 ||
        base >= DP_TRACE_WORDS || record_words > DP_TRACE_WORDS - base ||
        header->seed_words != state_words || header->record_words != record_words ||
        header->capacity != (DP_TRACE_WORDS - base) / record_words || header->recorded > header->capacity ||
        header->discontinuity > 1 || header->invalid_launch > 1 ||
        header->checkpoints > (DP_TRACE_WORDS - base) / checkpoint_words ||
        base + static_cast<unsigned long long>(header->recorded) * record_words >
            DP_TRACE_WORDS - static_cast<unsigned long long>(header->checkpoints) * checkpoint_words) {
        header->invalid_launch = 1;
        return false;
    }
    return true;
}

extern "C" __global__ void fn_exl3_dcheckpoint(const unsigned* st, int lines, unsigned* trace, int transfer) {
    if (trace == nullptr || threadIdx.x != 0 || blockIdx.x != 0) return;
    auto* header = reinterpret_cast<DpTraceHeader*>(trace);
    if (!dp_trace_valid(header, lines)) return;
    if (transfer != 0 && transfer != 1) {
        header->invalid_launch = 1;
        return;
    }
    unsigned* seed = trace + 16;
    unsigned* terminal = seed + header->seed_words;
    const bool initialized = header->recorded != 0 || header->checkpoints != 0;
    if (transfer == 0) {
        if (initialized)
            for (unsigned index = 0; index < header->seed_words; ++index)
                if (terminal[index] != st[index]) header->discontinuity = 1;
        return;
    }
    const unsigned stride = 4 + 2 * header->seed_words;
    const unsigned end = 16 + 2 * header->seed_words + header->recorded * header->record_words;
    const unsigned start = DP_TRACE_WORDS - header->checkpoints * stride;
    if (start - end < stride || header->dropped_lo != 0 || header->dropped_hi != 0 || header->dropped_checkpoints != 0) {
        if (header->dropped_checkpoints != 0xffffffffu) ++header->dropped_checkpoints;
        return;
    }
    unsigned* checkpoint = trace + start - stride;
    checkpoint[0] = header->recorded;
    checkpoint[1] = 1;
    checkpoint[2] = header->seed_words;
    checkpoint[3] = header->checkpoints;
    for (unsigned index = 0; index < header->seed_words; ++index) {
        if (!initialized) seed[index] = terminal[index] = st[index];
        checkpoint[4 + index] = terminal[index];
        checkpoint[4 + header->seed_words + index] = st[index];
        terminal[index] = st[index];
    }
    ++header->checkpoints;
}

__device__ __forceinline__ unsigned* dp_trace_begin(const int* picks, int n, unsigned* trace,
                                                    const unsigned* st, int recent_enabled, int lines) {
    if (trace == nullptr) return nullptr;
    DpTraceHeader* header = reinterpret_cast<DpTraceHeader*>(trace);
    if (!dp_trace_valid(header, lines)) return nullptr;
    if (n <= 0 || n % 11 != 0 || static_cast<unsigned>(n / 11) > header->max_rows) {
        header->invalid_launch = 1;
        return nullptr;
    }
    const unsigned end = 16 + 2 * header->seed_words + header->recorded * header->record_words;
    const unsigned start = DP_TRACE_WORDS - header->checkpoints * (4 + 2 * header->seed_words);
    if (start - end < header->record_words || header->dropped_checkpoints != 0) {
        unsigned long long dropped = static_cast<unsigned long long>(header->dropped_hi) << 32 | header->dropped_lo;
        if (dropped != ~0ull) ++dropped;
        header->dropped_lo = static_cast<unsigned>(dropped);
        header->dropped_hi = static_cast<unsigned>(dropped >> 32);
        return nullptr;
    }
    unsigned* seed = trace + 16;
    unsigned* terminal = seed + header->seed_words;
    for (unsigned index = 0; index < header->seed_words; ++index) {
        if (header->recorded == 0 && header->checkpoints == 0) seed[index] = st[index];
        else if (terminal[index] != st[index]) header->discontinuity = 1;
    }
    unsigned* round = terminal + header->seed_words + header->recorded * header->record_words;
    round[0] = static_cast<unsigned>(n / 11);
    round[1] = 0;
    round[2] = 0;
    round[3] = static_cast<unsigned>(recent_enabled);
    unsigned seen[DP_EXPERTS / 32] = {};
    unsigned* raw = round + 15;
    for (int pick = 0; pick < n; ++pick) {
        const int expert = picks[pick];
        raw[pick] = static_cast<unsigned>(expert);
        if (expert < 0 || expert >= DP_SHARED) continue;
        round[1] += 1;
        const unsigned bit = 1u << (expert & 31);
        if ((seen[expert >> 5] & bit) == 0) round[2] += 1;
        seen[expert >> 5] |= bit;
    }
    return round;
}

__device__ __forceinline__ void dp_trace_end(unsigned* trace, unsigned* round, const unsigned* st,
                                             const DpMisc* misc, const int* picks, int n,
                                             const int* ovf, int n_ovf) {
    if (round == nullptr) return;
    DpTraceHeader* header = reinterpret_cast<DpTraceHeader*>(trace);
    const unsigned* counters = reinterpret_cast<const unsigned*>(misc);
    for (unsigned index = 0; index < 10; ++index) round[4 + index] = counters[index];
    unsigned mapped_experts[DP_EXPERTS / 32] = {};
    for (int mapped = 0; mapped < n_ovf; ++mapped) {
        const int expert = ovf[mapped];
        mapped_experts[expert >> 5] |= 1u << (expert & 31);
    }
    unsigned mapped_picks = 0;
    for (int pick = 0; pick < n; ++pick) {
        const int expert = picks[pick];
        if (expert >= 0 && expert < DP_SHARED && (mapped_experts[expert >> 5] & (1u << (expert & 31))))
            ++mapped_picks;
    }
    round[14] = mapped_picks;
    unsigned* terminal = trace + 16 + header->seed_words;
    for (unsigned index = 0; index < header->seed_words; ++index) terminal[index] = st[index];
    header->recorded += 1;
}

__device__ __forceinline__ bool dp_colder(unsigned ca, unsigned sa, int ia, unsigned cb, unsigned sb, int ib) {
    if (ca != cb) return ca < cb;
    if (sa != sb) return sa < sb;
    return ia > ib;
}

// One block of 256 threads. picks [n] int32 (-1 skipped, 512 the shared expert, masked to the rank's ids at tp=2).
extern "C" __global__ void __launch_bounds__(256) fn_exl3_dplan(
    const int* __restrict__ picks, int n, unsigned* __restrict__ st, int lines, int recent_enabled, long long* __restrict__ tab0,
    long long* __restrict__ tab1, long long* __restrict__ tab2, int* __restrict__ k2t0, int* __restrict__ k2t1,
    int* __restrict__ k2t2, const long long* __restrict__ src, const long long* __restrict__ lens,
    const int* __restrict__ k2, unsigned long long line_base, unsigned long long line_bytes, unsigned long long off0,
    unsigned long long off1, unsigned long long off2, long long* __restrict__ work, int* __restrict__ work_n,
    int* __restrict__ hit_picks, int* __restrict__ miss_picks, int* __restrict__ ovf, int* __restrict__ ovf_n,
    unsigned* __restrict__ trace) {
    int* line_id = reinterpret_cast<int*>(st);
    unsigned* stamp = st + lines;
    int* line_of = reinterpret_cast<int*>(st + 2 * lines);
    unsigned* counts = st + 2 * lines + DP_EXPERTS;
    unsigned* recent_counts = counts + DP_EXPERTS;
    unsigned* recent_rows = recent_counts + DP_EXPERTS;
    DpMisc* misc = reinterpret_cast<DpMisc*>(recent_rows + DP_RECENT_WORDS);
    const unsigned* score = recent_enabled ? recent_counts : counts;

    __shared__ unsigned short uniq[DP_MAXU];
    __shared__ unsigned short app[DP_MAXU];
    __shared__ unsigned short miss[DP_MAXU];
    __shared__ unsigned keep[DP_EXPERTS / 32];
    __shared__ int n_uniq, n_miss, n_copy, n_ovf;
    __shared__ unsigned now;
    __shared__ unsigned rkey_c[256], rkey_s[256];
    __shared__ int rkey_i[256];
    const int t = threadIdx.x;
    unsigned* trace_round = nullptr;

    for (int word = t; word < DP_EXPERTS / 32; word += blockDim.x) keep[word] = 0;
    __syncthreads();
    if (t == 0) {
        trace_round = dp_trace_begin(picks, n, trace, st, recent_enabled, lines);
        now = ++misc->now;
        misc->rounds += 1;
        if (recent_enabled) {
            const int rows = n / 11;
            for (int row = 0; row < rows; ++row) {
                unsigned* expired = recent_rows + misc->recent_cursor * DP_RECENT_WORDS_PER_ROW;
                for (int word = 0; word < DP_RECENT_WORDS_PER_ROW; ++word) {
                    unsigned removed = expired[word];
                    while (removed != 0) {
                        const int bit = __ffs(removed) - 1;
                        const int expert = word * 32 + bit;
                        recent_counts[expert] -= 1;
                        removed &= removed - 1;
                    }
                    expired[word] = 0;
                }
                for (int slot = 0; slot < 11; ++slot) {
                    const int expert = picks[row * 11 + slot];
                    if (expert < 0 || expert >= DP_SHARED) continue;
                    const unsigned bit = 1u << (expert & 31);
                    unsigned& word = expired[expert >> 5];
                    if ((word & bit) == 0) {
                        word |= bit;
                        recent_counts[expert] += 1;
                    }
                }
                misc->recent_cursor = (misc->recent_cursor + 1) % DP_RECENT_WINDOW;
                if (misc->recent_filled < DP_RECENT_WINDOW) misc->recent_filled += 1;
            }
        }
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
                const unsigned cx = score[uniq[x]], cy = score[uniq[y]];
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
    // the split (optional): picks held when the round started (and the shared expert) vs the misses this round copies.
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
            const unsigned c = score[id], s = stamp[i];
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
                // no line is free of this round's experts: the expert is read in place from the mapped host bank.
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
        dp_trace_end(trace, trace_round, st, misc, picks, n, ovf, n_ovf);
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
extern "C" __global__ void fn_exl3_draft_token(const float* __restrict__ best, const unsigned* __restrict__ ids,
                                               int* __restrict__ token) {
    if (threadIdx.x == 0 && blockIdx.x == 0) token[0] = static_cast<int>(ids[__float_as_int(best[2])]);
}

// The chain's step decision on the device: the next token as fn_exl3_draft_token, then upstream's stop rule
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
