// The Metal engine's keyed draws (host references: lanes/gpu_rule.zig, gpu_full.zig), one 1,024-thread block a row.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace tf_sample {

constexpr unsigned TG = 1024;
constexpr unsigned NSG = TG / 32;
constexpr unsigned C = 1024;
constexpr unsigned FULL = 0xFFFFFFFFu;
// tokens this far (in logits / T) below the row's max never win the race: its noise spans under 19.5
constexpr float NEVER = 24.0f;

// A sequence's rule: its seed, 1 / T, top_p, the candidates' near window, ln(min_p) and top_k (0: off).
struct Rule {
    unsigned long long seed;
    float inv_t, top_p, near, min_log;
    unsigned top_k, pad;
};

__device__ __forceinline__ unsigned key(float v) {
    const unsigned b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ float val(unsigned k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k);
}

__device__ __forceinline__ unsigned long long mix(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float uniform(unsigned long long seed, unsigned pos, unsigned id) {
    unsigned long long x = mix(seed + 0x9E3779B97F4A7C15ull);
    x = mix(x ^ (static_cast<unsigned long long>(pos) * 0xD1B54A32D192ED03ull));
    x = mix(x ^ static_cast<unsigned long long>(id));
    return (static_cast<float>(static_cast<unsigned>(x >> 40)) + 0.5f) * (1.0f / 16777216.0f);
}

// uniform kept below 1: its top value rounds to 1.0, a score of +inf that would win from anywhere in the vocabulary
__device__ __forceinline__ float race_uniform(unsigned long long seed, unsigned pos, unsigned id) {
    return fminf(uniform(seed, pos, id), __uint_as_float(0x3F7FFFFFu));
}

// Metal's simd_sum as the host reference has it: a 32-lane butterfly, every lane ending with the sum.
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) x += __shfl_xor_sync(FULL, x, off);
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) x = fmaxf(x, __shfl_xor_sync(FULL, x, off));
    return x;
}

__device__ __forceinline__ unsigned warp_sum_u(unsigned x) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) x += __shfl_xor_sync(FULL, x, off);
    return x;
}

__device__ __forceinline__ unsigned warp_min_u(unsigned x) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) x = min(x, __shfl_xor_sync(FULL, x, off));
    return x;
}

__device__ __forceinline__ unsigned warp_prefix_exclusive(unsigned x, unsigned lane) {
    unsigned incl = x;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const unsigned y = __shfl_up_sync(FULL, incl, off);
        if (lane >= static_cast<unsigned>(off)) incl += y;
    }
    return incl - x;
}

// A place in the (value desc, id asc) order; END is past every token.
struct Mark {
    unsigned key, id;
};

__device__ __forceinline__ Mark end_mark() { return Mark{0u, 0xFFFFFFFFu}; }

__device__ __forceinline__ bool at_or_before(unsigned k, unsigned i, Mark m) { return k > m.key || (k == m.key && i <= m.id); }

__device__ __forceinline__ bool after(unsigned k, unsigned i, Mark m) { return k < m.key || (k == m.key && i > m.id); }

__device__ __forceinline__ bool beats(float s, unsigned k, unsigned i, float bs, unsigned bk, unsigned bi) {
    return s > bs || (s == bs && (k > bk || (k == bk && i < bi)));
}

struct Shared {
    float fsh[NSG], fsh2[NSG];
    unsigned ush[NSG], ush2[NSG];
    unsigned hist[256];
    unsigned st[4];
    unsigned ck[C], ci[C];
    unsigned fill[2];
};

// One row of logits as the block reads it, and the thread's place in it.
struct Row {
    const __nv_bfloat16* L;
    unsigned V;
    float inv_t;
    unsigned t, lane, sg;
    __device__ __forceinline__ float v(unsigned i) const { return __bfloat162float(L[i]) * inv_t; }
};

// A block sum in tf_gpu_sample's order: warp butterflies, then warps in order; every thread gets it.
__device__ __forceinline__ float group_sum(float x, const Row& r, Shared& s) {
    x = warp_sum(x);
    if (r.lane == 0) s.fsh[r.sg] = x;
    __syncthreads();
    float z = 0.0f;
    for (unsigned i = 0; i < NSG; i++) z += s.fsh[i];
    __syncthreads();
    return z;
}

__device__ __forceinline__ float row_max(const Row& r, Shared& s) {
    float lm = -INFINITY;
    for (unsigned i = r.t; i < r.V; i += TG) lm = fmaxf(lm, r.v(i));
    lm = warp_max(lm);
    if (r.lane == 0) s.fsh[r.sg] = lm;
    __syncthreads();
    float m = -INFINITY;
    for (unsigned i = 0; i < NSG; i++) m = fmaxf(m, s.fsh[i]);
    __syncthreads();
    return m;
}

// The radix select of the `want` largest: the key they reach, how many lie above it, ties' id cut.
struct Top {
    unsigned key, above, need, idcut;
};

__device__ Top top_keys(const Row& r, unsigned want, Shared& s) {
    const unsigned t = r.t;
    unsigned prefix = 0u, pmask = 0u, need = min(want, r.V), above = 0u, ties = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (t < 256) s.hist[t] = 0u;
        __syncthreads();
        for (unsigned i = t; i < r.V; i += TG) {
            const unsigned k = key(r.v(i));
            if ((k & pmask) == prefix) atomicAdd(&s.hist[(k >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (t == 0) {
            unsigned cum = 0u;
            int b = 255;
            for (; b > 0; b--) {
                const unsigned c = s.hist[b];
                if (cum + c >= need) break;
                cum += c;
            }
            s.st[0] = static_cast<unsigned>(b);
            s.st[1] = cum;
            s.st[2] = s.hist[b];
        }
        __syncthreads();
        prefix |= s.st[0] << shift;
        pmask |= 255u << shift;
        above += s.st[1];
        need -= s.st[1];
        ties = s.st[2];
        __syncthreads();
    }
    unsigned idcut = 0xFFFFFFFFu;
    if (ties > need) {
        unsigned ipre = 0u, imask = 0u, ineed = need;
        for (int shift = 16; shift >= 0; shift -= 8) {
            if (t < 256) s.hist[t] = 0u;
            __syncthreads();
            for (unsigned i = t; i < r.V; i += TG) {
                if (key(r.v(i)) == prefix && (i & imask) == ipre) atomicAdd(&s.hist[(i >> shift) & 255u], 1u);
            }
            __syncthreads();
            if (t == 0) {
                unsigned cum = 0u;
                unsigned b = 0u;
                for (; b < 255u; b++) {
                    const unsigned c = s.hist[b];
                    if (cum + c >= ineed) break;
                    cum += c;
                }
                s.st[0] = b;
                s.st[1] = cum;
            }
            __syncthreads();
            ipre |= s.st[0] << shift;
            imask |= 255u << shift;
            ineed -= s.st[1];
            __syncthreads();
        }
        idcut = ipre;
    }
    return Top{prefix, above, need, idcut};
}

// The candidates, sorted (value desc, id asc): a window near the max holding the rule's tokens, else the C largest.
__device__ unsigned first_block(const Row& r, float m, float top_p, float near, unsigned kw, float& z, Shared& s) {
    const unsigned t = r.t;
    float ls = 0.0f, lnear[3] = {0.0f, 0.0f, 0.0f};
    unsigned near_count[3] = {0u, 0u, 0u};
    for (unsigned i = t; i < r.V; i += TG) {
        const float v = r.v(i);
        const float e = expf(v - m);
        ls += e;
#pragma unroll
        for (int w = 0; w < 3; w++) {
            if (v >= m - near / static_cast<float>(1 << w)) {
                lnear[w] += e;
                near_count[w]++;
            }
        }
    }
    z = group_sum(ls, r, s);
    int window = -1;
    unsigned offset = 0u, n_near = 0u;
#pragma unroll
    for (int w = 0; w < 3; w++) {
        const float znear_part = warp_sum(lnear[w]);
        const unsigned before_in_warp = warp_prefix_exclusive(near_count[w], r.lane);
        const unsigned warp_count = warp_sum_u(near_count[w]);
        if (r.lane == 0) {
            s.fsh2[r.sg] = znear_part;
            s.ush[r.sg] = warp_count;
        }
        __syncthreads();
        float znear = 0.0f;
        unsigned off = before_in_warp, count = 0u;
        for (unsigned g = 0; g < NSG; g++) {
            znear += s.fsh2[g];
            if (g < r.sg) off += s.ush[g];
            count += s.ush[g];
        }
        __syncthreads();
        const bool holds = count <= C && (kw == 0u ? (top_p > 0.0f && top_p < 1.0f && znear >= (top_p + 1e-4f) * z)
                                                   : count >= min(kw, C));
        if (window < 0 && holds) {
            window = w;
            offset = off;
            n_near = count;
        }
    }
    const float floor_v = window < 0 ? INFINITY : m - near / static_cast<float>(1 << window);
    unsigned n_cand = 0u;
    unsigned sort_n = C;
    s.ck[t] = 0u;
    s.ci[t] = 0xFFFFFFFFu;
    __syncthreads();
    if (window >= 0) {
        unsigned at = offset;
        for (unsigned i = t; i < r.V; i += TG) {
            const float v = r.v(i);
            if (v >= floor_v) {
                s.ck[at] = key(v);
                s.ci[at] = i;
                at++;
            }
        }
        n_cand = n_near;
        sort_n = 32u;
        while (sort_n < n_near) sort_n <<= 1;
    } else {
        const Top top = top_keys(r, C, s);
        if (t == 0) {
            s.fill[0] = 0u;
            s.fill[1] = 0u;
        }
        __syncthreads();
        for (unsigned i = t; i < r.V; i += TG) {
            const unsigned k = key(r.v(i));
            if (k > top.key) {
                const unsigned at = atomicAdd(&s.fill[0], 1u);
                s.ck[at] = k;
                s.ci[at] = i;
            } else if (k == top.key && i <= top.idcut) {
                const unsigned at = top.above + atomicAdd(&s.fill[1], 1u);
                s.ck[at] = k;
                s.ci[at] = i;
            }
        }
        n_cand = top.above + top.need;
    }
    __syncthreads();
    for (unsigned k = 2; k <= sort_n; k <<= 1) {
        for (unsigned j = k >> 1; j > 0; j >>= 1) {
            const unsigned p = t ^ j;
            if (p > t && p < sort_n) {
                const unsigned ka = s.ck[t], kb = s.ck[p], ia = s.ci[t], ib = s.ci[p];
                const bool a_first = ka > kb || (ka == kb && ia < ib);
                if (a_first != ((t & k) == 0)) {
                    s.ck[t] = kb;
                    s.ck[p] = ka;
                    s.ci[t] = ib;
                    s.ci[p] = ia;
                }
            }
            __syncthreads();
        }
    }
    return min(n_cand, C);
}

// The probability mass (over `norm`) of the tokens after `b`, inside `top`, at or above `lo`, and inside `cut`.
__device__ float mass(const Row& r, float m, float norm, float lo, Mark b, Mark top, Mark cut, bool strict, Shared& s) {
    float x = 0.0f;
    for (unsigned i = r.t; i < r.V; i += TG) {
        const float v = r.v(i);
        if (!(v >= lo)) continue;
        const unsigned k = key(v);
        const bool inside = strict ? (k > cut.key || (k == cut.key && i < cut.id)) : at_or_before(k, i, cut);
        if (after(k, i, b) && at_or_before(k, i, top) && inside) x += expf(v - m) / norm;
    }
    return group_sum(x, r, s);
}

// The nucleus's last token past the block: the first place whose prefix reaches top_p, by bits of key then id.
__device__ Mark extend(const Row& r, float m, float norm, float lo, float top_p, float cum, Mark b, Mark top, Shared& s) {
    if (!(cum + mass(r, m, norm, lo, b, top, end_mark(), false, s) >= top_p)) return end_mark();
    unsigned k = 0u;
    for (int bit = 31; bit >= 0; bit--) {
        const unsigned cand = k | (1u << bit);
        if (cum + mass(r, m, norm, lo, b, top, Mark{cand, 0xFFFFFFFFu}, false, s) >= top_p) k = cand;
    }
    unsigned id = 0u;
    for (int bit = 31 - __clz(r.V - 1u); bit >= 0; bit--) {
        const unsigned cand = id | (1u << bit);
        if (cand >= r.V) continue;
        if (cum + mass(r, m, norm, lo, b, top, Mark{k, cand}, true, s) < top_p) id = cand;
    }
    return Mark{k, id};
}

// The nucleus's last token: the sequential sum over the first block, extended past it when it falls short.
__device__ Mark nucleus(const Row& r, float m, float lo, float top_p, float near, unsigned kc, bool whole, Mark top, Shared& s) {
    float z = 0.0f;
    const unsigned n = first_block(r, m, top_p, near, whole ? 0u : kc, z, s);
    const float norm = whole ? z : mass(r, m, 1.0f, -INFINITY, Mark{0xFFFFFFFFu, 0xFFFFFFFFu}, top, end_mark(), false, s);
    if (r.t == 0) {
        float cum = 0.0f;
        unsigned keep = 0u;
        for (unsigned j = 0; j < n; j++) {
            cum += expf(val(s.ck[j]) - m) / norm;
            if (cum >= top_p) {
                keep = j + 1;
                break;
            }
        }
        s.st[0] = keep;
        s.fsh2[0] = cum;
    }
    __syncthreads();
    const unsigned keep = s.st[0];
    const float cum = s.fsh2[0];
    const Mark last = keep > 0u ? Mark{s.ck[keep - 1u], s.ci[keep - 1u]} : Mark{s.ck[n - 1u], s.ci[n - 1u]};
    __syncthreads();
    return keep > 0u ? last : extend(r, m, norm, lo, top_p, cum, last, top, s);
}

// tf_gpu_sample: the top_k candidates, the top_p and min_p cuts, then the race; PROB (if any) gets the pick's share.
template <bool MAP>
__device__ void gpu_sample(const __nv_bfloat16* L, unsigned V, const Rule& rl, unsigned position, const long long* IDS,
                           unsigned* TOK, float* PROB, Shared& s) {
    const unsigned t = threadIdx.x, lane = t & 31u, sg = t >> 5;
    const Row r{L, V, rl.inv_t, t, lane, sg};
    const unsigned kc = rl.top_k;
    const unsigned cap = (kc == 0u || kc > C) ? C : kc;
    const float m = row_max(r, s);
    float z = 0.0f;
    const unsigned n_cand = first_block(r, m, rl.top_p, rl.near, kc, z, s);
    if (t == 0) {
        const unsigned n = min(n_cand, cap);
        float norm = z;
        if (kc != 0u) {
            norm = 0.0f;
            for (unsigned j = 0; j < n; j++) norm += expf(val(s.ck[j]) - m);
        }
        unsigned keep = n;
        if (rl.top_p > 0.0f && rl.top_p < 1.0f) {
            float cum = 0.0f;
            for (unsigned j = 0; j < n; j++) {
                cum += expf(val(s.ck[j]) - m) / norm;
                if (cum >= rl.top_p) {
                    keep = j + 1;
                    break;
                }
            }
        }
        if (rl.min_log > -INFINITY) {
            const float floor_p = m + rl.min_log;
            unsigned j = 0;
            while (j < keep && val(s.ck[j]) >= floor_p) j++;
            keep = j;
        }
        s.st[0] = keep;
        s.fsh2[0] = norm;
    }
    __syncthreads();
    const unsigned keep = s.st[0];
    const float norm = s.fsh2[0];
    float score = -INFINITY;
    unsigned best = 0xFFFFFFFFu;
    if (t < keep) {
        const unsigned id = MAP ? static_cast<unsigned>(IDS[s.ci[t]]) : s.ci[t];
        score = val(s.ck[t]) - logf(-logf(uniform(rl.seed, position, id)));
        best = t;
    }
    const float sm = warp_max(score);
    const unsigned pick = warp_min_u(score == sm ? best : 0xFFFFFFFFu);
    __syncthreads();
    if (lane == 0) {
        s.fsh[sg] = sm;
        s.ush[sg] = pick;
    }
    __syncthreads();
    if (t == 0) {
        float bs = -INFINITY;
        unsigned bj = 0xFFFFFFFFu;
        for (unsigned g = 0; g < NSG; g++) {
            if (s.fsh[g] > bs || (s.fsh[g] == bs && s.ush[g] < bj)) {
                bs = s.fsh[g];
                bj = s.ush[g];
            }
        }
        *TOK = MAP ? static_cast<unsigned>(IDS[s.ci[bj]]) : s.ci[bj];
        if (PROB) *PROB = expf(val(s.ck[bj]) - m) / norm;
    }
}

// The kept tokens' race over the whole row (value - log(-log(u)), ties to the earlier in the order).
template <bool MAP>
__device__ void race(const Row& r, float lo, Mark top, Mark cut, unsigned long long seed, unsigned position, const long long* IDS,
                     unsigned* TOK, Shared& s) {
    float bs = -INFINITY;
    unsigned bk = 0u, bi = 0xFFFFFFFFu;
    for (unsigned i = r.t; i < r.V; i += TG) {
        const float v = r.v(i);
        if (!(v >= lo)) continue;
        const unsigned k = key(v);
        if (!at_or_before(k, i, top) || !at_or_before(k, i, cut)) continue;
        const float score = v - logf(-logf(race_uniform(seed, position, MAP ? static_cast<unsigned>(IDS[i]) : i)));
        if (beats(score, k, i, bs, bk, bi)) {
            bs = score;
            bk = k;
            bi = i;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float os = __shfl_xor_sync(FULL, bs, off);
        const unsigned ok = __shfl_xor_sync(FULL, bk, off), oi = __shfl_xor_sync(FULL, bi, off);
        if (beats(os, ok, oi, bs, bk, bi)) {
            bs = os;
            bk = ok;
            bi = oi;
        }
    }
    __syncthreads();
    if (r.lane == 0) {
        s.fsh[r.sg] = bs;
        s.ush[r.sg] = bk;
        s.ush2[r.sg] = bi;
    }
    __syncthreads();
    if (r.t == 0) {
        for (unsigned g = 0; g < NSG; g++) {
            if (beats(s.fsh[g], s.ush[g], s.ush2[g], bs, bk, bi)) {
                bs = s.fsh[g];
                bk = s.ush[g];
                bi = s.ush2[g];
            }
        }
        const unsigned pick = min(bi, r.V - 1u);
        *TOK = MAP ? static_cast<unsigned>(IDS[pick]) : pick;
        s.st[3] = bk;
    }
}

// Full-row filters precede the keyed race; PROB reports the selected probability.
template <bool MAP>
__device__ void sample_full(const __nv_bfloat16* L, unsigned V, const Rule& rl, unsigned position, const long long* IDS,
                            unsigned* TOK, float* PROB, Shared& s) {
    const unsigned t = threadIdx.x, lane = t & 31u, sg = t >> 5;
    const Row r{L, V, rl.inv_t, t, lane, sg};
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
    if (r.t == 0) *PROB = expf(pick - m) / z; // the pick's share of the top_k mass at T
}

// The Metal engine's choice: tf_sample_full for top_k off or past 1,024, tf_gpu_sample for the rest.
template <bool MAP>
__device__ void draw(const __nv_bfloat16* L, unsigned V, const Rule* rule, const int* meta, int offset, unsigned* TOK,
                     const long long* IDS, float* PROB) {
    __shared__ Shared s;
    const unsigned row = blockIdx.x;
    const Rule rl = *rule;
    const unsigned position = static_cast<unsigned>(meta[0] + static_cast<int>(row) + 1 + offset);
    const __nv_bfloat16* L_row = L + static_cast<size_t>(row) * V;
    if (rl.top_k == 0u || rl.top_k > C) {
        sample_full<MAP>(L_row, V, rl, position, IDS, TOK + row, PROB ? PROB + row : nullptr, s);
    } else {
        gpu_sample<MAP>(L_row, V, rl, position, IDS, TOK + row, PROB ? PROB + row : nullptr, s);
    }
}

}  // namespace tf_sample

// Rows of bf16 logits [rows, V] drawn under RULE; row r keys position meta[0] + r + 1 + offset; PROB optional.
extern "C" __global__ void __launch_bounds__(1024) tf_draw(const __nv_bfloat16* L, unsigned V, const tf_sample::Rule* rule,
                                                           const int* meta, int offset, unsigned* TOK, float* PROB) {
    tf_sample::draw<false>(L, V, rule, meta, offset, TOK, nullptr, PROB);
}

// The same over a column subset whose token ids are IDS (int64, a draft head's vocabulary): noise and token by id.
extern "C" __global__ void __launch_bounds__(1024) tf_draw_ids(const __nv_bfloat16* L, unsigned V, const tf_sample::Rule* rule,
                                                               const int* meta, int offset, unsigned* TOK, const long long* IDS,
                                                               float* PROB) {
    tf_sample::draw<true>(L, V, rule, meta, offset, TOK, IDS, PROB);
}
