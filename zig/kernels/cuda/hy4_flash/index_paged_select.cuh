// Hy4-flash compressed indexer: radix top-511 + own-group selection and direct 2,048-token expansion.
#pragma once

#include "../common/qmm_frag.cuh"
#include "index_paged_common.cuh"

namespace tf_hy4_idx {

constexpr int kTopG = 511;
constexpr int kListW = 512;
constexpr int kPer = 8;
constexpr int kTSel = 1024;
constexpr int kSCap = 8192;
constexpr int kCap = 4096;
constexpr int kTScan = 256;
constexpr int kSpan = 2048;
constexpr int kNS8 = 2048;
constexpr int kFloorPasses = 2;
constexpr int kTarget = 2048;

// Score bits as uint32 whose unsigned order matches scores (-0 as +0, every NaN above +inf; kBadNan drops NaN to 0).
template <int F>
__device__ __forceinline__ uint32_t order_key(float s) {
    if constexpr (F == kBadNan) {
        if (s != s) s = 0.0f;
    }
    if (s != s) return 0xFFFFFFFFu;
    const int b = __float_as_int(__fadd_rn(s, 0.0f));
    return static_cast<uint32_t>(b ^ ((b >> 31) | static_cast<int>(0x80000000u)));
}

template <int P>
__device__ __forceinline__ void hist_add_n(int* hist, const int (&b)[P]) {
#pragma unroll
    for (int j = 0; j < P; ++j)
        if (b[j] < 256) atomicAdd(&hist[b[j]], 1);
}

__device__ __forceinline__ void digit_of(const int* hist, int need, uint32_t prefix, int shift, uint32_t* sh_prefix,
                                         int* sh_need, int* sh_whole = nullptr) {
    const int lane = threadIdx.x & 31;
    const int4 lo = *reinterpret_cast<const int4*>(hist + 248 - 8 * lane);
    const int4 hi = *reinterpret_cast<const int4*>(hist + 252 - 8 * lane);
    const int h[8] = {hi.w, hi.z, hi.y, hi.x, lo.w, lo.z, lo.y, lo.x};
    int mine = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) mine += h[j];
    int upto = mine;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, upto, o);
        if (lane >= o) upto += v;
    }
    const unsigned hit = __ballot_sync(0xffffffffu, upto >= need);
    if (lane == (hit ? __ffs(hit) - 1 : 31)) {
        int above = upto - mine, digit = 0, cnt = 0;
        bool found = false;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (!found && above + h[j] >= need) {
                digit = 255 - 8 * lane - j;
                cnt = h[j];
                found = true;
            }
            if (!found) above += h[j];
        }
        *sh_need = need - above;
        *sh_prefix = prefix | (static_cast<uint32_t>(digit) << shift);
        if (sh_whole) *sh_whole = found && cnt == need - above;
    }
}

template <int T>
__device__ __forceinline__ int block_scan(int v, int (*wsum)[T / 32], int slot, int* total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += u;
    }
    if (lane == 31) wsum[slot][warp] = inc;
    __syncthreads();
    const int ws = lane < T / 32 ? wsum[slot][lane] : 0;
    int wi = ws;
#pragma unroll
    for (int o = 1; o < T / 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, wi, o);
        if (lane >= o) wi += u;
    }
    *total = __shfl_sync(0xffffffffu, wi, T / 32 - 1);
    return __shfl_sync(0xffffffffu, wi - ws, warp) + inc - v;
}

template <int T>
struct __align__(16) Sel {
    int hist[256];
    int wsum[2][T / 32];
    uint32_t prefix;
    int need, whole;
};

template <int F>
struct RowSrc {
    const float* s;
    int n;
    bool vec;
    __device__ __forceinline__ uint32_t key(int i) const { return order_key<F>(s[i]); }
    __device__ __forceinline__ int pool(int i) const { return i; }
};

struct SmemSrc {
    const uint32_t* key_;
    __device__ __forceinline__ uint32_t key(int i) const { return key_[i]; }
    __device__ __forceinline__ int pool(int i) const { return i; }
};

struct CandSrc {
    const uint32_t* key_;
    const int* idx;
    int n;
    __device__ __forceinline__ uint32_t key(int i) const { return key_[i]; }
    __device__ __forceinline__ int pool(int i) const { return idx[i]; }
};

template <int T, int F, class Src>
__device__ void select_core(Sel<T>& sh, const Src& src, int n_in, int K, int* __restrict__ out) {
    const int t = threadIdx.x;
    const int n = F == kBadTail ? (n_in > 64 ? n_in - 64 : max(0, n_in - 1)) : n_in;
    uint32_t prefix = 0, fixed = 0, thr;
    int need = K;
    for (int shift = 24;; shift -= 8) {
        for (int b = t; b < 256; b += T) sh.hist[b] = 0;
        __syncthreads();
        for (int c = 0; c < n; c += kPer * T) {
            int b[kPer];
#pragma unroll
            for (int j = 0; j < kPer; ++j) {
                const int i = c + t + j * T;
                const uint32_t u = i < n ? src.key(i) : 0u;
                b[j] = i < n && (u & fixed) == prefix ? static_cast<int>((u >> shift) & 0xFF) : 256;
            }
            hist_add_n<kPer>(sh.hist, b);
        }
        __syncthreads();
        if (t < 32) digit_of(sh.hist, need, prefix, shift, &sh.prefix, &sh.need, &sh.whole);
        __syncthreads();
        need = sh.need;
        prefix = sh.prefix;
        fixed |= 0xFFu << shift;
        if (shift == 0) {
            thr = prefix;
            break;
        }
        if (sh.whole && prefix != 0) {
            thr = prefix - 1;
            need = 0;
            break;
        }
    }
    int skip_eq = 0;
    if constexpr (F == kBadTies) {
        int eq_all = 0;
        for (int i = t; i < n; i += T) eq_all += (src.key(i) == thr);
        int tot_eq = 0;
        block_scan<T>(eq_all, sh.wsum, 0, &tot_eq);
        __syncthreads();
        skip_eq = max(0, tot_eq - need);
    }
    int gt_run = 0, eq_run = 0, slot = 0;
    for (int c = 0; c < n; c += kPer * T) {
        const int len = min(kPer * T, n - c), P = (len + T - 1) / T, i0 = c + t * P;
        uint32_t u[kPer];
#pragma unroll
        for (int j = 0; j < kPer; ++j) u[j] = j < P && i0 + j < c + len ? src.key(i0 + j) : 0u;
        int gcount = 0, ecount = 0;
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            const bool ok = j < P && i0 + j < c + len;
            gcount += ok && u[j] > thr;
            ecount += ok && u[j] == thr;
        }
        int total;
        const int ex = block_scan<T>((gcount << 16) | ecount, sh.wsum, slot, &total);
        slot ^= 1;
        int gb = gt_run + (ex >> 16), eb = eq_run + (ex & 0xFFFF);
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            if (j >= P || i0 + j >= c + len) break;
            if (u[j] > thr) {
                const int eq_taken = F == kBadTies ? min(max(0, eb - skip_eq), need) : min(eb, need);
                out[gb + eq_taken] = src.pool(i0 + j);
                gb++;
            } else if (u[j] == thr) {
                if constexpr (F == kBadTies) {
                    if (eb >= skip_eq && eb < skip_eq + need) out[gb + (eb - skip_eq)] = src.pool(i0 + j);
                } else {
                    if (eb < need) out[gb + eb] = src.pool(i0 + j);
                }
                eb++;
            }
        }
        gt_run += total >> 16;
        eq_run += total & 0xFFFF;
    }
    __syncthreads();
}

template <int T, int F>
__device__ __forceinline__ void select_row(Sel<T>& sh, uint32_t* __restrict__ keys, int cap,
                                           const float* __restrict__ row, int own, int* __restrict__ out) {
    const int t = threadIdx.x, K = min(kTopG, own);
    for (int k = t; k < kTopG; k += T) out[k] = -1;
    if (t == 0) out[kTopG] = own;
    if (K <= 0) return;
    const bool vec = (reinterpret_cast<uintptr_t>(row) & 15) == 0;
    if (own <= cap) {
        const int padded = (own + kPer - 1) / kPer * kPer;
        for (int c = 4 * t; c < padded; c += 4 * T) {
            if (vec && c + 4 <= own) {
                const float4 v = *reinterpret_cast<const float4*>(row + c);
                *reinterpret_cast<uint4*>(keys + c) = make_uint4(order_key<F>(v.x), order_key<F>(v.y),
                                                                 order_key<F>(v.z), order_key<F>(v.w));
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) keys[c + j] = c + j < own ? order_key<F>(row[c + j]) : 0u;
            }
        }
        select_core<T, F>(sh, SmemSrc{keys}, own, K, out);
    } else {
        select_core<T, F>(sh, RowSrc<F>{row, own, vec}, own, K, out);
    }
}

__device__ __forceinline__ void emit_selected(int i, int pos, const int* __restrict__ out_smem,
                                              int* __restrict__ groups_out, int* __restrict__ tokens_out) {
    __syncthreads();
    const int t = threadIdx.x;
    if (groups_out != nullptr && t < kListW) groups_out[static_cast<size_t>(i) * kListW + t] = out_smem[t];
    if (tokens_out != nullptr) {
        int* tok_dst = tokens_out + static_cast<size_t>(i) * (kListW * 4);
        for (int e = t; e < kListW * 4; e += kTSel) {
            const int g = out_smem[e >> 2];
            const int tok = 4 * g + (e & 3);
            tok_dst[e] = (g >= 0 && tok <= pos) ? tok : -1;
        }
    }
}

template <int T>
struct Merge {
    uint32_t key[kCap];
    int idx[kCap];
    Sel<T> sel;
};

template <int F>
__device__ __forceinline__ void scan_body(const int* __restrict__ SEL, const int* __restrict__ ROW_POS,
                                          const float* __restrict__ SC, int gcap, uint32_t* __restrict__ bits,
                                          int words) {
    __shared__ __align__(16) int hist[256];
    __shared__ uint32_t sh_prefix;
    __shared__ int sh_need;
    bump<F>(SC);
    const int i = blockIdx.y, r = SEL != nullptr ? SEL[i] : i, NP = ROW_POS[r] / 4, p0 = blockIdx.x * kSpan;
    if (NP <= kCap || p0 >= NP) return;
    const float* row = SC + static_cast<size_t>(i) * gcap;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    constexpr int PT = kNS8 / kTScan, WARPS = kTScan / 32, WPW = kSpan / 32 / WARPS;
    const int w0 = p0 / 32, nwc = min(kSpan / 32, (NP + 31) / 32 - w0);
    qmm_frag::grid_wait();
    qmm_frag::grid_launch();
    if constexpr (F == kBadTail) {
        if ((blockIdx.x + 1) * kSpan >= NP) {
            uint32_t* gt = bits + static_cast<size_t>(i) * 2 * words;
            for (int wi = t; wi < nwc; wi += kTScan) {
                gt[w0 + wi] = 0u;
                gt[words + w0 + wi] = 0u;
            }
            return;
        }
    }
    float v[WPW];
#pragma unroll
    for (int m = 0; m < WPW; ++m) {
        const int wi = warp + m * WARPS, k = (w0 + wi) * 32 + lane;
        v[m] = wi < nwc && k < NP ? row[k] : 0.0f;
    }
    uint32_t u[PT];
#pragma unroll
    for (int j = 0; j < PT; ++j) {
        const int s = t + j * kTScan;
        const int64_t k = static_cast<int64_t>(s / 8) * NP / (kNS8 / 8) / 8 * 8 + s % 8;
        u[j] = order_key<F>(k < NP ? row[k] : 0.0f);
    }
    uint32_t prefix = 0, fixed = 0;
    int need = max(static_cast<int>(static_cast<int64_t>(kTarget) * kNS8 / NP), 1);
#pragma unroll 1
    for (int pass = 0; pass < kFloorPasses; ++pass) {
        const int shift = 24 - 8 * pass;
        hist[t] = 0;
        __syncthreads();
        int b[PT];
#pragma unroll
        for (int j = 0; j < PT; ++j) b[j] = (u[j] & fixed) == prefix ? static_cast<int>((u[j] >> shift) & 0xFF) : 256;
        hist_add_n<PT>(hist, b);
        __syncthreads();
        if (warp == 0) digit_of(hist, need, prefix, shift, &sh_prefix, &sh_need);
        __syncthreads();
        prefix = sh_prefix;
        need = sh_need;
        fixed |= 0xFFu << shift;
    }
    const uint32_t lo = prefix;
    uint32_t* gt = bits + static_cast<size_t>(i) * 2 * words;
#pragma unroll
    for (int m = 0; m < WPW; ++m) {
        const int wi = warp + m * WARPS, k = (w0 + wi) * 32 + lane;
        const bool in = wi < nwc && k < NP;
        const uint32_t key = order_key<F>(v[m]);
        const unsigned gb = __ballot_sync(0xffffffffu, in && key > lo);
        const unsigned eb = __ballot_sync(0xffffffffu, in && key == lo);
        if (lane == 0 && wi < nwc) {
            gt[w0 + wi] = gb;
            gt[words + w0 + wi] = eb;
        }
    }
}

template <int F>
__device__ __forceinline__ void select_1cta_body(const int* __restrict__ SEL, const int* __restrict__ ROW_POS,
                                                 const float* __restrict__ SC, int gcap, int* __restrict__ groups_out,
                                                 int* __restrict__ tokens_out) {
    __shared__ __align__(16) uint32_t keys[kSCap];
    __shared__ Sel<kTSel> sh;
    __shared__ int out_smem[kListW];
    bump<F>(SC);
    const int i = blockIdx.x, r = SEL != nullptr ? SEL[i] : i, pos = ROW_POS[r];
    qmm_frag::grid_wait();
    select_row<kTSel, F>(sh, keys, kSCap, SC + static_cast<size_t>(i) * gcap, pos / 4, out_smem);
    emit_selected(i, pos, out_smem, groups_out, tokens_out);
}

template <int F>
__device__ __forceinline__ void merge_body(Merge<kTSel>& sm, int* __restrict__ out_smem, const int* __restrict__ SEL,
                                           const int* __restrict__ ROW_POS, const float* __restrict__ SC, int gcap,
                                           const uint32_t* __restrict__ bits, int words, int* __restrict__ groups_out,
                                           int* __restrict__ tokens_out, int* __restrict__ PATH) {
    bump<F>(SC);
    const int i = blockIdx.x, r = SEL != nullptr ? SEL[i] : i, pos = ROW_POS[r], NP = pos / 4, t = threadIdx.x;
    const float* row = SC + static_cast<size_t>(i) * gcap;
    qmm_frag::grid_wait();
    if (NP <= kCap) {
        select_row<kTSel, F>(sm.sel, sm.key, kCap, row, NP, out_smem);
        if (PATH != nullptr && t == 0) PATH[i] = 0;
        emit_selected(i, pos, out_smem, groups_out, tokens_out);
        return;
    }
    constexpr int K = kTopG;
    if (t < kTopG) out_smem[t] = -1;
    if (t == 0) out_smem[kTopG] = NP;
    const uint32_t* gt = bits + static_cast<size_t>(i) * 2 * words;
    const uint32_t* eq = gt + words;
    const int nw = (NP + 31) / 32, per = (nw + kTSel - 1) / kTSel;
    const int wa = min(nw, t * per), wb = min(nw, wa + per);
    int cg = 0, ce = 0;
    for (int w = wa; w < wb; ++w) {
        cg += __popc(gt[w]);
        ce += __popc(eq[w]);
    }
    int n_gt, n_eq;
    const int gt_before = block_scan<kTSel>(cg, sm.sel.wsum, 0, &n_gt);
    const int eq_before = block_scan<kTSel>(ce, sm.sel.wsum, 1, &n_eq);
    const int n_ge = n_gt + n_eq;
    int path;
    if (n_gt < K && K <= n_ge) {
        const int need = K - n_gt;
        const int eq_lo = F == kBadTies ? (n_eq - need) : 0;
        const int eq_hi = F == kBadTies ? n_eq : need;
        int seen = eq_before, cs = 0;
        for (int w = wa; w < wb; ++w) {
            const uint32_t e = eq[w];
            uint32_t take = 0;
            for (uint32_t m = e; m != 0; m &= m - 1, ++seen)
                if (seen >= eq_lo && seen < eq_hi) take |= m & (0u - m);
            cs += __popc(gt[w] | take);
        }
        int all;
        int at = block_scan<kTSel>(cs, sm.sel.wsum, 0, &all);
        seen = eq_before;
        for (int w = wa; w < wb; ++w) {
            const uint32_t e = eq[w];
            uint32_t take = 0;
            for (uint32_t m = e; m != 0; m &= m - 1, ++seen)
                if (seen >= eq_lo && seen < eq_hi) take |= m & (0u - m);
            for (uint32_t s = gt[w] | take; s != 0; s &= s - 1) {
                const int g = w * 32 + (__ffs(s) - 1);
                if (F == kBadTail && (g >= NP - 64 || at >= K - 1)) continue;
                out_smem[at++] = g;
            }
        }
        path = 3;
    } else if ((K <= n_ge && n_ge <= kCap) || (K <= n_gt && n_gt <= kCap)) {
        const bool with_eq = n_ge <= kCap;
        const int n = with_eq ? n_ge : n_gt;
        int at = with_eq ? gt_before + eq_before : gt_before;
        for (int w = wa; w < wb; ++w)
            for (uint32_t s = gt[w] | (with_eq ? eq[w] : 0u); s != 0; s &= s - 1) sm.idx[at++] = w * 32 + (__ffs(s) - 1);
        __syncthreads();
        for (int c = t; c < n; c += kTSel) sm.key[c] = order_key<F>(row[sm.idx[c]]);
        __syncthreads();
        select_core<kTSel, F>(sm.sel, CandSrc{sm.key, sm.idx, n}, n, K, out_smem);
        path = with_eq ? 1 : 4;
    } else {
        select_core<kTSel, F>(sm.sel, RowSrc<F>{row, NP, (reinterpret_cast<uintptr_t>(row) & 15) == 0}, NP, K,
                              out_smem);
        path = 2;
    }
    if (PATH != nullptr && t == 0) PATH[i] = path;
    emit_selected(i, pos, out_smem, groups_out, tokens_out);
}

}  // namespace tf_hy4_idx
