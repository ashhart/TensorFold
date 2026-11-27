// Inkling 2's DSpark drafter, every stream of a pass in one launch a step (Mini's batched pass; dspark_many.zig).
#pragma once

// One stream of a batched pass (dspark_many.zig's Stream): its slot's caches by layer, f_rnn, anchor, keys, race.
struct DspStream {
    uint16_t* kc[8];
    uint16_t* vc[8];
    const uint16_t* frnn;
    unsigned long long seed;
    int p, race;
    float inv_t;
    int lo[8];
    int pad;
};

namespace ink2m {

__device__ __forceinline__ bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

// sample.cu's keys (as ops.cu's tf_ink2_race1)
__device__ __forceinline__ unsigned long long mix(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float race_uniform(unsigned long long seed, unsigned pos, unsigned id) {
    unsigned long long x = mix(seed + 0x9E3779B97F4A7C15ull);
    x = mix(x ^ (static_cast<unsigned long long>(pos) * 0xD1B54A32D192ED03ull));
    x = mix(x ^ static_cast<unsigned long long>(id));
    const float u = (static_cast<float>(static_cast<unsigned>(x >> 40)) + 0.5f) * (1.0f / 16777216.0f);
    return fminf(u, __uint_as_float(0x3F7FFFFFu));
}

}  // namespace ink2m

// tf_ink2_dsp_qk over every stream's rows: row r is stream at[2r]'s position at[2r + 1], into that slot's caches.
extern "C" __global__ void __launch_bounds__(64) tf_ink2_dsp_qk_many(QkArgs a, const DspStream* __restrict__ st,
                                                                     const int* __restrict__ at, int layer) {
    using namespace ink2d;
    __shared__ float sh[32];
    const int row = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const bool isq = h < a.hq;
    const uint16_t* x = a.x + row * a.ld + static_cast<long long>(h) * kD;
    const float x0 = bf(x[i]), x1 = bf(x[i + 64]);
    const float inv = 1.0f / sqrtf(blockSum(x0 * x0 + x1 * x1, sh) / kD + a.eps);
    const uint16_t* w = isq ? a.qw : a.kw;
    const float y0 = bfr(x0 * inv * bf(w[i])), y1 = bfr(x1 * inv * bf(w[i + 64]));
    const int pos = at[2 * row + 1];
    const float f = static_cast<float>(pos) * (1.0f / powf(a.theta, static_cast<float>(2 * i) / kD));
    const float co = cosf(f), si = sinf(f);
    const uint16_t o0 = tobf(y0 * co - y1 * si), o1 = tobf(y1 * co + y0 * si);
    if (isq) {
        uint16_t* q = a.q + (static_cast<size_t>(row) * a.hq + h) * kD;
        q[i] = o0;
        q[i + 64] = o1;
        return;
    }
    const DspStream& s = st[at[2 * row]];
    uint16_t* kc = s.kc[layer];
    uint16_t* vc = s.vc[layer];
    const int j = h - a.hq;
    const size_t o = (static_cast<size_t>(pos % a.ring) * a.hk + j) * kD;
    kc[o + i] = o0;
    kc[o + i + 64] = o1;
    const uint16_t* v = a.x + row * a.ld + static_cast<long long>(a.hq + a.hk + j) * kD;
    vc[o + i] = v[i];
    vc[o + i + 64] = v[i + 64];
}

// tf_ink2_dsp_attn for every stream's block (blockIdx.z): its keys [lo, p + T - 1], partials maxc chunks apart.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_dsp_attn_many(AttnArgs a, const DspStream* __restrict__ st, int layer,
                                                                        int maxc) {
    using namespace ink2d;
    __shared__ float sq[kMaxQ][kD];
    __shared__ uint16_t sk[kKT][kD + 8];
    __shared__ uint16_t sv[kKT][kD];
    __shared__ float sp[kMaxQ][kKT];
    __shared__ float sm[kMaxQ], sl[kMaxQ], sa[kMaxQ];
    const DspStream& S = st[blockIdx.z];
    const int lo = S.lo[layer], hi = S.p + a.T - 1;
    if (static_cast<int>(blockIdx.x) > hi / kChunk - lo / kChunk) return;
    const int j = blockIdx.y, c = lo / kChunk + blockIdx.x, tid = threadIdx.x;
    const int g = a.hq / a.hk, nq = a.T * g, nqt = a.T * a.hq;
    const uint16_t* aq = a.q + static_cast<size_t>(blockIdx.z) * nqt * kD;
    const uint16_t* kc = S.kc[layer];
    const uint16_t* vc = S.vc[layer];
    float* po = a.po + static_cast<size_t>(blockIdx.z) * maxc * nqt * kD;
    float* pml = a.pml + static_cast<size_t>(blockIdx.z) * maxc * nqt * 2;
    const int k0 = max(lo, c * kChunk), k1 = min(hi + 1, (c + 1) * kChunk);
    for (int idx = tid; idx < nq * kD; idx += blockDim.x) {
        const int qi = idx / kD, d = idx % kD;
        sq[qi][d] = bf(aq[(static_cast<size_t>(qi / g) * a.hq + j * g + qi % g) * kD + d]);
    }
    for (int qi = tid; qi < nq; qi += blockDim.x) {
        sm[qi] = -INFINITY;
        sl[qi] = 0.0f;
    }
    const int d = tid & (kD - 1), half = tid >> 7;
    float o[kMaxQ / 2];
    for (int r = 0; r < kMaxQ / 2; ++r) o[r] = 0.0f;
    __syncthreads();
    for (int kt = k0; kt < k1; kt += kKT) {
        const int nk = min(kKT, k1 - kt);
        for (int idx = tid; idx < nk * kD; idx += blockDim.x) {
            const int kk = idx / kD, dd = idx % kD;
            const size_t at = (static_cast<size_t>((kt + kk) % a.ring) * a.hk + j) * kD + dd;
            sk[kk][dd] = kc[at];
            sv[kk][dd] = vc[at];
        }
        __syncthreads();
        for (int idx = tid; idx < nq * kKT; idx += blockDim.x) {
            const int qi = idx / kKT, kk = idx % kKT;
            float s = -INFINITY;
            if (kk < nk) {
                s = 0.0f;
                for (int dd = 0; dd < kD; ++dd) s += sq[qi][dd] * bf(sk[kk][dd]);
                s *= a.scale;
            }
            sp[qi][kk] = s;
        }
        __syncthreads();
        for (int qi = tid; qi < nq; qi += blockDim.x) {
            float mx = sm[qi];
            for (int kk = 0; kk < nk; ++kk) mx = fmaxf(mx, sp[qi][kk]);
            const float al = expf(sm[qi] - mx);
            float sum = 0.0f;
            for (int kk = 0; kk < nk; ++kk) {
                const float p = expf(sp[qi][kk] - mx);
                sp[qi][kk] = p;
                sum += p;
            }
            sl[qi] = sl[qi] * al + sum;
            sm[qi] = mx;
            sa[qi] = al;
        }
        __syncthreads();
        for (int r = 0; r < kMaxQ / 2; ++r) {
            const int qi = half * (kMaxQ / 2) + r;
            if (qi >= nq) break;
            float acc = o[r] * sa[qi];
            for (int kk = 0; kk < nk; ++kk) acc += sp[qi][kk] * bf(sv[kk][d]);
            o[r] = acc;
        }
        __syncthreads();
    }
    for (int r = 0; r < kMaxQ / 2; ++r) {
        const int qi = half * (kMaxQ / 2) + r;
        if (qi >= nq) break;
        const size_t rh = static_cast<size_t>(blockIdx.x) * nqt + (qi / g) * a.hq + j * g + qi % g;
        po[rh * kD + d] = o[r];
        if (d == 0) {
            pml[rh * 2] = sm[qi];
            pml[rh * 2 + 1] = sl[qi];
        }
    }
}

// tf_ink2_dsp_merge for every stream (blockIdx.y): its chunks folded in order into its block's rows of out.
extern "C" __global__ void __launch_bounds__(128) tf_ink2_dsp_merge_many(const float* __restrict__ po, const float* __restrict__ pml,
                                                                         const DspStream* __restrict__ st, int layer, int T,
                                                                         int maxc, int nqt, uint16_t* __restrict__ out) {
    using namespace ink2d;
    const DspStream& S = st[blockIdx.y];
    const int chunks = (S.p + T - 1) / kChunk - S.lo[layer] / kChunk + 1;
    po += static_cast<size_t>(blockIdx.y) * maxc * nqt * kD;
    pml += static_cast<size_t>(blockIdx.y) * maxc * nqt * 2;
    out += static_cast<size_t>(blockIdx.y) * nqt * kD;
    const size_t rh = blockIdx.x;
    const int d = threadIdx.x;
    float mx = -INFINITY;
    for (int c = 0; c < chunks; ++c) mx = fmaxf(mx, pml[(c * nqt + rh) * 2]);
    float l = 0.0f, acc = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        const float m = pml[(c * nqt + rh) * 2];
        const float wt = m == -INFINITY ? 0.0f : expf(m - mx);
        l += pml[(c * nqt + rh) * 2 + 1] * wt;
        acc += po[(c * nqt + rh) * kD + d] * wt;
    }
    out[rh * kD + d] = tobf(acc / l);
}

// tf_ink2_dsp_ctx for every stream (blockIdx.y): its hidden rows hstride elements apart, its slot's f_rnn.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_dsp_ctx_many(const uint16_t* __restrict__ hidden, long long hstride,
                                                                       const DspStream* __restrict__ st, const uint16_t* __restrict__ W,
                                                                       const uint16_t* __restrict__ bias, float* __restrict__ out,
                                                                       int rows, int C, int R) {
    using namespace ink2d;
    const int o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (o >= 3 * R) return;
    hidden += blockIdx.y * hstride;
    out += static_cast<size_t>(blockIdx.y) * rows * 3 * R;
    const uint16_t* frnn = st[blockIdx.y].frnn;
    const uint16_t* w = W + static_cast<size_t>(o) * (2 * R + 2 * C) + 2 * R;
    float acc[kMaxRows];
    for (int r = 0; r < kMaxRows; ++r) acc[r] = 0.0f;
    for (int k = lane; k < 2 * C; k += 32) {
        const float wk = bf(w[k]);
        if (k < C) {
            for (int r = 0; r < kMaxRows; ++r)
                if (r < rows) acc[r] += wk * bf(hidden[static_cast<size_t>(r) * C + k]);
        } else {
            const float f = wk * bf(frnn[k - C]);
            for (int r = 0; r < kMaxRows; ++r) acc[r] += f;
        }
    }
    for (int r = 0; r < kMaxRows; ++r) {
        const float s = tf_common::warp_sum(acc[r]);
        if (lane == 0 && r < rows) out[static_cast<size_t>(r) * 3 * R + o] = s + bf(bias[o]);
    }
}

// tf_ink2_dsp_rnn for every stream (a block each): ctx rows cstride apart, walks T apart, states and latents R apart.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_dsp_rnn_many(const float* __restrict__ ctx, long long cstride,
                                                                       const uint16_t* __restrict__ W, const uint16_t* __restrict__ w1,
                                                                       const int* __restrict__ walk, int T, int j,
                                                                       uint16_t* __restrict__ state, uint16_t* __restrict__ latent,
                                                                       int R, int C) {
    using namespace ink2d;
    __shared__ float in[512];
    __shared__ float raw[768];
    const int c = threadIdx.x, lane = c & 31, wp = c >> 5, nw = blockDim.x >> 5;
    ctx += blockIdx.x * cstride;
    walk += blockIdx.x * T;
    state += blockIdx.x * R;
    latent += blockIdx.x * R;
    const size_t prev = static_cast<size_t>(walk[j]);
    in[c] = bf(state[c]);
    in[R + c] = bf(w1[prev * R + c]);
    __syncthreads();
    const int ld = 2 * R + 2 * C;
    for (int o = wp; o < 3 * R; o += nw) {
        float s = 0.0f;
        for (int k = lane; k < 2 * R; k += 32) s += bf(W[static_cast<size_t>(o) * ld + k]) * in[k];
        s = tf_common::warp_sum(s);
        if (lane == 0) raw[o] = s + ctx[o];
    }
    __syncthreads();
    const float gate = bfr(1.0f / (1.0f + expf(-bfr(raw[c]))));
    const float cand = bfr(tanhf(bfr(raw[R + c])));
    const float keep = bfr(gate * in[c]);
    const float upd = bfr(bfr(1.0f - gate) * cand);
    state[c] = tobf(keep + upd);
    latent[c] = tobf(tanhf(bfr(raw[2 * R + c])));
}

// Each stream's step pick (a block each): greedy over nv columns or its keyed race over valid, into picks 16 B apart.
extern "C" __global__ void __launch_bounds__(1024) tf_ink2_dsp_pick_many(const float* __restrict__ scores, int nv, int valid,
                                                                         int base, const DspStream* __restrict__ st, int step,
                                                                         float* __restrict__ picks) {
    using namespace ink2d;
    using ink2m::better;
    __shared__ float sv[32];
    __shared__ int si[32];
    const DspStream& S = st[blockIdx.x];
    const float* r = scores + static_cast<size_t>(blockIdx.x) * nv;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    if (S.race) {
        const unsigned pos = static_cast<unsigned>(S.p + 1 + step);
        for (int i = threadIdx.x; i < valid; i += blockDim.x) {
            const float v = bfr(r[i]) * S.inv_t - logf(-logf(ink2m::race_uniform(S.seed, pos, static_cast<unsigned>(base + i))));
            if (better(v, i, bv, bi)) {
                bv = v;
                bi = i;
            }
        }
    } else {
        for (int i = threadIdx.x; i < nv; i += blockDim.x) {
            if (better(r[i], i, bv, bi)) {
                bv = r[i];
                bi = i;
            }
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (warp != 0) return;
    const int nw = blockDim.x >> 5;
    bv = lane < nw ? sv[lane] : -INFINITY;
    bi = lane < nw ? si[lane] : 0x7fffffff;
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        picks[blockIdx.x * 4] = bv;
        reinterpret_cast<int*>(picks)[blockIdx.x * 4 + 1] = (bi == 0x7fffffff ? 0 : bi) + base;
    }
}
