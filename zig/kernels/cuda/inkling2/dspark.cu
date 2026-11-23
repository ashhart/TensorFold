// Inkling 2's DSpark drafter: tap mix, q/k norm + RoPE into the draft caches, block attention, the Markov head.
#include <cstdint>
#include <cuda_bf16.h>

#include "common/dspark_pick.cuh"
#include "common/reduce.cuh"

namespace ink2d {

constexpr int kD = 128;
constexpr int kMaxQ = 48;
constexpr int kKT = 32;
constexpr int kChunk = 1024;
constexpr int kMaxRows = 8;

__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }

__device__ __forceinline__ uint16_t tobf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }

__device__ __forceinline__ float bfr(float f) { return bf(tobf(f)); }

// One sum over the block in a fixed order: warp shuffles, then warp 0 over the warps' partials.
__device__ __forceinline__ float blockSum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = tf_common::warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    const float t = lane < nw ? sh[lane] : 0.0f;
    return __shfl_sync(0xffffffffu, tf_common::warp_sum(t), 0);
}

}  // namespace ink2d

// out[row] = RMSNorm(sum_t alpha[t] * taps[row][t]) * w, the mix in fp32 in tap order (vLLM's mix_rmsnorm).
extern "C" __global__ void __launch_bounds__(512) tf_ink2_dsp_mix(const uint16_t* __restrict__ taps, const float* __restrict__ alpha,
                                                                  const uint16_t* __restrict__ w, uint16_t* __restrict__ out,
                                                                  int taps_n, int C, float eps) {
    using namespace ink2d;
    __shared__ float sh[32];
    const size_t row = blockIdx.x;
    const uint16_t* x = taps + row * taps_n * C;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float m = 0.0f;
        for (int t = 0; t < taps_n; ++t) m += bf(x[t * C + c]) * alpha[t];
        ss += m * m;
    }
    const float r = rsqrtf(blockSum(ss, sh) / C + eps);
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float m = 0.0f;
        for (int t = 0; t < taps_n; ++t) m += bf(x[t * C + c]) * alpha[t];
        out[row * C + c] = tobf(m * r * bf(w[c]));
    }
}

// Unlike core's add-norms: resid stays bf16 (+= fp32 partials in rank order); out = RMSNorm of the unrounded sum * w.
extern "C" __global__ void __launch_bounds__(512) tf_ink2_dsp_addnorm(uint16_t* __restrict__ resid, const float* __restrict__ parts,
                                                                      int nparts, long long part_stride, const uint16_t* __restrict__ w,
                                                                      uint16_t* __restrict__ out, int C, float eps) {
    using namespace ink2d;
    __shared__ float sh[32];
    const size_t row = blockIdx.x;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float s = 0.0f;
        for (int p = 0; p < nparts; ++p) s += parts[p * part_stride + row * C + c];
        const float v = bf(resid[row * C + c]) + s;
        ss += v * v;
    }
    const float r = rsqrtf(blockSum(ss, sh) / C + eps);
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float s = 0.0f;
        for (int p = 0; p < nparts; ++p) s += parts[p * part_stride + row * C + c];
        const float v = bf(resid[row * C + c]) + s;
        if (nparts > 0) resid[row * C + c] = tobf(v);
        out[row * C + c] = tobf(v * r * bf(w[c]));
    }
}

struct QkArgs {
    const uint16_t* x;  // bf16 [rows][ld]: q heads, then key heads, then value heads (128 wide each)
    const uint16_t* qw;
    const uint16_t* kw;
    uint16_t* q;   // bf16 [rows][hq][128]
    uint16_t* kc;  // bf16 [ring][hk][128]
    uint16_t* vc;
    long long ld;
    int hq, hk, ring, pos0;
    float eps, theta;
};

// Per (row, head): RMSNorm * weight rounded to bf16, then neox RoPE at pos0 + row; key and value heads to the caches.
extern "C" __global__ void __launch_bounds__(64) tf_ink2_dsp_qk(QkArgs a) {
    using namespace ink2d;
    __shared__ float sh[32];
    const int row = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const bool isq = h < a.hq;
    const uint16_t* x = a.x + row * a.ld + static_cast<long long>(h) * kD;
    const float x0 = bf(x[i]), x1 = bf(x[i + 64]);
    const float inv = 1.0f / sqrtf(blockSum(x0 * x0 + x1 * x1, sh) / kD + a.eps);
    const uint16_t* w = isq ? a.qw : a.kw;
    const float y0 = bfr(x0 * inv * bf(w[i])), y1 = bfr(x1 * inv * bf(w[i + 64]));
    const int pos = a.pos0 + row;
    const float f = static_cast<float>(pos) * (1.0f / powf(a.theta, static_cast<float>(2 * i) / kD));
    const float co = cosf(f), si = sinf(f);
    const uint16_t o0 = tobf(y0 * co - y1 * si), o1 = tobf(y1 * co + y0 * si);
    if (isq) {
        uint16_t* q = a.q + (static_cast<size_t>(row) * a.hq + h) * kD;
        q[i] = o0;
        q[i + 64] = o1;
        return;
    }
    const int j = h - a.hq;
    const size_t at = (static_cast<size_t>(pos % a.ring) * a.hk + j) * kD;
    a.kc[at + i] = o0;
    a.kc[at + i + 64] = o1;
    const uint16_t* v = a.x + row * a.ld + static_cast<long long>(a.hq + a.hk + j) * kD;
    a.vc[at + i] = v[i];
    a.vc[at + i + 64] = v[i + 64];
}

struct AttnArgs {
    const uint16_t* q;  // bf16 [T][hq][128]
    const uint16_t* kc;  // bf16 [ring][hk][128]
    const uint16_t* vc;
    float* po;   // fp32 [chunks][T * hq][128]: unnormalised sums
    float* pml;  // fp32 [chunks][T * hq][2]: max, total
    int T, hq, hk, ring, lo, hi, c0;
    float scale;
};

// A block's rows against keys [lo, hi], unmasked (the block sees itself whole); a 1,024-key chunk and key head a block.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_dsp_attn(AttnArgs a) {
    using namespace ink2d;
    __shared__ float sq[kMaxQ][kD];
    __shared__ uint16_t sk[kKT][kD + 8];
    __shared__ uint16_t sv[kKT][kD];
    __shared__ float sp[kMaxQ][kKT];
    __shared__ float sm[kMaxQ], sl[kMaxQ], sa[kMaxQ];
    const int j = blockIdx.y, c = a.c0 + blockIdx.x, tid = threadIdx.x;
    const int g = a.hq / a.hk, nq = a.T * g, nqt = a.T * a.hq;
    const int k0 = max(a.lo, c * kChunk), k1 = min(a.hi + 1, (c + 1) * kChunk);
    for (int idx = tid; idx < nq * kD; idx += blockDim.x) {
        const int qi = idx / kD, d = idx % kD;
        sq[qi][d] = bf(a.q[(static_cast<size_t>(qi / g) * a.hq + j * g + qi % g) * kD + d]);
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
            sk[kk][dd] = a.kc[at];
            sv[kk][dd] = a.vc[at];
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
        a.po[rh * kD + d] = o[r];
        if (d == 0) {
            a.pml[rh * 2] = sm[qi];
            a.pml[rh * 2 + 1] = sl[qi];
        }
    }
}

// Each (row, head)'s chunks folded in order: out bf16 [T][hq][128].
extern "C" __global__ void __launch_bounds__(128) tf_ink2_dsp_merge(const float* __restrict__ po, const float* __restrict__ pml,
                                                                    int chunks, int nqt, uint16_t* __restrict__ out) {
    using namespace ink2d;
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

// The Markov head's context: out fp32 [rows][3R] = bias + rnn_proj[:, 2R:] . [hidden_row, f_rnn], a warp an output.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_dsp_ctx(const uint16_t* __restrict__ hidden, const uint16_t* __restrict__ frnn,
                                                                  const uint16_t* __restrict__ W, const uint16_t* __restrict__ bias,
                                                                  float* __restrict__ out, int rows, int C, int R) {
    using namespace ink2d;
    const int o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (o >= 3 * R) return;
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

// One recurrence step (one block, R threads): raw = ctx + rnn_proj[:, :2R] . [state, w1[prev]]; gated as the bf16 ops.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_dsp_rnn(const float* __restrict__ ctx, const uint16_t* __restrict__ W,
                                                                  const uint16_t* __restrict__ w1, const int* __restrict__ walk, int j,
                                                                  uint16_t* __restrict__ state, uint16_t* __restrict__ latent, int R, int C) {
    using namespace ink2d;
    __shared__ float in[512];
    __shared__ float raw[768];
    const int c = threadIdx.x, lane = c & 31, wp = c >> 5, nw = blockDim.x >> 5;
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

// A step's logits over this rank's vocabulary: softcap(bf16(base + bf16(fold_bias + fold . latent))), -inf past valid.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_dsp_vocab(const uint16_t* __restrict__ base, const uint16_t* __restrict__ latent,
                                                                    const uint16_t* __restrict__ fold, const uint16_t* __restrict__ fbias,
                                                                    float cap, int valid, int nv, float* __restrict__ out) {
    using namespace ink2d;
    const int v = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (v >= nv) return;
    const uint16_t* f = fold + static_cast<size_t>(v) * 256;
    float s = 0.0f;
    for (int k = lane; k < 256; k += 32) s += bf(latent[k]) * bf(f[k]);
    s = tf_common::warp_sum(s);
    if (lane != 0) return;
    const float b = bfr(s + bf(fbias[v]));
    const float x = bfr(bf(base[v]) + b);
    out[v] = v < valid ? bfr(tanhf(x / cap) * cap) : -INFINITY;
}

// tf_ink2_dsp_vocab for `rows` streams at once (base rows base_ld apart): each fold row read once, sums as its order.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_dsp_vocab_rows(const uint16_t* __restrict__ base, long long base_ld,
                                                                         const uint16_t* __restrict__ latent,
                                                                         const uint16_t* __restrict__ fold,
                                                                         const uint16_t* __restrict__ fbias, float cap, int valid,
                                                                         int nv, int rows, float* __restrict__ out) {
    using namespace ink2d;
    const int v = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (v >= nv) return;
    const uint16_t* f = fold + static_cast<size_t>(v) * 256;
    float w[8];
    for (int q = 0; q < 8; ++q) w[q] = bf(f[lane + 32 * q]);
    const float fb = bf(fbias[v]);
    for (int r = 0; r < rows; ++r) {
        const uint16_t* lt = latent + static_cast<size_t>(r) * 256;
        float s = 0.0f;
        for (int q = 0; q < 8; ++q) s += bf(lt[lane + 32 * q]) * w[q];
        s = tf_common::warp_sum(s);
        if (lane == 0) {
            const float b = bfr(s + fb);
            const float x = bfr(bf(base[r * base_ld + v]) + b);
            out[static_cast<size_t>(r) * nv + v] = v < valid ? bfr(tanhf(x / cap) * cap) : -INFINITY;
        }
    }
}

// tf_ink2_dsp_pick_merge for `rows` streams, a block a stream: rank q's pick of row r at (q * rows + r) * 16 bytes.
extern "C" __global__ void tf_ink2_dsp_pick_rows(const unsigned char* __restrict__ picks, int world, int rows,
                                                 int* __restrict__ walk, int T, int j) {
    const int r = blockIdx.x;
    tf_common::dspark_pick<0x7fffffff>(picks + (size_t)r * 16, world, rows * 16, walk + (size_t)r * T, j);
}

// The ranks' (value, id) picks to walk[j + 1]: MiniMax M3's merge, with 0x7fffffff when no value beats -Inf.
extern "C" __global__ void tf_ink2_dsp_pick_merge(const unsigned char* __restrict__ picks, int world, int stride,
                                                  int* __restrict__ walk, int j) {
    tf_common::dspark_pick<0x7fffffff>(picks, world, stride, walk, j);
}

#include "inkling2/dspark_many.cuh"
