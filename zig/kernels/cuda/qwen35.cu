// Qwen3.5's hybrid text models on CUDA (the 2B and Qwen3.8-27B): the Metal engine's kernels (zig/kernels/metal/qwen3_5)
// with the same bf16 rounding points, their widths from the model's Geo. Rows of one forward belong to segments (one
// stream's window each); a segment's state lives in its stream's blocks.

#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_q35 {

constexpr int DK = 128;   // linear-attention key and value head width
constexpr int TAPS = 4;   // causal convolution taps
constexpr int HD = 256;   // attention head width
constexpr int MAXQ = 32;  // attention query heads at most

// The model's widths: hidden, linear key heads, linear value heads, attention query and key/value heads.
struct Geo {
    int hid, nk, nv, qh, kvh;
    __device__ int conv() const { return (2 * nk + nv) * DK; }  // q, k and v channels the convolution sees
    __device__ size_t rec() const { return (size_t)nv * DK * DK; }  // floats of one layer's recurrence
    __device__ size_t conv_rows() const { return (size_t)(TAPS - 1) * conv(); }
};

// One stream's window in this forward: its state blocks, cache length before the window, and rows.
struct Seg {
    unsigned long long rec;   // float recurrences, one rec() block a linear layer
    unsigned long long conv;  // bf16 convolution tails, conv_rows() a linear layer
    unsigned long long kv;    // bf16 keys then values, kvh * cap * HD each, an attention layer at a time
    int cap, len, row0, rows;
};

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int m = 16; m; m >>= 1) x += __shfl_xor_sync(0xffffffffu, x, m);
    return x;
}

}  // namespace tf_q35

using namespace tf_q35;

// MLX affine 4-bit token rows: out = bf16(s * q + b), D wide.
extern "C" __global__ void __launch_bounds__(256) tf_q35_embed(const int* __restrict__ ids, const uint8_t* __restrict__ W,
                                                                 const __nv_bfloat16* __restrict__ S, const __nv_bfloat16* __restrict__ B,
                                                                 __nv_bfloat16* __restrict__ out, int D) {
    const size_t id = ids[blockIdx.x];
    for (int b = threadIdx.x; b < D / 2; b += blockDim.x) {
        const uint8_t v = W[id * (D / 2) + b];
        const size_t g = id * (D / 64) + (2 * b) / 64;
        const float s = bf(S[g]), z = bf(B[g]);
        out[(size_t)blockIdx.x * D + 2 * b] = __float2bfloat16_rn(s * (float)(v & 15) + z);
        out[(size_t)blockIdx.x * D + 2 * b + 1] = __float2bfloat16_rn(s * (float)(v >> 4) + z);
    }
}

// h = bf16(h + r) (r may be null), x = bf16(w * (h * inv)), D wide (a multiple of 256, at most 8192). One 256-thread
// block a row; each thread's elements in index order, warps summed in order.
extern "C" __global__ void __launch_bounds__(256) tf_q35_norm(__nv_bfloat16* __restrict__ H, const __nv_bfloat16* __restrict__ R,
                                                               const __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ X, float eps, int D) {
    constexpr int MAXE = 32;
    __shared__ float red[8];
    const size_t row = blockIdx.x;
    const int t = threadIdx.x;
    const int E = D / 256;
    float hv[MAXE];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < MAXE; i++) {
        if (i >= E) break;
        const size_t at = row * D + t * E + i;
        float h = bf(H[at]);
        if (R != nullptr) {
            const __nv_bfloat16 hb = __float2bfloat16_rn(h + bf(R[at]));
            H[at] = hb;
            h = bf(hb);
        }
        hv[i] = h;
        ss = fmaf(h, h, ss);
    }
    ss = warp_sum(ss);
    if ((t & 31) == 0) red[t >> 5] = ss;
    __syncthreads();
    float total = 0.0f;
    for (int i = 0; i < 8; i++) total += red[i];
    const float inv = rsqrtf(total / (float)D + eps);
#pragma unroll
    for (int i = 0; i < MAXE; i++) {
        if (i >= E) break;
        const int d = t * E + i;
        X[row * D + d] = __float2bfloat16_rn(bf(W[d]) * (hv[i] * inv));
    }
}

// The 64-group sums of bf16 rows D wide (the decode projections' input sums). Grid (rows), a thread a group.
extern "C" __global__ void tf_q35_sums(const __nv_bfloat16* __restrict__ X, float* __restrict__ XS, int D) {
    const size_t row = blockIdx.x;
    for (int g = threadIdx.x; g < D / 64; g += blockDim.x) {
        const __nv_bfloat16* x = X + row * D + g * 64;
        float s = 0.0f;
        for (int j = 0; j < 64; j++) s += bf(x[j]);
        XS[row * (D / 64) + g] = s;
    }
}

// The causal convolution, silu, q/k normalization and the decay and write strength of each row's linear heads.
// Grid (1, 2 * nk + nv, rows), a warp a (row, head): heads [0, nk) are q, [nk, 2nk) k, then nv of v.
extern "C" __global__ void __launch_bounds__(32) tf_q35_gdn_pre(Geo geo, const __nv_bfloat16* __restrict__ QKV, const Seg* __restrict__ segs,
                                                                  const int* __restrict__ rowseg, int layer, const __nv_bfloat16* __restrict__ CW,
                                                                  const __nv_bfloat16* __restrict__ Ain, const __nv_bfloat16* __restrict__ Bin,
                                                                  const float* __restrict__ ALOG, const __nv_bfloat16* __restrict__ DT,
                                                                  __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K,
                                                                  __nv_bfloat16* __restrict__ V, float* __restrict__ G, __nv_bfloat16* __restrict__ BETA) {
    const int lane = threadIdx.x;
    const int head = blockIdx.y;
    const int w = blockIdx.z;
    const int NK = geo.nk, NV = geo.nv, C = geo.conv();
    const Seg sg = segs[rowseg[w]];
    const int r = w - sg.row0;
    const __nv_bfloat16* CS = reinterpret_cast<const __nv_bfloat16*>(sg.conv) + (size_t)layer * geo.conv_rows();
    const bool isq = head < NK, isk = !isq && head < 2 * NK;
    const int c0 = head * DK;  // q, k and v channels are consecutive heads of DK
    constexpr int PER = DK / 32;
    float vals[PER];
#pragma unroll
    for (int j = 0; j < PER; j++) {
        const int c = c0 + lane * PER + j;
        float acc = 0.0f;
        for (int tap = 0; tap < TAPS; tap++) {
            const int at = r + tap;  // in [state rows (TAPS - 1)] ++ [this segment's rows]
            const float xv = at < TAPS - 1 ? bf(CS[(size_t)at * C + c]) : bf(QKV[(size_t)(sg.row0 + at - (TAPS - 1)) * C + c]);
            acc += bf(CW[c * TAPS + tap]) * xv;
        }
        const float conv = rbf(acc);
        const float sig = rbf(1.0f / (1.0f + expf(-conv)));
        vals[j] = rbf(conv * sig);
    }
    if (isq || isk) {
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; j++) ss += vals[j] * vals[j];
        ss = warp_sum(ss);
        const float inv = rsqrtf(ss / (float)DK + (1e-6f / (float)DK));
        const float scale = isq ? rbf(1.0f / (float)DK) : rbf(rsqrtf((float)DK));
        __nv_bfloat16* out = isq ? Q + ((size_t)w * NK + head) * DK : K + ((size_t)w * NK + head - NK) * DK;
#pragma unroll
        for (int j = 0; j < PER; j++) out[lane * PER + j] = __float2bfloat16_rn(scale * rbf(vals[j] * inv));
    } else {
        const int hv = head - 2 * NK;
#pragma unroll
        for (int j = 0; j < PER; j++) V[((size_t)w * NV + hv) * DK + lane * PER + j] = __float2bfloat16_rn(vals[j]);
        if (lane == 0) {
            const float s = rbf(bf(Ain[w * NV + hv]) + bf(DT[hv]));
            const float sp = rbf(fmaxf(s, 0.0f) + logf(1.0f + expf(-fabsf(s))));
            G[w * NV + hv] = expf(-expf(ALOG[hv]) * sp);
            BETA[w * NV + hv] = __float2bfloat16_rn(1.0f / (1.0f + expf(-bf(Bin[w * NV + hv]))));
        }
    }
}

// Each segment's convolution tail after its rows: the last TAPS - 1 of [tail ++ rows]. A thread a (segment, channel).
// SNAP, if given, also takes the tail after each row (forward row m at m * conv_rows()), read before the update.
extern "C" __global__ void tf_q35_conv_commit(Geo geo, const __nv_bfloat16* __restrict__ QKV, const Seg* __restrict__ segs, int layer,
                                                __nv_bfloat16* __restrict__ SNAP) {
    const int C = geo.conv();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const Seg sg = segs[blockIdx.y];
    __nv_bfloat16* CS = reinterpret_cast<__nv_bfloat16*>(sg.conv) + (size_t)layer * geo.conv_rows();
    if (SNAP != nullptr)
        for (int j = 0; j < sg.rows; j++)
            for (int r = 0; r < TAPS - 1; r++) {
                const int at = r + j + 1;
                SNAP[((size_t)(sg.row0 + j) * (TAPS - 1) + r) * C + c] =
                    at < TAPS - 1 ? CS[(size_t)at * C + c] : QKV[(size_t)(sg.row0 + at - (TAPS - 1)) * C + c];
            }
    // ascending: slot r takes [r + rows], which is either a later state slot (not yet written) or a row
    for (int r = 0; r < TAPS - 1; r++) {
        const int at = r + sg.rows;
        CS[(size_t)r * C + c] = at < TAPS - 1 ? CS[(size_t)at * C + c] : QKV[(size_t)(sg.row0 + at - (TAPS - 1)) * C + c];
    }
}

// The gated delta rule over each segment's rows, its recurrence updated in place; SNAP, if given, takes the state after
// each row (forward row m at m * rec()). Value head hv reads key head hv / (nv / nk). Grid (1, DK / 4, nv * segments),
// block (32, 4): a warp a (value head, value row), its 32 lanes over the key width.
extern "C" __global__ void __launch_bounds__(128) tf_q35_gdn_chain(Geo geo, const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                                                                     const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                                                                     const __nv_bfloat16* __restrict__ beta, const Seg* __restrict__ segs, int layer,
                                                                     __nv_bfloat16* __restrict__ y, float* __restrict__ SNAP) {
    const int NK = geo.nk, NV = geo.nv;
    const int hv = blockIdx.z % NV;
    const int hk = hv / (NV / NK);
    const Seg sg = segs[blockIdx.z / NV];
    const int dk = threadIdx.x;
    const int dv = blockIdx.y * 4 + threadIdx.y;
    constexpr int PT = DK / 32;
    const size_t at = ((size_t)hv * DK + dv) * DK + PT * dk;
    float* st = reinterpret_cast<float*>(sg.rec) + (size_t)layer * geo.rec() + at;
    float state[PT];
#pragma unroll
    for (int i = 0; i < PT; i++) state[i] = st[i];
    for (int r = 0; r < sg.rows; r++) {
        const size_t row = sg.row0 + r;
        float kc[PT], qc[PT];
#pragma unroll
        for (int i = 0; i < PT; i++) {
            kc[i] = bf(k[(row * NK + hk) * DK + PT * dk + i]);
            qc[i] = bf(q[(row * NK + hk) * DK + PT * dk + i]);
        }
        const float vc = bf(v[(row * NV + hv) * DK + dv]);
        const float gc = g[row * NV + hv];
        const float bc = bf(beta[row * NV + hv]);
        float kv_mem = 0.0f;
#pragma unroll
        for (int i = 0; i < PT; i++) {
            state[i] = state[i] * gc;
            kv_mem += state[i] * kc[i];
        }
        kv_mem = warp_sum(kv_mem);
        const float delta = (vc - kv_mem) * bc;
        float out = 0.0f;
#pragma unroll
        for (int i = 0; i < PT; i++) {
            state[i] = state[i] + kc[i] * delta;
            out += state[i] * qc[i];
        }
        out = warp_sum(out);
        if (dk == 0) y[(row * NV + hv) * DK + dv] = __float2bfloat16_rn(out);
        if (SNAP != nullptr) {
            float* sn = SNAP + row * geo.rec() + at;
#pragma unroll
            for (int i = 0; i < PT; i++) sn[i] = state[i];
        }
    }
#pragma unroll
    for (int i = 0; i < PT; i++) st[i] = state[i];
}

// out = bf16(silu(z) * bf16(w * rmsnorm(y))) per linear value head; rows nv * DK wide. Grid (1, nv, rows), a warp each.
extern "C" __global__ void __launch_bounds__(32) tf_q35_gdn_post(Geo geo, const __nv_bfloat16* __restrict__ Y, const __nv_bfloat16* __restrict__ Z,
                                                                   const __nv_bfloat16* __restrict__ NW, float eps, __nv_bfloat16* __restrict__ OUT) {
    const int lane = threadIdx.x, hv = blockIdx.y;
    const size_t m = blockIdx.z;
    const size_t W = (size_t)geo.nv * DK;
    constexpr int PER = DK / 32;
    float yv[PER], ss = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; j++) {
        yv[j] = bf(Y[m * W + hv * DK + lane * PER + j]);
        ss += yv[j] * yv[j];
    }
    ss = warp_sum(ss);
    const float inv = rsqrtf(ss / (float)DK + eps);
#pragma unroll
    for (int j = 0; j < PER; j++) {
        const int d = lane * PER + j;
        const float x = rbf(bf(NW[d]) * (yv[j] * inv));
        const float zf = bf(Z[m * W + hv * DK + d]);
        OUT[m * W + hv * DK + d] = __float2bfloat16_rn(zf / (1.0f + expf(-zf)) * x);
    }
}

// Per-head RMSNorm of attention rows: X's head h of row m at m * stride + h * hstride, out packed (rows, heads, HD).
// Grid (rows * heads), 64 threads of 4 values.
extern "C" __global__ void __launch_bounds__(64) tf_q35_head_norm(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                    __nv_bfloat16* __restrict__ OUT, int heads, int stride, int hstride, float eps) {
    __shared__ float sums[2];
    const int head = blockIdx.x % heads;
    const size_t m = blockIdx.x / heads;
    const int t = threadIdx.x;
    const __nv_bfloat16* x = X + m * stride + head * hstride;
    float v[4], acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = bf(x[4 * t + i]);
        acc += v[i] * v[i];
    }
    acc = warp_sum(acc);
    if ((t & 31) == 0) sums[t >> 5] = acc;
    __syncthreads();
    const float inv = rsqrtf((sums[0] + sums[1]) / (float)HD + eps);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int d = 4 * t + i;
        OUT[(m * heads + head) * HD + d] = __float2bfloat16_rn(bf(W[d]) * (v[i] * inv));
    }
}

// Text RoPE on the first 64 of HD: pairs (i, i + 32) turned by position * theta^(-i / 32).
__device__ __forceinline__ __nv_bfloat16 rotate(const __nv_bfloat16* x, int d, int position) {
    if (d >= 64) return x[d];
    const int i = d % 32;
    const float angle = (float)position * powf(10000000.0f, -(float)i / 32.0f);
    float sn, cs;
    sincosf(angle, &sn, &cs);
    const float a = bf(x[i]), b = bf(x[i + 32]);
    return __float2bfloat16_rn(d < 32 ? a * cs - b * sn : b * cs + a * sn);
}

// Queries in place at their positions; keys rotated and values copied into their segment's cache.
// Grid (rows), HD threads.
extern "C" __global__ void __launch_bounds__(256) tf_q35_rope(Geo geo, __nv_bfloat16* __restrict__ Qn, const __nv_bfloat16* __restrict__ Kn,
                                                                const __nv_bfloat16* __restrict__ Vr, const Seg* __restrict__ segs,
                                                                const int* __restrict__ rowseg, int layer) {
    const int QH = geo.qh, KVH = geo.kvh;
    const int d = threadIdx.x;
    const size_t m = blockIdx.x;
    const Seg sg = segs[rowseg[m]];
    const int pos = sg.len + (int)m - sg.row0;
    __shared__ __nv_bfloat16 qs[MAXQ * HD];
    for (int h = 0; h < QH; h++) qs[h * HD + d] = Qn[(m * QH + h) * HD + d];
    __syncthreads();
    for (int h = 0; h < QH; h++) Qn[(m * QH + h) * HD + d] = rotate(qs + h * HD, d, pos);
    __nv_bfloat16* keys = reinterpret_cast<__nv_bfloat16*>(sg.kv) + (size_t)layer * 2 * KVH * sg.cap * HD;
    __nv_bfloat16* values = keys + (size_t)KVH * sg.cap * HD;
    for (int h = 0; h < KVH; h++) {
        const size_t dst = ((size_t)h * sg.cap + pos) * HD + d;
        keys[dst] = rotate(Kn + (m * KVH + h) * HD, d, pos);
        values[dst] = Vr[(m * KVH + h) * HD + d];
    }
}

// Causal attention over a segment's cache, ck keys a block: grid (chunks, kvh, rows), block qh / kvh warps, a warp a
// query head of the kv head's group. Partials (max, sum, out) at slot (row, qh, chunk); empty chunks leave sum 0.
extern "C" __global__ void tf_q35_attn_partial(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                 const int* __restrict__ rowseg, int layer, int nch, int ck, float scale,
                                                 float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    constexpr int DPL = HD / 32;
    const int QH = geo.qh, KVH = geo.kvh;
    const int lane = threadIdx.x & 31, g = threadIdx.x >> 5;
    const int c = blockIdx.x, h = blockIdx.y;
    const size_t m = blockIdx.z;
    const int qh = h * (QH / KVH) + g;
    const Seg sg = segs[rowseg[m]];
    const int last = sg.len + (int)m - sg.row0;  // the row's own position, the last key it sees
    const int k0 = c * ck, k1 = min(k0 + ck, last + 1);
    const size_t slot = (m * QH + qh) * nch + c;
    if (k0 >= k1) {
        if (lane == 0) {
            PM[slot] = -INFINITY;
            PL[slot] = 0.0f;
        }
        return;
    }
    const __nv_bfloat16* K = reinterpret_cast<const __nv_bfloat16*>(sg.kv) + (size_t)layer * 2 * KVH * sg.cap * HD;
    const __nv_bfloat16* V = K + (size_t)KVH * sg.cap * HD;
    float q[DPL], o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        q[i] = bf(Q[(m * QH + qh) * HD + lane * DPL + i]);
        o[i] = 0.0f;
    }
    float mx = -INFINITY, l = 0.0f;
    for (int pos = k0; pos < k1; pos++) {
        const uint4 kw = *reinterpret_cast<const uint4*>(K + ((size_t)h * sg.cap + pos) * HD + lane * DPL);
        const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kw);
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < DPL; i++) dot = fmaf(q[i], bf(kb[i]), dot);
        const float s = warp_sum(dot) * scale;
        const float mn = fmaxf(mx, s);
        const float a = __expf(mx - mn), b = __expf(s - mn);
        l = l * a + b;
        const uint4 vw = *reinterpret_cast<const uint4*>(V + ((size_t)h * sg.cap + pos) * HD + lane * DPL);
        const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&vw);
#pragma unroll
        for (int i = 0; i < DPL; i++) o[i] = fmaf(b, bf(vb[i]), o[i] * a);
        mx = mn;
    }
    if (lane == 0) {
        PM[slot] = mx;
        PL[slot] = l;
    }
#pragma unroll
    for (int i = 0; i < DPL; i++) PO[slot * HD + lane * DPL + i] = o[i];
}

// The chunks' partials into each (row, query head)'s output, rows qh * HD wide. Grid (qh, rows), a warp each.
extern "C" __global__ void __launch_bounds__(32) tf_q35_attn_merge(Geo geo, const float* __restrict__ PM, const float* __restrict__ PL,
                                                                     const float* __restrict__ PO, int nch, __nv_bfloat16* __restrict__ OUT) {
    constexpr int DPL = HD / 32;
    const int QH = geo.qh;
    const int lane = threadIdx.x, qh = blockIdx.x;
    const size_t m = blockIdx.y;
    const size_t base = (m * QH + qh) * nch;
    float mx = -INFINITY;
    for (int c = 0; c < nch; c++)
        if (PL[base + c] > 0.0f) mx = fmaxf(mx, PM[base + c]);
    float lsum = 0.0f, acc[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) acc[i] = 0.0f;
    for (int c = 0; c < nch; c++) {
        if (!(PL[base + c] > 0.0f)) continue;
        const float e = __expf(PM[base + c] - mx);
        lsum = fmaf(PL[base + c], e, lsum);
#pragma unroll
        for (int i = 0; i < DPL; i++) acc[i] = fmaf(PO[(base + c) * HD + lane * DPL + i], e, acc[i]);
    }
#pragma unroll
    for (int i = 0; i < DPL; i++) OUT[(m * QH + qh) * HD + lane * DPL + i] = __float2bfloat16_rn(acc[i] / lsum);
}

// out = bf16(y * bf16(sigmoid(gate))), the gate the second half of each query head's q_proj rows; rows qh * HD wide.
extern "C" __global__ void tf_q35_attn_gate(Geo geo, const __nv_bfloat16* __restrict__ Y, const __nv_bfloat16* __restrict__ QRAW,
                                              __nv_bfloat16* __restrict__ OUT, int rows) {
    const size_t W = (size_t)geo.qh * HD;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * W) return;
    const size_t r = i / W;
    const int d = (int)(i % W), head = d / HD;
    const float gate = bf(QRAW[r * 2 * W + head * 2 * HD + HD + d % HD]);
    const float s = rbf(1.0f / (1.0f + expf(-gate)));
    OUT[i] = __float2bfloat16_rn(bf(Y[i]) * s);
}

// out = bf16(silu(gate) * up) over `total` values.
extern "C" __global__ void tf_q35_mlp_act(const __nv_bfloat16* __restrict__ GATE, const __nv_bfloat16* __restrict__ UP,
                                            __nv_bfloat16* __restrict__ OUT, long long total) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float gf = bf(GATE[i]);
    OUT[i] = __float2bfloat16_rn(gf / (1.0f + expf(-gf)) * bf(UP[i]));
}

// Each row's greedy token: the largest logit, a tie to the lower id (the host draw's rule). Grid (rows), 1024 threads.
extern "C" __global__ void __launch_bounds__(1024) tf_q35_argmax(const __nv_bfloat16* __restrict__ L, int V, unsigned* __restrict__ TOK) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const __nv_bfloat16* x = L + (size_t)blockIdx.x * V;
    float best = -INFINITY;
    int id = 0x7fffffff;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float v = bf(x[i]);
        if (v > best) {
            best = v;
            id = i;
        }
    }
#pragma unroll
    for (int m = 16; m; m >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, m);
        const int oi = __shfl_xor_sync(0xffffffffu, id, m);
        if (ov > best || (ov == best && oi < id)) {
            best = ov;
            id = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        bv[threadIdx.x >> 5] = best;
        bi[threadIdx.x >> 5] = id;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 32; w++)
            if (bv[w] > best || (bv[w] == best && bi[w] < id)) {
                best = bv[w];
                id = bi[w];
            }
        TOK[blockIdx.x] = (unsigned)id;
    }
}
