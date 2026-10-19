// Qwen3.5's hybrids (2B, 27B) on CUDA: the Metal kernels' arithmetic and rounding; a segment is one stream's window.

#include "qwen35.cuh"

using namespace tf_q35;

// P (device counts, one graph for every shape): chunks, width, rows, windows, recorded; rowpar/rowanc give the tree.
__device__ __forceinline__ bool past_windows(const int* P, int s) { return P != nullptr && s >= P[3]; }
__device__ __forceinline__ bool recorded(const int* P) { return P != nullptr && P[4] != 0; }

// One 64-entry group's sum, its entries added in order (x 16-byte aligned).
__device__ __forceinline__ float group_sum(const __nv_bfloat16* x) {
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 64; j += 8) {
        const uint4 w = *reinterpret_cast<const uint4*>(x + j);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&w);
#pragma unroll
        for (int k = 0; k < 8; k++) s += bf(b[k]);
    }
    return s;
}

// MLX affine 4-bit token rows: out = bf16(s * q + b), D wide.
extern "C" __global__ void __launch_bounds__(256) tf_q35_embed(const int* __restrict__ ids, const uint8_t* __restrict__ W,
                                                                 const __nv_bfloat16* __restrict__ S, const __nv_bfloat16* __restrict__ B,
                                                                 __nv_bfloat16* __restrict__ out, int D, const int* __restrict__ P) {
    if (past_rows(P, blockIdx.x)) return;
    const size_t id = ids[blockIdx.x];
    for (int b = threadIdx.x; b < D / 2; b += blockDim.x) {
        const uint8_t v = W[id * (D / 2) + b];
        const size_t g = id * (D / 64) + (2 * b) / 64;
        const float s = bf(S[g]), z = bf(B[g]);
        out[(size_t)blockIdx.x * D + 2 * b] = __float2bfloat16_rn(s * (float)(v & 15) + z);
        out[(size_t)blockIdx.x * D + 2 * b + 1] = __float2bfloat16_rn(s * (float)(v >> 4) + z);
    }
}

// MLX affine 8-bit token rows (a mixed checkpoint's, or widened to 8 at load): out = bf16(s * q + b), D wide.
extern "C" __global__ void __launch_bounds__(256) tf_q35_embed8(const int* __restrict__ ids, const uint8_t* __restrict__ W,
                                                                  const __nv_bfloat16* __restrict__ S, const __nv_bfloat16* __restrict__ B,
                                                                  __nv_bfloat16* __restrict__ out, int D, const int* __restrict__ P) {
    if (past_rows(P, blockIdx.x)) return;
    const size_t id = ids[blockIdx.x];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const size_t g = id * (D / 64) + d / 64;
        out[(size_t)blockIdx.x * D + d] = __float2bfloat16_rn(bf(S[g]) * (float)W[id * D + d] + bf(B[g]));
    }
}

// Image rows into a prompt chunk's embeddings: row m of H (D wide) takes feature row src[m] where src[m] >= 0.
extern "C" __global__ void __launch_bounds__(256) tf_q35_splice(__nv_bfloat16* __restrict__ H, const __nv_bfloat16* __restrict__ F,
                                                                  const int* __restrict__ src, int D, const int* __restrict__ P) {
    if (past_rows(P, blockIdx.x)) return;
    const int s = src[blockIdx.x];
    if (s < 0) return;
    for (int i = threadIdx.x; i < D; i += blockDim.x) H[(size_t)blockIdx.x * D + i] = F[(size_t)s * D + i];
}

// h = bf16(h + r) (r may be null), x = bf16(w * (h * inv)), XS (may be null) x's group sums; D 256k up to 8192.
extern "C" __global__ void __launch_bounds__(256) tf_q35_norm(__nv_bfloat16* __restrict__ H, const __nv_bfloat16* __restrict__ R,
                                                               const __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ X, float eps, int D,
                                                               const int* __restrict__ P, float* __restrict__ XS) {
    constexpr int MAXE = 32;
    __shared__ float red[8];
    __shared__ __align__(16) __nv_bfloat16 xrow[8192];
    if (past_rows(P, blockIdx.x)) return;
    const size_t row = blockIdx.x;
    const int t = threadIdx.x;
    const int E = D / 256;
    if ((E & 3) == 0) {
        // h + r and w * (h * inv) move 8 entries a lane through xrow; the squares keep a thread's E entries in order
        uint4 wv[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int d = (t + 256 * j) * 8;
            if (d < D) wv[j] = *reinterpret_cast<const uint4*>(W + d);
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int d = (t + 256 * j) * 8;
            if (d >= D) break;
            const size_t at = row * D + d;
            uint4 hw = *reinterpret_cast<const uint4*>(H + at);
            if (R != nullptr) {
                const uint4 rw = *reinterpret_cast<const uint4*>(R + at);
                __nv_bfloat16* hb = reinterpret_cast<__nv_bfloat16*>(&hw);
                const __nv_bfloat16* rb = reinterpret_cast<const __nv_bfloat16*>(&rw);
#pragma unroll
                for (int k = 0; k < 8; k++) hb[k] = __float2bfloat16_rn(bf(hb[k]) + bf(rb[k]));
                *reinterpret_cast<uint4*>(H + at) = hw;
            }
            *reinterpret_cast<uint4*>(xrow + d) = hw;
        }
        __syncthreads();
        float sq = 0.0f;
#pragma unroll
        for (int i = 0; i < MAXE; i++) {
            if (i >= E) break;
            const float h = bf(xrow[t * E + i]);
            sq = fmaf(h, h, sq);
        }
        sq = warp_sum(sq);
        if ((t & 31) == 0) red[t >> 5] = sq;
        __syncthreads();
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) sum += red[i];
        const float scale = rsqrtf(sum / (float)D + eps);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int d = (t + 256 * j) * 8;
            if (d >= D) break;
            const uint4 hw = *reinterpret_cast<const uint4*>(xrow + d);
            const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&hw);
            const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv[j]);
            uint4 xw;
            __nv_bfloat16* xb = reinterpret_cast<__nv_bfloat16*>(&xw);
#pragma unroll
            for (int k = 0; k < 8; k++) xb[k] = __float2bfloat16_rn(bf(wb[k]) * (bf(hb[k]) * scale));
            *reinterpret_cast<uint4*>(X + row * D + d) = xw;
            *reinterpret_cast<uint4*>(xrow + d) = xw;
        }
        if (XS == nullptr) return;
        __syncthreads();
        for (int g = t; g < D / 64; g += blockDim.x) XS[row * (D / 64) + g] = group_sum(xrow + g * 64);
        return;
    }
    // E not a multiple of 4: a thread's entries one at a time
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
        const __nv_bfloat16 x = __float2bfloat16_rn(bf(W[d]) * (hv[i] * inv));
        X[row * D + d] = x;
        xrow[d] = x;
    }
    if (XS == nullptr) return;
    __syncthreads();
    for (int g = t; g < D / 64; g += blockDim.x) XS[row * (D / 64) + g] = group_sum(xrow + g * 64);
}

// 64-group sums of bf16 rows D wide (the decode projections' input sums). Grid (rows, groups / 64), a thread a group.
extern "C" __global__ void __launch_bounds__(64) tf_q35_sums(const __nv_bfloat16* __restrict__ X, float* __restrict__ XS, int D, const int* __restrict__ P) {
    if (past_rows(P, blockIdx.x)) return;
    const size_t row = blockIdx.x;
    const int g = blockIdx.y * blockDim.x + threadIdx.x;
    if (g < D / 64) XS[row * (D / 64) + g] = group_sum(X + row * D + g * 64);
}

// Conv, silu, q/k norms, the linear heads' decay and write: a warp a (row, head) over q, k, v; taps from the parents.
extern "C" __global__ void __launch_bounds__(32) tf_q35_gdn_pre(Geo geo, const __nv_bfloat16* __restrict__ QKV, const Seg* __restrict__ segs,
                                                                  const int* __restrict__ rowseg, const int* __restrict__ rowpar, int layer,
                                                                  const __nv_bfloat16* __restrict__ CW,
                                                                  const __nv_bfloat16* __restrict__ Ain, const __nv_bfloat16* __restrict__ Bin,
                                                                  const float* __restrict__ ALOG, const __nv_bfloat16* __restrict__ DT,
                                                                  __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K,
                                                                  __nv_bfloat16* __restrict__ V, float* __restrict__ G, __nv_bfloat16* __restrict__ BETA,
                                                                  const int* __restrict__ P) {
    const int lane = threadIdx.x;
    const int head = blockIdx.y;
    const int w = blockIdx.z;
    if (past_rows(P, w)) return;
    const int NK = geo.nk, NV = geo.nv, C = geo.conv();
    const Seg sg = segs[rowseg[w]];
    const __nv_bfloat16* CS = reinterpret_cast<const __nv_bfloat16*>(sg.conv) + (size_t)layer * geo.conv_rows();
    const bool isq = head < NK, isk = !isq && head < 2 * NK;
    const int c0 = head * DK;  // q, k and v channels are consecutive heads of DK
    int src[TAPS];  // tap t reads the row TAPS - 1 - t tokens back: a forward row, or -1 - (tail row)
#pragma unroll
    for (int tap = 0; tap < TAPS; tap++) src[tap] = back(rowpar, w, TAPS - 1 - tap);
    constexpr int PER = DK / 32;
    // a recorded window's row logs the tail after it (taps 1..TAPS-1) here, so conv_commit only moves the stream's
    __nv_bfloat16* tail = recorded(P) ? log_row(geo, sg, layer, w - sg.row0).conv : nullptr;
    float vals[PER];
#pragma unroll
    for (int j = 0; j < PER; j++) {
        const int c = c0 + lane * PER + j;
        float acc = 0.0f;
#pragma unroll
        for (int tap = 0; tap < TAPS; tap++) {
            const int at = src[tap];
            const __nv_bfloat16 xb = at < 0 ? CS[(size_t)(-1 - at) * C + c] : QKV[(size_t)at * C + c];
            if (tail != nullptr && tap > 0) tail[(size_t)(tap - 1) * C + c] = xb;
            acc += bf(CW[c * TAPS + tap]) * bf(xb);
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

// A segment's conv tail: a recorded window leaves the stream's (gdn_pre logged its rows'); others update it in place.
extern "C" __global__ void tf_q35_conv_commit(Geo geo, const __nv_bfloat16* __restrict__ QKV, const Seg* __restrict__ segs,
                                                const int* __restrict__ rowpar, int layer, const int* __restrict__ P) {
    const int C = geo.conv();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C || past_windows(P, blockIdx.y)) return;
    const Seg sg = segs[blockIdx.y];
    __nv_bfloat16* CS = reinterpret_cast<__nv_bfloat16*>(sg.conv) + (size_t)layer * geo.conv_rows();
    if (recorded(P)) {
        // gdn_pre logged each row's tail
        if (sg.rows != 1) return;
        __nv_bfloat16* before = reinterpret_cast<__nv_bfloat16*>(sg.before + geo.nl * geo.rec() * sizeof(float)) + (size_t)layer * geo.conv_rows();
        for (int r = 0; r < TAPS - 1; ++r) before[(size_t)r * C + c] = CS[(size_t)r * C + c];
    }
    // ascending: slot r takes [r + rows], which is either a later state slot (not yet written) or a row
    for (int r = 0; r < TAPS - 1; r++) {
        const int at = r + sg.rows;
        CS[(size_t)r * C + c] = at < TAPS - 1 ? CS[(size_t)at * C + c] : QKV[(size_t)(sg.row0 + at - (TAPS - 1)) * C + c];
    }
}

// Gated delta rule, each row's state its parent's: a chain keeps it in registers, a branch rebuilds it from the log.
extern "C" __global__ void __launch_bounds__(128) tf_q35_gdn_chain(Geo geo, const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                                                                     const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                                                                     const __nv_bfloat16* __restrict__ beta, const Seg* __restrict__ segs,
                                                                     const int* __restrict__ rowpar, int layer, __nv_bfloat16* __restrict__ y,
                                                                     const int* __restrict__ P) {
    const int NK = geo.nk, NV = geo.nv;
    if (past_windows(P, blockIdx.z / NV)) return;
    const int hv = blockIdx.z % NV;
    const int hk = hv / (NV / NK);
    const Seg sg = segs[blockIdx.z / NV];
    const int dk = threadIdx.x;
    const int dv = blockIdx.y * 4 + threadIdx.y;
    constexpr int PT = DK / 32;
    const size_t at = ((size_t)hv * DK + dv) * DK + PT * dk;
    float* st = reinterpret_cast<float*>(sg.rec) + (size_t)layer * geo.rec() + at;
    const bool record = recorded(P);
    float state[PT];
#pragma unroll
    for (int i = 0; i < PT; i++) state[i] = st[i];
    int prev = -1;  // the row the registers' state follows
    // row r's inputs, loaded while row r - 1 computes
    float kn[PT], qn[PT], vn = 0.0f, gn = 0.0f, bn = 0.0f;
    int pn = -1;
    auto load = [&](int row) {
        const uint2 kw = *reinterpret_cast<const uint2*>(k + ((size_t)row * NK + hk) * DK + PT * dk);
        const uint2 qw = *reinterpret_cast<const uint2*>(q + ((size_t)row * NK + hk) * DK + PT * dk);
        const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kw);
        const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(&qw);
#pragma unroll
        for (int i = 0; i < PT; i++) {
            kn[i] = bf(kb[i]);
            qn[i] = bf(qb[i]);
        }
        vn = bf(v[((size_t)row * NV + hv) * DK + dv]);
        gn = g[row * NV + hv];
        bn = bf(beta[row * NV + hv]);
        pn = rowpar[row];
    };
    if (sg.rows > 0) load(sg.row0);
    for (int r = 0; r < sg.rows; r++) {
        const int row = sg.row0 + r;
        float kc[PT], qc[PT];
#pragma unroll
        for (int i = 0; i < PT; i++) {
            kc[i] = kn[i];
            qc[i] = qn[i];
        }
        const float vc = vn, gc = gn, bc = bn;
        const int parent = pn;
        if (r + 1 < sg.rows) load(row + 1);
        if (parent != prev) {
            // a branch: its parent's state from the stream's and the ancestors' logged updates, oldest first
            int anc[MAXW];
            int n = 0;
            for (int a = parent; a >= 0; a = rowpar[a]) anc[n++] = a;
#pragma unroll
            for (int i = 0; i < PT; i++) state[i] = st[i];
            for (int j = n - 1; j >= 0; j--) {
                const int a = anc[j];
                const float ga = g[a * NV + hv], da = log_row(geo, sg, layer, a - sg.row0).delta[hv * DK + dv];
#pragma unroll
                for (int i = 0; i < PT; i++) {
                    state[i] = state[i] * ga;
                    state[i] = fmaf(bf(k[((size_t)a * NK + hk) * DK + PT * dk + i]), da, state[i]);
                }
            }
        }
        prev = row;
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
            state[i] = fmaf(kc[i], delta, state[i]);
            out += state[i] * qc[i];
        }
        out = warp_sum(out);
        if (dk == 0) y[((size_t)row * NV + hv) * DK + dv] = __float2bfloat16_rn(out);
        if (record) {
            const LogRow l = log_row(geo, sg, layer, r);
            if (dk == 0) {
                l.delta[hv * DK + dv] = delta;
                if (dv == 0) l.g[hv] = gc;
            }
            if (dv == 0 && hv % (NV / NK) == 0)
#pragma unroll
                for (int i = 0; i < PT; i++) l.k[hk * DK + PT * dk + i] = k[((size_t)row * NK + hk) * DK + PT * dk + i];
        }
    }
    if (record && sg.rows != 1) return;
#pragma unroll
    for (int i = 0; i < PT; i++) st[i] = state[i];
}

// out = bf16(silu(z) * bf16(w * rmsnorm(y))) per linear value head; rows nv * DK wide. Grid (1, nv, rows), a warp each.
extern "C" __global__ void __launch_bounds__(32) tf_q35_gdn_post(Geo geo, const __nv_bfloat16* __restrict__ Y, const __nv_bfloat16* __restrict__ Z,
                                                                   const __nv_bfloat16* __restrict__ NW, float eps, __nv_bfloat16* __restrict__ OUT,
                                                                   const int* __restrict__ P) {
    const int lane = threadIdx.x, hv = blockIdx.y;
    if (past_rows(P, blockIdx.z)) return;
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

// Per-head RMSNorm of attention rows, out packed (rows, heads, HD); grid (rows * heads), 64 threads of 4 values.
extern "C" __global__ void __launch_bounds__(64) tf_q35_head_norm(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                    __nv_bfloat16* __restrict__ OUT, int heads, int stride, int hstride, float eps,
                                                                    const int* __restrict__ P) {
    __shared__ float sums[2];
    const int head = blockIdx.x % heads;
    if (past_rows(P, blockIdx.x / heads)) return;
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

// A key and value as bf16 rows in their cache slot (qwen35.cu's layout: a layer's keys then values, kvh blocks of cap).
struct StoreBf16 {
    int layer, kvh;
    __device__ __forceinline__ void operator()(const Seg& sg, int h, int slot, int d, __nv_bfloat16 k, __nv_bfloat16 v) const {
        __nv_bfloat16* keys = reinterpret_cast<__nv_bfloat16*>(sg.kv) + (size_t)layer * 2 * kvh * sg.cap * HD;
        const size_t dst = ((size_t)h * sg.cap + slot) * HD + d;
        keys[dst] = k;
        keys[(size_t)kvh * sg.cap * HD + dst] = v;
    }
};

// Queries rotated in place, keys rotated and values copied into the row's cache slot; grid (rows), HD threads.
extern "C" __global__ void __launch_bounds__(256) tf_q35_rope(Geo geo, __nv_bfloat16* __restrict__ Qn, const __nv_bfloat16* __restrict__ Kn,
                                                                const __nv_bfloat16* __restrict__ Vr, const Seg* __restrict__ segs,
                                                                const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                const int* __restrict__ P, const int* __restrict__ M) {
    rope_rows(geo, Qn, Kn, Vr, segs, rowseg, rowanc, P, M, StoreBf16{layer, geo.kvh});
}

// Causal attention in chunks of chunk_keys(the row's keys), a warp a query head; grid (rows, key heads, chunks).
extern "C" __global__ void tf_q35_attn_partial(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                 const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                 const int* __restrict__ P, float scale,
                                                 float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    constexpr int DPL = HD / 32;
    const int QH = geo.qh, KVH = geo.kvh;
    const int lane = threadIdx.x & 31, g = threadIdx.x >> 5;
    const int c = blockIdx.z, h = blockIdx.y;
    if (c >= MAXCH || past_rows(P, blockIdx.x)) return;
    const size_t m = blockIdx.x;
    const int qh = h * (QH / KVH) + g;
    const Seg sg = segs[rowseg[m]];
    const Ancestor anc = rowanc[m];
    // chunks cut at logical positions (a window key's: the stream's length plus its ancestors), as a step's are
    const int keys = sg.len + rows_seen(anc), ck = chunk_keys(keys);
    const int k0 = c * ck, k1 = min(k0 + ck, keys);
    const size_t slot = (m * QH + qh) * MAXCH + c;
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
    for (int t = k0; t < k1; t++) {
        const int pos = t < sg.len ? t : sg.len + seen_row(anc, t - sg.len);
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

// A row's chunks' partials into each (row, query head)'s output, rows qh * HD wide. Grid (qh, rows), a warp each.
extern "C" __global__ void __launch_bounds__(32) tf_q35_attn_merge(Geo geo, const float* __restrict__ PM, const float* __restrict__ PL,
                                                                     const float* __restrict__ PO, const Seg* __restrict__ segs,
                                                                     const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc,
                                                                     const int* __restrict__ P, __nv_bfloat16* __restrict__ OUT) {
    constexpr int DPL = HD / 32;
    const int QH = geo.qh;
    const int lane = threadIdx.x, qh = blockIdx.x;
    if (past_rows(P, blockIdx.y)) return;
    const size_t m = blockIdx.y;
    const int keys = segs[rowseg[m]].len + rows_seen(rowanc[m]);
    const int ck = chunk_keys(keys), nch = (keys + ck - 1) / ck;
    const size_t base = (m * QH + qh) * MAXCH;
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
                                              __nv_bfloat16* __restrict__ OUT, int rows, const int* __restrict__ P) {
    const size_t W = (size_t)geo.qh * HD;
    if (P != nullptr) rows = P[2];
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * W) return;
    const size_t r = i / W;
    const int d = (int)(i % W), head = d / HD;
    const float gate = bf(QRAW[r * 2 * W + head * 2 * HD + HD + d % HD]);
    const float s = rbf(1.0f / (1.0f + expf(-gate)));
    OUT[i] = __float2bfloat16_rn(bf(Y[i]) * s);
}

// A tapped layer's residual bf16(h + r) into the segment's tap ring at the row's position; a 256-thread block a row.
extern "C" __global__ void __launch_bounds__(256) tf_q35_tap(const __nv_bfloat16* __restrict__ H, const __nv_bfloat16* __restrict__ R,
                                                              const Seg* __restrict__ segs, const int* __restrict__ rowseg, int tap, int D,
                                                              const int* __restrict__ P) {
    if (past_rows(P, blockIdx.x)) return;
    const size_t m = blockIdx.x;
    const Seg sg = segs[rowseg[m]];
    if (sg.tap == 0) return;
    const long long pos = sg.len + (long long)m - sg.row0;
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(sg.tap) + ((size_t)(pos % sg.tap_rows) * sg.taps + tap) * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[i] = __float2bfloat16_rn(bf(H[m * D + i]) + bf(R[m * D + i]));
}

// out = bf16(silu(gate) * up) over `total` values.
extern "C" __global__ void tf_q35_mlp_act(const __nv_bfloat16* __restrict__ GATE, const __nv_bfloat16* __restrict__ UP,
                                            __nv_bfloat16* __restrict__ OUT, long long total, const int* __restrict__ P, int D) {
    if (P != nullptr) total = (long long)P[2] * D;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float gf = bf(GATE[i]);
    OUT[i] = __float2bfloat16_rn(gf / (1.0f + expf(-gf)) * bf(UP[i]));
}

#include "qwen35_slots.cuh"
#include "qwen35_argmax.cuh"
