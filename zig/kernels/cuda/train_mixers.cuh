// Training kernels for Nemotron's Mamba-2 and attention mixers (train_mixers.metal), included by train.cu.

// ---- the Mamba-2 layer ----------------------------------------------------------------------------------------------

// A Mamba layer's shape (train_ops.Shape): rows, heads, head dim, groups, state, inner, conv and projection widths.
struct Shape {
    int rows, heads, dh, groups, n, inner, conv, proj;
};

#define CK 16  // scan steps between checkpoints

// Sum over the 16 lanes holding one state row (lanes 0..15 or 16..31 of a warp).
__device__ __forceinline__ float row_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

// Sum over all 64 state rows for each of n state columns: lanes l, l+16 first, then the 32 warps in turn.
__device__ __forceinline__ void column_sums(const float (&v)[8], float* scratch, int lane, int w, int t, int n, float* out) {
    float s[8];
    for (int i = 0; i < 8; i++) s[i] = v[i] + __shfl_down_sync(0xffffffffu, v[i], 16);
    if (lane < 16)
        for (int i = 0; i < 8; i++) scratch[w * n + lane * 8 + i] = s[i];
    __syncthreads();
    if (t < n) {
        float x = 0;
        for (int g = 0; g < 32; g++) x += scratch[g * n + t];
        out[t] = x;
    }
    __syncthreads();
}

// The causal conv (conv_w fp32 [taps][conv]) before its bias, and its SiLU with it, from position 0's xBC columns.
extern "C" __global__ void tf_train_conv_pre(const bf16* proj, const float* conv_w, const float* conv_b, bf16* conv_out, bf16* act,
                                             Shape s, int taps) {
    const int c = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y;
    if (c >= s.conv || r >= s.rows) return;
    float acc = 0;
    for (int j = 0; j < taps; j++) {
        const int at = r - (taps - 1) + j;
        if (at < 0) continue;
        acc += conv_w[j * s.conv + c] * f32(proj[(size_t)at * s.proj + s.inner + c]);
    }
    conv_out[(size_t)r * s.conv + c] = b16(acc);
    act[(size_t)r * s.conv + c] = b16(silu(f32(b16(acc)) + conv_b[c]));
}

// dt[t, h] = softplus(raw + bias) from the in-projection's last columns; grid (heads, rows).
extern "C" __global__ void tf_train_dt(const bf16* proj, const float* bias, float* dt, Shape s) {
    const int h = threadIdx.x, t = blockIdx.x;
    if (h >= s.heads || t >= s.rows) return;
    dt[t * s.heads + h] = softplus(f32(proj[(size_t)t * s.proj + s.inner + s.conv + h]) + bias[h]);
}

// The scan for one head, a block of 1024 (64 rows x 16 lanes of 8 columns): y with the D skip, and checkpoints.
extern "C" __global__ void __launch_bounds__(1024) tf_train_ssm_fwd(const bf16* act, const float* dt, const float* a_neg,
                                                                   const float* d_skip, float* y, float* ckpt, Shape s) {
    const int h = blockIdx.x, t = threadIdx.x, lane = t & 31;
    const int p = t / 16, n0 = (t % 16) * 8, g = h / (s.heads / s.groups), size = s.dh * s.n;
    const int b0 = s.inner + g * s.n, c0 = s.inner + s.groups * s.n + g * s.n;
    float st[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float* mine = ckpt + (size_t)h * ((s.rows + CK - 1) / CK + 1) * size;
    for (int i = 0; i < 8; i++) mine[p * s.n + n0 + i] = 0;
    for (int r = 0; r < s.rows; r++) {
        const bf16* row = act + (size_t)r * s.conv;
        const float d = dt[r * s.heads + h], a = expf(d * a_neg[h]), x = f32(row[h * s.dh + p]);
        float acc = 0;
        for (int i = 0; i < 8; i++) {
            st[i] = a * st[i] + d * x * f32(row[b0 + n0 + i]);
            acc += st[i] * f32(row[c0 + n0 + i]);
        }
        acc = row_sum(acc);
        if (lane % 16 == 0) y[(size_t)r * s.inner + h * s.dh + p] = acc + d_skip[h] * x;
        if ((r + 1) % CK == 0)
            for (int i = 0; i < 8; i++) mine[((r + 1) / CK) * size + p * s.n + n0 + i] = st[i];
    }
}

// The scan backward for one head: dx (with the D skip) into dact, per-head dB and dC partials, and ddt.
extern "C" __global__ void __launch_bounds__(1024) tf_train_ssm_back(const bf16* act, const float* dt, const float* a_neg,
                                                                    const float* d_skip, const float* dy, const float* ckpt,
                                                                    float* states, float* dact, float* db_part, float* dc_part,
                                                                    float* ddt, Shape s) {
    __shared__ float scratch[32 * 128];
    const int h = blockIdx.x, t = threadIdx.x, lane = t & 31, w = t >> 5;
    const int p = t / 16, n0 = (t % 16) * 8, g = h / (s.heads / s.groups), size = s.dh * s.n;
    const int b0 = s.inner + g * s.n, c0 = s.inner + s.groups * s.n + g * s.n;
    const float* mine = ckpt + (size_t)h * ((s.rows + CK - 1) / CK + 1) * size;
    float* chunk = states + (size_t)h * CK * size;
    float ds[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int c = (s.rows - 1) / CK; c >= 0; c--) {
        const int first = c * CK, last = min(first + CK, s.rows);
        float st[8];
        for (int i = 0; i < 8; i++) st[i] = mine[c * size + p * s.n + n0 + i];
        for (int r = first; r < last; r++) {
            const bf16* row = act + (size_t)r * s.conv;
            const float d = dt[r * s.heads + h], a = expf(d * a_neg[h]), x = f32(row[h * s.dh + p]);
            for (int i = 0; i < 8; i++) {
                st[i] = a * st[i] + d * x * f32(row[b0 + n0 + i]);
                chunk[(r - first) * size + p * s.n + n0 + i] = st[i];
            }
        }
        __syncthreads();
        for (int r = last - 1; r >= first; r--) {
            const bf16* row = act + (size_t)r * s.conv;
            const float d = dt[r * s.heads + h], a = expf(d * a_neg[h]), x = f32(row[h * s.dh + p]);
            const float gy = dy[(size_t)r * s.inner + h * s.dh + p];
            float cur[8], prev[8], v[8];
            for (int i = 0; i < 8; i++) {
                cur[i] = chunk[(r - first) * size + p * s.n + n0 + i];
                prev[i] = r > first ? chunk[(r - first - 1) * size + p * s.n + n0 + i] : mine[c * size + p * s.n + n0 + i];
                ds[i] += gy * f32(row[c0 + n0 + i]);
                v[i] = gy * cur[i];
            }
            column_sums(v, scratch, lane, w, t, s.n, dc_part + ((size_t)r * s.heads + h) * s.n);
            float dx = 0, dd = 0;
            for (int i = 0; i < 8; i++) {
                const float bn = f32(row[b0 + n0 + i]);
                dx += ds[i] * bn;
                dd += ds[i] * (x * bn + a_neg[h] * a * prev[i]);
                v[i] = d * ds[i] * x;
            }
            column_sums(v, scratch, lane, w, t, s.n, db_part + ((size_t)r * s.heads + h) * s.n);
            dx = row_sum(dx);
            if (lane % 16 == 0) dact[(size_t)r * s.conv + h * s.dh + p] = d * dx + d_skip[h] * gy;
            dd = block_sum(dd, scratch, 32);
            if (t == 0) ddt[r * s.heads + h] = dd;
            for (int i = 0; i < 8; i++) ds[i] *= a;
        }
        __syncthreads();
    }
}

// dact's B and C columns: each group's per-head partials summed in head order; grid (groups, rows), block n.
extern "C" __global__ void tf_train_ssm_bc(const float* db_part, const float* dc_part, float* dact, Shape s) {
    const int n = threadIdx.x, g = blockIdx.x, r = blockIdx.y, per = s.heads / s.groups;
    if (n >= s.n || g >= s.groups || r >= s.rows) return;
    float b = 0, c = 0;
    for (int h = g * per; h < (g + 1) * per; h++) {
        b += db_part[((size_t)r * s.heads + h) * s.n + n];
        c += dc_part[((size_t)r * s.heads + h) * s.n + n];
    }
    dact[(size_t)r * s.conv + s.inner + g * s.n + n] = b;
    dact[(size_t)r * s.conv + s.inner + s.groups * s.n + g * s.n + n] = c;
}

// The gate and grouped RMS norm undone: dy into the SSM's output, dz into the projection's z columns; grid (groups, rows).
extern "C" __global__ void __launch_bounds__(256) tf_train_gate_back(const float* y, const bf16* proj, const bf16* w, const float* dn,
                                                                    float* dy, float* dproj, Shape s, float eps) {
    __shared__ float scratch[32];
    const int t = threadIdx.x, g = blockIdx.x, r = blockIdx.y, width = s.inner / s.groups, j0 = g * width;
    float sq = 0, dot = 0;
    for (int j = j0 + t; j < j0 + width; j += 256) {
        const float gated = y[(size_t)r * s.inner + j] * silu(f32(proj[(size_t)r * s.proj + j]));
        sq += gated * gated;
        dot += f32(w[j]) * dn[(size_t)r * s.inner + j] * gated;
    }
    sq = block_sum(sq, scratch, 8);
    dot = block_sum(dot, scratch, 8);
    const float inv = rsqrtf(sq / (float)width + eps), k = inv * inv * inv * dot / (float)width;
    for (int j = j0 + t; j < j0 + width; j += 256) {
        const float z = f32(proj[(size_t)r * s.proj + j]), yv = y[(size_t)r * s.inner + j];
        const float gated = yv * silu(z), dg = inv * f32(w[j]) * dn[(size_t)r * s.inner + j] - k * gated;
        dy[(size_t)r * s.inner + j] = dg * silu(z);
        dproj[(size_t)r * s.proj + j] = dg * yv * silu_grad(z);
    }
}

// The depthwise causal conv and its SiLU undone into dproj's xBC columns; grid (conv / 256, rows).
extern "C" __global__ void tf_train_conv_back(const float* dact, const bf16* conv_out, const float* conv_w, const float* conv_b,
                                              float* dproj, Shape s, int taps) {
    const int c = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y;
    if (c >= s.conv || r >= s.rows) return;
    float acc = 0;
    for (int j = 0; j < taps; j++) {
        const int at = r + taps - 1 - j;
        if (at >= s.rows) continue;
        const float pre = f32(conv_out[(size_t)at * s.conv + c]) + conv_b[c];
        acc += conv_w[j * s.conv + c] * dact[(size_t)at * s.conv + c] * silu_grad(pre);
    }
    dproj[(size_t)r * s.proj + s.inner + c] = acc;
}

// d(raw dt) = ddt sigmoid(raw + bias) into dproj's last columns; grid (rows), block heads.
extern "C" __global__ void tf_train_dt_back(const bf16* proj, const float* bias, const float* ddt, float* dproj, Shape s) {
    const int h = threadIdx.x, r = blockIdx.x;
    if (h >= s.heads || r >= s.rows) return;
    const size_t at = (size_t)r * s.proj + s.inner + s.conv + h;
    const float v = f32(proj[at]) + bias[h];
    dproj[at] = ddt[r * s.heads + h] / (1.0f + expf(-v));
}

// ---- the attention layer --------------------------------------------------------------------------------------------

// Attention's shape: rows, heads, kv heads, head dim, the q|k|v row width; scale 1/sqrt(head dim).
struct Heads {
    int rows, heads, kv_heads, dim, qkv;
    float scale;
};

// One query row of one head (q, k, v the qkv projection's columns): probabilities and their gradient kept, and dq.
extern "C" __global__ void __launch_bounds__(256) tf_train_attn_q(const bf16* qkv, const float* dout, float* probs, float* dscores,
                                                                 float* dq, Heads a) {
    __shared__ float qi[256], gi[256], ds[256], scratch[32];
    const int t = threadIdx.x, h = blockIdx.x, i = blockIdx.y, kv = h / (a.heads / a.kv_heads), qw = a.heads * a.dim;
    const bf16* q = qkv;
    const bf16* k = qkv + qw;
    const bf16* v = qkv + qw + a.kv_heads * a.dim;
    for (int d = t; d < a.dim; d += 256) {
        qi[d] = f32(q[(size_t)i * a.qkv + h * a.dim + d]);
        gi[d] = dout[(size_t)i * qw + h * a.dim + d];
    }
    __syncthreads();
    const bool live = t <= i && t < a.rows;
    float score = -INFINITY, dp = 0;
    if (live) {
        float sc = 0, dv = 0;
        for (int d = 0; d < a.dim; d++) {
            sc += qi[d] * f32(k[(size_t)t * a.qkv + kv * a.dim + d]);
            dv += gi[d] * f32(v[(size_t)t * a.qkv + kv * a.dim + d]);
        }
        score = sc * a.scale;
        dp = dv;
    }
    const float top = block_max(score, scratch, 8);
    const float e = live ? expf(score - top) : 0.0f;
    const float total = block_sum(e, scratch, 8);
    const float pr = e / total;
    const float mix = block_sum(pr * dp, scratch, 8);
    const float g = pr * (dp - mix);
    ds[t] = g;
    if (t < a.rows) {
        probs[((size_t)h * a.rows + i) * a.rows + t] = pr;
        dscores[((size_t)h * a.rows + i) * a.rows + t] = g;
    }
    __syncthreads();
    for (int d = t; d < a.dim; d += 256) {
        float acc = 0;
        for (int j = 0; j <= i; j++) acc += ds[j] * f32(k[(size_t)j * a.qkv + kv * a.dim + d]);
        dq[(size_t)i * qw + h * a.dim + d] = acc * a.scale;
    }
}

// One key row of one kv head: dk and dv summed over the heads that read it and the query rows after it; block `dim`.
extern "C" __global__ void tf_train_attn_kv(const bf16* qkv, const float* dout, const float* probs, const float* dscores, float* dk,
                                            float* dv, Heads a) {
    const int d = threadIdx.x, kv = blockIdx.x, j = blockIdx.y, per = a.heads / a.kv_heads, qw = a.heads * a.dim,
              kw = a.kv_heads * a.dim;
    if (d >= a.dim) return;
    float gk = 0, gv = 0;
    for (int h = kv * per; h < (kv + 1) * per; h++)
        for (int i = j; i < a.rows; i++) {
            const size_t at = ((size_t)h * a.rows + i) * a.rows + j;
            gk += dscores[at] * f32(qkv[(size_t)i * a.qkv + h * a.dim + d]);
            gv += probs[at] * dout[(size_t)i * qw + h * a.dim + d];
        }
    dk[(size_t)j * kw + kv * a.dim + d] = gk * a.scale;
    dv[(size_t)j * kw + kv * a.dim + d] = gv;
}
