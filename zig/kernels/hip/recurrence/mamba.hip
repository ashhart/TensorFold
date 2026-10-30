// Nemotron's Mamba-2 mixer: the conv, dt, one fp32 state recurrence at every length, and the group norm.

#include "ops/common.hpp"

#include <hip/hip_runtime.h>

namespace {

constexpr int kMambaTaps = 4;    // conv_kernel
constexpr int kMambaKept = 3;    // conv state rows: the last kMambaTaps - 1 raw inputs
constexpr int kMambaState = 128; // ssm_state_size
constexpr int kMambaRows = 16;   // rows a conv thread slides over
constexpr int kScanRows = 32;    // value rows a scan block walks
constexpr int kScanStage = 16;   // rows of B, C, x and the gate a scan block stages at once
constexpr int kScanLanes = 4;    // threads a value row, each kMambaState / kScanLanes states

__device__ __forceinline__ float wave_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x += __shfl_xor(x, mask, 32);
    return x;
}

}  // namespace

// xc = round(silu(round(cb + w . [state_in rows; P's x columns]))), cw channel-major; state_out may be state_in.
extern "C" __global__ void __launch_bounds__(256) tf_mamba_conv(const void* P, int kind, const void* state_in, void* xc,
                                                                const float* cw, const float* cb, void* state_out,
                                                                int rows, int proj, int xoff, int cd) {
    const int ch = blockIdx.x * 256 + threadIdx.x;
    if (ch >= cd) return;
    const int r0 = blockIdx.y * kMambaRows;
    // input t of [state_in's kMambaKept rows; P's rows], one channel
    auto raw = [&](int t) -> float {
        if (t < kMambaKept) return load(state_in, kind, static_cast<long long>(t) * cd + ch);
        return load(P, kind, static_cast<long long>(t - kMambaKept) * proj + xoff + ch);
    };
    float next[kMambaKept];
    if (blockIdx.y == 0) {
        for (int j = 0; j < kMambaKept; ++j) next[j] = raw(rows + j);
    }
    float t0 = raw(r0), t1 = raw(r0 + 1), t2 = raw(r0 + 2);
    const float w0 = cw[ch * kMambaTaps], w1 = cw[ch * kMambaTaps + 1];
    const float w2 = cw[ch * kMambaTaps + 2], w3 = cw[ch * kMambaTaps + 3];
    const float bias = cb[ch];
    const int r1 = r0 + kMambaRows < rows ? r0 + kMambaRows : rows;
    for (int r = r0; r < r1; ++r) {
        const float cur = raw(r + kMambaKept);
        float acc = bias;
        acc = fmaf(w0, t0, acc);
        acc = fmaf(w1, t1, acc);
        acc = fmaf(w2, t2, acc);
        acc = fmaf(w3, cur, acc);
        store(xc, kind, static_cast<long long>(r) * cd + ch, silu(rounded(acc, kind)));
        t0 = t1;
        t1 = t2;
        t2 = cur;
    }
    if (blockIdx.y == 0) {
        for (int j = 0; j < kMambaKept; ++j) store(state_out, kind, static_cast<long long>(j) * cd + ch, next[j]);
    }
}

// dt[r][h] = clamp(softplus(P[r][dtoff + h] + dt_bias[h]), lo, hi), softplus as max(v, 0) + log1p(e^-|v|).
extern "C" __global__ void tf_mamba_dt(const void* P, int kind, const float* dt_bias, float* dt, int rows, int proj,
                                       int dtoff, int heads, float lo, float hi) {
    const long long i = gid();
    if (i >= static_cast<long long>(rows) * heads) return;
    const int r = static_cast<int>(i / heads), h = static_cast<int>(i % heads);
    const float v = load(P, kind, static_cast<long long>(r) * proj + dtoff + h) + dt_bias[h];
    const float sp = fmaxf(v, 0.0f) + log1pf(expf(-fabsf(v)));
    dt[i] = sp != sp ? sp : fminf(fmaxf(sp, lo), hi);  // a NaN stays NaN, as torch's clamp keeps it
}

// A block (head, 32 value rows) walks the rows in order: s = s e^(A dt) + x dt B, y = round(silu(z)) round(x D + s C).
extern "C" __global__ void __launch_bounds__(kScanRows * kScanLanes)
tf_mamba_scan(const void* P, const void* xc, int kind, const float* dt, float* state, const float* a_log,
              const float* dsk, void* y, int rows, int proj, int xd, int cd, int heads, int dh, int groups) {
    constexpr int DS = kMambaState, TPR = kScanLanes, NJ = DS / TPR / 4, T = kScanRows * kScanLanes;
    __shared__ float4 bs[kScanStage][DS / 4], cs[kScanStage][DS / 4];
    __shared__ float xin[kScanStage][kScanRows], gate[kScanStage][kScanRows], dts[kScanStage];
    const int h = blockIdx.x, g = h / (heads / groups);
    const int tid = threadIdx.x, local = tid / TPR, q = tid % TPR, d = blockIdx.y * kScanRows + local;
    const float a = -expf(a_log[h]), dskip = dsk[h];
    float* s0 = state + (static_cast<size_t>(h) * dh + d) * DS;
    float s[4 * NJ];
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const float4 t = *reinterpret_cast<const float4*>(s0 + 4 * (TPR * j + q));
        s[4 * j] = t.x;
        s[4 * j + 1] = t.y;
        s[4 * j + 2] = t.z;
        s[4 * j + 3] = t.w;
    }
    const long long boff = xd + static_cast<long long>(g) * DS, coff = boff + static_cast<long long>(groups) * DS;
    for (int t0 = 0; t0 < rows; t0 += kScanStage) {
        const int n = rows - t0 < kScanStage ? rows - t0 : kScanStage;
        __syncthreads();
        for (int t = tid; t < n * (DS / 4); t += T) {
            const int i = t / (DS / 4), c = t - i * (DS / 4);
            const long long base = static_cast<long long>(t0 + i) * cd + 4 * c;
            bs[i][c] = make_float4(load(xc, kind, base + boff), load(xc, kind, base + boff + 1),
                                   load(xc, kind, base + boff + 2), load(xc, kind, base + boff + 3));
            cs[i][c] = make_float4(load(xc, kind, base + coff), load(xc, kind, base + coff + 1),
                                   load(xc, kind, base + coff + 2), load(xc, kind, base + coff + 3));
        }
        for (int t = tid; t < n * kScanRows; t += T) {
            const int i = t / kScanRows, l = t - i * kScanRows, dd = blockIdx.y * kScanRows + l;
            xin[i][l] = load(xc, kind, static_cast<long long>(t0 + i) * cd + h * dh + dd);
            gate[i][l] = rounded(silu(load(P, kind, static_cast<long long>(t0 + i) * proj + h * dh + dd)), kind);
        }
        if (tid < n) dts[tid] = dt[static_cast<long long>(t0 + tid) * heads + h];
        __syncthreads();
        for (int i = 0; i < n; ++i) {
            const float dtv = dts[i], x = xin[i][local];
            const float da = expf(a * dtv), xdt = x * dtv;
            float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                const float4 b = bs[i][TPR * j + q], c = cs[i][TPR * j + q];
                s[4 * j] = fmaf(xdt, b.x, s[4 * j] * da);
                s[4 * j + 1] = fmaf(xdt, b.y, s[4 * j + 1] * da);
                s[4 * j + 2] = fmaf(xdt, b.z, s[4 * j + 2] * da);
                s[4 * j + 3] = fmaf(xdt, b.w, s[4 * j + 3] * da);
                m[0] = fmaf(s[4 * j], c.x, m[0]);
                m[1] = fmaf(s[4 * j + 1], c.y, m[1]);
                m[2] = fmaf(s[4 * j + 2], c.z, m[2]);
                m[3] = fmaf(s[4 * j + 3], c.w, m[3]);
            }
            float out = (m[0] + m[1]) + (m[2] + m[3]);
            out = out + __shfl_xor(out, 1, 32);
            out = out + __shfl_xor(out, 2, 32);
            if (q != 0) continue;
            const float yv = rounded(fmaf(x, dskip, out), kind);
            store(y, kind, static_cast<long long>(t0 + i) * xd + h * dh + d, gate[i][local] * yv);
        }
    }
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        *reinterpret_cast<float4*>(s0 + 4 * (TPR * j + q)) =
            make_float4(s[4 * j], s[4 * j + 1], s[4 * j + 2], s[4 * j + 3]);
    }
}

// A block a (row, group) of gs <= 1024: out = round(w * round(v / sqrt(mean(v^2) + eps))), element i on lane i % 128.
extern "C" __global__ void __launch_bounds__(128) tf_mamba_gnorm(const void* x, int kind, const float* w, void* out,
                                                                 int xd, int gs, float eps) {
    constexpr int T = 128, M = 8;
    __shared__ float red[T / 32];
    const long long at = static_cast<long long>(blockIdx.x) * xd + blockIdx.y * gs;
    float v[M];
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < M; ++k) {
        const int i = threadIdx.x + k * T;
        if (i >= gs) continue;
        v[k] = load(x, kind, at + i);
        acc = fmaf(v[k], v[k], acc);
    }
    acc = wave_sum(acc);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    float ss = red[0];
    for (int i = 1; i < T / 32; ++i) ss = ss + red[i];
    const float inv = 1.0f / sqrtf(ss / static_cast<float>(gs) + eps);
#pragma unroll
    for (int k = 0; k < M; ++k) {
        const int i = threadIdx.x + k * T;
        if (i >= gs) continue;
        store(out, kind, at + i, w[blockIdx.y * gs + i] * rounded(v[k] * inv, kind));
    }
}

// tf_mamba_scan's walk, each state but the last row's into slot r + 1 of `snap` (`stride` floats apart).
__device__ void scan_snap(const void* P, const void* xc, int kind, const float* dt, float* state, const float* a_log,
                            const float* dsk, void* y, int rows, int proj, int xd, int cd, int heads, int dh,
                            int groups, float* snap, long long stride) {
    constexpr int DS = kMambaState, TPR = kScanLanes, NJ = DS / TPR / 4, T = kScanRows * kScanLanes;
    __shared__ float4 bs[kScanStage][DS / 4], cs[kScanStage][DS / 4];
    __shared__ float xin[kScanStage][kScanRows], gate[kScanStage][kScanRows], dts[kScanStage];
    const int h = blockIdx.x, g = h / (heads / groups);
    const int tid = threadIdx.x, local = tid / TPR, q = tid % TPR, d = blockIdx.y * kScanRows + local;
    const float a = -expf(a_log[h]), dskip = dsk[h];
    float* s0 = state + (static_cast<size_t>(h) * dh + d) * DS;
    float s[4 * NJ];
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const float4 t = *reinterpret_cast<const float4*>(s0 + 4 * (TPR * j + q));
        s[4 * j] = t.x;
        s[4 * j + 1] = t.y;
        s[4 * j + 2] = t.z;
        s[4 * j + 3] = t.w;
    }
    const long long boff = xd + static_cast<long long>(g) * DS, coff = boff + static_cast<long long>(groups) * DS;
    for (int t0 = 0; t0 < rows; t0 += kScanStage) {
        const int n = rows - t0 < kScanStage ? rows - t0 : kScanStage;
        __syncthreads();
        for (int t = tid; t < n * (DS / 4); t += T) {
            const int i = t / (DS / 4), c = t - i * (DS / 4);
            const long long base = static_cast<long long>(t0 + i) * cd + 4 * c;
            bs[i][c] = make_float4(load(xc, kind, base + boff), load(xc, kind, base + boff + 1),
                                   load(xc, kind, base + boff + 2), load(xc, kind, base + boff + 3));
            cs[i][c] = make_float4(load(xc, kind, base + coff), load(xc, kind, base + coff + 1),
                                   load(xc, kind, base + coff + 2), load(xc, kind, base + coff + 3));
        }
        for (int t = tid; t < n * kScanRows; t += T) {
            const int i = t / kScanRows, l = t - i * kScanRows, dd = blockIdx.y * kScanRows + l;
            xin[i][l] = load(xc, kind, static_cast<long long>(t0 + i) * cd + h * dh + dd);
            gate[i][l] = rounded(silu(load(P, kind, static_cast<long long>(t0 + i) * proj + h * dh + dd)), kind);
        }
        if (tid < n) dts[tid] = dt[static_cast<long long>(t0 + tid) * heads + h];
        __syncthreads();
        for (int i = 0; i < n; ++i) {
            const float dtv = dts[i], x = xin[i][local];
            const float da = expf(a * dtv), xdt = x * dtv;
            float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                const float4 b = bs[i][TPR * j + q], c = cs[i][TPR * j + q];
                s[4 * j] = fmaf(xdt, b.x, s[4 * j] * da);
                s[4 * j + 1] = fmaf(xdt, b.y, s[4 * j + 1] * da);
                s[4 * j + 2] = fmaf(xdt, b.z, s[4 * j + 2] * da);
                s[4 * j + 3] = fmaf(xdt, b.w, s[4 * j + 3] * da);
                m[0] = fmaf(s[4 * j], c.x, m[0]);
                m[1] = fmaf(s[4 * j + 1], c.y, m[1]);
                m[2] = fmaf(s[4 * j + 2], c.z, m[2]);
                m[3] = fmaf(s[4 * j + 3], c.w, m[3]);
            }
            float out = (m[0] + m[1]) + (m[2] + m[3]);
            out = out + __shfl_xor(out, 1, 32);
            out = out + __shfl_xor(out, 2, 32);
            if (t0 + i + 1 < rows) {
                float* at = snap + (t0 + i + 1) * stride + (s0 - state);
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    *reinterpret_cast<float4*>(at + 4 * (TPR * j + q)) =
                        make_float4(s[4 * j], s[4 * j + 1], s[4 * j + 2], s[4 * j + 3]);
                }
            }
            if (q != 0) continue;
            const float yv = rounded(fmaf(x, dskip, out), kind);
            store(y, kind, static_cast<long long>(t0 + i) * xd + h * dh + d, gate[i][local] * yv);
        }
    }
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        *reinterpret_cast<float4*>(s0 + 4 * (TPR * j + q)) =
            make_float4(s[4 * j], s[4 * j + 1], s[4 * j + 2], s[4 * j + 3]);
    }
}

// A window's launches: its rows over one stream's state, each row's state kept in a snapshot slot.
struct MambaWindow {
    const void* P;  // in_proj rows: [z | x B C | dt]
    int kind;
    void* conv;     // the conv tail (kMambaKept, cd), in place
    float* ssm;     // the SSM state (heads, dh, kMambaState), in place
    void* xc;
    float* dt;
    void* y;
    const float* cw;
    const float* cb;
    const float* dt_bias;
    const float* a_log;
    const float* dsk;
    void* snap_conv;  // where slot 0's conv tail would be (slot 0 is never written), or null for no snapshots
    float* snap_ssm;  // where slot 0's SSM state would be
    long long stride; // bytes from a slot to the next
    int rows, proj, xd, cd, heads, dh, groups;
    float lo, hi;
};

// tf_mamba_conv and tf_mamba_dt in one launch (the blocks past the conv's do dt), each tail but the last kept.
extern "C" __global__ void __launch_bounds__(256) tf_mamba_conv_window(MambaWindow m) {
    const int conv_blocks = (m.cd + 255) / 256;
    if (static_cast<int>(blockIdx.x) >= conv_blocks) {
        const long long i = static_cast<long long>(blockIdx.x - conv_blocks) * 256 + threadIdx.x;
        if (i >= static_cast<long long>(m.rows) * m.heads) return;
        const int r = static_cast<int>(i / m.heads), h = static_cast<int>(i % m.heads);
        const float v = load(m.P, m.kind, static_cast<long long>(r) * m.proj + m.xd + m.cd + h) + m.dt_bias[h];
        const float sp = fmaxf(v, 0.0f) + log1pf(expf(-fabsf(v)));
        m.dt[i] = sp != sp ? sp : fminf(fmaxf(sp, m.lo), m.hi);  // as tf_mamba_dt
        return;
    }
    const int ch = blockIdx.x * 256 + threadIdx.x;
    if (ch >= m.cd) return;
    const int kind = m.kind, cd = m.cd;
    auto raw = [&](int t) -> float {
        if (t < kMambaKept) return load(m.conv, kind, static_cast<long long>(t) * cd + ch);
        return load(m.P, kind, static_cast<long long>(t - kMambaKept) * m.proj + m.xd + ch);
    };
    auto keep = [&](void* to, float a, float b, float c) {
        store(to, kind, ch, a);
        store(to, kind, cd + ch, b);
        store(to, kind, 2LL * cd + ch, c);
    };
    char* snap = static_cast<char*>(m.snap_conv);
    float next[kMambaKept];
    for (int j = 0; j < kMambaKept; ++j) next[j] = raw(m.rows + j);
    float t0 = raw(0), t1 = raw(1), t2 = raw(2);
    const float w0 = m.cw[ch * kMambaTaps], w1 = m.cw[ch * kMambaTaps + 1];
    const float w2 = m.cw[ch * kMambaTaps + 2], w3 = m.cw[ch * kMambaTaps + 3];
    const float bias = m.cb[ch];
    for (int r = 0; r < m.rows; ++r) {
        const float cur = raw(r + kMambaKept);
        float acc = bias;
        acc = fmaf(w0, t0, acc);
        acc = fmaf(w1, t1, acc);
        acc = fmaf(w2, t2, acc);
        acc = fmaf(w3, cur, acc);
        store(m.xc, kind, static_cast<long long>(r) * cd + ch, silu(rounded(acc, kind)));
        t0 = t1;
        t1 = t2;
        t2 = cur;
        if (snap != nullptr && r + 1 < m.rows) keep(snap + (r + 1) * m.stride, t0, t1, t2);
    }
    for (int j = 0; j < kMambaKept; ++j) store(m.conv, kind, static_cast<long long>(j) * cd + ch, next[j]);
}

// tf_mamba_scan over a window's rows, each state but the last into its slot (the group norm is its own launch).
extern "C" __global__ void __launch_bounds__(kScanRows * kScanLanes) tf_mamba_scan_snap(MambaWindow m) {
    scan_snap(m.P, m.xc, m.kind, m.dt, m.ssm, m.a_log, m.dsk, m.y, m.rows, m.proj, m.xd, m.cd, m.heads, m.dh, m.groups,
              m.snap_ssm, m.stride / static_cast<long long>(sizeof(float)));
}
