// Owned learner algebra with generated Metal bindings; regenerate with tools/zig/metal_learning_sources.py.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bf16;
struct GdnShape {
    int rows, nk, nv, conv, proj;
};
constant constexpr int DK = 128, DV = 128, TAPS = 4;
constant constexpr int CK = 4;
constant constexpr int SPLIT = 4;
constant constexpr int ROWS = DV / SPLIT;
constant constexpr int WARPS = ROWS / 4;

namespace tf_metal_learn {
struct Context {
  uint3 threadIdx, blockIdx, blockDim;
  threadgroup float* shared;
  inline float tf_log1p(float x) { return metal::log(1.0f + x); }
  inline float warp_sum(float v) {
    for (ushort mask = 16; mask > 0; mask >>= 1) v += simd_shuffle_xor(v, mask);
    return v;
  }
  inline float warp_max(float v) {
    for (ushort mask = 16; mask > 0; mask >>= 1) v = max(v, simd_shuffle_xor(v, mask));
    return v;
  }
  inline float block_sum_tree(float v, threadgroup float* scratch, int warps) {
    const uint lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) scratch[warp] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = warp == 0 && lane < uint(warps) ? scratch[lane] : 0.0f;
    if (warp == 0) total = warp_sum(total);
    if (threadIdx.x == 0) scratch[0] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
  }
  inline float block_max_tree(float v, threadgroup float* scratch, int warps) {
    const uint lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) scratch[warp] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = warp == 0 && lane < uint(warps) ? scratch[lane] : -INFINITY;
    if (warp == 0) total = warp_max(total);
    if (threadIdx.x == 0) scratch[0] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
  }
inline float f32(bf16 v) { return float(v); }
inline bf16 b16(float v) { return bfloat(v); }
inline float bf(float x) { return f32(b16(x)); }

inline float sigmoidf_(float x) { return 1.0f / (1.0f + precise::exp(-x)); }
inline float softplusf_(float x) { return x > 20.0f ? x : tf_log1p(precise::exp(x)); }

inline float silu_grad(float v) {
    v = bf(v);
    const float s = sigmoidf_(v);
    return bf(s) + v * s * (1.0f - s);
}

inline int chunks(int rows) { return (rows + CK - 1) / CK; }

inline float conv_acc(const device bf16* P, const device bf16* hist, const device bf16* cw, int r, int c, thread const GdnShape& s) {
    float acc = 0.0f;
#pragma unroll
    for (int tap = 0; tap < TAPS; ++tap) {
        const int at = r - (TAPS - 1) + tap;
        const float x = at >= 0 ? f32(P[(size_t)at * s.proj + c])
                                : (hist != nullptr ? f32(hist[(size_t)(at + TAPS - 1) * s.conv + c]) : 0.0f);
        acc = acc + f32(cw[c * TAPS + tap]) * x;
    }
    return acc;
}

inline void load_state(thread float (&s)[4][4], const device float* base, int warp, int lane) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 v = base != nullptr ? *reinterpret_cast<const device float4*>(base + (warp * 4 + j) * DK + lane * 4)
                                         : float4(0.0f, 0.0f, 0.0f, 0.0f);
        s[j][0] = v.x; s[j][1] = v.y; s[j][2] = v.z; s[j][3] = v.w;
    }
}

inline void store_state(thread const float (&s)[4][4], device float* base, int warp, int lane) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
        *reinterpret_cast<device float4*>(base + (warp * 4 + j) * DK + lane * 4) = float4(s[j][0], s[j][1], s[j][2], s[j][3]);
}

inline void update(thread float (&s)[4][4], thread const float (&kk)[4], const threadgroup float* vrow, int warp, float g,
                                       float beta) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[j][i] = s[j][i] * g;
            kv = kv + s[j][i] * kk[i];
        }
        kv = warp_sum(kv);
        const float delta = (vrow[warp * 4 + j] - kv) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
    }
}
  void tf_gdn_norm_back(const device bf16*  y, const device bf16*  P,
                                                                   const device bf16*  norm_w, const device float*  dout,
                                                                   device float*  dy, device float*  dproj, GdnShape s,
                                                                   float eps) {

    const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    threadgroup float* ys = shared + 0;
    threadgroup float* scratch = shared + 128;
    threadgroup float& rinv = shared[160];
    const size_t row = ((size_t)r * s.nv + h) * DV;
    ys[t] = f32(y[row + t]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float v = ys[lane * 4 + i]; ss = ss + v * v; }
        ss = warp_sum(ss);
        if (lane == 0) rinv = 1.0f / precise::sqrt(ss / (float)DV + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float w = f32(norm_w[t]), yv = ys[t], ri = rinv;
    const float yn = bf(yv * ri * w);
    const size_t zc = (size_t)r * s.proj + s.conv + h * DV + t;
    const float z = f32(P[zc]), sg = sigmoidf_(z), d = dout[row + t];
    dproj[zc] = d * yn * sg * (1.0f + z * (1.0f - sg));
    const float dn = d * z * sg * w;
    const float dot = block_sum_tree(yv * dn, scratch, 4);
    dy[row + t] = ri * dn - ri * ri * ri * yv * dot / (float)DV;

  }
  void tf_gdn_scan_back(const device bf16*  act, const device float*  q,
                                                                   const device float*  k, const device float*  g,
                                                                   const device float*  beta, const device float*  ckpt,
                                                                   const device float*  dy, device float*  work,
                                                                   device float*  dqp, device float*  dkp,
                                                                   device float*  dact, device float*  dg,
                                                                   device float*  dbeta, GdnShape s, int from) {

    threadgroup float* qs = shared + 0;
    threadgroup float* ks = shared + 128;
    threadgroup float* vs = shared + 256;
    threadgroup float* dos = shared + 288;
    threadgroup float (*red_q)[DK] = reinterpret_cast<threadgroup float (*)[DK]>(shared + 320);
    threadgroup float (*red_k)[DK] = reinterpret_cast<threadgroup float (*)[DK]>(shared + 1344);
    threadgroup float* red_g = shared + 2368;
    threadgroup float* red_b = shared + 2376;
    const int h = blockIdx.x / SPLIT, part = blockIdx.x % SPLIT, rb = part * ROWS, hk = h / (s.nv / s.nk);
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const size_t size = (size_t)DV * DK, mine_size = (size_t)ROWS * DK;
    const int v0 = 2 * s.nk * DK + h * DV + rb, nch = chunks(s.rows);
    const device float* saved = ckpt + (size_t)h * size + rb * DK;
    device float* mine = work + (size_t)blockIdx.x * CK * mine_size;
    float ds[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) ds[j][i] = 0.0f;
    for (int c = nch - 1; c >= from / CK; --c) {
        const int first = c * CK, last = min(first + CK, s.rows);
        {

            float st[4][4];
            load_state(st, saved + (size_t)c * s.nv * size, warp, lane);
            float in_kv = 0.0f, in_g = 0.0f, in_b = 0.0f;
            auto fetch = [&](int r) {
                if (t < DK) in_kv = k[((size_t)r * s.nk + hk) * DK + t];
                else if (t < DK + ROWS) in_kv = f32(act[(size_t)r * s.conv + v0 + t - DK]);
                in_g = g[r * s.nv + h];
                in_b = beta[r * s.nv + h];
            };
            if (first + 1 < last) fetch(first);
            for (int r = first; r < last; ++r) {
                store_state(st, mine + (r - first) * mine_size, warp, lane);
                if (r + 1 == last) break;
                if (t < DK) ks[t] = in_kv;
                else if (t < DK + ROWS) vs[t - DK] = in_kv;
                const float gg = in_g, bb = in_b;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (r + 2 < last) fetch(r + 1);
                float kk[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) kk[i] = ks[lane * 4 + i];
                update(st, kk, vs, warp, gg, bb);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        float in_qk, in_vd = 0.0f, in_g, in_b;
        auto fetch = [&](int r) {
            in_qk = t < DK ? q[((size_t)r * s.nk + hk) * DK + t] : k[((size_t)r * s.nk + hk) * DK + t - DK];
            if (t < ROWS) in_vd = f32(act[(size_t)r * s.conv + v0 + t]);
            else if (t < 2 * ROWS) in_vd = dy[((size_t)r * s.nv + h) * DV + rb + t - ROWS];
            in_g = g[r * s.nv + h];
            in_b = beta[r * s.nv + h];
        };
        fetch(last - 1);
        for (int r = last - 1; r >= first; --r) {
            if (t < DK) qs[t] = in_qk;
            else ks[t - DK] = in_qk;
            if (t < ROWS) vs[t] = in_vd;
            else if (t < 2 * ROWS) dos[t - ROWS] = in_vd;
            const float gg = in_g, bb = in_b;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (r > first) fetch(r - 1);
            float kk[4], qq[4], pq[4], pk[4], pg = 0.0f, pb = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                kk[i] = ks[lane * 4 + i];
                qq[i] = qs[lane * 4 + i];
                pq[i] = 0.0f;
                pk[i] = 0.0f;
            }
            const device float* prev = mine + (r - first) * mine_size;
            float4 p4[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) p4[j] = *reinterpret_cast<const device float4*>(prev + (warp * 4 + j) * DK + lane * 4);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int row = warp * 4 + j;
                const float sp[4] = {p4[j].x, p4[j].y, p4[j].z, p4[j].w};
                float sd[4], kv = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sd[i] = sp[i] * gg;
                    kv = kv + sd[i] * kk[i];
                }
                kv = warp_sum(kv);
                const float err = vs[row] - kv, delta = err * bb, dj = dos[row];
                float dd = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    pq[i] += dj * (sd[i] + kk[i] * delta);
                    ds[j][i] += dj * qq[i];
                    dd += ds[j][i] * kk[i];
                }
                dd = warp_sum(dd);
                const float dkv = -bb * dd;
                if (lane == 0) {
                    dact[(size_t)r * s.conv + v0 + row] = bb * dd;
                    pb += dd * err;
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    pk[i] += delta * ds[j][i] + dkv * sd[i];
                    const float dsp = ds[j][i] + dkv * kk[i];
                    pg += dsp * sp[i];
                    ds[j][i] = gg * dsp;
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                red_q[warp][lane * 4 + i] = pq[i];
                red_k[warp][lane * 4 + i] = pk[i];
            }
            pg = warp_sum(pg);
            if (lane == 0) {
                red_g[warp] = pg;
                red_b[warp] = pb;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const size_t out = ((size_t)r * s.nv + h) * SPLIT + part;
            if (t < DK) {
                float x = 0.0f;
#pragma unroll
                for (int w = 0; w < WARPS; ++w) x += red_q[w][t];
                dqp[out * DK + t] = x;
                if (t == 0) {
                    float e = 0.0f;
#pragma unroll
                    for (int w = 0; w < WARPS; ++w) e += red_g[w];
                    dg[out] = e;
                }
            } else {
                float x = 0.0f;
#pragma unroll
                for (int w = 0; w < WARPS; ++w) x += red_k[w][t - DK];
                dkp[out * DK + t - DK] = x;
                if (t == DK) {
                    float e = 0.0f;
#pragma unroll
                    for (int w = 0; w < WARPS; ++w) e += red_b[w];
                    dbeta[out] = e;
                }
            }
        }
    }

  }
  void tf_gdn_qk_back(const device bf16*  act, const device float*  dqp,
                                                                 const device float*  dkp, device float*  dact,
                                                                 GdnShape s) {

    threadgroup float* scratch = shared + 0;
    const int r = blockIdx.x, hk = blockIdx.y, t = threadIdx.x, per = s.nv / s.nk;
    for (int which = 0; which < 2; ++which) {
        const device float* part = which == 0 ? dqp : dkp;
        float d = 0.0f;
        for (int h = hk * per; h < (hk + 1) * per; ++h)
            for (int p = 0; p < SPLIT; ++p) d += part[(((size_t)r * s.nv + h) * SPLIT + p) * DK + t];
        const int c = which * s.nk * DK + hk * DK + t;
        const float x = f32(act[(size_t)r * s.conv + c]);
        const float ss = block_sum_tree(x * x, scratch, 4);
        const float dot = block_sum_tree(x * d, scratch, 4);
        const float inv = 1.0f / precise::sqrt(ss / float(DK) + 1e-6f), sc = which == 0 ? bf(1.0f / float(DK)) : bf(1.0f / precise::sqrt(float(DK)));
        dact[(size_t)r * s.conv + c] = sc * inv * (d - inv * inv * x * dot / float(DK));
    }

  }
  void tf_gdn_gates_back(const device bf16*  P, const device float*  a_log,
                                             const device float*  dt_bias, const device float*  g,
                                             const device float*  dg, const device float*  dbeta,
                                             device float*  dproj, GdnShape s) {

    const int h = threadIdx.x, r = blockIdx.x;
    if (h >= s.nv || r >= s.rows) return;
    const size_t at = (size_t)r * s.proj + s.conv + s.nv * DV;
    const float b = f32(P[at + h]), x = bf(f32(P[at + s.nv + h]) + dt_bias[h]);
    const float sb = sigmoidf_(b), sx = sigmoidf_(x);
    float db = 0.0f, dgs = 0.0f;
    for (int p = 0; p < SPLIT; ++p) {
        db += dbeta[(r * s.nv + h) * SPLIT + p];
        dgs += dg[(r * s.nv + h) * SPLIT + p];
    }
    dproj[at + h] = db * sb * (1.0f - sb);
    dproj[at + s.nv + h] = -dgs * g[r * s.nv + h] * precise::exp(a_log[h]) * sx;

  }
  void tf_gdn_conv_back(const device bf16*  P, const device bf16*  hist,
                                            const device bf16*  cw, const device float*  dact,
                                            device float*  dproj, GdnShape s) {

    const int c = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y;
    if (c >= s.conv || r >= s.rows) return;
    float acc = 0.0f;
#pragma unroll
    for (int tap = 0; tap < TAPS; ++tap) {
        const int at = r + TAPS - 1 - tap;
        if (at >= s.rows) continue;
        const float pre = conv_acc(P, hist, cw, at, c, s);
        acc += f32(cw[c * TAPS + tap]) * dact[(size_t)at * s.conv + c] * silu_grad(pre);
    }
    dproj[(size_t)r * s.proj + c] = acc;

  }
};
}
