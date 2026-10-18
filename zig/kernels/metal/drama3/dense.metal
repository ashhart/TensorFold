// Drama 3 dense rows: bf16 projections, zero-centred RMS norms, frame embeddings, SwiGLU, MoE combine, tower mix.
#include "d3_common.h"

struct RowsArgs {
  uint K, N, rows, xs, ys, f32out;
};

// y[r, n] = x[r] . W[n]: 4 columns a simdgroup, R rows; lane chains then simd_sum, so bits do not depend on R.
template <int R>
inline void rows_body(device const bf16* W, device const bf16* X, device void* Y, constant RowsArgs& a, uint2 tg,
                      uint sg, uint lane) {
  const uint n0 = (tg.x * 4 + sg) * 4;
  const uint r0 = tg.y * R;
  if (n0 >= a.N) return;
  const uint nr = min(uint(R), a.rows - r0);
  device const uint4* w[4];
  for (uint j = 0; j < 4; ++j) w[j] = (device const uint4*)(W + (ulong)min(n0 + j, a.N - 1) * a.K);
  float acc[R][4];
  for (int r = 0; r < R; ++r)
    for (int j = 0; j < 4; ++j) acc[r][j] = 0.0f;
  for (uint c = lane; c < a.K / 8; c += 32) {
    uint4 wv[4];
    for (int j = 0; j < 4; ++j) wv[j] = w[j][c];
    for (int r = 0; r < R; ++r) {
      if (uint(r) >= nr) break;
      const uint4 xv = ((device const uint4*)(X + (ulong)(r0 + r) * a.xs))[c];
      const float4 x0 = bf4(xv.xy), x1 = bf4(xv.zw);
      for (int j = 0; j < 4; ++j) {
        const float4 w0 = bf4(wv[j].xy), w1 = bf4(wv[j].zw);
        float s = acc[r][j];
        s = fma(w0.x, x0.x, s);
        s = fma(w0.y, x0.y, s);
        s = fma(w0.z, x0.z, s);
        s = fma(w0.w, x0.w, s);
        s = fma(w1.x, x1.x, s);
        s = fma(w1.y, x1.y, s);
        s = fma(w1.z, x1.z, s);
        s = fma(w1.w, x1.w, s);
        acc[r][j] = s;
      }
    }
  }
  for (int r = 0; r < R; ++r) {
    if (uint(r) >= nr) break;
    for (uint j = 0; j < 4; ++j) {
      const float v = simd_sum(acc[r][j]);
      if (lane == 0 && n0 + j < a.N) {
        const ulong at = (ulong)(r0 + r) * a.ys + n0 + j;
        if (a.f32out) ((device float*)Y)[at] = rb(v);  // logits: rounded to bf16 as the bf16 heads return them
        else ((device bf16*)Y)[at] = f2b(v);
      }
    }
  }
}

[[max_total_threads_per_threadgroup(128)]] kernel void d3_rows_r1(device const bf16* W [[buffer(0)]], device const bf16* X [[buffer(1)]],
                       device void* Y [[buffer(2)]], constant RowsArgs& a [[buffer(3)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  rows_body<1>(W, X, Y, a, tg, sg, lane);
}

[[max_total_threads_per_threadgroup(128)]] kernel void d3_rows_r8(device const bf16* W [[buffer(0)]], device const bf16* X [[buffer(1)]],
                       device void* Y [[buffer(2)]], constant RowsArgs& a [[buffer(3)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  rows_body<8>(W, X, Y, a, tg, sg, lane);
}

struct NormArgs {
  uint xs, ys, D, zero_centred;
  float eps;
};

// y = bf16(x * rsqrt(mean(x^2) + eps) * g), g = 1 + w when zero_centred else w, fp32 inside; a group a row.
[[max_total_threads_per_threadgroup(256)]] kernel void d3_rmsnorm(device const bf16* x [[buffer(0)]], device const bf16* w [[buffer(1)]],
                       device bf16* y [[buffer(2)]], constant NormArgs& a [[buffer(3)]],
                       uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                       uint nt [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  device const bf16* xr = x + (ulong)row * a.xs;
  device bf16* yr = y + (ulong)row * a.ys;
  float ss = 0.0f;
  for (uint i = t; i < a.D; i += nt) {
    const float v = b2f(xr[i]);
    ss += v * v;
  }
  const float inv = rsqrt(group_sum(ss, scratch, sg, lane, nt / 32) / float(a.D) + a.eps);
  for (uint i = t; i < a.D; i += nt) {
    const float g = a.zero_centred ? 1.0f + b2f(w[i]) : b2f(w[i]);
    yr[i] = f2b(b2f(xr[i]) * inv * g);
  }
}

struct AddNormArgs {
  uint as, D;
  float eps;
};

// r = bf16(r + a) (a's rows `as` apart), then y = zero-centred rmsnorm(r); r and y are [rows, D] contiguous.
[[max_total_threads_per_threadgroup(256)]] kernel void d3_add_rmsnorm(device bf16* r [[buffer(0)]], device const bf16* add [[buffer(1)]],
                           device const bf16* w [[buffer(2)]], device bf16* y [[buffer(3)]],
                           constant AddNormArgs& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                           uint t [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  device bf16* rr = r + (ulong)row * a.D;
  device const bf16* ar = add + (ulong)row * a.as;
  device bf16* yr = y + (ulong)row * a.D;
  float ss = 0.0f;
  for (uint i = t; i < a.D; i += nt) {
    const float v = rb(b2f(rr[i]) + b2f(ar[i]));
    rr[i] = f2b(v);
    ss += v * v;
  }
  const float inv = rsqrt(group_sum(ss, scratch, sg, lane, nt / 32) / float(a.D) + a.eps);
  for (uint i = t; i < a.D; i += nt) yr[i] = f2b(b2f(rr[i]) * inv * (1.0f + b2f(w[i])));
}

struct EmbedArgs {
  uint D;
  int sem_begin, sem_end, cb;
};

// Frame rows [rows, 11] to embeddings: text rows embed[main], semantic rows bf16(bf16(embed + cb1[code1]) / sqrt 2).
[[max_total_threads_per_threadgroup(256)]] kernel void d3_embed_backbone(device const int* frames [[buffer(0)]], device const bf16* embed [[buffer(1)]],
                              device const bf16* cb_emb [[buffer(2)]], device bf16* out [[buffer(3)]],
                              constant EmbedArgs& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                              uint t [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
  device const int* f = frames + (ulong)row * 11;
  const int tok = f[0];
  const bool sem = is_semantic(tok, a.sem_begin, a.sem_end);
  const int code1 = clamp(f[2], 0, a.cb - 1);
  device const bf16* e = embed + (ulong)tok * a.D;
  device const bf16* c = cb_emb + (ulong)(a.cb + code1) * a.D;
  device bf16* o = out + (ulong)row * a.D;
  const float scale = 0.70710678118654752f;
  for (uint i = t; i < a.D; i += nt) o[i] = sem ? f2b(rb(b2f(e[i]) + b2f(c[i])) * scale) : e[i];
}

struct SwigluArgs {
  uint stride, g_off, u_off, N;
};

// y [T, N] = bf16(bf16(silu(g)) * u) with g, u at offsets of rows `stride` apart; grid (N / 256, T).
[[max_total_threads_per_threadgroup(256)]] kernel void d3_swiglu(device const bf16* x [[buffer(0)]], device bf16* y [[buffer(1)]],
                      constant SwigluArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.x, t = gid.y;
  if (i >= a.N) return;
  const float g = b2f(x[(ulong)t * a.stride + a.g_off + i]), u = b2f(x[(ulong)t * a.stride + a.u_off + i]);
  y[(ulong)t * a.N + i] = f2b(rb(d3_silu(g)) * u);
}

struct CombineArgs {
  uint ss, sg_off, D;
};

// r = bf16(r + bf16(bf16(sum_k w_k down_k) + bf16(bf16(sigmoid(sg)) * shared))); a group a row.
[[max_total_threads_per_threadgroup(256)]] kernel void d3_moe_combine(device bf16* r [[buffer(0)]], device const float* down [[buffer(1)]],
                           device const float* w [[buffer(2)]], device const bf16* shared_out [[buffer(3)]],
                           device const bf16* sp [[buffer(4)]], constant CombineArgs& a [[buffer(5)]],
                           uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                           uint nt [[threads_per_threadgroup]]) {
  const float sgate = rb(d3_sigmoid(b2f(sp[(ulong)row * a.ss + a.sg_off])));
  float wk[8];
  for (int k = 0; k < 8; ++k) wk[k] = w[row * 8 + k];
  for (uint i = t; i < a.D; i += nt) {
    float acc = 0.0f;
    for (int k = 0; k < 8; ++k) acc = fma(wk[k], down[((ulong)row * 8 + k) * a.D + i], acc);
    const float sh = rb(sgate * b2f(shared_out[(ulong)row * a.D + i]));
    const float moe = rb(rb(acc) + sh);
    r[(ulong)row * a.D + i] = f2b(b2f(r[(ulong)row * a.D + i]) + moe);
  }
}

struct MixArgs {
  uint lfs, code_col, with_shared, D;
  int sem_begin, sem_end, cb;
  float eps;
};

// Tower input [enorm(e) | hnorm(h)]: e = table[label code], plus on layer 0 the input frame's code rows 2..9.
[[max_total_threads_per_threadgroup(256)]] kernel void d3_tower_mix(device const int* lf [[buffer(0)]], device const int* inf [[buffer(1)]],
                         device const bf16* table [[buffer(2)]], device const bf16* cb_emb [[buffer(3)]],
                         device const bf16* h [[buffer(4)]], device const bf16* enorm_w [[buffer(5)]],
                         device const bf16* hnorm_w [[buffer(6)]], device bf16* out [[buffer(7)]],
                         constant MixArgs& a [[buffer(8)]], uint row [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float ev[2048];
  threadgroup float scratch[32];
  device const int* l = lf + (ulong)row * a.lfs;
  device const int* in = inf + (ulong)row * 11;
  const bool lsem = is_semantic(l[0], a.sem_begin, a.sem_end);
  const int code = clamp(l[a.code_col], 0, a.cb - 1);
  const bool shared_codes = a.with_shared && is_semantic(in[0], a.sem_begin, a.sem_end);
  float se = 0.0f, sh = 0.0f;
  for (uint i = t; i < a.D; i += nt) {
    float e = lsem ? b2f(table[(ulong)code * a.D + i]) : 0.0f;
    if (shared_codes) {
      float s = 0.0f;
      for (int c = 2; c < 10; ++c) s += b2f(cb_emb[(ulong)(c * a.cb + clamp(in[1 + c], 0, a.cb - 1)) * a.D + i]);
      e = rb(e + rb(s));
    }
    ev[i] = e;
    se += e * e;
    const float hv = b2f(h[(ulong)row * a.D + i]);
    sh += hv * hv;
  }
  const float ie = rsqrt(group_sum(se, scratch, sg, lane, nt / 32) / float(a.D) + a.eps);
  const float ih = rsqrt(group_sum(sh, scratch + 16, sg, lane, nt / 32) / float(a.D) + a.eps);
  device bf16* o = out + (ulong)row * 2 * a.D;
  for (uint i = t; i < a.D; i += nt) {
    o[i] = f2b(ev[i] * ie * (1.0f + b2f(enorm_w[i])));
    o[a.D + i] = f2b(b2f(h[(ulong)row * a.D + i]) * ih * (1.0f + b2f(hnorm_w[i])));
  }
}
