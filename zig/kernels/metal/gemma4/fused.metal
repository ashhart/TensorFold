// Gemma 4's decode glue inside the core reg lane kernel (TF_FUSED), one row, f.x = eps: glue.metal's arithmetic.
#include <metal_stdlib>
using namespace metal;
constant constexpr int FT = 256; // the tails' threads, the glue kernels' T (their reductions' order)
constant constexpr int FKG = TF_K / TF_GROUP;

inline float gf_reduce(float acc, threadgroup float* p, uint lane, uint sg) {
  const float s = simd_sum(acc);
  if (lane == 0) p[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float out = 0.0f;
  for (int q = 0; q < FT / 32; q++) out += p[q];
  return out;
}

// gemma_row_sums over the row in threadgroup memory: each group's values added in order from zero.
inline void gf_sums(const threadgroup bfloat* row, threadgroup float* xsum, uint tid) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int g = int(tid); g < FKG; g += FT) {
    float acc = 0.0f;
    for (int i = 0; i < TF_GROUP; i++) acc += float(row[g * TF_GROUP + i]);
    xsum[g] = acc;
  }
}

#if TF_FUSE == 1
static_assert(TF_THREADS == FT, "the dense tail reduces over 256 threads");
// TF_FUSE 1, gemma_dense_tail (the qkv input): a0 H, a1 Y, a2 post-MLP norm, a3 scalar, a4 next norm; o0 HN, o1 tap
inline void tf_prologue(tf_fused fa, const device bfloat* X, threadgroup bfloat* xs, threadgroup float* xsum, threadgroup float* p, bool first, uint tid) {
  constexpr int PER = TF_K / FT;
  threadgroup float* p1 = p;
  threadgroup float* p2 = p + FT / 32;
  const uint lane = tid & 31, sg = tid >> 5;
  float yv[PER], hin[PER], wp[PER], wn[PER];
  for (int i = 0; i < PER; i++) {
    const int c = int(tid) + i * FT;
    yv[i] = float(fa.a1[c]); hin[i] = float(fa.a0[c]);
    wp[i] = float(fa.a2[c]); wn[i] = float(fa.a4[c]);
  }
  const float sc = float(fa.a3[0]);
  float ss1 = 0.0f;
  for (int i = 0; i < PER; i++) ss1 = fma(yv[i], yv[i], ss1);
  const float inv1 = metal::precise::rsqrt(gf_reduce(ss1, p1, lane, sg) / float(TF_K) + fa.f.x);
  float hv[PER];
  float ss2 = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int c = int(tid) + i * FT;
    const float b = float(bfloat(wp[i] * float(bfloat(yv[i] * inv1))));
    const bfloat hs = bfloat(hin[i] + b);
    const bfloat hn = bfloat(float(hs) * sc);
    if (first) fa.o0[c] = hn;
    if (first && fa.i.x) fa.o1[c] = hn;
    hv[i] = float(hn);
    ss2 = fma(hv[i], hv[i], ss2);
  }
  const float inv2 = metal::precise::rsqrt(gf_reduce(ss2, p2, lane, sg) / float(TF_K) + fa.f.x);
  for (int i = 0; i < PER; i++) {
    const int c = int(tid) + i * FT;
    xs[c] = bfloat(wn[i] * float(bfloat(hv[i] * inv2)));
  }
  gf_sums(xs, xsum, tid);
}
#endif

#if TF_FUSE == 2
static_assert(TF_THREADS == FT, "the attention tail reduces over 256 threads");
// TF_FUSE 2, gemma_attn_tail (the gate/up input): a0 H, a1 O, a2 post-attention norm, a3 pre-MLP norm; o0 HN.
inline void tf_prologue(tf_fused fa, const device bfloat* X, threadgroup bfloat* xs, threadgroup float* xsum, threadgroup float* p, bool first, uint tid) {
  constexpr int PER = TF_K / FT;
  threadgroup float* p1 = p;
  threadgroup float* p2 = p + FT / 32;
  const uint lane = tid & 31, sg = tid >> 5;
  float ov[PER], hin[PER], wa[PER], w1[PER];
  for (int i = 0; i < PER; i++) {
    const int c = int(tid) + i * FT;
    ov[i] = float(fa.a1[c]); hin[i] = float(fa.a0[c]);
    wa[i] = float(fa.a2[c]); w1[i] = float(fa.a3[c]);
  }
  float ss = 0.0f;
  for (int i = 0; i < PER; i++) ss = fma(ov[i], ov[i], ss);
  const float inv1 = metal::precise::rsqrt(gf_reduce(ss, p1, lane, sg) / float(TF_K) + fa.f.x);
  float hv[PER];
  float ss2 = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int c = int(tid) + i * FT;
    const float a = float(bfloat(wa[i] * float(bfloat(ov[i] * inv1))));
    const bfloat hn = bfloat(hin[i] + a);
    if (first) fa.o0[c] = hn;
    hv[i] = float(hn);
    ss2 = fma(hv[i], hv[i], ss2);
  }
  const float inv2 = metal::precise::rsqrt(gf_reduce(ss2, p2, lane, sg) / float(TF_K) + fa.f.x);
  for (int i = 0; i < PER; i++) {
    const int c = int(tid) + i * FT;
    xs[c] = bfloat(w1[i] * float(bfloat(hv[i] * inv2)));
  }
  gf_sums(xs, xsum, tid);
}

// gemma_geglu over the gate/up stack's 16-row interleaved blocks: C[i] a gate column, C[i + 8] its up column; Y = act.
inline void tf_epilogue(tf_fused fa, thread const float* C, int M, int mb, int n, int tile, device bfloat* Y) {
  for (int i = 0; i < 8; i++) {
    const int m = mb + 8 * ((i >> 2) & 1), c = tile * 16 + (n - tile * 32) + (i & 3);
    if (m >= M) continue;
    const bfloat x = bfloat(C[i]), up = bfloat(C[i + 8]);
    const bfloat cube = bfloat(metal::pow(float(x), 3.0f));
    const bfloat inner = bfloat(float(x) + float(bfloat(float(bfloat(0.044715f)) * float(cube))));
    const bfloat th = bfloat(metal::precise::tanh(float(bfloat(float(bfloat(0.7978845608028654f)) * float(inner)))));
    const bfloat half_x = bfloat(0.5f * float(x));
    const bfloat gl = bfloat(float(half_x) * float(bfloat(1.0f + float(th))));
    Y[(size_t)m * (TF_N / 2) + c] = bfloat(float(gl) * float(up));
  }
}
#endif

#if TF_FUSE == 3
// TF_FUSE 3 (the down input): gemma_geglu's group sums over the act row X, each group's values added in order.
inline void tf_prologue(tf_fused fa, const device bfloat* X, threadgroup bfloat* xs, threadgroup float* xsum, threadgroup float* p, bool first, uint tid) {
  for (int g = int(tid); g < FKG; g += TF_THREADS) {
    float acc = 0.0f;
    for (int i = 0; i < TF_GROUP; i++) acc += float(X[g * TF_GROUP + i]);
    xsum[g] = acc;
  }
}
#endif
