// Gemma 4 E4B's bf16 linears (the converter's --dense-ple: per-layer gate, projection and model projection).
#include <metal_stdlib>
using namespace metal;

#ifndef GELU_C0
#define GELU_C0 0.7978845608028654f
#endif
#ifndef GELU_C1
#define GELU_C1 0.044715f
#endif

// Columns a simdgroup computes in the bf16 kernel.
constant constexpr int QMV_COLS = 4;
constant constexpr int QMV_ROWS = 8;

// y = x W^T for a bf16 W [N, K]: QMV_COLS columns a simdgroup, a lane 8 values of K a step (K a multiple of 256),
// QMV_ROWS rows sharing each step's weights. GATE: E4B's per-layer gate, each bf16 output through the tanh GELU
// times the layer's per-layer input PLI[r * ple.y + ple.x + n].
template <bool GATE>
void dense4(const device bfloat* X, const device bfloat* W, device bfloat* Y, int3 dims, const device bfloat* PLI,
            int2 ple, uint lane, uint sg, uint tg, uint sgs) {
  const int rows = dims.x, N = dims.y, K = dims.z;
  const int n0 = (int(tg * sgs) + int(sg)) * QMV_COLS;
  if (n0 >= N) return;
  for (int r0 = 0; r0 < rows; r0 += QMV_ROWS) {
    const int nr = min(QMV_ROWS, rows - r0);
    float acc[QMV_ROWS][QMV_COLS];
    for (int r = 0; r < QMV_ROWS; r++)
      for (int c = 0; c < QMV_COLS; c++) acc[r][c] = 0.0f;
    for (int k0 = int(lane) * 8; k0 < K; k0 += 256) {
      for (int c = 0; c < QMV_COLS; c++) {
        const device bfloat* w = W + size_t(n0 + c) * K + k0;
        float wv[8];
        for (int i = 0; i < 8; i++) wv[i] = float(w[i]);
        for (int r = 0; r < nr; r++) {
          const device bfloat* x = X + size_t(r0 + r) * K + k0;
          for (int i = 0; i < 8; i++) acc[r][c] = fma(float(x[i]), wv[i], acc[r][c]);
        }
      }
    }
    for (int r = 0; r < nr; r++)
      for (int c = 0; c < QMV_COLS; c++) {
        const float v = simd_sum(acc[r][c]);
        if (lane != 0) continue;
        const size_t at = size_t(r0 + r) * N + n0 + c;
        if (!GATE) {
          Y[at] = bfloat(v);
          continue;
        }
        // DeepMind _modules.py Block PLE step: gelu(gate) * per-layer input, tanh GELU with bf16 rounding after each op.
        const float g = float(bfloat(v));
        const float x3 = float(bfloat(g * g * g));
        const float t = float(bfloat(float(bfloat(GELU_C1)) * x3));
        const float s = float(bfloat(g + t));
        const float u = float(bfloat(float(bfloat(GELU_C0)) * s));
        const float th = float(bfloat(precise::tanh(u)));
        const float a = float(bfloat(1.0f + th));
        const float ge = float(bfloat(float(bfloat(0.5f * g)) * a));
        const float pli = float(PLI[size_t(r0 + r) * ple.y + ple.x + n0 + c]);
        Y[at] = bfloat(ge * pli);
      }
  }
}

kernel void gemma_dense4(const device bfloat* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
                         device bfloat* Y [[buffer(2)]], constant int3& dims [[buffer(3)]],
                         uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint tg [[threadgroup_position_in_grid]], uint sgs [[simdgroups_per_threadgroup]]) {
  dense4<false>(X, W, Y, dims, X, int2(0), lane, sg, tg, sgs);
}

kernel void gemma_dense4_gate(const device bfloat* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
                              device bfloat* Y [[buffer(2)]], constant int3& dims [[buffer(3)]],
                              const device bfloat* PLI [[buffer(4)]], constant int2& ple [[buffer(5)]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint tg [[threadgroup_position_in_grid]], uint sgs [[simdgroups_per_threadgroup]]) {
  dense4<true>(X, W, Y, dims, PLI, ple, lane, sg, tg, sgs);
}
