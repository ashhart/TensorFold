// The Neural Engine's MLP share for prompt chunks of up to 128 rows: its fp16 input planes, SwiGLU on its planes.
#include <metal_stdlib>
using namespace metal;
Q27_SPECIALIZE

// X bf16 [rows][H] into fp16 planes [H][128]; rows past the chunk are zero.
[[kernel]] void q27_ane_input(const device bfloat* X [[buffer(0)]], const device int32_t* dims [[buffer(1)]], device half* Y [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
  const uint m = pos.x, k = pos.y;
  Y[k * 128 + m] = int(m) < dims[0] ? half(float(X[m * H + k])) : half(0.0h);
}

// q27_mlp_xs over 64-column groups from `first`; gate and up below A come from the Neural Engine's fp16 planes.
[[kernel]] void q27_mlp_ane(const device bfloat* GATE [[buffer(0)]], const device bfloat* UP [[buffer(1)]], const device int32_t* dims [[buffer(2)]], device bfloat* HOUT [[buffer(3)]], device float* XS [[buffer(4)]], const device half* PLANES [[buffer(5)]], constant uint& first [[buffer(6)]], uint3 tpt [[thread_position_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const uint t = tpt.x, g = tg.x + first, m = tg.y;
  const int M = dims[0], MP = dims[1];
  threadgroup bfloat hb[64];
  if (int(m) >= M) {
    if (t == 0) XS[g * MP + m] = 0.0f;
    return;
  }
  const int j = int(g) * 64 + int(t);
  float gf, uf;
  if (j < A) {
    gf = float(PLANES[j * 128 + m]);
    uf = float(PLANES[(A + j) * 128 + m]);
  } else {
    const int fused = int(m) * 2 * N + j;
    gf = float(GATE[fused]);
    uf = float(UP[fused]);
  }
  const bfloat h = bfloat(gf / (1.0f + metal::exp(-gf)) * uf);
  HOUT[int(m) * N + j] = h;
  hb[t] = h;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float acc = 0.0f;
    for (int i = 0; i < 64; i++) acc += float(hb[i]);
    XS[g * MP + m] = acc;
  }
}

// The down projection: the GPU's fp32 partial plus the Neural Engine's fp16 partial planes, rounded once to bf16.
[[kernel]] void q27_add_ane(const device float* P [[buffer(0)]], const device half* PLANES [[buffer(1)]], device bfloat* Y [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
  const uint o = pos.x, m = pos.y;
  Y[m * H + o] = bfloat(P[m * H + o] + float(PLANES[o * 128 + m]));
}
