// GLM-5.3-Flash's products at any MLX width: core/row_projection.metal's row sums (compiled first), any window.

struct GlmRowDims {
  int k, n, group, rows;  // a matrix [n, k] at `group`; x rows
  int heads, ldx, hsx;    // matrices back to back (one a head); x's row and head strides
  int ldo, hso, hsw;      // out's row and head strides; a head's matrix rows
};

struct GlmPickDims {
  int k, n, group, slots;  // slots: (row, pick) pairs, or sorted pairs
  int repeat, experts, ldo; // a row's picks (x row = slot / repeat); experts in W; out's row stride
  float lim;                // the SwiGLU clip
};

// The MoE kernels' SwiGLU of a gate and up sum: glm_swiglu's, its sigmoid_fast put first (wide.zig glue_math).
inline bfloat glm_rows_glu(float gv, float uv, float limit) {
  const float lim = float(bfloat(limit));
  const bfloat gt = bfloat(metal::min(float(bfloat(gv)), lim));
  const bfloat up = bfloat(metal::min(metal::max(float(bfloat(uv)), -lim), lim));
  return (gt * sigmoid_fast(gt)) * up;
}

// out[m, h, c] = x[m, h] . W[h][c]: 4 columns a simdgroup, 2 simdgroups a threadgroup; grid (n / 8, heads).
template <int BITS>
[[kernel]] void glm_rows(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                         const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                         constant GlmRowDims& d [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint3 tg [[threadgroup_position_in_grid]]) {
  const int row0 = (int(tg.x) * 2 + int(sg)) * 4;
  if (row0 >= d.n) return;
  const int h = int(tg.y);
  const size_t at = size_t(h) * d.hsw + row0;
  const device uint8_t* wr = (const device uint8_t*)W + at * (d.k * BITS / 8);
  const device bfloat* sr = S + at * (d.k / d.group);
  const device bfloat* br = B + at * (d.k / d.group);
  const device bfloat* x0 = X + size_t(h) * d.hsx;
  device bfloat* o0 = OUT + size_t(h) * d.hso + row0;
  const int cols = metal::min(4, d.n - row0);
  int m = 0;
  if (d.ldx == d.k) {
    #pragma clang loop unroll(disable)
    for (; m + 2 <= d.rows; m += 2) {
      float acc[2][4];
      tf_rp_rows<4, 2, BITS, false>(wr, sr, br, x0 + size_t(m) * d.k, d.k, d.group, lane, acc);
      if (lane == 0)
        for (int b = 0; b < 2; b++)
          for (int j = 0; j < cols; j++) o0[size_t(m + b) * d.ldo + j] = bfloat(acc[b][j]);
    }
  }
  #pragma clang loop unroll(disable)
  for (; m < d.rows; m++) {
    float acc[1][4];
    tf_rp_rows<4, 1, BITS, false>(wr, sr, br, x0 + size_t(m) * d.ldx, d.k, d.group, lane, acc);
    if (lane == 0)
      for (int j = 0; j < cols; j++) o0[size_t(m) * d.ldo + j] = bfloat(acc[0][j]);
  }
}

// Slot y through expert ids[y] on x row y / repeat (GLU: gate W and up W2, SwiGLU'd); grid (n / 8, slots).
template <int BITS, bool GLU>
[[kernel]] void glm_rows_ids(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                             const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                             constant GlmPickDims& d [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                             const device int* ids [[buffer(6)]], const device uint32_t* W2 [[buffer(7)]],
                             const device bfloat* S2 [[buffer(8)]], const device bfloat* B2 [[buffer(9)]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                             uint3 tg [[threadgroup_position_in_grid]]) {
  const int row0 = (int(tg.x) * 2 + int(sg)) * 4;
  if (row0 >= d.n) return;
  const int slot = int(tg.y);
  const int expert = ids[slot];
  device bfloat* out = OUT + size_t(slot) * d.ldo + row0;
  const int cols = metal::min(4, d.n - row0);
  if (expert < 0 || expert >= d.experts) {
    if (int(lane) < cols) out[lane] = bfloat(0.0f);
    return;
  }
  const size_t at = size_t(expert) * d.n + row0;
  const size_t kb = size_t(d.k) * BITS / 8, kg = size_t(d.k / d.group);
  const device bfloat* x = X + size_t(slot / d.repeat) * d.k;
  float acc[1][4];
  tf_rp_rows<4, 1, BITS, false>((const device uint8_t*)W + at * kb, S + at * kg, B + at * kg, x, d.k, d.group, lane, acc);
  if (GLU) {
    float up[1][4];
    tf_rp_rows<4, 1, BITS, false>((const device uint8_t*)W2 + at * kb, S2 + at * kg, B2 + at * kg, x, d.k, d.group, lane, up);
    if (lane == 0)
      for (int j = 0; j < cols; j++) out[j] = glm_rows_glu(acc[0][j], up[0][j], d.lim);
  } else if (lane == 0) {
    for (int j = 0; j < cols; j++) out[j] = bfloat(acc[0][j]);
  }
}

// Slots as glm_rows_ids (BITS), then X2's rows through the shared expert (SBITS; GLU: gate then up rows) into OUT2.
template <int BITS, int SBITS, bool GLU, int RPS = 4, int KC = 0> // RPS columns a simdgroup; KC: k compiled in
[[kernel]] void glm_moe_rows(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                             const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                             constant GlmPickDims& d [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                             const device int* ids [[buffer(6)]], const device uint32_t* W2 [[buffer(7)]],
                             const device bfloat* S2 [[buffer(8)]], const device bfloat* B2 [[buffer(9)]],
                             const device bfloat* X2 [[buffer(10)]], const device uint32_t* SW [[buffer(11)]],
                             const device bfloat* SS [[buffer(12)]], const device bfloat* SB [[buffer(13)]],
                             device bfloat* OUT2 [[buffer(14)]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                             uint3 tg [[threadgroup_position_in_grid]]) {
  const int row0 = (int(tg.x) * 2 + int(sg)) * RPS;
  if (row0 >= d.n) return;
  const int y = int(tg.y);
  const int cols = metal::min(RPS, d.n - row0);
  const int K = KC > 0 ? KC : d.k;
  const size_t kg = size_t(K / d.group);
  const device uint8_t* gw;
  const device bfloat *gs, *gb, *x;
  device bfloat* out;
  size_t up_at = 0; // the up rows' offset from the gate rows, in rows
  if (y < d.slots) {
    const int expert = ids[y];
    out = OUT + size_t(y) * d.ldo + row0;
    if (expert < 0 || expert >= d.experts) {
      if (int(lane) < cols) out[lane] = bfloat(0.0f);
      return;
    }
    const size_t at = size_t(expert) * d.n + row0;
    const size_t kb = size_t(K) * BITS / 8;
    x = X + size_t(y / d.repeat) * K;
    float acc[1][RPS];
    tf_rp_rows<RPS, 1, BITS, false>((const device uint8_t*)W + at * kb, S + at * kg, B + at * kg, x, K, d.group, lane,
                                    acc);
    if (GLU) {
      float up[1][RPS];
      tf_rp_rows<RPS, 1, BITS, false>((const device uint8_t*)W2 + at * kb, S2 + at * kg, B2 + at * kg, x, K, d.group,
                                      lane, up);
      if (lane == 0)
        for (int j = 0; j < cols; j++) out[j] = glm_rows_glu(acc[0][j], up[0][j], d.lim);
    } else if (lane == 0) {
      for (int j = 0; j < cols; j++) out[j] = bfloat(acc[0][j]);
    }
    return;
  }
  const int r = y - d.slots;
  const size_t skb = size_t(K) * SBITS / 8;
  x = X2 + size_t(r) * K;
  out = OUT2 + size_t(r) * d.n + row0;
  gw = (const device uint8_t*)SW + size_t(row0) * skb;
  gs = SS + size_t(row0) * kg;
  gb = SB + size_t(row0) * kg;
  float acc[1][RPS];
  tf_rp_rows<RPS, 1, SBITS, false>(gw, gs, gb, x, K, d.group, lane, acc);
  if (GLU) {
    up_at = size_t(d.n);
    float up[1][RPS];
    tf_rp_rows<RPS, 1, SBITS, false>(gw + up_at * skb, gs + up_at * kg, gb + up_at * kg, x, K, d.group, lane, up);
    if (lane == 0)
      for (int j = 0; j < cols; j++) out[j] = glm_rows_glu(acc[0][j], up[0][j], d.lim);
  } else if (lane == 0) {
    for (int j = 0; j < cols; j++) out[j] = bfloat(acc[0][j]);
  }
}

// Expert e's sorted pairs: GLU x row order[i] / repeat to out row i, else x row i to row order[i]; grid (n / 8, e).
template <int BITS, bool GLU>
[[kernel]] void glm_rows_sorted(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                                const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                                constant GlmPickDims& d [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                                const device int* offsets [[buffer(6)]], const device int* order [[buffer(7)]],
                                const device uint32_t* W2 [[buffer(8)]], const device bfloat* S2 [[buffer(9)]],
                                const device bfloat* B2 [[buffer(10)]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                uint3 tg [[threadgroup_position_in_grid]]) {
  const int row0 = (int(tg.x) * 2 + int(sg)) * 4;
  if (row0 >= d.n) return;
  const int e = int(tg.y);
  const int lo = offsets[e], hi = e + 1 < d.experts ? offsets[e + 1] : d.slots;
  const size_t at = size_t(e) * d.n + row0;
  const size_t kb = size_t(d.k) * BITS / 8, kg = size_t(d.k / d.group);
  const device uint8_t* w = (const device uint8_t*)W + at * kb;
  const int cols = metal::min(4, d.n - row0);
  #pragma clang loop unroll(disable)
  for (int i = lo; i < hi; i++) {
    const int src = GLU ? order[i] / d.repeat : i;
    const int dst = GLU ? i : order[i];
    float acc[1][4];
    tf_rp_rows<4, 1, BITS, false>(w, S + at * kg, B + at * kg, X + size_t(src) * d.k, d.k, d.group, lane, acc);
    device bfloat* out = OUT + size_t(dst) * d.ldo + row0;
    if (GLU) {
      float up[1][4];
      tf_rp_rows<4, 1, BITS, false>((const device uint8_t*)W2 + at * kb, S2 + at * kg, B2 + at * kg, X + size_t(src) * d.k, d.k, d.group, lane, up);
      if (lane == 0)
        for (int j = 0; j < cols; j++) out[j] = glm_rows_glu(acc[0][j], up[0][j], d.lim);
    } else if (lane == 0) {
      for (int j = 0; j < cols; j++) out[j] = bfloat(acc[0][j]);
    }
  }
}

// out[m, c] = x[m] . W[c] for bf16 W [n, k] (an unquantized eh_proj): fp32 sums, lanes over 16-input pieces in order.
[[kernel]] void glm_rows_bf16(const device bfloat* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
                              constant GlmRowDims& d [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                              uint3 tg [[threadgroup_position_in_grid]]) {
  const int row0 = (int(tg.x) * 2 + int(sg)) * 4;
  if (row0 >= d.n) return;
  const int cols = metal::min(4, d.n - row0);
  #pragma clang loop unroll(disable)
  for (int m = 0; m < d.rows; m++) {
    const device bfloat* x = X + size_t(m) * d.ldx;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k0 = int(lane) * 16; k0 < d.k; k0 += 512) {
      float xt[16];
      for (int i = 0; i < 16; i++) xt[i] = float(x[k0 + i]);
      for (int j = 0; j < cols; j++) {
        const device bfloat* w = W + size_t(row0 + j) * d.k + k0;
        float s = 0.0f;
        for (int i = 0; i < 16; i++) s += xt[i] * float(w[i]);
        acc[j] += s;
      }
    }
    for (int j = 0; j < 4; j++) acc[j] = simd_sum(acc[j]);
    if (lane == 0)
      for (int j = 0; j < cols; j++) OUT[size_t(m) * d.ldo + row0 + j] = bfloat(acc[j]);
  }
}

#define GLM_ROWS_AT(B)                                                                                                 \
  template [[host_name("glm_rows_q" #B)]] [[kernel]] decltype(glm_rows<B>) glm_rows<B>;                                \
  template [[host_name("glm_rows_ids_q" #B)]] [[kernel]] decltype(glm_rows_ids<B, false>) glm_rows_ids<B, false>;      \
  template [[host_name("glm_rows_glu_ids_q" #B)]] [[kernel]] decltype(glm_rows_ids<B, true>) glm_rows_ids<B, true>;    \
  template [[host_name("glm_rows_sorted_q" #B)]] [[kernel]] decltype(glm_rows_sorted<B, false>) glm_rows_sorted<B, false>; \
  template [[host_name("glm_rows_glu_sorted_q" #B)]] [[kernel]] decltype(glm_rows_sorted<B, true>) glm_rows_sorted<B, true>;
GLM_ROWS_AT(2)
GLM_ROWS_AT(3)
GLM_ROWS_AT(4)
GLM_ROWS_AT(5)
GLM_ROWS_AT(6)
GLM_ROWS_AT(8)

#define GLM_MOE_AT(B)                                                                                                  \
  template [[host_name("glm_moe_rows_q" #B "_s8")]] [[kernel]]                                                         \
      decltype(glm_moe_rows<B, 8, false>) glm_moe_rows<B, 8, false>;                                                   \
  template [[host_name("glm_moe_rows_glu_q" #B "_s8")]] [[kernel]]                                                     \
      decltype(glm_moe_rows<B, 8, true>) glm_moe_rows<B, 8, true>;
GLM_MOE_AT(2)
GLM_MOE_AT(3)
GLM_MOE_AT(4)
GLM_MOE_AT(5)
GLM_MOE_AT(6)
GLM_MOE_AT(8)
