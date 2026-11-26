// EXL3 projections of 2 to 16 unrouted rows and the gate/up SwiGLU epilogue; compiled after exl3_projection.metal.

// q lanes share a column's decode, rp rows a lane (rows tg.y*q*rp on); each row is tf_exl3_matmul_row's chain.
template <uint RP, uint Q>
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_exl3_matmul_rows(const device half* x [[buffer(0)]],
                                                                           const device uchar* trellis [[buffer(1)]],
                                                                           device float* sums [[buffer(2)]],
                                                                           constant TfExl3Args& a [[buffer(3)]],
                                                                           uint lane [[thread_index_in_simdgroup]],
                                                                           uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr uint C = 32 / Q, S = 16 / Q;
  const uint c = lane % C, q = lane / C, n = tg.x * C + c, col = n % 16, ring = 128 * a.halves;
  const uint r0 = (tg.y * Q + q) * RP;
  uint maps[S];
  for (uint j = 0; j < S; ++j) {
    // tf_exl3_matmul_row's table: the stream position tf_exl3_matmul's placement puts at (row r, column col)
    const uint r = q * S + j;
    const uint p = 8 * (4 * (col % 8) + (r % 8) / 2) + 4 * (col / 8) + 2 * (r / 8) + (r % 2);
    const uint end = ((p + 1) * a.halves - ((a.halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
    const uint start = (end + ring - 16) % ring;
    const uint wi = start / 32, next = wi + 1 == 4 * a.halves ? 0 : wi + 1;
    maps[j] = wi | ((start % 32) << 6) | (next << 11);
  }
  const device uchar* column = trellis + ulong(n / 16) * a.tile_n_stride;
  float high[RP], low[RP];
  for (uint i = 0; i < RP; ++i) high[i] = low[i] = 0;
  for (uint kb = 0; kb < a.k; kb += 16) {
    const device ushort* tile = reinterpret_cast<const device ushort*>(column + ulong(kb / 16) * a.tile_k_stride);
    float own[S];
    for (uint j = 0; j < S; ++j) own[j] = float(tf_exl3_value(tf_exl3_mapped_state(tile, maps[j]), a.codebook));
    half4 xv[RP][4];
    for (uint i = 0; i < RP; ++i) {
      const device half4* xr = reinterpret_cast<const device half4*>(x + ulong(r0 + i) * a.k + kb);
      for (uint v = 0; v < 4; ++v) xv[i][v] = r0 + i < a.rows ? xr[v] : half4(0);
    }
    for (uint r = 0; r < 16; ++r) {
      // weight (r, col) once for the lane's rows; each row's chain is the one-row kernel's
      const float w = Q == 1 ? own[r % S] : simd_shuffle(own[r % S], ushort(c + C * (r / S)));
      for (uint i = 0; i < RP; ++i) {
        const float product = float(xv[i][r / 4][r % 4]) * w;
        const float next = high[i] + product;
        const float part = next - high[i];
        low[i] += (high[i] - (next - part)) + (product - part);
        high[i] = next;
      }
    }
  }
  for (uint i = 0; i < RP; ++i)
    if (r0 + i < a.rows) sums[ulong(r0 + i) * a.n + n] = high[i] + low[i];
}

#define TF_EXL3_ROWS(RP, Q)                                                                                       \
  template [[host_name("tf_exl3_matmul_rows" #RP "x" #Q)]] [[kernel]] decltype(tf_exl3_matmul_rows<RP, Q>)       \
      tf_exl3_matmul_rows<RP, Q>;
TF_EXL3_ROWS(1, 2)
TF_EXL3_ROWS(1, 4)
TF_EXL3_ROWS(1, 8)
TF_EXL3_ROWS(2, 8)

// DeepSeek V4.1's fused.metal helpers, the same expressions: the clamp that keeps a NaN, and act_quant's round trip.
inline float tf_exl3_clip(float x, float lo, float hi) { return x != x ? x : clamp(x, lo, hi); }

inline float tf_exl3_pow2(int e) { return as_type<float>(uint(e + 127) << 23); }

inline int tf_exl3_log2c(float v) {
  const uint b = as_type<uint>(v);
  return int((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) != 0u ? 1 : 0);
}

inline uint tf_exl3_e4m3(float a) {
  uint b = as_type<uint>(a);
  b += 0x7FFFFu + ((b >> 20) & 1u);
  return a < 0.015625f ? uint(rint(a * 512.0f)) : (((b >> 23) - 120u) << 3) | ((b >> 20) & 7u);
}

inline float tf_exl3_e4m3v(uint c) { return float(as_type<half>(ushort((c & 0x7Fu) << 7))) * 256.0f; }

// tf_f_actq: the value a lane holds, its 32-value block being the simdgroup's 32 lanes.
inline bfloat tf_exl3_actq(float v) {
  const bool bad = simd_any(!isfinite(v));
  const float amax = simd_max(fabs(v));
  const int e = tf_exl3_log2c(max(amax, 1e-4f) * (1.0f / 448.0f));
  const float q = tf_exl3_e4m3v(tf_exl3_e4m3(min(fabs(v) * tf_exl3_pow2(-e), 448.0f))) * tf_exl3_pow2(e);
  return bfloat(bad ? NAN : (signbit(v) ? -q : q));
}

// tf_exl3_rotate_out of one row's gate and up sums to bf16, then tf_dsv41_f_swiglu's clamped SwiGLU and act_quant.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_exl3_swiglu(
    const device float* gs [[buffer(0)]], const device float* us [[buffer(1)]], const device ushort* gsv [[buffer(2)]],
    const device ushort* usv [[buffer(3)]], const device half* gb [[buffer(4)]], const device half* ub [[buffer(5)]],
    device bfloat* y [[buffer(6)]], device bfloat* yq [[buffer(7)]], constant TfExl3Args& ag [[buffer(8)]],
    constant TfExl3Args& au [[buffer(9)]], constant float& limit [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y, base = tg.x * 128;
  float g[4], u[4];
  for (uint j = 0; j < 4; ++j) {
    g[j] = gs[ulong(row) * ag.n + base + lane + 32 * j];
    u[j] = us[ulong(row) * au.n + base + lane + 32 * j];
  }
  tf_exl3_hadamard(g, lane);
  tf_exl3_hadamard(u, lane);
  for (uint j = 0; j < 4; ++j) {
    const uint n = base + lane + 32 * j;
    float go = (g[j] * 0.08838834764831845f) * tf_exl3_scale(gsv, n, ag.sv_signs);
    if (ag.has_bias) go += float(gb[n]);
    float uo = (u[j] * 0.08838834764831845f) * tf_exl3_scale(usv, n, au.sv_signs);
    if (au.has_bias) uo += float(ub[n]);
    const float gc = tf_exl3_clip(float(bfloat(go)), -INFINITY, limit);
    const float uc = tf_exl3_clip(float(bfloat(uo)), -limit, limit);
    const bfloat v = bfloat(precise::divide(gc, 1.0f + precise::exp(-gc)) * uc);
    const ulong at = ulong(row) * ag.y_stride + n;
    y[at] = v;
    yq[at] = tf_exl3_actq(float(v));
  }
}
