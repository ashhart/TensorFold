// Routed experts with an expert's member rows MB at a time: each weight word unpacked once for MB rows, each row's sums as tf_rowdot.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

inline float tf_load16(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
  }
  return sum;
}
inline float tf_qdot16(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += (xt[4 * i] * (ws[i] & 0x000f) + xt[4 * i + 1] * (ws[i] & 0x00f0) +
              xt[4 * i + 2] * (ws[i] & 0x0f00) + xt[4 * i + 3] * (ws[i] & 0xf000));
  return scale * accum + sum * bias;
}
template <int K, int GS, int RPS>
inline void tf_rowdot(const device uint8_t* w, const device bfloat* sc, const device bfloat* bi,
                      const device bfloat* x, uint lane, thread float* acc) {
  constexpr int KB = K / 2;
  constexpr int KG = K / GS;
  constexpr int FULL = K / 512 * 512;
  w += lane * 8;
  sc += lane / (GS / 16);
  bi += lane / (GS / 16);
  x += lane * 16;
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < FULL; k0 += 512) {
    float xt[16];
    const float sum = tf_load16(x, xt);
    for (int j = 0; j < RPS; j++) acc[j] += tf_qdot16(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    w += 256; sc += 512 / GS; bi += 512 / GS; x += 512;
  }
  if (FULL < K && int(lane) < (K - FULL) / 16) {
    float xt[16];
    const float sum = tf_load16(x, xt);
    for (int j = 0; j < RPS; j++) acc[j] += tf_qdot16(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
  }
  for (int j = 0; j < RPS; j++) acc[j] = simd_sum(acc[j]);
}

// tf_qdot16 for MB inputs of one weight word group: the masked words read once, each input's sums in tf_qdot16's order.
template <int MB>
inline void tf_qdot16_rows(const device uint8_t* w, const thread float (*xt)[16], float scale, float bias,
                           const thread float* sum, thread float* out) {
  const device uint16_t* ws = (const device uint16_t*)w;
  int q[16];
  for (int i = 0; i < 4; i++) {
    const int v = ws[i];
    q[4 * i] = v & 0x000f; q[4 * i + 1] = v & 0x00f0; q[4 * i + 2] = v & 0x0f00; q[4 * i + 3] = v & 0xf000;
  }
  for (int b = 0; b < MB; b++) {
    float accum = 0.0f;
    for (int i = 0; i < 4; i++)
      accum += (xt[b][4 * i] * q[4 * i] + xt[b][4 * i + 1] * q[4 * i + 1] +
                xt[b][4 * i + 2] * q[4 * i + 2] + xt[b][4 * i + 3] * q[4 * i + 3]);
    out[b] += scale * accum + sum[b] * bias;
  }
}

// tf_rowdot for MB input rows at once: the same chunk order and the same simd_sum per (row, output).
template <int K, int GS, int RPS, int MB>
inline void tf_rowdot_rows(const device uint8_t* w, const device bfloat* sc, const device bfloat* bi,
                           const thread int* xoff, const device bfloat* x0, uint lane, thread float (*acc)[RPS]) {
  constexpr int KB = K / 2;
  constexpr int KG = K / GS;
  constexpr int FULL = K / 512 * 512;
  w += lane * 8;
  sc += lane / (GS / 16);
  bi += lane / (GS / 16);
  const int xl = int(lane) * 16;
  float col[MB];
  for (int j = 0; j < RPS; j++)
    for (int b = 0; b < MB; b++) acc[b][j] = 0.0f;
  for (int k0 = 0; k0 <= FULL; k0 += 512) {
    if (k0 == FULL && !(FULL < K && int(lane) < (K - FULL) / 16)) break;
    float xt[MB][16];
    float sum[MB];
    for (int b = 0; b < MB; b++) sum[b] = tf_load16(x0 + xoff[b] + xl + k0, xt[b]);
    for (int j = 0; j < RPS; j++) {
      for (int b = 0; b < MB; b++) col[b] = acc[b][j];
      tf_qdot16_rows<MB>(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum, col);
      for (int b = 0; b < MB; b++) acc[b][j] = col[b];
    }
    w += 256; sc += 512 / GS; bi += 512 / GS;
  }
  for (int j = 0; j < RPS; j++)
    for (int b = 0; b < MB; b++) acc[b][j] = simd_sum(acc[b][j]);
}

// One threadgroup an (expert, 8 outputs): its member pairs MB at a time, then one at a time (UP: x rows by token, relu2).
template <int K, int N, int GS, int RPS, int SG, int TOPK, int MB, bool UP>
[[kernel]] void tf_experts_rows(
  const device bfloat16_t* X [[buffer(0)]],
  const device uint32_t* UIDS [[buffer(1)]],
  const device int32_t* START [[buffer(2)]],
  const device int32_t* COUNT [[buffer(3)]],
  const device int32_t* MEMBERS [[buffer(4)]],
  const constant int32_t* UCOUNT [[buffer(5)]],
  const device uint32_t* W [[buffer(6)]],
  const device bfloat16_t* S [[buffer(7)]],
  const device bfloat16_t* B [[buffer(8)]],
  device bfloat16_t* OUT [[buffer(9)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  const uint lane = thread_index_in_simdgroup;
  const int u = int(threadgroup_position_in_grid.z);
  if (u >= UCOUNT[0]) return;
  const size_t e = size_t(UIDS[u]);
  const int row0 = (int(threadgroup_position_in_grid.y) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
  const size_t at = e * N + size_t(row0);
  const int first = START[u], last = START[u] + COUNT[u];
  const device uint8_t* wr = (const device uint8_t*)W + at * (K / 2);
  int m = first;
  #pragma clang loop unroll(disable)
  for (; m + MB <= last; m += MB) {
    int p[MB], xoff[MB];
    for (int b = 0; b < MB; b++) {
      p[b] = MEMBERS[m + b];
      xoff[b] = (UP ? p[b] / TOPK : p[b]) * K;
    }
    float acc[MB][RPS];
    tf_rowdot_rows<K, GS, RPS, MB>(wr, S + at * (K / GS), B + at * (K / GS), xoff, X, lane, acc);
    if (lane == 0)
      for (int b = 0; b < MB; b++)
        for (int j = 0; j < RPS; j++) {
          if (UP) {
            const float h = metal::max(float(bfloat(acc[b][j])), 0.0f);
            OUT[size_t(p[b]) * N + row0 + j] = bfloat(h * h);
          } else OUT[size_t(p[b]) * N + row0 + j] = bfloat(acc[b][j]);
        }
  }
  #pragma clang loop unroll(disable)
  for (; m < last; m++) {
    const int p = MEMBERS[m];
    float acc[RPS];
    tf_rowdot<K, GS, RPS>(wr, S + at * (K / GS), B + at * (K / GS), X + size_t(UP ? p / TOPK : p) * K, lane, acc);
    if (lane == 0)
      for (int j = 0; j < RPS; j++) {
        if (UP) {
          const float h = metal::max(float(bfloat(acc[j])), 0.0f);
          OUT[size_t(p) * N + row0 + j] = bfloat(h * h);
        } else OUT[size_t(p) * N + row0 + j] = bfloat(acc[j]);
      }
  }
}

template [[host_name("tf_xup_rows2")]] [[kernel]] decltype(tf_experts_rows<2688, 1856, 64, 4, 2, 6, 2, true>) tf_experts_rows<2688, 1856, 64, 4, 2, 6, 2, true>;
template [[host_name("tf_xdown_rows2")]] [[kernel]] decltype(tf_experts_rows<1856, 2688, 64, 4, 2, 6, 2, false>) tf_experts_rows<1856, 2688, 64, 4, 2, 6, 2, false>;
template [[host_name("tf_xup_rows4")]] [[kernel]] decltype(tf_experts_rows<2688, 1856, 64, 4, 2, 6, 4, true>) tf_experts_rows<2688, 1856, 64, 4, 2, 6, 4, true>;
template [[host_name("tf_xdown_rows4")]] [[kernel]] decltype(tf_experts_rows<1856, 2688, 64, 4, 2, 6, 4, false>) tf_experts_rows<1856, 2688, 64, 4, 2, 6, 4, false>;

// A 16-group's four weight words as the outer loop: four activations a member live, each sum in tf_qdot16's order.
template <int K, int GS, int RPS, int MB>
inline void tf_rowdot_w(const device uint8_t* w, const device bfloat* sc, const device bfloat* bi, const thread int* xoff,
                        const device bfloat* x0, uint lane, thread float (*acc)[RPS]) {
  constexpr int KB = K / 2;
  constexpr int KG = K / GS;
  constexpr int FULL = K / 512 * 512;
  w += lane * 8;
  sc += lane / (GS / 16);
  bi += lane / (GS / 16);
  const int xl = int(lane) * 16;
  for (int j = 0; j < RPS; j++)
    for (int b = 0; b < MB; b++) acc[b][j] = 0.0f;
  for (int k0 = 0; k0 <= FULL; k0 += 512) {
    if (k0 == FULL && !(FULL < K && int(lane) < (K - FULL) / 16)) break;
    float accum[MB][RPS];
    float sum[MB];
    for (int b = 0; b < MB; b++) {
      sum[b] = 0.0f;
      for (int j = 0; j < RPS; j++) accum[b][j] = 0.0f;
    }
    for (int i = 0; i < 4; i++) {
      float xt[MB][4];
      for (int b = 0; b < MB; b++) {
        const device bfloat* x = x0 + xoff[b] + xl + k0 + 4 * i;
        const bfloat a = x[0], bb = x[1], c = x[2], d = x[3];
        sum[b] += float(bfloat(float(bfloat(float(bfloat(float(a) + float(bb))) + float(c))) + float(d)));
        xt[b][0] = float(a); xt[b][1] = float(bb) / 16.0f; xt[b][2] = float(c) / 256.0f; xt[b][3] = float(d) / 4096.0f;
      }
      for (int j = 0; j < RPS; j++) {
        const int v = ((const device uint16_t*)(w + j * KB))[i];
        const int q0 = v & 0x000f, q1 = v & 0x00f0, q2 = v & 0x0f00, q3 = v & 0xf000;
        for (int b = 0; b < MB; b++)
          accum[b][j] += (xt[b][0] * q0 + xt[b][1] * q1 + xt[b][2] * q2 + xt[b][3] * q3);
      }
    }
    for (int j = 0; j < RPS; j++) {
      const float scale = float(sc[j * KG]), bias = float(bi[j * KG]);
      for (int b = 0; b < MB; b++) acc[b][j] += scale * accum[b][j] + sum[b] * bias;
    }
    w += 256; sc += 512 / GS; bi += 512 / GS;
  }
  for (int j = 0; j < RPS; j++)
    for (int b = 0; b < MB; b++) acc[b][j] = simd_sum(acc[b][j]);
}

// One threadgroup an (expert, SG*RPS outputs): members MB at a time through tf_rowdot_w, the rest through tf_rowdot.
template <int K, int N, int GS, int RPS, int SG, int TOPK, int MB, bool UP>
[[kernel]] void tf_experts_w(
  const device bfloat16_t* X [[buffer(0)]],
  const device uint32_t* UIDS [[buffer(1)]],
  const device int32_t* START [[buffer(2)]],
  const device int32_t* COUNT [[buffer(3)]],
  const device int32_t* MEMBERS [[buffer(4)]],
  const constant int32_t* UCOUNT [[buffer(5)]],
  const device uint32_t* W [[buffer(6)]],
  const device bfloat16_t* S [[buffer(7)]],
  const device bfloat16_t* B [[buffer(8)]],
  device bfloat16_t* OUT [[buffer(9)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  const uint lane = thread_index_in_simdgroup;
  const int u = int(threadgroup_position_in_grid.z);
  if (u >= UCOUNT[0]) return;
  const size_t e = size_t(UIDS[u]);
  const int row0 = (int(threadgroup_position_in_grid.y) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
  const size_t at = e * N + size_t(row0);
  const int first = START[u], last = START[u] + COUNT[u];
  const device uint8_t* wr = (const device uint8_t*)W + at * (K / 2);
  int m = first;
  #pragma clang loop unroll(disable)
  for (; m + MB <= last; m += MB) {
    int p[MB], xoff[MB];
    for (int b = 0; b < MB; b++) {
      p[b] = MEMBERS[m + b];
      xoff[b] = (UP ? p[b] / TOPK : p[b]) * K;
    }
    float acc[MB][RPS];
    tf_rowdot_w<K, GS, RPS, MB>(wr, S + at * (K / GS), B + at * (K / GS), xoff, X, lane, acc);
    if (lane == 0)
      for (int b = 0; b < MB; b++)
        for (int j = 0; j < RPS; j++) {
          if (UP) {
            const float h = metal::max(float(bfloat(acc[b][j])), 0.0f);
            OUT[size_t(p[b]) * N + row0 + j] = bfloat(h * h);
          } else OUT[size_t(p[b]) * N + row0 + j] = bfloat(acc[b][j]);
        }
  }
  #pragma clang loop unroll(disable)
  for (; m < last; m++) {
    const int p = MEMBERS[m];
    float acc[RPS];
    tf_rowdot<K, GS, RPS>(wr, S + at * (K / GS), B + at * (K / GS), X + size_t(UP ? p / TOPK : p) * K, lane, acc);
    if (lane == 0)
      for (int j = 0; j < RPS; j++) {
        if (UP) {
          const float h = metal::max(float(bfloat(acc[j])), 0.0f);
          OUT[size_t(p) * N + row0 + j] = bfloat(h * h);
        } else OUT[size_t(p) * N + row0 + j] = bfloat(acc[j]);
      }
  }
}

template [[host_name("tf_xup_w3")]] [[kernel]] decltype(tf_experts_w<2688, 1856, 64, 4, 4, 6, 3, true>) tf_experts_w<2688, 1856, 64, 4, 4, 6, 3, true>;
template [[host_name("tf_xdown_w3")]] [[kernel]] decltype(tf_experts_w<1856, 2688, 64, 4, 4, 6, 3, false>) tf_experts_w<1856, 2688, 64, 4, 4, 6, 3, false>;
