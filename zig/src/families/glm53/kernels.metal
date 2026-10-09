// Full GLM-5.3 (glm_moe_dsa) decode kernels: fp32 activations as mlx-lm runs this checkpoint (f16 scales promote
// every quantized matmul to fp32), affine 6/8-bit weights in groups of 64, bf16 dense weights, fp32 KV caches.
// Every reduction runs in a fixed order, so a row's bits do not depend on how many Macs share the model.
#include <metal_stdlib>
using namespace metal;
#ifndef KVT
#define KVT float
#endif

// ---------------------------------------------------------------- quantized matvec (MLX qmv arithmetic) ----------
template <int BITS> struct qpack {
  static constant constexpr int vals = BITS == 6 ? 4 : 32 / BITS;
  static constant constexpr int bytes = BITS == 6 ? 3 : 4;
};

// A lane's VALS inputs pre-divided to meet unshifted weight fields (MLX load_vector), and their plain sum.
template <int BITS, int VALS>
inline float q_load(const device float* x, thread float* xs) {
  float total = 0.0f;
  if (BITS == 8) {
    for (int i = 0; i < VALS; ++i) {
      total += x[i];
      xs[i] = x[i];
    }
    return total;
  }
  for (int i = 0; i < VALS; i += 4) {
    const float a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    total += a + b + c + d;
    xs[i] = a;
    xs[i + 1] = b / (BITS == 4 ? 16.0f : 64.0f);
    xs[i + 2] = c / (BITS == 4 ? 256.0f : 16.0f);
    xs[i + 3] = d / (BITS == 4 ? 4096.0f : 4.0f);
  }
  return total;
}

template <int BITS, int VALS>
inline float q_dot(const device uint8_t* w, const thread float* xs, float scale, float bias, float total) {
  float acc = 0.0f;
  if (BITS == 4) {
    const device uint16_t* h = (const device uint16_t*)w;
    for (int i = 0; i < VALS / 4; ++i) {
      const uint16_t q = h[i];
      const thread float* v = xs + 4 * i;
      acc += (v[0] * (q & 0x000f) + v[1] * (q & 0x00f0) + v[2] * (q & 0x0f00) + v[3] * (q & 0xf000));
    }
  } else if (BITS == 6) {
    for (int i = 0; i < VALS / 4; ++i) {
      const device uint8_t* p = w + 3 * i;
      const thread float* v = xs + 4 * i;
      acc += v[0] * (p[0] & 0x3f);
      acc += v[1] * (p[0] & 0xc0);
      acc += (v[1] * 256.0f) * (p[1] & 0x0f);
      acc += v[2] * (p[1] & 0xf0);
      acc += (v[2] * 256.0f) * (p[2] & 0x03);
      acc += v[3] * (p[2] & 0xfc);
    }
  } else {
    for (int i = 0; i < VALS; ++i) acc += xs[i] * w[i];
  }
  return scale * acc + total * bias;
}

// Four rows' dots for one input over [0, K): MLX's qmv (PACKS 1, a ragged last step) or qmv_fast (PACKS 2).
template <int BITS, int PACKS>
inline void q_rows4(const device uint8_t* w, const device half* s, const device half* b, const device float* x,
                    int K, int ldw, int lds, uint lane, thread float* acc) {
  constexpr int VALS = PACKS * qpack<BITS>::vals;
  constexpr int STEP = 32 * VALS;
  constexpr int WSTEP = STEP / qpack<BITS>::vals * qpack<BITS>::bytes;
  const device uint8_t* wl = w + lane * PACKS * qpack<BITS>::bytes;
  const device half* sl = s + lane * VALS / 64;
  const device half* bl = b + lane * VALS / 64;
  const device float* xl = x + lane * VALS;
  int k = 0;
  for (; k < (PACKS == 2 ? K : K - STEP); k += STEP) {
    float xs[VALS];
    const float total = q_load<BITS, VALS>(xl, xs);
    for (int r = 0; r < 4; ++r)
      acc[r] += q_dot<BITS, VALS>(wl + r * ldw, xs, float(sl[r * lds]), float(bl[r * lds]), total);
    wl += WSTEP;
    sl += STEP / 64;
    bl += STEP / 64;
    xl += STEP;
  }
  if (PACKS == 1 && k + int(lane) * VALS < K) {
    float xs[VALS];
    const float total = q_load<BITS, VALS>(xl, xs);
    for (int r = 0; r < 4; ++r)
      acc[r] += q_dot<BITS, VALS>(wl + r * ldw, xs, float(sl[r * lds]), float(bl[r * lds]), total);
  }
}

// q_dot on prepared fields: the same terms in the same order.
template <int BITS, int VALS>
inline float q_dotf(const thread float* f, const thread float* xs, float scale, float bias, float total) {
  float acc = 0.0f;
  if (BITS == 6) {
    for (int i = 0; i < VALS / 4; ++i) {
      const thread float* v = xs + 4 * i;
      const thread float* g = f + 6 * i;
      acc += v[0] * g[0];
      acc += v[1] * g[1];
      acc += (v[1] * 256.0f) * g[2];
      acc += v[2] * g[3];
      acc += (v[2] * 256.0f) * g[4];
      acc += v[3] * g[5];
    }
  } else {
    for (int i = 0; i < VALS; ++i) acc += xs[i] * f[i];
  }
  return scale * acc + total * bias;
}

// Four rows' dots for `cnt` (<= MB) inputs over [0, K): each input's sums in q_rows4's lane split and order. A step
// reads every input's values once, then each row's weight fields once for all inputs.
template <int BITS, int VALS, int MB>
inline void q_stepM(const device uint8_t* wl, const device half* sl, const device half* bl, const thread long* xo,
                    const device float* X, int xl, int cnt, int ldw, int lds, thread float (*acc)[4]) {
  float xs[MB][VALS];
  float tot[MB];
  for (int j = 0; j < MB; ++j)
    if (j < cnt) tot[j] = q_load<BITS, VALS>(X + xo[j] + xl, xs[j]);
  for (int r = 0; r < 4; ++r) {
    constexpr int NF = BITS == 6 ? VALS / 4 * 6 : VALS;
    float f[NF];
    const device uint8_t* wr = wl + r * ldw;
    if (BITS == 6) {
      for (int i = 0; i < VALS / 4; ++i) {
        const device uint8_t* p = wr + 3 * i;
        f[6 * i + 0] = float(p[0] & 0x3f);
        f[6 * i + 1] = float(p[0] & 0xc0);
        f[6 * i + 2] = float(p[1] & 0x0f);
        f[6 * i + 3] = float(p[1] & 0xf0);
        f[6 * i + 4] = float(p[2] & 0x03);
        f[6 * i + 5] = float(p[2] & 0xfc);
      }
    } else {
      for (int i = 0; i < VALS; ++i) f[i] = float(wr[i]);
    }
    const float sc = float(sl[r * lds]), bi = float(bl[r * lds]);
    for (int j = 0; j < MB; ++j)
      if (j < cnt) acc[j][r] += q_dotf<BITS, VALS>(f, xs[j], sc, bi, tot[j]);
  }
}

template <int BITS, int PACKS, int MB>
inline void q_rowsM(const device uint8_t* w, const device half* s, const device half* b, const thread long* xo,
                    const device float* X, int cnt, int K, int ldw, int lds, uint lane, thread float (*acc)[4]) {
  if (MB == 1) {
    q_rows4<BITS, PACKS>(w, s, b, X + xo[0], K, ldw, lds, lane, acc[0]);
    return;
  }
  constexpr int VALS = PACKS * qpack<BITS>::vals;
  constexpr int STEP = 32 * VALS;
  constexpr int WSTEP = STEP / qpack<BITS>::vals * qpack<BITS>::bytes;
  const device uint8_t* wl = w + lane * PACKS * qpack<BITS>::bytes;
  const device half* sl = s + lane * VALS / 64;
  const device half* bl = b + lane * VALS / 64;
  int xl = int(lane) * VALS;
  int k = 0;
  for (; k < (PACKS == 2 ? K : K - STEP); k += STEP) {
    q_stepM<BITS, VALS, MB>(wl, sl, bl, xo, X, xl, cnt, ldw, lds, acc);
    wl += WSTEP;
    sl += STEP / 64;
    bl += STEP / 64;
    xl += STEP;
  }
  if (PACKS == 1 && k + int(lane) * VALS < K) q_stepM<BITS, VALS, MB>(wl, sl, bl, xo, X, xl, cnt, ldw, lds, acc);
}

// Strides of one matrix in a stack: row bytes, scale-row groups; a z step's weight, scale, input and output offsets.
// Items (threadgroup x) are input rows or (row, expert) pairs; z (threadgroup z) steps a stack (heads).
struct QArgs {
  int K;      // the reduction length this launch covers
  int N;      // output rows (a multiple of 8)
  int ldw;    // bytes between weight rows
  int lds;    // scale/bias groups between rows
  int zw;     // bytes between stacked matrices (experts or heads)
  int zs;     // groups between stacked scale matrices
  int xr;     // floats between input rows
  int zx;     // floats between inputs of consecutive z
  int yr;     // floats between output rows
  int zy;     // floats between outputs of consecutive z
  int ids;    // 1: item i's matrix is ids[i] (+ z)
  int xids;   // 1: item i's input row is xids[i]
  int yids;   // 1: item i's output row is yids[i]
  int items;  // items in all
  int segs;   // 1: threadgroup x's items are segment x of SEG ([start, count] pairs, SEG's count at SEGN); 0: MB a group
};

// Threadgroup x's items: [start, start + cnt). Segments share one matrix (picks grouped by expert).
inline int2 q_batch(constant QArgs& a, uint x, const device uint* SEG, int MB) {
  if (a.segs) {
    if (int(x) >= int(SEG[0])) return int2(0, 0);
    return int2(int(SEG[1 + 2 * x]), int(SEG[2 + 2 * x]));
  }
  const int start = int(x) * MB;
  return int2(start, max(0, min(MB, a.items - start)));
}

// y[item][z][n] = sum_k W_e[n][k] x[item][z][k] (fp32); threadgroups [batches, N/8, Z] of [32, 2, 1], up to MB items a
// batch sharing one matrix.
template <int BITS, int PACKS, int MB>
[[kernel]] void g53_qmv(const device uint8_t* W [[buffer(0)]], const device half* S [[buffer(1)]],
                        const device half* B [[buffer(2)]], const device float* X [[buffer(3)]],
                        device float* Y [[buffer(4)]], constant QArgs& a [[buffer(5)]],
                        const device uint* IDS [[buffer(6)]], const device uint* XIDS [[buffer(7)]],
                        const device uint* YIDS [[buffer(8)]], const device uint* SEG [[buffer(9)]],
                        uint3 tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const int2 bt = q_batch(a, tg.x, SEG, MB);
  if (bt.y == 0) return;
  const uint z = tg.z;
  const long e = (a.ids ? long(IDS[bt.x]) : 0) + long(z);
  long xo[MB], yo[MB];
  for (int j = 0; j < MB; ++j) {
    const int it = bt.x + min(j, bt.y - 1);
    xo[j] = (a.xids ? long(XIDS[it]) : long(it)) * a.xr + long(z) * a.zx;
    yo[j] = (a.yids ? long(YIDS[it]) : long(it)) * a.yr + long(z) * a.zy;
  }
  const int n0 = int(tg.y) * 8 + int(sg) * 4;
  const device uint8_t* w = W + e * long(a.zw) + long(n0) * a.ldw;
  const device half* s = S + e * long(a.zs) + long(n0) * a.lds;
  const device half* b = B + e * long(a.zs) + long(n0) * a.lds;
  float acc[MB][4];
  for (int j = 0; j < MB; ++j)
    for (int r = 0; r < 4; ++r) acc[j][r] = 0.0f;
  q_rowsM<BITS, PACKS, MB>(w, s, b, xo, X, bt.y, a.K, a.ldw, a.lds, lane, acc);
  for (int j = 0; j < MB; ++j) {
    if (j < bt.y) {
      for (int r = 0; r < 4; ++r) {
        const float v = simd_sum(acc[j][r]);
        if (lane == 0) Y[yo[j] + n0 + r] = v;
      }
    }
  }
}

// act[item][n] = silu(gate . x) * (up . x): two stacked matrices with the same layout; batches as g53_qmv.
template <int BITS, int PACKS, int MB>
[[kernel]] void g53_gateup(const device uint8_t* GW [[buffer(0)]], const device half* GS [[buffer(1)]],
                           const device half* GB [[buffer(2)]], const device uint8_t* UW [[buffer(3)]],
                           const device half* US [[buffer(4)]], const device half* UB [[buffer(5)]],
                           const device float* X [[buffer(6)]], device float* Y [[buffer(7)]],
                           constant QArgs& a [[buffer(8)]], const device uint* IDS [[buffer(9)]],
                           const device uint* XIDS [[buffer(10)]], const device uint* YIDS [[buffer(11)]],
                           const device uint* SEG [[buffer(12)]],
                           uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const int2 bt = q_batch(a, tg.x, SEG, MB);
  if (bt.y == 0) return;
  const long e = a.ids ? long(IDS[bt.x]) : 0;
  long xo[MB], yo[MB];
  for (int j = 0; j < MB; ++j) {
    const int it = bt.x + min(j, bt.y - 1);
    xo[j] = (a.xids ? long(XIDS[it]) : long(it)) * a.xr;
    yo[j] = (a.yids ? long(YIDS[it]) : long(it)) * a.yr;
  }
  const int n0 = int(tg.y) * 8 + int(sg) * 4;
  const long wo = e * long(a.zw) + long(n0) * a.ldw;
  const long so = e * long(a.zs) + long(n0) * a.lds;
  float g[MB][4], u[MB][4];
  for (int j = 0; j < MB; ++j)
    for (int r = 0; r < 4; ++r) g[j][r] = u[j][r] = 0.0f;
  q_rowsM<BITS, PACKS, MB>(GW + wo, GS + so, GB + so, xo, X, bt.y, a.K, a.ldw, a.lds, lane, g);
  q_rowsM<BITS, PACKS, MB>(UW + wo, US + so, UB + so, xo, X, bt.y, a.K, a.ldw, a.lds, lane, u);
  for (int j = 0; j < MB; ++j) {
    if (j < bt.y) {
      for (int r = 0; r < 4; ++r) {
        const float gv = simd_sum(g[j][r]);
        const float uv = simd_sum(u[j][r]);
        if (lane == 0) {
          const float sig = 1.0f / (1.0f + metal::exp(-gv));
          Y[yo[j] + n0 + r] = (gv * sig) * uv;
        }
      }
    }
  }
}

template [[host_name("g53_qmv_b6_p1_m1")]] [[kernel]] void g53_qmv<6, 1, 1>(const device uint8_t*, const device half*, const device half*, const device float*, device float*, constant QArgs&, const device uint*, const device uint*, const device uint*, const device uint*, uint3, uint, uint);
template [[host_name("g53_qmv_b6_p2_m1")]] [[kernel]] void g53_qmv<6, 2, 1>(const device uint8_t*, const device half*, const device half*, const device float*, device float*, constant QArgs&, const device uint*, const device uint*, const device uint*, const device uint*, uint3, uint, uint);
template [[host_name("g53_gateup_b6_p2_m1")]] [[kernel]] void g53_gateup<6, 2, 1>(const device uint8_t*, const device half*, const device half*, const device uint8_t*, const device half*, const device half*, const device float*, device float*, constant QArgs&, const device uint*, const device uint*, const device uint*, const device uint*, uint3, uint, uint);
template [[host_name("g53_qmv_b8_p1_m1")]] [[kernel]] void g53_qmv<8, 1, 1>(const device uint8_t*, const device half*, const device half*, const device float*, device float*, constant QArgs&, const device uint*, const device uint*, const device uint*, const device uint*, uint3, uint, uint);
template [[host_name("g53_qmv_b8_p2_m1")]] [[kernel]] void g53_qmv<8, 2, 1>(const device uint8_t*, const device half*, const device half*, const device float*, device float*, constant QArgs&, const device uint*, const device uint*, const device uint*, const device uint*, uint3, uint, uint);
template [[host_name("g53_gateup_b8_p2_m1")]] [[kernel]] void g53_gateup<8, 2, 1>(const device uint8_t*, const device half*, const device half*, const device uint8_t*, const device half*, const device half*, const device float*, device float*, constant QArgs&, const device uint*, const device uint*, const device uint*, const device uint*, uint3, uint, uint);

// Split-K one-input matvec for short N: a threadgroup of 8 simdgroups takes 4 rows, simdgroup g the K range
// [g K/8, (g+1) K/8) in q_rows4's lane split, the 8 partials added in order. threadgroups [items, N/4] of 256.
template <int BITS, int PACKS>
[[kernel]] void g53_qmv_sk(const device uint8_t* W [[buffer(0)]], const device half* S [[buffer(1)]],
                           const device half* B [[buffer(2)]], const device float* X [[buffer(3)]],
                           device float* Y [[buffer(4)]], constant QArgs& a [[buffer(5)]],
                           const device uint* IDS [[buffer(6)]], const device uint* XIDS [[buffer(7)]],
                           const device uint* YIDS [[buffer(8)]], const device uint* SEG [[buffer(9)]],
                           uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint lid [[thread_index_in_threadgroup]]) {
  threadgroup float part[8 * 4];
  const uint it = tg.x;
  const long xrow = a.xids ? long(XIDS[it]) : long(it);
  const long yrow = a.yids ? long(YIDS[it]) : long(it);
  const int n0 = int(tg.y) * 4;
  const int kp = a.K / 8;
  const long ko = long(sg) * kp;
  const device uint8_t* w = W + long(n0) * a.ldw + ko * BITS / 8;
  const device half* s = S + long(n0) * a.lds + ko / 64;
  const device half* b = B + long(n0) * a.lds + ko / 64;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  q_rows4<BITS, PACKS>(w, s, b, X + xrow * a.xr + ko, kp, a.ldw, a.lds, lane, acc);
  for (int r = 0; r < 4; ++r) {
    const float v = simd_sum(acc[r]);
    if (lane == 0) part[sg * 4 + r] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 4) {
    float t = part[lid];
    for (int g = 1; g < 8; ++g) t += part[g * 4 + lid];
    Y[yrow * a.yr + n0 + lid] = t;
  }
}

// silu(gate x) * (up x) split-K, as g53_qmv_sk.
template <int BITS, int PACKS>
[[kernel]] void g53_gateup_sk(const device uint8_t* GW [[buffer(0)]], const device half* GS [[buffer(1)]],
                              const device half* GB [[buffer(2)]], const device uint8_t* UW [[buffer(3)]],
                              const device half* US [[buffer(4)]], const device half* UB [[buffer(5)]],
                              const device float* X [[buffer(6)]], device float* Y [[buffer(7)]],
                              constant QArgs& a [[buffer(8)]], const device uint* IDS [[buffer(9)]],
                              const device uint* XIDS [[buffer(10)]], const device uint* YIDS [[buffer(11)]],
                              const device uint* SEG [[buffer(12)]], uint3 tg [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                              uint lid [[thread_index_in_threadgroup]]) {
  threadgroup float part[2][8 * 4];
  const uint it = tg.x;
  const long xrow = a.xids ? long(XIDS[it]) : long(it);
  const long yrow = a.yids ? long(YIDS[it]) : long(it);
  const int n0 = int(tg.y) * 4;
  const int kp = a.K / 8;
  const long ko = long(sg) * kp;
  const long wo = long(n0) * a.ldw + ko * BITS / 8;
  const long so = long(n0) * a.lds + ko / 64;
  float g[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float u[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const device float* x = X + xrow * a.xr + ko;
  q_rows4<BITS, PACKS>(GW + wo, GS + so, GB + so, x, kp, a.ldw, a.lds, lane, g);
  q_rows4<BITS, PACKS>(UW + wo, US + so, UB + so, x, kp, a.ldw, a.lds, lane, u);
  for (int r = 0; r < 4; ++r) {
    const float gv = simd_sum(g[r]), uv = simd_sum(u[r]);
    if (lane == 0) {
      part[0][sg * 4 + r] = gv;
      part[1][sg * 4 + r] = uv;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 4) {
    float gt = part[0][lid], ut = part[1][lid];
    for (int k = 1; k < 8; ++k) {
      gt += part[0][k * 4 + lid];
      ut += part[1][k * 4 + lid];
    }
    const float sig = 1.0f / (1.0f + metal::exp(-gt));
    Y[yrow * a.yr + n0 + lid] = (gt * sig) * ut;
  }
}

#define QSK_ARGS (const device uint8_t*, const device half*, const device half*, const device float*, device float*, constant QArgs&, const device uint*, const device uint*, const device uint*, const device uint*, uint3, uint, uint, uint)
#define GSK_ARGS (const device uint8_t*, const device half*, const device half*, const device uint8_t*, const device half*, const device half*, const device float*, device float*, constant QArgs&, const device uint*, const device uint*, const device uint*, const device uint*, uint3, uint, uint, uint)
template [[host_name("g53_qmv_sk_b8")]] [[kernel]] void g53_qmv_sk<8, 2> QSK_ARGS;
template [[host_name("g53_qmv_sk_b6")]] [[kernel]] void g53_qmv_sk<6, 2> QSK_ARGS;
template [[host_name("g53_gateup_sk_b8")]] [[kernel]] void g53_gateup_sk<8, 2> GSK_ARGS;
template [[host_name("g53_gateup_sk_b6")]] [[kernel]] void g53_gateup_sk<6, 2> GSK_ARGS;

// ---------------------------------------------------------------- bf16 weights, fp32 input ------------------------
struct GArgs {
  int K;    // input length (a multiple of 2048)
  int N;    // output values a row
  int ldw;  // bf16 values between weight rows
  int xr;   // floats between input rows
  int yr;   // floats between output rows
};

// y[n] = sum_k W[n][k] x[k] for bf16 W: a threadgroup takes 4 rows, its 8 simdgroups an eighth of K each (lanes 8
// values a step), the 8 partials added in order. K a multiple of 2048. Threadgroups [ceil(N/4)] of 256.
template <int RT>
[[kernel]] void g53_gemv_bf16(const device bfloat* W [[buffer(0)]], const device float* X [[buffer(1)]],
                              device float* Y [[buffer(2)]], constant GArgs& a [[buffer(3)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint lid [[thread_index_in_threadgroup]]) {
  threadgroup float part[8 * 4];
  X += long(tg.y) * a.xr;
  Y += long(tg.y) * a.yr;
  const int n0 = int(tg.x) * RT;
  const int rows = min(RT, a.N - n0);
  const int kp = a.K / 8;
  const int k0 = int(sg) * kp;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int k = k0 + int(lane) * 8; k < k0 + kp; k += 256) {
    float xv[8];
    for (int i = 0; i < 8; ++i) xv[i] = X[k + i];
    for (int r = 0; r < 4; ++r) {
      if (r < rows) {
        const device bfloat* w = W + long(n0 + r) * a.ldw + k;
        float pr = 0.0f;
        for (int i = 0; i < 8; ++i) pr += float(w[i]) * xv[i];
        acc[r] += pr;
      }
    }
  }
  for (int r = 0; r < 4; ++r) {
    const float v = simd_sum(acc[r]);
    if (lane == 0) part[sg * 4 + r] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(lid) < rows) {
    float t = part[lid];
    for (int g = 1; g < 8; ++g) t += part[g * 4 + lid];
    Y[n0 + lid] = t;
  }
}
template [[host_name("g53_gemv_bf16")]] [[kernel]] void g53_gemv_bf16<4>(const device bfloat*, const device float*, device float*, constant GArgs&, uint2, uint, uint, uint);
template [[host_name("g53_gemv1_bf16")]] [[kernel]] void g53_gemv_bf16<1>(const device bfloat*, const device float*, device float*, constant GArgs&, uint2, uint, uint, uint);

// g53_gemv_bf16 over up to NR input rows a threadgroup (G53_GROWS=1): each weight read serves every row; per row the
// arithmetic is g53_gemv_bf16's exactly (the same 8-value products summed in the same order into the same lane
// partial, the same simd_sum, the same in-order 8-partial sum). Threadgroups [ceil(N/RT), ceil(rows/NR)] of 256.
// Pattern: Ash Hart's TensorFold glm qmv_rows (rows in one threadgroup share a weight read).
template <int RT, int NR>
[[kernel]] void g53_gemv_rows_bf16(const device bfloat* W [[buffer(0)]], const device float* X [[buffer(1)]],
                                   device float* Y [[buffer(2)]], constant GArgs& a [[buffer(3)]],
                                   constant int& nrows [[buffer(4)]],
                                   uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]], uint lid [[thread_index_in_threadgroup]]) {
  threadgroup float part[NR * 8 * 4];
  const int j0 = int(tg.y) * NR;
  const int nj = min(NR, nrows - j0);
  X += long(j0) * a.xr;
  Y += long(j0) * a.yr;
  const int n0 = int(tg.x) * RT;
  const int rows = min(RT, a.N - n0);
  const int kp = a.K / 8;
  const int k0 = int(sg) * kp;
  float acc[NR][4];
  for (int j = 0; j < NR; ++j)
    for (int r = 0; r < 4; ++r) acc[j][r] = 0.0f;
  for (int k = k0 + int(lane) * 8; k < k0 + kp; k += 256) {
    for (int r = 0; r < 4; ++r) {
      if (r < rows) {
        const device bfloat* w = W + long(n0 + r) * a.ldw + k;
        float wv[8];
        for (int i = 0; i < 8; ++i) wv[i] = float(w[i]);
        for (int j = 0; j < NR; ++j) {
          if (j < nj) {
            const device float* xj = X + long(j) * a.xr + k;
            float pr = 0.0f;
            for (int i = 0; i < 8; ++i) pr += wv[i] * xj[i];
            acc[j][r] += pr;
          }
        }
      }
    }
  }
  for (int j = 0; j < NR; ++j) {
    for (int r = 0; r < 4; ++r) {
      const float v = simd_sum(acc[j][r]);
      if (lane == 0) part[(j * 8 + sg) * 4 + r] = v;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int t = int(lid); t < NR * 4; t += 256) {
    const int j = t / 4, r = t % 4;
    if (j < nj && r < rows) {
      float s = part[(j * 8) * 4 + r];
      for (int g = 1; g < 8; ++g) s += part[(j * 8 + g) * 4 + r];
      Y[long(j) * a.yr + n0 + r] = s;
    }
  }
}
template [[host_name("g53_gemv_rows_bf16")]] [[kernel]] void g53_gemv_rows_bf16<4, 8>(const device bfloat*, const device float*, device float*, constant GArgs&, constant int&, uint2, uint, uint, uint);
template [[host_name("g53_gemv1_rows_bf16")]] [[kernel]] void g53_gemv_rows_bf16<1, 8>(const device bfloat*, const device float*, device float*, constant GArgs&, constant int&, uint2, uint, uint, uint);

// ---------------------------------------------------------------- norms -------------------------------------------
struct NArgs {
  int D;
  float eps;
  int round_bf16;  // 1: x is bf16-valued and the norm rounds as MLX's bf16 rms does (layer 0's input norm)
  int ld;          // floats between rows (threadgroup x = row)
};

// One threadgroup of 1024 threads: fixed-order fp32 sum of squares (per-thread runs, simd sums, then 32 partials).
inline float block_sum(float acc, threadgroup float* part, uint lid, uint lane, uint sg) {
  acc = simd_sum(acc);
  if (lane == 0) part[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = (lid < 32) ? part[lid] : 0.0f;
  if (sg == 0) t = simd_sum(t);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) part[0] = t;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float r = part[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}

// out = w * (x * rsqrt(mean(x^2) + eps)); in place allowed. Threadgroup [1024].
[[kernel]] void g53_rms(const device float* X [[buffer(0)]], const device bfloat* Wt [[buffer(1)]],
                        device float* O [[buffer(2)]], constant NArgs& a [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[32];
  X += long(row) * a.ld;
  O += long(row) * a.ld;
  float acc = 0.0f;
  for (int i = int(lid); i < a.D; i += 1024) acc += X[i] * X[i];
  const float inv = metal::precise::rsqrt(block_sum(acc, part, lid, lane, sg) / float(a.D) + a.eps);
  for (int i = int(lid); i < a.D; i += 1024) {
    if (a.round_bf16) O[i] = float(bfloat(float(Wt[i]) * float(bfloat(X[i] * inv))));
    else O[i] = float(Wt[i]) * (X[i] * inv);
  }
}

// h += sum of S fp32 partials (slot order 0..S-1); then normed = rms(h) * w. Threadgroup [1024].
struct SArgs {
  int D;
  int S;        // partial slots
  int stride;   // floats between slots
  float eps;
  int norm;     // 0: only the residual add
};
[[kernel]] void g53_sum_res_rms(const device float* P [[buffer(0)]], device float* H [[buffer(1)]],
                                const device bfloat* Wt [[buffer(2)]], device float* O [[buffer(3)]],
                                constant SArgs& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[32];
  P += long(row) * a.D;
  H += long(row) * a.D;
  O += long(row) * a.D;
  float acc = 0.0f;
  for (int i = int(lid); i < a.D; i += 1024) {
    float r = P[i];
    for (int s = 1; s < a.S; ++s) r += P[long(s) * a.stride + i];
    const float h = H[i] + r;
    H[i] = h;
    acc += h * h;
  }
  if (a.norm == 0) return;
  const float inv = metal::precise::rsqrt(block_sum(acc, part, lid, lane, sg) / float(a.D) + a.eps);
  for (int i = int(lid); i < a.D; i += 1024) O[i] = float(Wt[i]) * (H[i] * inv);
}

// g53_sum_res_rms with the exchange's wait folded in (G53_FWAIT=1): thread 0 of every threadgroup polls each peer's
// flag (as g53_wait), then the same slot-ordered sum. Slots are read as relaxed atomic words (peers' bytes land from
// the NIC mid-buffer), the values and the order of the adds are g53_sum_res_rms's exactly. Pattern: Ash Hart's
// TensorFold glm_ep.metal ep_rsum / flashnext tp_plain (wait in the consuming launch, peer words read as atomics).
struct WArgs2 {
  uint seq;
  uint ranks;
  uint me;
};
[[kernel]] void g53_wsum_res_rms(device atomic_uint* PA [[buffer(0)]], device float* H [[buffer(1)]],
                                 const device bfloat* Wt [[buffer(2)]], device float* O [[buffer(3)]],
                                 constant SArgs& a [[buffer(4)]], device atomic_uint* flags [[buffer(5)]],
                                 device atomic_uint* gaveup [[buffer(6)]], constant WArgs2& w [[buffer(7)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint lid [[thread_position_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[32];
  if (lid == 0) {
    for (uint r = 0; r < w.ranks; ++r) {
      if (r == w.me) continue;
      uint polls = 0;
      while (int(atomic_load_explicit(&flags[2 * r], memory_order_relaxed) - w.seq) < 0) {
        if (++polls > 400000000u) {
          atomic_fetch_add_explicit(gaveup, 1u, memory_order_relaxed);
          break;
        }
      }
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device); // the flags before the slots
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  PA += long(row) * a.D;
  H += long(row) * a.D;
  O += long(row) * a.D;
  float acc = 0.0f;
  for (int i = int(lid); i < a.D; i += 1024) {
    float r = as_type<float>(atomic_load_explicit(&PA[i], memory_order_relaxed));
    for (int s = 1; s < a.S; ++s) r += as_type<float>(atomic_load_explicit(&PA[long(s) * a.stride + i], memory_order_relaxed));
    const float h = H[i] + r;
    H[i] = h;
    acc += h * h;
  }
  if (a.norm == 0) return;
  const float inv = metal::precise::rsqrt(block_sum(acc, part, lid, lane, sg) / float(a.D) + a.eps);
  for (int i = int(lid); i < a.D; i += 1024) O[i] = float(Wt[i]) * (H[i] * inv);
}

// The embedding row (bf16) as fp32 values.
[[kernel]] void g53_embed(const device bfloat* T [[buffer(0)]], const device uint* tok [[buffer(1)]],
                          device float* H [[buffer(2)]], constant int& D [[buffer(3)]],
                          uint2 p [[thread_position_in_grid]]) {
  if (int(p.x) < D) H[long(p.y) * D + p.x] = float(T[long(tok[p.y]) * D + p.x]);
}

// ---------------------------------------------------------------- rope (MLX fast rope, traditional) ---------------
// Rotate interleaved pairs of `dims` values at `off` in each of `rows` rows `stride` apart, at position pos.
struct RArgs {
  int rows;
  int stride;
  int off;
  int dims;
  int pos;
  float log2base;
  int hpr;      // rows a token (a token's heads): row r sits at position pos + r / hpr
};
inline float2 rope_vals(float x1, float x2, int i, int half_dims, int pos, float log2base) {
  const float d = float(i) / float(half_dims);
  const float inv_freq = metal::exp2(-d * log2base);
  const float theta = float(pos) * inv_freq;
  const float c = metal::fast::cos(theta), s = metal::fast::sin(theta);
  return float2(x1 * c - x2 * s, x1 * s + x2 * c);
}
inline void rope_pair(device float* r, int i, int half_dims, int pos, float log2base) {
  const float d = float(i) / float(half_dims);
  const float inv_freq = metal::exp2(-d * log2base);
  const float theta = float(pos) * inv_freq;
  const float c = metal::fast::cos(theta), s = metal::fast::sin(theta);
  const float x1 = r[2 * i], x2 = r[2 * i + 1];
  r[2 * i] = x1 * c - x2 * s;
  r[2 * i + 1] = x1 * s + x2 * c;
}
// threads [dims/2, rows].
[[kernel]] void g53_rope(device float* X [[buffer(0)]], constant RArgs& a [[buffer(1)]],
                         uint2 p [[thread_position_in_grid]]) {
  if (int(p.x) >= a.dims / 2 || int(p.y) >= a.rows) return;
  rope_pair(X + long(p.y) * a.stride + a.off, int(p.x), a.dims / 2, a.pos + int(p.y) / a.hpr, a.log2base);
}

// The latent cache row at pos: rms(kv[0:512]) * w and the roped kv[512:576]. Threadgroup [512].
struct CArgs {
  int pos;
  float eps;
  float log2base;
};
[[kernel]] void g53_kv_store(const device float* KV [[buffer(0)]], const device bfloat* Wt [[buffer(1)]],
                             device KVT* C [[buffer(2)]], constant CArgs& a [[buffer(3)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[32];
  KV += long(row) * 576;
  const int pos = a.pos + int(row);
  const float x = KV[lid];
  float acc = x * x;
  acc = simd_sum(acc);
  if (lane == 0) part[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = (lid < 16) ? part[lid] : 0.0f;
  if (sg == 0) t = simd_sum(t);
  if (lid == 0) part[0] = t;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = metal::precise::rsqrt(part[0] / 512.0f + a.eps);
  device KVT* cr = C + long(pos) * 576;
  cr[lid] = KVT(float(Wt[lid]) * (x * inv));
  if (lid < 32) {
    const float2 r = rope_vals(KV[512 + 2 * lid], KV[513 + 2 * lid], int(lid), 32, pos, a.log2base);
    cr[512 + 2 * lid] = KVT(r.x);
    cr[513 + 2 * lid] = KVT(r.y);
  }
}

// Indexer key at pos: LayerNorm(k) * w + b over 128, rope on the first 64, into the cache. Threadgroup [128].
[[kernel]] void g53_idx_k_store(const device float* K [[buffer(0)]], const device bfloat* Wt [[buffer(1)]],
                                const device bfloat* Bs [[buffer(2)]], device KVT* C [[buffer(3)]],
                                constant CArgs& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[4];
  threadgroup float stat[2];
  threadgroup float kn[128];
  const int pos = a.pos + int(row);
  const float x = K[long(row) * 128 + lid];
  float s = simd_sum(x);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) stat[0] = (((part[0] + part[1]) + part[2]) + part[3]) / 128.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float d = x - stat[0];
  float v = simd_sum(d * d);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) stat[1] = metal::precise::rsqrt((((part[0] + part[1]) + part[2]) + part[3]) / 128.0f + a.eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device KVT* cr = C + long(pos) * 128;
  const float y = (d * stat[1]) * float(Wt[lid]) + float(Bs[lid]);
  kn[lid] = y;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid >= 64) cr[lid] = KVT(y);
  if (lid < 32) {
    const float2 r = rope_vals(kn[2 * lid], kn[2 * lid + 1], int(lid), 32, pos, a.log2base);
    cr[2 * lid] = KVT(r.x);
    cr[2 * lid + 1] = KVT(r.y);
  }
}

// ---------------------------------------------------------------- indexer scores and top-k --------------------------
struct IArgs {
  int p0;       // row r's keys are [0, p0 + r + 1)
  float wscale; // n_heads^-0.5 * head_dim^-0.5
  int cap;      // floats between rows' scores
  int keys;     // index_topk: rows with no more keys than this select nothing
};
// score[t] = sum_h (w[h] * wscale) * max(0, q[h] . k[t]): a simdgroup a key, lane h a head; threadgroups of 256 (8 keys).
[[kernel]] void g53_idx_scores(const device float* Q [[buffer(0)]], const device float* Wh [[buffer(1)]],
                               const device KVT* KC [[buffer(2)]], device float* Sc [[buffer(3)]],
                               constant IArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float q[32 * 129];  // padded rows: lane h reads row h
  const int n = a.p0 + int(tg.y) + 1;
  if (n <= a.keys || int(tg.x) * 8 >= n) return;
  Q += long(tg.y) * 4096;
  Wh += long(tg.y) * 32;
  Sc += long(tg.y) * a.cap;
  for (int i = int(lid); i < 32 * 128; i += 256) q[(i / 128) * 129 + i % 128] = Q[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int t = int(tg.x) * 8 + int(sg);
  if (t >= n) return;
  const device KVT* k = KC + long(t) * 128;
  const threadgroup float* qh = q + lane * 129;
  float s = 0.0f;
  for (int d = 0; d < 128; ++d) s += qh[d] * k[d];
  const float v = simd_sum(max(s, 0.0f) * (Wh[lane] * a.wscale));
  if (lane == 0) Sc[t] = v;
}

// The same scores as g53_idx_scores, bit for bit: every key's dot product runs d = 0..127 in order in lane h (head h),
// then max(s, 0) * (w[h] * wscale) and the same simd_sum over the 32 heads. What changes is the work around it:
// g53_idx_scores stages a row's whole query (32 x 128 fp32 = 16 KB) into threadgroup memory to score 8 keys, so a long
// prompt reads ~2 KB of query a (row, key) — the cost that makes prompt reading O(n^2) with a large constant. Here a
// threadgroup stages the query once for `span` keys, and each simdgroup keeps 8 keys in flight (8 independent FMA
// chains, the keys read 4 dims at a time). Threadgroups [rows, ceil(last_n / span)] of 256: rows vary fastest, so the
// rows of a block walk the same span of keys together and share it in cache.
struct IArgs2 {
  int p0;       // row r's keys are [0, p0 + r + 1)
  float wscale; // n_heads^-0.5 * head_dim^-0.5
  int cap;      // floats between rows' scores
  int keys;     // index_topk: rows with no more keys than this select nothing
  int span;     // keys a threadgroup scores (a multiple of 64)
  int rows;     // the block's rows (g53_idx_scores3: a missing second row)
};
[[kernel]] void g53_idx_scores2(const device float* Q [[buffer(0)]], const device float* Wh [[buffer(1)]],
                                const device KVT* KC [[buffer(2)]], device float* Sc [[buffer(3)]],
                                constant IArgs2& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float q[32 * 129];  // padded rows: lane h reads row h (as g53_idx_scores)
  const int row = int(tg.x);
  const int n = a.p0 + row + 1;
  const int t0 = int(tg.y) * a.span;
  if (n <= a.keys || t0 >= n) return;
  Q += long(row) * 4096;
  Wh += long(row) * 32;
  Sc += long(row) * a.cap;
  for (int i = int(lid); i < 32 * 128; i += 256) q[(i / 128) * 129 + i % 128] = Q[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const threadgroup float* qh = q + lane * 129;
  const float wl = Wh[lane] * a.wscale;
  const int t1 = min(n, t0 + a.span);
  typedef vec<KVT, 4> kv4;
  // simdgroup sg takes keys [t, t + 8) of every group of 64 in the span
  for (int t = t0 + int(sg) * 8; t < t1; t += 64) {
    const device kv4* k[8];
    for (int j = 0; j < 8; ++j) k[j] = (const device kv4*)(KC + long(min(t + j, t1 - 1)) * 128);
    float s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int d4 = 0; d4 < 32; ++d4) {
      const float q0 = qh[4 * d4], q1 = qh[4 * d4 + 1], q2 = qh[4 * d4 + 2], q3 = qh[4 * d4 + 3];
      for (int j = 0; j < 8; ++j) {
        const kv4 kk = k[j][d4];
        float x = s[j];
        x += q0 * float(kk.x);
        x += q1 * float(kk.y);
        x += q2 * float(kk.z);
        x += q3 * float(kk.w);
        s[j] = x;
      }
    }
    float v[8];
    for (int j = 0; j < 8; ++j) v[j] = simd_sum(max(s[j], 0.0f) * wl);
    if (lane < 8 && t + int(lane) < t1) {
      float mine = v[0];
      for (int j = 1; j < 8; ++j) mine = (int(lane) == j) ? v[j] : mine;
      Sc[t + int(lane)] = mine;
    }
  }
}

inline uint order_key(float f) {
  const uint u = as_type<uint>(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// The `top` largest of n scores as ascending key indices (ties at the threshold: highest indices). One threadgroup of 1024.
struct TArgs {
  int p0;   // row r selects among [0, p0 + r + 1)
  int top;
  int cap;  // floats between rows' scores
};
[[kernel]] void g53_topk(const device float* Sc [[buffer(0)]], device uint* Out [[buffer(1)]],
                         constant TArgs& a [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                         uint lid [[thread_position_in_threadgroup]]) {
  const struct { int n; int top; } a_ = {a.p0 + int(row) + 1, a.top};
  if (a_.n <= a_.top) return;
  Sc += long(row) * a.cap;
  Out += long(row) * a.top;
  threadgroup atomic_uint hist[256];
  threadgroup uint prefix_key;
  threadgroup uint need;
  threadgroup uint dsum[8];
  threadgroup uint found_digit;
  threadgroup uint found_left;
  threadgroup uint counts_gt[1024];
  threadgroup uint counts_eq[1024];
  if (lid == 0) {
    prefix_key = 0;
    need = uint(a_.top);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint mask = 0;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (lid < 256) atomic_store_explicit(&hist[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint pk = prefix_key;
    for (int i = int(lid); i < a_.n; i += 1024) {
      const uint k = order_key(Sc[i]);
      if ((k & mask) == pk) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {  // thread t < 256 holds digit 255 - t; the inclusive count from 255 down finds the digit holding the need-th key
      const uint t = lid, lane = lid % 32, sgi = lid / 32;
      const uint c = t < 256 ? atomic_load_explicit(&hist[255 - t], memory_order_relaxed) : 0u;
      const uint xin = simd_prefix_inclusive_sum(c);
      if (t < 256 && lane == 31) dsum[sgi] = xin;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint off = 0;
      for (uint g = 0; g < sgi && g < 8; ++g) off += dsum[g];
      const uint cum = xin + off;
      const uint nd = need;
      if (t < 256 && cum >= nd && cum - c < nd) {
        found_digit = 255 - t;
        found_left = nd - (cum - c);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lid == 0) {
        need = found_left;
        prefix_key = pk | (found_digit << shift);
      }
    }
    mask |= 255u << shift;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // prefix_key is the threshold key; `need` of the keys equal to it are taken, lowest index first
  const uint thr = prefix_key;
  const int chunk = (a_.n + 1023) / 1024;
  const int lo = min(a_.n, int(lid) * chunk), hi = min(a_.n, lo + chunk);
  uint gt = 0, eq = 0;
  for (int i = lo; i < hi; ++i) {
    const uint k = order_key(Sc[i]);
    gt += k > thr;
    eq += k == thr;
  }
  threadgroup uint total_eq;
  {  // exclusive scans in thread order: within simdgroups, then across the 32 simdgroup totals
    const uint lane = lid % 32, sgi = lid / 32;
    const uint xg = simd_prefix_exclusive_sum(gt), xe = simd_prefix_exclusive_sum(eq);
    threadgroup uint tg_g[32], tg_e[32];
    if (lane == 31) {
      tg_g[sgi] = xg + gt;
      tg_e[sgi] = xe + eq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi == 0) {
      const uint a2 = tg_g[lane], b2 = tg_e[lane];
      const uint pa = simd_prefix_exclusive_sum(a2), pb = simd_prefix_exclusive_sum(b2);
      tg_g[lane] = pa;
      tg_e[lane] = pb;
      if (lane == 31) total_eq = pb + b2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    counts_gt[lid] = tg_g[sgi] + xg;
    counts_eq[lid] = tg_e[sgi] + xe;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // ties at the threshold go to the highest indices, as MLX's argpartition (an ascending sort, last k) takes them:
  // of the eq keys, the first `skip` (lowest indices) are left out
  const uint skip = total_eq - need;
  uint eq_before = counts_eq[lid];
  uint sel_before = counts_gt[lid] + (eq_before > skip ? eq_before - skip : 0u);
  for (int i = lo; i < hi; ++i) {
    const uint k = order_key(Sc[i]);
    bool take = k > thr;
    if (k == thr) {
      take = eq_before >= skip;
      eq_before += 1;
    }
    if (take) {
      Out[sel_before] = uint(i);
      sel_before += 1;
    }
  }
}

// ---------------------------------------------------------------- key-split decode indexer (G53_KSPLIT=1) ---------
// Decode/verify rows at long context: rank j scores only keys [lo, hi) of the block (a quarter), keeps that range's top
// `top` by g53_topk's own order (key value, ties to the highest index) as (score, index) candidates, the ranks swap
// candidates, and every rank takes the top `top` of the union in ascending index order. Every key's score is the same
// kernel's (bits per key do not depend on which keys a threadgroup takes); the union holds the global top set (a range's
// local top holds all of its keys above the global threshold and its highest-index ties), and the merge applies the same
// threshold and tie rule, so e.idx is g53_topk's exactly.
constant constexpr int KCAND = 4 * 2048; // candidates a row (4 ranks x top)

// Every rank's candidates for row r, in rank order (= ascending index), into Sc/Ix [rows][KCAND]; M[r] = their count.
// Threadgroups [rows] of 1024.
[[kernel]] void g53_kconcat(const device uint* Slots [[buffer(0)]], device float* Sc [[buffer(1)]], device uint* Ix [[buffer(2)]],
                            device uint* M [[buffer(3)]], constant int4& a [[buffer(4)]],  // ranks, slot stride (u32), rstride
                            uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]]) {
  const int ranks = a.x, sstride = a.y, rstride = a.z;
  int off = 0;
  for (int j = 0; j < ranks; ++j) {
    const device uint* C = Slots + long(j) * sstride + long(row) * rstride;
    const int n = min(int(C[0]), min(a.w, KCAND - off));  // a peer's count is clamped (a failed exchange cannot write wild)
    for (int i = int(lid); i < n; i += 1024) {
      Sc[long(row) * KCAND + off + i] = as_type<float>(C[4 + 2 * i]);
      Ix[long(row) * KCAND + off + i] = C[4 + 2 * i + 1];
    }
    off += n;
  }
  if (lid == 0) M[row] = uint(off);
}
[[kernel]] void g53_kmerge(const device float* Sc [[buffer(0)]], const device uint* Ix [[buffer(1)]],
                           const device uint* M [[buffer(2)]], device uint* Out [[buffer(3)]], constant int& top [[buffer(4)]],
                           uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]]) {
  const struct { int n; int top; } a_ = {min(int(M[row]), KCAND), top};
  Sc += long(row) * KCAND;
  Ix += long(row) * KCAND;
  Out += long(row) * top;
  if (a_.n <= a_.top) {
    for (int i = int(lid); i < a_.n; i += 1024) Out[i] = Ix[i];
    return;
  }
  threadgroup atomic_uint hist[256];
  threadgroup uint prefix_key;
  threadgroup uint need;
  threadgroup uint dsum[8];
  threadgroup uint found_digit;
  threadgroup uint found_left;
  threadgroup uint counts_gt[1024];
  threadgroup uint counts_eq[1024];
  if (lid == 0) {
    prefix_key = 0;
    need = uint(a_.top);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint mask = 0;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (lid < 256) atomic_store_explicit(&hist[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint pk = prefix_key;
    for (int i = int(lid); i < a_.n; i += 1024) {
      const uint k = order_key(Sc[i]);
      if ((k & mask) == pk) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {  // thread t < 256 holds digit 255 - t; the inclusive count from 255 down finds the digit holding the need-th key
      const uint t = lid, lane = lid % 32, sgi = lid / 32;
      const uint c = t < 256 ? atomic_load_explicit(&hist[255 - t], memory_order_relaxed) : 0u;
      const uint xin = simd_prefix_inclusive_sum(c);
      if (t < 256 && lane == 31) dsum[sgi] = xin;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint off = 0;
      for (uint g = 0; g < sgi && g < 8; ++g) off += dsum[g];
      const uint cum = xin + off;
      const uint nd = need;
      if (t < 256 && cum >= nd && cum - c < nd) {
        found_digit = 255 - t;
        found_left = nd - (cum - c);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lid == 0) {
        need = found_left;
        prefix_key = pk | (found_digit << shift);
      }
    }
    mask |= 255u << shift;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // prefix_key is the threshold key; `need` of the keys equal to it are taken, lowest index first
  const uint thr = prefix_key;
  const int chunk = (a_.n + 1023) / 1024;
  const int lo = min(a_.n, int(lid) * chunk), hi = min(a_.n, lo + chunk);
  uint gt = 0, eq = 0;
  for (int i = lo; i < hi; ++i) {
    const uint k = order_key(Sc[i]);
    gt += k > thr;
    eq += k == thr;
  }
  threadgroup uint total_eq;
  {  // exclusive scans in thread order: within simdgroups, then across the 32 simdgroup totals
    const uint lane = lid % 32, sgi = lid / 32;
    const uint xg = simd_prefix_exclusive_sum(gt), xe = simd_prefix_exclusive_sum(eq);
    threadgroup uint tg_g[32], tg_e[32];
    if (lane == 31) {
      tg_g[sgi] = xg + gt;
      tg_e[sgi] = xe + eq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi == 0) {
      const uint a2 = tg_g[lane], b2 = tg_e[lane];
      const uint pa = simd_prefix_exclusive_sum(a2), pb = simd_prefix_exclusive_sum(b2);
      tg_g[lane] = pa;
      tg_e[lane] = pb;
      if (lane == 31) total_eq = pb + b2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    counts_gt[lid] = tg_g[sgi] + xg;
    counts_eq[lid] = tg_e[sgi] + xe;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // ties at the threshold go to the highest indices, as MLX's argpartition (an ascending sort, last k) takes them:
  // of the eq keys, the first `skip` (lowest indices) are left out
  const uint skip = total_eq - need;
  uint eq_before = counts_eq[lid];
  uint sel_before = counts_gt[lid] + (eq_before > skip ? eq_before - skip : 0u);
  for (int i = lo; i < hi; ++i) {
    const uint k = order_key(Sc[i]);
    bool take = k > thr;
    if (k == thr) {
      take = eq_before >= skip;
      eq_before += 1;
    }
    if (take) {
      Out[sel_before] = Ix[i];
      sel_before += 1;
    }
  }
}

// g53_idx_scores2 with two rows a threadgroup: each key read once for two rows (the bits per row and key unchanged:
// the same d order, the same simd_sum). The two queries sit transposed, q[d][h] (lane h reads consecutive words, no
// padding needed), 2 x 16 KB = all of threadgroup memory. Threadgroups [ceil(rows / 2), ceil(last_n / span)] of 256.
[[kernel]] void g53_idx_scores3(const device float* Q [[buffer(0)]], const device float* Wh [[buffer(1)]],
                                const device KVT* KC [[buffer(2)]], device float* Sc [[buffer(3)]],
                                constant IArgs2& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float q[2 * 128 * 32];
  const int r0 = int(tg.x) * 2;
  const int n0 = a.p0 + r0 + 1, n1 = n0 + 1;
  const bool has1 = r0 + 1 < a.rows;
  const int t0 = int(tg.y) * a.span;
  const bool live0 = n0 > a.keys && t0 < n0;
  const bool live1 = has1 && n1 > a.keys && t0 < n1;
  if (!live0 && !live1) return;
  for (int i = int(lid); i < (has1 ? 2 : 1) * 4096; i += 256) {  // i = rr * 4096 + d * 32 + h: tg stores in order
    const int rr = i / 4096, d = (i % 4096) / 32, h = i % 32;
    q[i] = Q[long(r0 + rr) * 4096 + h * 128 + d];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float wl0 = Wh[long(r0) * 32 + lane] * a.wscale;
  const float wl1 = has1 ? Wh[long(r0 + 1) * 32 + lane] * a.wscale : 0.0f;
  const int t1 = min(has1 ? n1 : n0, t0 + a.span);  // row r0 + 1 has one more key; row r0's extra score is never written
  typedef vec<KVT, 4> kv4;
  for (int t = t0 + int(sg) * 8; t < t1; t += 64) {
    const device kv4* k[8];
    for (int j = 0; j < 8; ++j) k[j] = (const device kv4*)(KC + long(min(t + j, t1 - 1)) * 128);
    float s0[8] = {0, 0, 0, 0, 0, 0, 0, 0}, s1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int d4 = 0; d4 < 32; ++d4) {
      const int d = 4 * d4;
      const float a0 = q[(d + 0) * 32 + lane], a1 = q[(d + 1) * 32 + lane], a2 = q[(d + 2) * 32 + lane], a3 = q[(d + 3) * 32 + lane];
      const float b0 = q[4096 + (d + 0) * 32 + lane], b1 = q[4096 + (d + 1) * 32 + lane], b2 = q[4096 + (d + 2) * 32 + lane], b3 = q[4096 + (d + 3) * 32 + lane];
      for (int j = 0; j < 8; ++j) {
        const kv4 kk = k[j][d4];
        const float k0 = float(kk.x), k1 = float(kk.y), k2 = float(kk.z), k3 = float(kk.w);
        float x = s0[j];
        x += a0 * k0;
        x += a1 * k1;
        x += a2 * k2;
        x += a3 * k3;
        s0[j] = x;
        float y = s1[j];
        y += b0 * k0;
        y += b1 * k1;
        y += b2 * k2;
        y += b3 * k3;
        s1[j] = y;
      }
    }
    float v0[8], v1[8];
    for (int j = 0; j < 8; ++j) {
      v0[j] = simd_sum(max(s0[j], 0.0f) * wl0);
      v1[j] = simd_sum(max(s1[j], 0.0f) * wl1);
    }
    if (lane < 8) {
      float m0 = v0[0], m1 = v1[0];
      for (int j = 1; j < 8; ++j) {
        m0 = (int(lane) == j) ? v0[j] : m0;
        m1 = (int(lane) == j) ? v1[j] : m1;
      }
      const int tt = t + int(lane);
      if (live0 && tt < min(n0, t1)) Sc[long(r0) * a.cap + tt] = m0;
      if (live1 && tt < t1) Sc[long(r0 + 1) * a.cap + tt] = m1;
    }
  }
}

// ---------------------------------------------------------------- top-k for a few rows (decode) --------------------
// The same picks as g53_topk (the same radix select on order_key, ties at the threshold to the highest indices, the
// picks in ascending order), spread over many threadgroups: g53_topk runs one threadgroup a row, so one decode row reads
// all n scores 6 times on one GPU core. Rounds: g53_tk_init, then for shift 24, 16, 8, 0 g53_tk_hist ([chunks, rows]:
// the chunk's digit counts of keys matching the prefix, added into the row's device histogram) and g53_tk_digit ([rows]:
// the digit holding the need-th key, the histogram cleared); then g53_tk_count ([chunks, rows]: keys above / equal to the
// threshold), g53_tk_scan ([rows]: every chunk's offsets) and g53_tk_write ([chunks, rows]: the picks in order).
// State a row: [0] prefix (the threshold key at the end), [1] need (keys equal to it to take), [2] mask, [3] total equal.
struct TkArgs {
  int p0;    // row r selects among [0, p0 + r + 1)
  int top;
  int cap;   // floats between rows' scores
  int chunk; // keys a chunk (a multiple of 256)
  int shift; // the round's digit (hist, digit)
  int fixed; // > 0: every row selects among [0, fixed) (KSPLIT: a rank's whole key range); 0: [0, p0 + r + 1)
  int lo;    // KSPLIT pairs: global index of key 0
  int rstride; // KSPLIT pairs: u32 words between rows' candidate lists
};
inline bool tk_rows(constant TkArgs& a, int row, thread int& n) {
  n = a.fixed > 0 ? a.fixed : a.p0 + row + 1;
  return n > a.top;
}
[[kernel]] void g53_tk_init(device uint* St [[buffer(0)]], device uint* Hist [[buffer(1)]],
                            constant TkArgs& a [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]]) {
  Hist[row * 256 + lid] = 0;
  if (lid == 0) {
    St[row * 4 + 0] = 0;
    St[row * 4 + 1] = uint(a.top);
    St[row * 4 + 2] = 0;
    St[row * 4 + 3] = 0;
  }
}
[[kernel]] void g53_tk_hist(const device float* Sc [[buffer(0)]], device atomic_uint* Hist [[buffer(1)]],
                            const device uint* St [[buffer(2)]], constant TkArgs& a [[buffer(3)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
  int n;
  const int row = int(tg.y);
  if (!tk_rows(a, row, n)) return;
  const int lo = int(tg.x) * a.chunk;
  if (lo >= n) return;
  const int hi = min(n, lo + a.chunk);
  threadgroup atomic_uint h[256];
  atomic_store_explicit(&h[lid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint pk = St[row * 4 + 0], mask = St[row * 4 + 2];
  Sc += long(row) * a.cap;
  for (int i = lo + int(lid); i < hi; i += 256) {
    const uint k = order_key(Sc[i]);
    if ((k & mask) == pk) atomic_fetch_add_explicit(&h[(k >> a.shift) & 255u], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint c = atomic_load_explicit(&h[lid], memory_order_relaxed);
  if (c) atomic_fetch_add_explicit(&Hist[row * 256 + lid], c, memory_order_relaxed);
}
// threadgroups [rows] of 256: thread t holds digit 255 - t; the inclusive count from 255 down finds the need-th key's
// digit (g53_topk's arithmetic), then the histogram is cleared for the next round.
[[kernel]] void g53_tk_digit(device uint* St [[buffer(0)]], device uint* Hist [[buffer(1)]],
                             constant TkArgs& a [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]]) {
  int n;
  if (!tk_rows(a, int(row), n)) return;
  threadgroup uint dsum[8];
  threadgroup uint found_digit, found_left;
  const uint t = lid, lane = lid % 32, sgi = lid / 32;
  const uint c = Hist[row * 256 + 255 - t];
  const uint xin = simd_prefix_inclusive_sum(c);
  if (lane == 31) dsum[sgi] = xin;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint off = 0;
  for (uint g = 0; g < sgi; ++g) off += dsum[g];
  const uint cum = xin + off;
  const uint nd = St[row * 4 + 1];
  if (cum >= nd && cum - c < nd) {
    found_digit = 255 - t;
    found_left = nd - (cum - c);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  Hist[row * 256 + t] = 0;
  if (lid == 0) {
    St[row * 4 + 0] = St[row * 4 + 0] | (found_digit << a.shift);
    St[row * 4 + 1] = found_left;
    St[row * 4 + 2] = St[row * 4 + 2] | (255u << a.shift);
  }
}
// threadgroups [chunks, rows] of 256: the chunk's keys above and equal to the threshold.
[[kernel]] void g53_tk_count(const device float* Sc [[buffer(0)]], const device uint* St [[buffer(1)]],
                             device uint* Cnt [[buffer(2)]], constant TkArgs& a [[buffer(3)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
  int n;
  const int row = int(tg.y);
  if (!tk_rows(a, row, n)) return;
  const int lo = int(tg.x) * a.chunk;
  if (lo >= n) return;
  const int hi = min(n, lo + a.chunk);
  const uint thr = St[row * 4 + 0];
  Sc += long(row) * a.cap;
  uint gt = 0, eq = 0;
  for (int i = lo + int(lid); i < hi; i += 256) {
    const uint k = order_key(Sc[i]);
    gt += k > thr;
    eq += k == thr;
  }
  threadgroup uint pg[8], pe[8];
  gt = simd_sum(gt);
  eq = simd_sum(eq);
  if (lid % 32 == 0) {
    pg[lid / 32] = gt;
    pe[lid / 32] = eq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    uint sg = 0, se = 0;
    for (int g = 0; g < 8; ++g) {
      sg += pg[g];
      se += pe[g];
    }
    Cnt[(long(row) * 1024 + tg.x) * 2 + 0] = sg;
    Cnt[(long(row) * 1024 + tg.x) * 2 + 1] = se;
  }
}
// threadgroups [rows] of 1024 (chunks <= 1024): exclusive offsets of every chunk's above / equal counts, total equal.
[[kernel]] void g53_tk_scan(device uint* St [[buffer(0)]], const device uint* Cnt [[buffer(1)]],
                            device uint* Base [[buffer(2)]], constant TkArgs& a [[buffer(3)]],
                            uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]]) {
  int n;
  if (!tk_rows(a, int(row), n)) return;
  const int chunks = (n + a.chunk - 1) / a.chunk;
  const uint lane = lid % 32, sgi = lid / 32;
  const uint g0 = int(lid) < chunks ? Cnt[(long(row) * 1024 + lid) * 2 + 0] : 0u;
  const uint e0 = int(lid) < chunks ? Cnt[(long(row) * 1024 + lid) * 2 + 1] : 0u;
  const uint xg = simd_prefix_exclusive_sum(g0), xe = simd_prefix_exclusive_sum(e0);
  threadgroup uint tg_g[32], tg_e[32];
  if (lane == 31) {
    tg_g[sgi] = xg + g0;
    tg_e[sgi] = xe + e0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgi == 0) {
    const uint a2 = tg_g[lane], b2 = tg_e[lane];
    const uint pa = simd_prefix_exclusive_sum(a2), pb = simd_prefix_exclusive_sum(b2);
    tg_g[lane] = pa;
    tg_e[lane] = pb;
    if (lane == 31) St[row * 4 + 3] = pb + b2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(lid) < chunks) {
    Base[(long(row) * 1024 + lid) * 2 + 0] = tg_g[sgi] + xg;
    Base[(long(row) * 1024 + lid) * 2 + 1] = tg_e[sgi] + xe;
  }
}
// threadgroups [chunks, rows] of 256: each thread a contiguous run of its chunk, in order; the picks land where
// g53_topk puts them (keys above the threshold, then of the equal ones all but the `skip` lowest indices).
[[kernel]] void g53_tk_write(const device float* Sc [[buffer(0)]], const device uint* St [[buffer(1)]],
                             const device uint* Base [[buffer(2)]], device uint* Out [[buffer(3)]],
                             constant TkArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]]) {
  int n;
  const int row = int(tg.y);
  if (!tk_rows(a, row, n)) return;
  const int c0 = int(tg.x) * a.chunk;
  if (c0 >= n) return;
  const int c1 = min(n, c0 + a.chunk);
  const uint thr = St[row * 4 + 0], need = St[row * 4 + 1], total_eq = St[row * 4 + 3];
  Sc += long(row) * a.cap;
  Out += long(row) * a.top;
  const int per = (c1 - c0 + 255) / 256;
  const int lo = min(c1, c0 + int(lid) * per), hi = min(c1, lo + per);
  uint gt = 0, eq = 0;
  for (int i = lo; i < hi; ++i) {
    const uint k = order_key(Sc[i]);
    gt += k > thr;
    eq += k == thr;
  }
  const uint lane = lid % 32, sgi = lid / 32;
  const uint xg = simd_prefix_exclusive_sum(gt), xe = simd_prefix_exclusive_sum(eq);
  threadgroup uint tg_g[8], tg_e[8];
  if (lane == 31) {
    tg_g[sgi] = xg + gt;
    tg_e[sgi] = xe + eq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint og = 0, oe = 0;
  for (uint g = 0; g < sgi; ++g) {
    og += tg_g[g];
    oe += tg_e[g];
  }
  const uint skip = total_eq - need;
  uint eq_before = Base[(long(row) * 1024 + tg.x) * 2 + 1] + oe + xe;
  uint sel_before = Base[(long(row) * 1024 + tg.x) * 2 + 0] + og + xg + (eq_before > skip ? eq_before - skip : 0u);
  for (int i = lo; i < hi; ++i) {
    const uint k = order_key(Sc[i]);
    bool take = k > thr;
    if (k == thr) {
      take = eq_before >= skip;
      eq_before += 1;
    }
    if (take) {
      Out[sel_before] = uint(i);
      sel_before += 1;
    }
  }
}

// g53_tk_write's picks as (score, global index) pairs with a count word (KSPLIT local candidates; same picks, same order).
[[kernel]] void g53_tk_wpairs(const device float* Sc [[buffer(0)]], const device uint* St [[buffer(1)]],
                             const device uint* Base [[buffer(2)]], device uint* Out [[buffer(3)]],
                             constant TkArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]]) {
  int n;
  const int row = int(tg.y);
  if (!tk_rows(a, row, n)) return;
  const int c0 = int(tg.x) * a.chunk;
  if (c0 >= n) return;
  const int c1 = min(n, c0 + a.chunk);
  const uint thr = St[row * 4 + 0], need = St[row * 4 + 1], total_eq = St[row * 4 + 3];
  Sc += long(row) * a.cap;
  device uint* Cnt = Out + long(row) * a.rstride;
  Out = Cnt + 4;
  if (tg.x == 0 && lid == 0) Cnt[0] = uint(a.top);
  const int per = (c1 - c0 + 255) / 256;
  const int lo = min(c1, c0 + int(lid) * per), hi = min(c1, lo + per);
  uint gt = 0, eq = 0;
  for (int i = lo; i < hi; ++i) {
    const uint k = order_key(Sc[i]);
    gt += k > thr;
    eq += k == thr;
  }
  const uint lane = lid % 32, sgi = lid / 32;
  const uint xg = simd_prefix_exclusive_sum(gt), xe = simd_prefix_exclusive_sum(eq);
  threadgroup uint tg_g[8], tg_e[8];
  if (lane == 31) {
    tg_g[sgi] = xg + gt;
    tg_e[sgi] = xe + eq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint og = 0, oe = 0;
  for (uint g = 0; g < sgi; ++g) {
    og += tg_g[g];
    oe += tg_e[g];
  }
  const uint skip = total_eq - need;
  uint eq_before = Base[(long(row) * 1024 + tg.x) * 2 + 1] + oe + xe;
  uint sel_before = Base[(long(row) * 1024 + tg.x) * 2 + 0] + og + xg + (eq_before > skip ? eq_before - skip : 0u);
  for (int i = lo; i < hi; ++i) {
    const uint k = order_key(Sc[i]);
    bool take = k > thr;
    if (k == thr) {
      take = eq_before >= skip;
      eq_before += 1;
    }
    if (take) {
      Out[2 * sel_before] = as_type<uint>(Sc[i]);
      Out[2 * sel_before + 1] = uint(a.lo + i);
      sel_before += 1;
    }
  }
}

// KSPLIT rows whose range has no more than `top` keys: every key is a candidate (the few-launch select skips them).
// Threadgroups [rows] of 256.
[[kernel]] void g53_tk_allpairs(const device float* Sc [[buffer(0)]], device uint* Out [[buffer(1)]],
                                constant TkArgs& a [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]]) {
  int n;
  if (tk_rows(a, int(row), n)) return;  // more than top keys: g53_tk_wpairs writes this row
  Sc += long(row) * a.cap;
  device uint* Cnt = Out + long(row) * a.rstride;
  for (int i = int(lid); i < max(n, 0); i += 256) {
    Cnt[4 + 2 * i] = as_type<uint>(Sc[i]);
    Cnt[4 + 2 * i + 1] = uint(a.lo + i);
  }
  if (lid == 0) Cnt[0] = uint(max(n, 0));
}

// ---------------------------------------------------------------- latent attention -------------------------------
// Scores s = scale * (q_lat . c) + (scale * q_pe) . k_pe over a key list, a softmax, o = sum p c (512 dims).
// Pass 1: threadgroups [blocks of 128 keys, head groups of 8] of 1024 threads write each block's max, sum and
// unnormalized output; pass 2 joins blocks in order.
struct AArgs {
  int p0;        // row r attends keys [0, p0 + r + 1), or (past `keys` of them) its idx list
  int sparse;    // 0: always every key (debug)
  float scale;
  int heads;     // heads in the q buffers
  int keys;      // index_topk
  int qp_stride; // floats between heads' q_pe
};
// Row r's key count, whether it reads its idx list, and its blocks of 32.
inline int3 attn_keys(constant AArgs& a, int row) {
  const int n = a.p0 + row + 1;
  const bool sp = a.sparse && n > a.keys;
  const int k = sp ? a.keys : n;
  return int3(k, int(sp), (k + 31) / 32);
}
// threadgroups [ceil(n / 32), ceil(heads / 8)] of 256 threads: 32 keys a block, 8 heads a group. Simdgroup p takes
// latent dims [64p, 64p + 64) and rope dims [8p, 8p + 8), lane j key j; the 8 parts are added in order.
[[kernel]] void g53_attn_block(const device float* QL [[buffer(0)]], const device float* QP [[buffer(1)]],
                               const device KVT* C [[buffer(2)]], const device uint* IDX [[buffer(3)]],
                               device float* PO [[buffer(4)]], device float* PM [[buffer(5)]],
                               device float* PL [[buffer(6)]], constant AArgs& a [[buffer(7)]],
                               uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float q[8 * 576];
  threadgroup float part[8 * 256];
  threadgroup int keys[32];
  const int blk = int(tg.x), h0 = int(tg.y) * 8, row = int(tg.z);
  const int3 kk = attn_keys(a, row);
  if (blk >= kk.z) return;
  QL += long(row) * a.heads * 512;
  QP += long(row) * a.heads * a.qp_stride;
  IDX += long(row) * a.keys;
  for (int i = int(lid); i < 8 * 576; i += 256) {
    const int h = i / 576, d = i % 576;
    q[i] = (h0 + h >= a.heads) ? 0.0f : d < 512 ? QL[(h0 + h) * 512 + d] : QP[(h0 + h) * a.qp_stride + d - 512] * a.scale;
  }
  if (lid < 32) {
    const int j = blk * 32 + int(lid);
    keys[lid] = j < kk.x ? (kk.y ? int(IDX[j]) : j) : -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    const int key = keys[lane];
    const int pp = int(sg);
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (key >= 0) {
      const device KVT* c = C + long(key) * 576;
      for (int i = 0; i < 64; ++i) {
        const float kd = c[64 * pp + i];
        for (int h = 0; h < 8; ++h) acc[h] += q[h * 576 + 64 * pp + i] * kd;
      }
      float ar[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      for (int i = 0; i < 8; ++i) {
        const float kd = c[512 + 8 * pp + i];
        for (int h = 0; h < 8; ++h) ar[h] += q[h * 576 + 512 + 8 * pp + i] * kd;
      }
      for (int h = 0; h < 8; ++h) acc[h] = acc[h] * a.scale + ar[h];
    }
    for (int h = 0; h < 8; ++h) part[pp * 256 + h * 32 + int(lane)] = acc[h];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {  // thread t: head t / 32, key t % 32
    float sc = part[lid];
    for (int pp = 1; pp < 8; ++pp) sc += part[pp * 256 + lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    part[lid] = keys[lid % 32] >= 0 ? sc : -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int base = (row * 64 + blk) * a.heads + h0;
  {  // simdgroup h: head h's softmax over the block's 32 keys
    const float v = part[sg * 32 + lane];
    const float mx = simd_max(v);
    const float e = v == -INFINITY ? 0.0f : metal::exp(v - mx);
    const float l = simd_sum(e);
    part[sg * 32 + lane] = e;
    if (lane == 0 && h0 + int(sg) < a.heads) {
      PM[base + sg] = mx;
      PL[base + sg] = l;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int d = int(lid); d < 512; d += 256) {
    float o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int j = 0; j < 32; ++j) {
      const int key = keys[j];
      if (key >= 0) {
        const float kv = C[long(key) * 576 + d];
        for (int h = 0; h < 8; ++h) o[h] += part[h * 32 + j] * kv;
      }
    }
    for (int h = 0; h < 8 && h0 + h < a.heads; ++h) PO[long(base + h) * 512 + d] = o[h];
  }
}

// g53_attn_block for 4 heads a threadgroup (G53_ATTN4H=1): threadgroups [blocks of 32 keys, ceil(heads / 4), rows] of
// 256; threadgroup memory 26.8 -> 13.4 KB (two resident threadgroups a core instead of one). Per head the same products,
// parts, sums and order as g53_attn_block; PO/PM/PL the same layout, g53_attn_join unchanged: the same bits.
[[kernel]] void g53_attn_block4h(const device float* QL [[buffer(0)]], const device float* QP [[buffer(1)]],
                                 const device KVT* C [[buffer(2)]], const device uint* IDX [[buffer(3)]],
                                 device float* PO [[buffer(4)]], device float* PM [[buffer(5)]],
                                 device float* PL [[buffer(6)]], constant AArgs& a [[buffer(7)]],
                                 uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float q[4 * 576];
  threadgroup float part[8 * 128];
  threadgroup int keys[32];
  const int blk = int(tg.x), h0 = int(tg.y) * 4, row = int(tg.z);
  const int3 kk = attn_keys(a, row);
  if (blk >= kk.z) return;
  QL += long(row) * a.heads * 512;
  QP += long(row) * a.heads * a.qp_stride;
  IDX += long(row) * a.keys;
  for (int i = int(lid); i < 4 * 576; i += 256) {
    const int h = i / 576, d = i % 576;
    q[i] = (h0 + h >= a.heads) ? 0.0f : d < 512 ? QL[(h0 + h) * 512 + d] : QP[(h0 + h) * a.qp_stride + d - 512] * a.scale;
  }
  if (lid < 32) {
    const int j = blk * 32 + int(lid);
    keys[lid] = j < kk.x ? (kk.y ? int(IDX[j]) : j) : -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    const int key = keys[lane];
    const int pp = int(sg);
    float acc[4] = {0, 0, 0, 0};
    if (key >= 0) {
      const device KVT* c = C + long(key) * 576;
      for (int i = 0; i < 64; ++i) {
        const float kd = c[64 * pp + i];
        for (int h = 0; h < 4; ++h) acc[h] += q[h * 576 + 64 * pp + i] * kd;
      }
      float ar[4] = {0, 0, 0, 0};
      for (int i = 0; i < 8; ++i) {
        const float kd = c[512 + 8 * pp + i];
        for (int h = 0; h < 4; ++h) ar[h] += q[h * 576 + 512 + 8 * pp + i] * kd;
      }
      for (int h = 0; h < 4; ++h) acc[h] = acc[h] * a.scale + ar[h];
    }
    for (int h = 0; h < 4; ++h) part[pp * 128 + h * 32 + int(lane)] = acc[h];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {  // thread t < 128: head t / 32, key t % 32
    float sc = 0.0f;
    if (lid < 128) {
      sc = part[lid];
      for (int pp = 1; pp < 8; ++pp) sc += part[pp * 128 + lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 128) part[lid] = keys[lid % 32] >= 0 ? sc : -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int base = (row * 64 + blk) * a.heads + h0;
  if (sg < 4) {  // simdgroup h: head h's softmax over the block's 32 keys
    const float v = part[sg * 32 + lane];
    const float mx = simd_max(v);
    const float e = v == -INFINITY ? 0.0f : metal::exp(v - mx);
    const float l = simd_sum(e);
    part[sg * 32 + lane] = e;
    if (lane == 0 && h0 + int(sg) < a.heads) {
      PM[base + sg] = mx;
      PL[base + sg] = l;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int d = int(lid); d < 512; d += 256) {
    float o[4] = {0, 0, 0, 0};
    for (int j = 0; j < 32; ++j) {
      const int key = keys[j];
      if (key >= 0) {
        const float kv = C[long(key) * 576 + d];
        for (int h = 0; h < 4; ++h) o[h] += part[h * 32 + j] * kv;
      }
    }
    for (int h = 0; h < 4 && h0 + h < a.heads; ++h) PO[long(base + h) * 512 + d] = o[h];
  }
}

// g53_attn_join with loads issued 8 blocks at a time (G53_JOINB=1): the same max, factors and block-order adds.
[[kernel]] void g53_attn_joinb(const device float* PO [[buffer(0)]], const device float* PM [[buffer(1)]],
                               const device float* PL [[buffer(2)]], device float* O [[buffer(3)]],
                               constant AArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint2 tp [[thread_position_in_threadgroup]]) {
  const int h = int(tg.x), row = int(tg.y), d = int(tp.x);
  const int blocks = attn_keys(a, row).z;
  const int b0 = row * 64;
  float m = -INFINITY;
  for (int b = 0; b < blocks; ++b) m = max(m, PM[(b0 + b) * a.heads + h]);
  float o = 0.0f, l = 0.0f;
  for (int bb = 0; bb < blocks; bb += 8) {
    float pm[8], pl[8], po[8];
    for (int j = 0; j < 8; ++j) {
      const int b = min(bb + j, blocks - 1);
      pm[j] = PM[(b0 + b) * a.heads + h];
      pl[j] = PL[(b0 + b) * a.heads + h];
      po[j] = PO[long((b0 + b) * a.heads + h) * 512 + d];
    }
    for (int j = 0; j < 8; ++j) {
      if (bb + j < blocks) {
        const float mb = pm[j];
        const float f = (mb == -INFINITY) ? 0.0f : metal::exp(mb - m);
        o += f * po[j];
        l += f * pl[j];
      }
    }
  }
  O[(long(row) * a.heads + h) * 512 + d] = o / l;
}

// out[h][d] = sum_b e^(m_b - m) o_b / sum_b e^(m_b - m) l_b; threadgroups [heads] of 512.
[[kernel]] void g53_attn_join(const device float* PO [[buffer(0)]], const device float* PM [[buffer(1)]],
                              const device float* PL [[buffer(2)]], device float* O [[buffer(3)]],
                              constant AArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint2 tp [[thread_position_in_threadgroup]]) {
  const int h = int(tg.x), row = int(tg.y), d = int(tp.x);
  const int blocks = attn_keys(a, row).z;
  const int b0 = row * 64;
  float m = -INFINITY;
  for (int b = 0; b < blocks; ++b) m = max(m, PM[(b0 + b) * a.heads + h]);
  float o = 0.0f, l = 0.0f;
  for (int b = 0; b < blocks; ++b) {
    const float mb = PM[(b0 + b) * a.heads + h];
    const float f = (mb == -INFINITY) ? 0.0f : metal::exp(mb - m);
    o += f * PO[long((b0 + b) * a.heads + h) * 512 + d];
    l += f * PL[(b0 + b) * a.heads + h];
  }
  O[(long(row) * a.heads + h) * 512 + d] = o / l;
}

// ---------------------------------------------------------------- MoE route and combine -------------------------
// noaux_tc, one group: sigmoid(logits), picks by sigmoid + bias (ties: lowest index), weights normalized, scaled.
struct MArgs {
  int experts;
  int top;
  float scale;
  int direct;  // one row: write the pick segments here (no sort)
};
[[kernel]] void g53_route(const device float* G [[buffer(0)]], const device float* Bias [[buffer(1)]],
                          device uint* Ids [[buffer(2)]], device float* Wts [[buffer(3)]],
                          constant MArgs& a [[buffer(4)]], device uint* SE [[buffer(5)]], device uint* SR [[buffer(6)]],
                          device uint* SS [[buffer(7)]], device uint* SEG [[buffer(8)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sig[256];
  G += long(row) * a.experts;
  Ids += long(row) * a.top;
  Wts += long(row) * a.top;
  threadgroup float bv[8];
  threadgroup int bi[8];
  threadgroup uint picks[8];
  // one thread an expert (a.experts == 256 == threads)
  const float x = G[lid];
  const float s = 1.0f / (1.0f + metal::exp(-x));
  sig[lid] = s;
  float sel = s + Bias[lid];
  for (int k = 0; k < a.top; ++k) {
    // the largest (value, then lowest index) in each simdgroup, then across the 8
    float v = sel;
    int i = int(lid);
    for (int off = 16; off > 0; off /= 2) {
      const float ov = simd_shuffle_down(v, ushort(off));
      const int oi = simd_shuffle_down(i, ushort(off));
      if (ov > v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
      }
    }
    if (lane == 0) {
      bv[sg] = v;
      bi[sg] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
      float b = bv[0];
      int at = bi[0];
      for (int g = 1; g < 8; ++g)
        if (bv[g] > b || (bv[g] == b && bi[g] < at)) {
          b = bv[g];
          at = bi[g];
        }
      picks[k] = uint(at);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(lid) == int(picks[k])) sel = -INFINITY;
  }
  if (lid == 0) {
    float total = 0.0f;
    for (int k = 0; k < a.top; ++k) total += sig[picks[k]];
    for (int k = 0; k < a.top; ++k) {
      Ids[k] = picks[k];
      Wts[k] = sig[picks[k]] / total * a.scale;
    }
  }
  if (a.direct && lid < uint(a.top)) {  // one row: each pick its own segment, in pick order
    SE[lid] = picks[lid];
    SR[lid] = 0;
    SS[lid] = lid;
    SEG[1 + 2 * lid] = lid;
    SEG[2 + 2 * lid] = 1;
    if (lid == 0) SEG[0] = uint(a.top);
  }
}

// out[d] = sum_k w[k] y[k][d] (k in order) + y[top][d] (the shared expert, weight 1): this Mac's MoE partial.
struct CbArgs {
  int D;
  int top;
};
// threads [D, rows]: routed y [rows * top][D], shared y [rows][D].
[[kernel]] void g53_moe_combine(const device float* Y [[buffer(0)]], const device float* Wts [[buffer(1)]],
                                device float* O [[buffer(2)]], constant CbArgs& a [[buffer(3)]],
                                const device float* YS [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
  const int i = int(p.x);
  const long row = long(p.y);
  if (i >= a.D) return;
  float acc = 0.0f;
  for (int k = 0; k < a.top; ++k) acc += Y[(row * a.top + k) * a.D + i] * Wts[row * a.top + k];
  O[row * a.D + i] = acc + YS[row * a.D + i];
}

// Picks (row-major, top a row) grouped by expert, stable: sorted expert, its row and its pick slot; and SEG: the
// segments of at most a.z same-expert picks ([count, start0, len0, ...]). One threadgroup of 1024; n <= 4096 picks.
inline uint scan256(uint v, uint lid, threadgroup uint* tmp, threadgroup uint* total) {
  // exclusive prefix over threads 0..255 (others pass 0); every thread returns its prefix
  const uint lane = lid % 32, sgi = lid / 32;
  const uint x = simd_prefix_exclusive_sum(v);
  if (lane == 31 && sgi < 8) tmp[sgi] = x + v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint off = 0;
  for (uint g = 0; g < sgi && g < 8; ++g) off += tmp[g];
  if (lid == 255) *total = off + x + v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return off + x;
}

[[kernel]] void g53_sort_picks(const device uint* Ids [[buffer(0)]], device uint* SE [[buffer(1)]],
                               device uint* SR [[buffer(2)]], device uint* SS [[buffer(3)]],
                               constant int3& a [[buffer(4)]], device uint* SEG [[buffer(5)]],
                               uint lid [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint cnt[256];
  threadgroup uint start[256];
  threadgroup uint tmp[8];
  threadgroup uint total;
  const int n = a.x, top = a.y;
  const uint smax = uint(a.z);
  if (lid < 256) atomic_store_explicit(&cnt[lid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(lid); i < n; i += 1024) atomic_fetch_add_explicit(&cnt[Ids[i]], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint c = lid < 256 ? atomic_load_explicit(&cnt[lid], memory_order_relaxed) : 0u;
  const uint st = scan256(c, lid, tmp, &total);
  if (lid < 256) start[lid] = st;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // a pick's place: its expert's start plus the earlier picks of the same expert (stable)
  for (int i = int(lid); i < n; i += 1024) {
    const uint e = Ids[i];
    uint before = 0;
    for (int j = 0; j < i; ++j) before += Ids[j] == e;
    const uint at = start[e] + before;
    SE[at] = e;
    SR[at] = uint(i / top);
    SS[at] = uint(i);
  }
  // segments: expert e's ceil(c / smax), numbered in expert order
  const uint ns = (c + smax - 1) / smax;
  const uint s0 = scan256(ns, lid, tmp, &total);
  if (lid < 256) {
    for (uint k = 0; k < ns; ++k) {
      SEG[1 + 2 * (s0 + k)] = st + k * smax;
      SEG[2 + 2 * (s0 + k)] = min(smax, c - k * smax);
    }
  }
  if (lid == 0) SEG[0] = total;
}

// ---------------------------------------------------------------- head argmax and exchange words ------------------
// The largest of n logits (lowest index on ties) as (value, base + index). One threadgroup of 1024.
[[kernel]] void g53_argmax(const device float* L [[buffer(0)]], device float* Out [[buffer(1)]],
                           constant int2& a [[buffer(2)]], uint lid [[thread_position_in_threadgroup]]) {
  threadgroup float bv[1024];
  threadgroup int bi[1024];
  float v = -INFINITY;
  int idx = 0x7fffffff;
  for (int i = int(lid); i < a.x; i += 1024) {
    const float x = L[i];
    if (x > v) {
      v = x;
      idx = i;
    }
  }
  bv[lid] = v;
  bi[lid] = idx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float best = -INFINITY;
    int at = 0x7fffffff;
    for (int t = 0; t < 1024; ++t)
      if (bv[t] > best || (bv[t] == best && bi[t] < at)) {
        best = bv[t];
        at = bi[t];
      }
    Out[0] = best;
    ((device uint*)Out)[1] = uint(at + a.y);
  }
}

// The global pick from each slot's (value, index) pair, in slot order (ties: lowest index); the token for the next step.
[[kernel]] void g53_pick(const device float* P [[buffer(0)]], device uint* Tok [[buffer(1)]],
                         constant int2& a [[buffer(2)]]) {
  float best = -INFINITY;
  uint at = 0xffffffffu;
  for (int s = 0; s < a.x; ++s) {
    const float v = P[s * a.y];
    const uint i = ((const device uint*)P)[s * a.y + 1];
    if (v > best || (v == best && i < at)) {
      best = v;
      at = i;
    }
  }
  Tok[0] = at;
}

// Read one word in every 16 KB of a buffer region (G53_TOUCH: right after a snapshot load, so the first real command
// buffer does not pay for the freshly written pages); the sum lands in Sink[0] so the reads are not optimized away.
[[kernel]] void g53_touch(const device uint* B [[buffer(0)]], device atomic_uint* Sink [[buffer(1)]], constant uint2& a [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
  if (i >= a.x) return;  // a.x = pages, a.y = words a page
  const uint v = B[ulong(i) * a.y];
  if (v == 0x7fc00001u) atomic_fetch_add_explicit(Sink, 1u, memory_order_relaxed);  // practically never: keeps the load
}

// GPU -> host: this exchange's partial is in the window.
[[kernel]] void g53_post(device atomic_uint* sync [[buffer(0)]], constant uint& seq [[buffer(1)]]) {
  atomic_store_explicit(sync, seq, memory_order_relaxed);
}

// Wait until every peer's flag word (u64 low half, 8 bytes apart) reaches seq; give-ups counted.
struct WArgs {
  uint seq;
  uint ranks;
  uint me;
};
[[kernel]] void g53_wait(device atomic_uint* flags [[buffer(0)]], device atomic_uint* gaveup [[buffer(1)]],
                         constant WArgs& a [[buffer(2)]]) {
  for (uint r = 0; r < a.ranks; ++r) {
    if (r == a.me) continue;
    uint polls = 0;
    while (int(atomic_load_explicit(&flags[2 * r], memory_order_relaxed) - a.seq) < 0) {
      if (++polls > 400000000u) {
        atomic_fetch_add_explicit(gaveup, 1u, memory_order_relaxed);
        return;
      }
    }
  }
}

// dst[i] = src[i] for n floats.
[[kernel]] void g53_copy(const device float* S [[buffer(0)]], device float* Dd [[buffer(1)]], constant int& n [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
  if (int(i) < n) Dd[i] = S[i];
}

// G53_LSPLIT: `rows` rows of `len` floats from a strided matrix into another: D[r * dld + d] = S[r * sld + d]
[[kernel]] void g53_copy_rows(const device float* S [[buffer(0)]], device float* Dd [[buffer(1)]], constant int4& a [[buffer(2)]],
                              uint2 t [[thread_position_in_grid]]) {  // a = {len, sld, dld, rows}
  if (int(t.x) < a.x && int(t.y) < a.w) Dd[long(t.y) * a.z + t.x] = S[long(t.y) * a.y + t.x];
}

[[kernel]] void g53_copy_u32(const device uint* S [[buffer(0)]], device uint* Dd [[buffer(1)]], constant int& n [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
  if (int(i) < n) Dd[i] = S[i];
}

// ---------------------------------------------------------------- prompt blocks: tiled matmuls ---------------------
// A block's inputs against a quantized (6/8-bit, groups of 64) or bf16 (BITS 16) matrix with 8x8 fp32 simdgroup
// matrices: 32 inputs x 32 outputs a threadgroup, K in steps of 64 (one quant group), 4 simdgroups of 16 x 16.
// Weights are dequantized to fp32 in threadgroup memory (scale * q + bias), so the sums differ from the row path's
// in order only. Items, stacks, ids and segments as in g53_qmv (segments up to 32 picks of one expert).
constant constexpr int QT = 32;      // inputs and outputs a tile
constant constexpr int QK = 64;      // K a step
constant constexpr int QLD = QK + 4; // padded row of a threadgroup tile

template <int BITS>
inline void qmm_wtile(const device uint8_t* w, const device half* s, const device half* b, int ldw, int lds, int k0,
                      uint lid, threadgroup float* wt) {
  // thread t: output row t / 4, values [16 (t % 4), +16) of this K step
  const int o = int(lid) / 4, c = (int(lid) % 4) * 16;
  threadgroup float* dst = wt + o * QLD + c;
  if (BITS == 16) {
    const device bfloat* wr = (const device bfloat*)w + long(o) * (ldw / 2) + k0 + c;
    for (int i = 0; i < 16; ++i) dst[i] = float(wr[i]);
    return;
  }
  const float sc = float(s[long(o) * lds + k0 / 64]), bi = float(b[long(o) * lds + k0 / 64]);
  const device uint8_t* wr = w + long(o) * ldw + (k0 + c) * BITS / 8;
  threadgroup float4* d4 = (threadgroup float4*)dst;
  if (BITS == 8) {
    const device uchar4* w4 = (const device uchar4*)wr;
    for (int i = 0; i < 4; ++i) d4[i] = sc * float4(w4[i]) + bi;
  } else {
    const device uint* w3 = (const device uint*)wr;  // 16 six-bit values = 12 bytes = 3 words
    const uint u0 = w3[0], u1 = w3[1], u2 = w3[2];
    const ulong lo = ulong(u0) | (ulong(u1) << 32);
    float v[16];
    for (int i = 0; i < 10; ++i) v[i] = float((lo >> (6 * i)) & 0x3f);
    const ulong hi = (ulong(u1) >> 28) | (ulong(u2) << 4);  // bits 60.. of the 96
    for (int i = 10; i < 16; ++i) v[i] = float((hi >> (6 * (i - 10))) & 0x3f);
    for (int i = 0; i < 4; ++i) d4[i] = sc * float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]) + bi;
  }
}

template <int BITS, bool GU>
inline void qmm_body(const device uint8_t* GW, const device half* GS, const device half* GB,
                     const device uint8_t* UW, const device half* US, const device half* UB,
                     const device float* X, device float* Y, constant QArgs& a, const device uint* IDS,
                     const device uint* XIDS, const device uint* YIDS, const device uint* SEG, uint3 tg, uint lid,
                     uint sg, threadgroup float* xt, threadgroup float* wt, threadgroup float* ut, threadgroup long* xo,
                     threadgroup long* yo) {
  int start, cnt;
  if (a.segs) {
    if (int(tg.x) >= int(SEG[0])) return;
    start = int(SEG[1 + 2 * tg.x]);
    cnt = int(SEG[2 + 2 * tg.x]);
  } else {
    start = int(tg.x) * QT;
    cnt = min(QT, a.items - start);
    if (cnt <= 0) return;
  }
  const uint z = tg.z;
  const long e = (a.ids ? long(IDS[start]) : 0) + long(z);
  if (lid < QT) {
    const int it = start + min(int(lid), cnt - 1);
    xo[lid] = (a.xids ? long(XIDS[it]) : long(it)) * a.xr + long(z) * a.zx;
    yo[lid] = (a.yids ? long(YIDS[it]) : long(it)) * a.yr + long(z) * a.zy;
  }
  const int n0 = int(tg.y) * QT;
  const long wo = e * long(a.zw) + long(n0) * a.ldw;
  const long so = e * long(a.zs) + long(n0) * a.lds;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int rb = (int(sg) / 2) * 16, ob = (int(sg) % 2) * 16;
  const long xr_l = xo[min(int(lid) / 4, cnt - 1)];  // this thread's X row for every K step
  simdgroup_float8x8 acc[2][2], accu[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      acc[i][j] = simdgroup_float8x8(0.0f);
      accu[i][j] = simdgroup_float8x8(0.0f);
    }
  for (int k0 = 0; k0 < a.K; k0 += QK) {
    {
      const int r = int(lid) / 4, c = (int(lid) % 4) * 16;
      threadgroup float4* d4 = (threadgroup float4*)(xt + r * QLD + c);
      if (r < cnt) {
        const device float4* s4 = (const device float4*)(X + xr_l + k0 + c);
        for (int i = 0; i < 4; ++i) d4[i] = s4[i];
      } else {
        for (int i = 0; i < 4; ++i) d4[i] = float4(0.0f);
      }
    }
    qmm_wtile<BITS>(GW + wo, GS + so, GB + so, a.ldw, a.lds, k0, lid, wt);
    if (GU) qmm_wtile<BITS>(UW + wo, US + so, UB + so, a.ldw, a.lds, k0, lid, ut);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < QK; kk += 8) {
      simdgroup_float8x8 a0, a1, b0, b1;
      simdgroup_load(a0, xt + (rb + 0) * QLD + kk, QLD);
      simdgroup_load(a1, xt + (rb + 8) * QLD + kk, QLD);
      simdgroup_load(b0, wt + (ob + 0) * QLD + kk, QLD, ulong2(0, 0), true);
      simdgroup_load(b1, wt + (ob + 8) * QLD + kk, QLD, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
      simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
      simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
      simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
      if (GU) {
        simdgroup_load(b0, ut + (ob + 0) * QLD + kk, QLD, ulong2(0, 0), true);
        simdgroup_load(b1, ut + (ob + 8) * QLD + kk, QLD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(accu[0][0], a0, b0, accu[0][0]);
        simdgroup_multiply_accumulate(accu[0][1], a0, b1, accu[0][1]);
        simdgroup_multiply_accumulate(accu[1][0], a1, b0, accu[1][0]);
        simdgroup_multiply_accumulate(accu[1][1], a1, b1, accu[1][1]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // results through threadgroup memory (xt, and wt for up), then each input row to its place
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      simdgroup_store(acc[i][j], xt + (rb + 8 * i) * QLD + ob + 8 * j, QLD);
      if (GU) simdgroup_store(accu[i][j], wt + (rb + 8 * i) * QLD + ob + 8 * j, QLD);
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int idx = int(lid); idx < QT * QT; idx += 128) {
    const int r = idx / QT, c = idx % QT;
    if (r < cnt) {
      float v = xt[r * QLD + c];
      if (GU) {
        const float sig = 1.0f / (1.0f + metal::exp(-v));
        v = (v * sig) * wt[r * QLD + c];
      }
      Y[yo[r] + n0 + c] = v;
    }
  }
}

template <int BITS>
[[kernel]] void g53_qmm(const device uint8_t* W [[buffer(0)]], const device half* S [[buffer(1)]],
                        const device half* B [[buffer(2)]], const device float* X [[buffer(3)]],
                        device float* Y [[buffer(4)]], constant QArgs& a [[buffer(5)]],
                        const device uint* IDS [[buffer(6)]], const device uint* XIDS [[buffer(7)]],
                        const device uint* YIDS [[buffer(8)]], const device uint* SEG [[buffer(9)]],
                        uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float xt[QT * QLD];
  threadgroup float wt[QT * QLD];
  threadgroup long xo[QT], yo[QT];
  qmm_body<BITS, false>(W, S, B, W, S, B, X, Y, a, IDS, XIDS, YIDS, SEG, tg, lid, sg, xt, wt, wt, xo, yo);
}

template <int BITS>
[[kernel]] void g53_qmm_gateup(const device uint8_t* GW [[buffer(0)]], const device half* GS [[buffer(1)]],
                               const device half* GB [[buffer(2)]], const device uint8_t* UW [[buffer(3)]],
                               const device half* US [[buffer(4)]], const device half* UB [[buffer(5)]],
                               const device float* X [[buffer(6)]], device float* Y [[buffer(7)]],
                               constant QArgs& a [[buffer(8)]], const device uint* IDS [[buffer(9)]],
                               const device uint* XIDS [[buffer(10)]], const device uint* YIDS [[buffer(11)]],
                               const device uint* SEG [[buffer(12)]], uint3 tg [[threadgroup_position_in_grid]],
                               uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float xt[QT * QLD];
  threadgroup float wt[QT * QLD];
  threadgroup float ut[QT * QLD];
  threadgroup long xo[QT], yo[QT];
  qmm_body<BITS, true>(GW, GS, GB, UW, US, UB, X, Y, a, IDS, XIDS, YIDS, SEG, tg, lid, sg, xt, wt, ut, xo, yo);
}

#define QMM_ARGS (const device uint8_t*, const device half*, const device half*, const device float*, device float*, constant QArgs&, const device uint*, const device uint*, const device uint*, const device uint*, uint3, uint, uint)
#define QMMG_ARGS (const device uint8_t*, const device half*, const device half*, const device uint8_t*, const device half*, const device half*, const device float*, device float*, constant QArgs&, const device uint*, const device uint*, const device uint*, const device uint*, uint3, uint, uint)
template [[host_name("g53_qmm_b6")]] [[kernel]] void g53_qmm<6> QMM_ARGS;
template [[host_name("g53_qmm_b8")]] [[kernel]] void g53_qmm<8> QMM_ARGS;
template [[host_name("g53_qmm_b16")]] [[kernel]] void g53_qmm<16> QMM_ARGS;
template [[host_name("g53_qmm_gateup_b6")]] [[kernel]] void g53_qmm_gateup<6> QMMG_ARGS;
template [[host_name("g53_qmm_gateup_b8")]] [[kernel]] void g53_qmm_gateup<8> QMMG_ARGS;

// ---- larger tiles: BM inputs x BN outputs, K in steps of 32, 4 simdgroups as 2 x 2 of (BM/2) x (BN/2); each thread
// writes its two accumulator elements straight out (MLX's fragment coordinates), so no staging tile.
constant constexpr int QK2 = 32;
constant constexpr int QLD2 = QK2 + 4;

template <int BITS, int BN>
inline void qmm2_wtile(const device uint8_t* w, const device half* s, const device half* b, int ldw, int lds, int k0,
                       uint lid, threadgroup float* wt) {
  // BN rows x 32 values: thread t: row t / (256 / BN)... (BN 64: two threads a row, 16 values each)
  constexpr int TPR = 128 / BN;  // threads a row
  constexpr int VPT = QK2 / TPR; // values a thread
  const int o = int(lid) / TPR, c = (int(lid) % TPR) * VPT;
  threadgroup float* dst = wt + o * QLD2 + c;
  if (BITS == 16) {
    const device bfloat* wr = (const device bfloat*)w + long(o) * (ldw / 2) + k0 + c;
    for (int i = 0; i < VPT; ++i) dst[i] = float(wr[i]);
    return;
  }
  const float sc = float(s[long(o) * lds + k0 / 64]), bi = float(b[long(o) * lds + k0 / 64]);
  const device uint8_t* wr = w + long(o) * ldw + (k0 + c) * BITS / 8;
  if (VPT == 16) {
    threadgroup float4* d4 = (threadgroup float4*)dst;
    if (BITS == 8) {
      const device uchar4* w4 = (const device uchar4*)wr;
      for (int i = 0; i < 4; ++i) d4[i] = sc * float4(w4[i]) + bi;
    } else {
      const device uint* w3 = (const device uint*)wr;
      const uint u0 = w3[0], u1 = w3[1], u2 = w3[2];
      const ulong lo = ulong(u0) | (ulong(u1) << 32);
      float v[16];
      for (int i = 0; i < 10; ++i) v[i] = float((lo >> (6 * i)) & 0x3f);
      const ulong hi = (ulong(u1) >> 28) | (ulong(u2) << 4);
      for (int i = 10; i < 16; ++i) v[i] = float((hi >> (6 * (i - 10))) & 0x3f);
      for (int i = 0; i < 4; ++i) d4[i] = sc * float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]) + bi;
    }
    return;
  }
  if (BITS == 8) {
    for (int i = 0; i < VPT; ++i) dst[i] = sc * float(wr[i]) + bi;
  } else {
    for (int i = 0; i < VPT / 4; ++i) {
      const device uint8_t* p = wr + 3 * i;
      dst[4 * i + 0] = sc * float(p[0] & 0x3f) + bi;
      dst[4 * i + 1] = sc * float(((p[0] >> 6) & 0x03) | ((p[1] & 0x0f) << 2)) + bi;
      dst[4 * i + 2] = sc * float(((p[1] >> 4) & 0x0f) | ((p[2] & 0x03) << 4)) + bi;
      dst[4 * i + 3] = sc * float((p[2] >> 2) & 0x3f) + bi;
    }
  }
}

template <int BITS, bool GU, int BM, int BN>
inline void qmm2_body(const device uint8_t* GW, const device half* GS, const device half* GB,
                      const device uint8_t* UW, const device half* US, const device half* UB,
                      const device float* X, device float* Y, constant QArgs& a, const device uint* IDS,
                      const device uint* XIDS, const device uint* YIDS, const device uint* SEG, uint3 tg, uint lid,
                      uint sg, uint lane, threadgroup float* xt, threadgroup float* wt, threadgroup float* ut,
                      threadgroup long* xo, threadgroup long* yo) {
  constexpr int SM = BM / 2, SN = BN / 2, FM = SM / 8, FN = SN / 8;
  int start, cnt;
  if (a.segs) {
    if (int(tg.x) >= int(SEG[0])) return;
    start = int(SEG[1 + 2 * tg.x]);
    cnt = int(SEG[2 + 2 * tg.x]);
  } else {
    start = int(tg.x) * BM;
    cnt = min(BM, a.items - start);
    if (cnt <= 0) return;
  }
  const uint z = tg.z;
  const long e = (a.ids ? long(IDS[start]) : 0) + long(z);
  if (int(lid) < BM) {
    const int it = start + min(int(lid), cnt - 1);
    xo[lid] = (a.xids ? long(XIDS[it]) : long(it)) * a.xr + long(z) * a.zx;
    yo[lid] = (a.yids ? long(YIDS[it]) : long(it)) * a.yr + long(z) * a.zy;
  }
  const int n0 = int(tg.y) * BN;
  const long wo = e * long(a.zw) + long(n0) * a.ldw;
  const long so = e * long(a.zs) + long(n0) * a.lds;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int rb = (int(sg) / 2) * SM, ob = (int(sg) % 2) * SN;
  simdgroup_float8x8 acc[FM][FN], accu[GU ? FM : 1][GU ? FN : 1];
  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
  if (GU)
    for (int i = 0; i < FM; ++i)
      for (int j = 0; j < FN; ++j) accu[i][j] = simdgroup_float8x8(0.0f);
  for (int k0 = 0; k0 < a.K; k0 += QK2) {
    {
      constexpr int VX = BM * QK2 / 128;  // values a thread (8 at BM 32, 16 at BM 64)
      const int r = int(lid) / (QK2 / VX), c = (int(lid) % (QK2 / VX)) * VX;
      threadgroup float4* d4 = (threadgroup float4*)(xt + r * QLD2 + c);
      if (r < cnt) {
        const device float4* s4 = (const device float4*)(X + xo[r] + k0 + c);
        for (int i = 0; i < VX / 4; ++i) d4[i] = s4[i];
      } else {
        for (int i = 0; i < VX / 4; ++i) d4[i] = float4(0.0f);
      }
    }
    qmm2_wtile<BITS, BN>(GW + wo, GS + so, GB + so, a.ldw, a.lds, k0, lid, wt);
    if (GU) qmm2_wtile<BITS, BN>(UW + wo, US + so, UB + so, a.ldw, a.lds, k0, lid, ut);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < QK2; kk += 8) {
      simdgroup_float8x8 av[FM], bv[FN];
      for (int i = 0; i < FM; ++i) simdgroup_load(av[i], xt + (rb + 8 * i) * QLD2 + kk, QLD2);
      for (int j = 0; j < FN; ++j) simdgroup_load(bv[j], wt + (ob + 8 * j) * QLD2 + kk, QLD2, ulong2(0, 0), true);
      for (int i = 0; i < FM; ++i)
        for (int j = 0; j < FN; ++j) simdgroup_multiply_accumulate(acc[i][j], av[i], bv[j], acc[i][j]);
      if (GU) {
        for (int j = 0; j < FN; ++j) simdgroup_load(bv[j], ut + (ob + 8 * j) * QLD2 + kk, QLD2, ulong2(0, 0), true);
        for (int i = 0; i < FM; ++i)
          for (int j = 0; j < FN; ++j) simdgroup_multiply_accumulate(accu[i][j], av[i], bv[j], accu[i][j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const short qid = short(lane) / 4;
  const short fm = (qid & 4) + ((short(lane) / 2) % 4);
  const short fn = (qid & 2) * 2 + (short(lane) % 2) * 2;
  for (int i = 0; i < FM; ++i) {
    const int r = rb + 8 * i + fm;
    if (r >= cnt) continue;
    for (int j = 0; j < FN; ++j) {
      const int c = ob + 8 * j + fn;
      auto g = acc[i][j].thread_elements();
      float v0 = g[0], v1 = g[1];
      if (GU) {
        auto u = accu[i][j].thread_elements();
        v0 = (v0 * (1.0f / (1.0f + metal::exp(-v0)))) * u[0];
        v1 = (v1 * (1.0f / (1.0f + metal::exp(-v1)))) * u[1];
      }
      Y[yo[r] + n0 + c] = v0;
      Y[yo[r] + n0 + c + 1] = v1;
    }
  }
}

template <int BITS, int BM>
[[kernel]] void g53_qmm2(const device uint8_t* W [[buffer(0)]], const device half* S [[buffer(1)]],
                         const device half* B [[buffer(2)]], const device float* X [[buffer(3)]],
                         device float* Y [[buffer(4)]], constant QArgs& a [[buffer(5)]],
                         const device uint* IDS [[buffer(6)]], const device uint* XIDS [[buffer(7)]],
                         const device uint* YIDS [[buffer(8)]], const device uint* SEG [[buffer(9)]],
                         uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xt[BM * QLD2];
  threadgroup float wt[64 * QLD2];
  threadgroup long xo[BM], yo[BM];
  qmm2_body<BITS, false, BM, 64>(W, S, B, W, S, B, X, Y, a, IDS, XIDS, YIDS, SEG, tg, lid, sg, lane, xt, wt, wt, xo, yo);
}

template <int BITS, int BM>
[[kernel]] void g53_qmm2_gateup(const device uint8_t* GW [[buffer(0)]], const device half* GS [[buffer(1)]],
                                const device half* GB [[buffer(2)]], const device uint8_t* UW [[buffer(3)]],
                                const device half* US [[buffer(4)]], const device half* UB [[buffer(5)]],
                                const device float* X [[buffer(6)]], device float* Y [[buffer(7)]],
                                constant QArgs& a [[buffer(8)]], const device uint* IDS [[buffer(9)]],
                                const device uint* XIDS [[buffer(10)]], const device uint* YIDS [[buffer(11)]],
                                const device uint* SEG [[buffer(12)]], uint3 tg [[threadgroup_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xt[BM * QLD2];
  threadgroup float wt[64 * QLD2];
  threadgroup float ut[64 * QLD2];
  threadgroup long xo[BM], yo[BM];
  qmm2_body<BITS, true, BM, 64>(GW, GS, GB, UW, US, UB, X, Y, a, IDS, XIDS, YIDS, SEG, tg, lid, sg, lane, xt, wt, ut, xo, yo);
}

#define QMM2_ARGS (const device uint8_t*, const device half*, const device half*, const device float*, device float*, constant QArgs&, const device uint*, const device uint*, const device uint*, const device uint*, uint3, uint, uint, uint)
#define QMM2G_ARGS (const device uint8_t*, const device half*, const device half*, const device uint8_t*, const device half*, const device half*, const device float*, device float*, constant QArgs&, const device uint*, const device uint*, const device uint*, const device uint*, uint3, uint, uint, uint)
template [[host_name("g53_qmm2_b6_m32")]] [[kernel]] void g53_qmm2<6, 32> QMM2_ARGS;
template [[host_name("g53_qmm2_b8_m32")]] [[kernel]] void g53_qmm2<8, 32> QMM2_ARGS;
template [[host_name("g53_qmm2_gateup_b6_m32")]] [[kernel]] void g53_qmm2_gateup<6, 32> QMM2G_ARGS;
template [[host_name("g53_qmm2_gateup_b8_m32")]] [[kernel]] void g53_qmm2_gateup<8, 32> QMM2G_ARGS;

// ---------------------------------------------------------------- MTP: every row's head pick, the draft head's input -
// g53_argmax per row (threadgroup x = row): row r's n logits at L + r*n, its (value, base + index) at Out + 2r.
[[kernel]] void g53_argmax_rows(const device float* L [[buffer(0)]], device float* Out [[buffer(1)]],
                                constant int2& a [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]]) {
  threadgroup float bv[1024];
  threadgroup int bi[1024];
  L += long(row) * a.x;
  Out += 2 * row;
  float v = -INFINITY;
  int idx = 0x7fffffff;
  for (int i = int(lid); i < a.x; i += 1024) {
    const float x = L[i];
    if (x > v) {
      v = x;
      idx = i;
    }
  }
  bv[lid] = v;
  bi[lid] = idx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float best = -INFINITY;
    int at = 0x7fffffff;
    for (int t = 0; t < 1024; ++t)
      if (bv[t] > best || (bv[t] == best && bi[t] < at)) {
        best = bv[t];
        at = bi[t];
      }
    Out[0] = best;
    ((device uint*)Out)[1] = uint(at + a.y);
  }
}

// g53_pick per row (thread = row): slot s's pair for row r at P + s*stride + 2r; the token into Tok[r].
[[kernel]] void g53_pick_rows(const device float* P [[buffer(0)]], device uint* Tok [[buffer(1)]],
                              constant int3& a [[buffer(2)]], uint row [[thread_position_in_grid]]) {
  if (int(row) >= a.z) return;
  float best = -INFINITY;
  uint at = 0xffffffffu;
  for (int s = 0; s < a.x; ++s) {
    const float v = P[s * a.y + 2 * row];
    const uint i = ((const device uint*)P)[s * a.y + 2 * row + 1];
    if (v > best || (v == best && i < at)) {
      best = v;
      at = i;
    }
  }
  Tok[row] = at;
}

// The draft head's input row: C[r] = [enorm * rms(E[r]), hnorm * rms(Hn[r])] (D each). Threadgroup [1024] a row.
[[kernel]] void g53_mtp_cat(const device float* E [[buffer(0)]], const device float* Hn [[buffer(1)]],
                            const device bfloat* We [[buffer(2)]], const device bfloat* Wh [[buffer(3)]],
                            device float* C [[buffer(4)]], constant NArgs& a [[buffer(5)]],
                            uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[32];
  E += long(row) * a.D;
  Hn += long(row) * a.D;
  C += long(row) * 2 * a.D;
  float acc = 0.0f;
  for (int i = int(lid); i < a.D; i += 1024) acc += E[i] * E[i];
  const float ie = metal::precise::rsqrt(block_sum(acc, part, lid, lane, sg) / float(a.D) + a.eps);
  acc = 0.0f;
  for (int i = int(lid); i < a.D; i += 1024) acc += Hn[i] * Hn[i];
  const float ih = metal::precise::rsqrt(block_sum(acc, part, lid, lane, sg) / float(a.D) + a.eps);
  for (int i = int(lid); i < a.D; i += 1024) {
    C[i] = float(We[i]) * (E[i] * ie);
    C[a.D + i] = float(Wh[i]) * (Hn[i] * ih);
  }
}
