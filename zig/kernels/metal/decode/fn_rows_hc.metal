// Flash Next hyper-connections without tensor units (0.6.6 rows.py _hc_rows): each row its own threadgroups.
#include <metal_stdlib>
// Defines: FZ_S, FZ_D, FZ_ND, FZ_LOW, FZ_KS, FZ_DB/DG and FZ_UB/UG (widths), FZ_Q4 (both 4/32), FZ_NIB (same bits).
using namespace metal;

// sum_i w_i x_i over one 32-value group: w = scale * q + bias
inline float qgroup_dot(const device uint32_t* w, float scale, float bias, const thread float* x) {
  float dq = 0.0f, dx = 0.0f;
  for (int word = 0; word < 4; word++) {
    const uint32_t bits = w[word];
    for (int n = 0; n < 8; n++) {
      const float xv = x[word * 8 + n];
      dq = fma(float((bits >> (4 * n)) & 0xFu), xv, dq);
      dx += xv;
    }
  }
  return fma(scale, dq, bias * dx);
}
// float(v) for v < 2^23 by the exponent trick: the same value as a convert
inline float nib(uint v) { return as_type<float>(0x4B000000u | v) - 8388608.0f; }
inline float qgroup_dotx(const device uint32_t* w, float scale, float bias, const thread float* x) {
  float dq = 0.0f, dx = 0.0f;
  for (int word = 0; word < 4; word++) {
    const uint32_t bits = w[word];
    for (int n = 0; n < 8; n++) {
      const float xv = x[word * 8 + n];
      dq = fma(nib((bits >> (4 * n)) & 0xFu), xv, dq);
      dx += xv;
    }
  }
  return fma(scale, dq, bias * dx);
}
// two 16-bit words' nibbles q (q < 1024 each half) as half with no convert
inline half2 nib2(uint q) { return as_type<half2>(0x64006400u | q) - half2(1024.0h); }
// qgroup_dot with half nibbles: the same fmas in the same order
inline float qgroup_doth(const device uint32_t* w, float scale, float bias, const thread float* x) {
  float dq = 0.0f, dx = 0.0f;
  for (int word = 0; word < 4; word++) {
    const uint32_t bits = w[word];
    const half2 q04 = nib2(bits & 0x000F000Fu), q15 = nib2((bits >> 4) & 0x000F000Fu),
                q26 = nib2((bits >> 8) & 0x000F000Fu), q37 = nib2((bits >> 12) & 0x000F000Fu);
    const float q[8] = {float(q04.x), float(q15.x), float(q26.x), float(q37.x),
                        float(q04.y), float(q15.y), float(q26.y), float(q37.y)};
    for (int n = 0; n < 8; n++) {
      const float xv = x[word * 8 + n];
      dq = fma(q[n], xv, dq);
      dx += xv;
    }
  }
  return fma(scale, dq, bias * dx);
}
#if FZ_NIB == 1
#define FZ_QGROUP qgroup_doth
#elif FZ_NIB == 2
#define FZ_QGROUP qgroup_dotx
#else
#define FZ_QGROUP qgroup_dot
#endif
// Elementwise ops as the checkpoint's training framework does them on bf16 tensors: fp32 math, one rounding.
inline float bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
inline float bsilu(float x) { return float(bfloat(x / (1.0f + metal::exp(-x)))); }
inline float stream_rinv(const device float* ssp, int r, int s, int nt, int streams, int dims, float eps) {
  float total = 0.0f;
  for (int j = 0; j < nt; j++) total += ssp[(r * nt + j) * streams + s];
  return metal::rsqrt(total / float(dims) + eps);
}
// values 8m .. 8m + 7 of the 32-value chunk at ``chunk`` (a chunk starts a word, and 8 values span at most 2 words)
template <int BITS>
inline void codes8(const device uint* chunk, int m, thread float* q) {
  const int bit = 8 * m * BITS, word = bit >> 5, shift = bit & 31;
  const uint hi = shift + 8 * BITS > 32 ? chunk[word + 1] : 0u;
  const ulong v = ((ulong(hi) << 32) | ulong(chunk[word])) >> shift;
  for (int i = 0; i < 8; i++) q[i] = float(uint(v >> (BITS * i)) & ((1u << BITS) - 1u));
}
// qgroup_dot for any width: one 32-value chunk, codes in order, then scale and bias
template <int BITS>
inline float qchunk_dot(const device uint* chunk, float scale, float bias, const thread float* x) {
  float dq = 0.0f, dx = 0.0f;
  for (int m = 0; m < 4; m++) {
    float q[8];
    codes8<BITS>(chunk, m, q);
    for (int n = 0; n < 8; n++) {
      dq = fma(q[n], x[8 * m + n], dq);
      dx += x[8 * m + n];
    }
  }
  return fma(scale, dq, bias * dx);
}

// split-K matvec of normed streams, simdgroup o of group (i, k, r) on groups 32 k ..: a lane normalises its own 32
[[kernel]] void qa_hc_down_split(const device bfloat* HN [[buffer(0)]], const device float* SSP [[buffer(1)]],
    const device float* NW [[buffer(2)]], const device uint32_t* QW [[buffer(3)]],
    const device bfloat* QS [[buffer(4)]],
    const device bfloat* QB [[buffer(5)]], const constant float* eps [[buffer(6)]],
    const constant int* rows [[buffer(7)]],
    device float* PART [[buffer(8)]], uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr int S = FZ_S, D = FZ_D, ND = FZ_ND, BITS = FZ_DB, GS = FZ_DG;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int R = rows[0];
  constexpr int W = S * D;
  constexpr int GROUPS = W / 32;
  const int o = int(threadgroup_position_in_grid.x) * 8 + int(g);
  const int k = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  if (r >= R || o >= ND) return; // a gated window's unverified rows, the outputs past the matrix
  const int grp = k * 32 + int(lane), e0 = grp * 32;
  const float rv = stream_rinv(SSP, r, e0 / D, D / 256, S, D, eps[0]);
  float x[32];
  for (int n = 0; n < 32; n++) x[n] = float(bfloat((float(HN[r * W + e0 + n]) * rv) * NW[e0 + n]));
#if FZ_Q4
  float acc = FZ_QGROUP(QW + (size_t(o) * GROUPS + grp) * 4, float(QS[o * GROUPS + grp]), float(QB[o * GROUPS + grp]), x);
#else
  float acc = qchunk_dot<BITS>(QW + size_t(o) * (W * BITS / 32) + grp * BITS, float(QS[o * (W / GS) + grp * 32 / GS]),
                               float(QB[o * (W / GS) + grp * 32 / GS]), x);
#endif
  acc = simd_sum(acc);
  if (lane == 0) PART[(k * R + r) * ND + o] = acc;
}

// one row's up projection for 8 dims of every stream: thread 10 i + q takes group q of up row i
[[kernel]] void qa_hc_up2(const device bfloat* HN [[buffer(0)]], const device float* SSP [[buffer(1)]],
    const device float* NW [[buffer(2)]], const device float* PART [[buffer(3)]],
    const device uint32_t* QW [[buffer(4)]],
    const device bfloat* QS [[buffer(5)]], const device bfloat* QB [[buffer(6)]],
    const constant float* eps [[buffer(7)]],
    const constant int* rows [[buffer(8)]], device bfloat* MIXED [[buffer(9)]], device bfloat* INJOUT [[buffer(10)]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  constexpr int S = FZ_S, D = FZ_D, ND = FZ_ND, LOW = FZ_LOW, KS = FZ_KS, BITS = FZ_UB, GS = FZ_UG;
  const uint t = thread_position_in_threadgroup.x;
  const int R = rows[0];
  const int r = int(threadgroup_position_in_grid.y);
  if (r >= R) return; // a gated window's unverified rows
  constexpr int W = S * D;
  constexpr int GPR = LOW / 32;
  constexpr int ROWS = S * 8;
  const int d0 = int(threadgroup_position_in_grid.x) * 8;
  threadgroup float act[GPR * 33];
  threadgroup float part[ROWS][GPR];
  threadgroup float prod[ROWS];
  const int i = int(t) / GPR, q = int(t) % GPR;
  const int s = i / 8, d = d0 + i % 8;
  const int row = s * D + d;
  for (int c = int(t); c < ND; c += ROWS * GPR) {
    float v = 0.0f;
    for (int k = 0; k < KS; k++) v += PART[(k * R + r) * ND + c];
    const float v4 = float(bfloat(float(bfloat(v)) / float(S)));
    if (c < LOW) act[(c / 32) * 33 + c % 32] = bsilu(v4);
    else if (threadgroup_position_in_grid.x == 0) INJOUT[r * S + (c - LOW)] = bfloat(2.0f * bsig(v4));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    float x[32];
    for (int n = 0; n < 32; n++) x[n] = act[q * 33 + n];
#if FZ_Q4
    part[i][q] = FZ_QGROUP(QW + (size_t(row) * GPR + q) * 4, float(QS[row * GPR + q]), float(QB[row * GPR + q]), x);
#else
    part[i][q] = qchunk_dot<BITS>(QW + size_t(row) * (LOW * BITS / 32) + q * BITS,
                                  float(QS[row * (LOW / GS) + q * 32 / GS]),
                                  float(QB[row * (LOW / GS) + q * 32 / GS]), x);
#endif
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (q == 0) {
    float u = 0.0f;
    for (int qq = 0; qq < GPR; qq++) u += part[i][qq];
    const float rv = stream_rinv(SSP, r, s, D / 256, S, D, eps[0]);
    const float normed = float(bfloat((float(HN[r * W + row]) * rv) * NW[row]));
    prod[i] = float(bfloat(bsig(float(bfloat(u))) * normed));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 8) {
    float total = 0.0f;
    for (int ss = 0; ss < S; ss++) total += prod[ss * 8 + int(t)];
    MIXED[r * D + d0 + int(t)] = bfloat(total / float(S));
  }
}

// lane l's place in an 8x8 tile: output fm, rows fn and fn + 1
inline int tile_fm(int l) { return ((l / 4) & 4) + ((l / 2) % 4); }
inline int tile_fn(int l) { return ((l / 4) & 2) * 2 + (l % 2) * 2; }

// qa_hc_down_split's bits for 8 rows a tile on the matrix units (4-bit group 32): threadgroup (i, k, t) takes outputs
[[kernel]] void q4_hc_down_tiles(const device bfloat* HN [[buffer(0)]], const device float* SSP [[buffer(1)]],
    const device float* NW [[buffer(2)]], const device uint32_t* QW [[buffer(3)]],
    const device bfloat* QS [[buffer(4)]],
    const device bfloat* QB [[buffer(5)]], const constant float* eps [[buffer(6)]],
    const constant int* rows [[buffer(7)]],
    device float* PART [[buffer(8)]], uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr int S = FZ_S, D = FZ_D, ND = FZ_ND;         // 8 i .., split k, rows 8 t ..; simdgroup c: groups 4 c ..
  const uint lane = thread_index_in_simdgroup;
  const int c = int(simdgroup_index_in_threadgroup);
  const uint t = thread_position_in_threadgroup.x;
  const int fm = tile_fm(int(lane)), fn = tile_fn(int(lane));
  const int R = rows[0];
  constexpr int W = S * D, GROUPS = W / 32;
  const int nb = int(threadgroup_position_in_grid.x) * 8;
  const int k = int(threadgroup_position_in_grid.y);
  const int rb = int(threadgroup_position_in_grid.z) * 8;
  threadgroup float rinv[8 * S];
  threadgroup float sums[8][32];
  threadgroup float red[64][33];
  if (t < 8 * S) rinv[t] = stream_rinv(SSP, min(rb + int(t) / S, R - 1), int(t) % S, D / 256, S, D, eps[0]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    const int r = int(lane) % 8, j = 4 * c + int(lane) / 8;
    const int row = min(rb + r, R - 1);
    const int e0 = 32 * (32 * k + j);
    float dx = 0.0f;
    for (int v = 0; v < 32; v++) {
      const int e = e0 + v;
      dx += float(bfloat((float(HN[size_t(row) * W + e]) * rinv[r * S + e / D]) * NW[e]));
    }
    sums[r][j] = dx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int o = min(nb + fm, ND - 1);
  const int ra = min(rb + fn, R - 1), rc = min(rb + fn + 1, R - 1);
  for (int jj = 0; jj < 4; jj++) {
    const int j = 4 * c + jj, g = 32 * k + j;
    const device uint* wq = QW + (size_t(o) * GROUPS + g) * 4;
    simdgroup_matrix<float, 8, 8> P = simdgroup_matrix<float, 8, 8>(0.0f);
    for (int st = 0; st < 4; st++) {
      const uint word = wq[st];
      const int e = 32 * g + 8 * st + fm;
      simdgroup_matrix<float, 8, 8> am, bm;
      am.thread_elements()[0] = nib((word >> (4 * fn)) & 0xFu);
      am.thread_elements()[1] = nib((word >> (4 * fn + 4)) & 0xFu);
      bm.thread_elements()[0] = float(bfloat((float(HN[size_t(ra) * W + e]) * rinv[(ra - rb) * S + e / D]) * NW[e]));
      bm.thread_elements()[1] = float(bfloat((float(HN[size_t(rc) * W + e]) * rinv[(rc - rb) * S + e / D]) * NW[e]));
      simdgroup_multiply_accumulate(P, am, bm, P);
    }
    const float sc = float(QS[o * GROUPS + g]), bi = float(QB[o * GROUPS + g]);
    red[fm * 8 + fn][j] = fma(sc, P.thread_elements()[0], bi * sums[ra - rb][j]);
    red[fm * 8 + fn + 1][j] = fma(sc, P.thread_elements()[1], bi * sums[rc - rb][j]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int p = 8 * c; p < 8 * c + 8; p++) {
    const float v = simd_sum(red[p][lane]);
    const int out = nb + p / 8, row = rb + p % 8;
    if (lane == 0 && out < ND && row < R) PART[(size_t(k) * R + row) * ND + out] = v;
  }
}

// qa_hc_up2's bits for 8 rows a tile on the matrix units (4-bit group 32): threadgroup (i, t) takes dims DT i ..
[[kernel]] void q4_hc_up_tiles(const device bfloat* HN [[buffer(0)]], const device float* SSP [[buffer(1)]],
    const device float* NW [[buffer(2)]], const device float* PART [[buffer(3)]],
    const device uint32_t* QW [[buffer(4)]],
    const device bfloat* QS [[buffer(5)]], const device bfloat* QB [[buffer(6)]],
    const constant float* eps [[buffer(7)]],
    const constant int* rows [[buffer(8)]], device bfloat* MIXED [[buffer(9)]], device bfloat* INJOUT [[buffer(10)]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr int S = FZ_S, D = FZ_D, ND = FZ_ND, LOW = FZ_LOW, KS = FZ_KS, DT = 32;  // of every stream, rows 8 t ..
  const uint lane = thread_index_in_simdgroup;
  const int s = int(simdgroup_index_in_threadgroup);
  const uint t = thread_position_in_threadgroup.x;
  const int fm = tile_fm(int(lane)), fn = tile_fn(int(lane));
  const int R = rows[0];
  constexpr int W = S * D, GPR = LOW / 32, NT = 32 * S, LP = LOW + 4;
  const int d0 = int(threadgroup_position_in_grid.x) * DT;
  const int rb = int(threadgroup_position_in_grid.y) * 8;
  const int nr = min(8, R - rb);
  threadgroup float act[8 * LP];
  threadgroup float sums[8][GPR];
  threadgroup float rinv[8 * S];
  threadgroup float prod[S][DT][8];
  if (t < 8 * S) rinv[t] = stream_rinv(SSP, min(rb + int(t) / S, R - 1), int(t) % S, D / 256, S, D, eps[0]);
  for (int i = int(t); i < 8 * ND; i += NT) {
    const int r = i / ND, cc = i % ND;
    const int row = min(rb + r, R - 1);
    float v = 0.0f;
    for (int kk = 0; kk < KS; kk++) v += PART[(size_t(kk) * R + row) * ND + cc];
    const float v4 = float(bfloat(float(bfloat(v)) / float(S)));
    if (cc < LOW) act[r * LP + cc] = bsilu(v4);
    else if (threadgroup_position_in_grid.x == 0 && r < nr) INJOUT[(rb + r) * S + (cc - LOW)] = bfloat(2.0f * bsig(v4));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(t); i < 8 * GPR; i += NT) {
    const int r = i / GPR, q = i % GPR;
    float dx = 0.0f;
    for (int v = 0; v < 32; v++) dx += act[r * LP + 32 * q + v];
    sums[r][q] = dx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int tt = 0; tt < DT / 8; tt++) {
    const int orow = s * D + d0 + 8 * tt + fm;
    const device uint* wq = QW + size_t(orow) * GPR * 4;
    float u0 = 0.0f, u1 = 0.0f;
    for (int q = 0; q < GPR; q++) {
      simdgroup_matrix<float, 8, 8> P = simdgroup_matrix<float, 8, 8>(0.0f);
      for (int st = 0; st < 4; st++) {
        const uint word = wq[4 * q + st];
        const int e = 32 * q + 8 * st + fm;
        simdgroup_matrix<float, 8, 8> am, bm;
        am.thread_elements()[0] = nib((word >> (4 * fn)) & 0xFu);
        am.thread_elements()[1] = nib((word >> (4 * fn + 4)) & 0xFu);
        bm.thread_elements()[0] = act[fn * LP + e];
        bm.thread_elements()[1] = act[(fn + 1) * LP + e];
        simdgroup_multiply_accumulate(P, am, bm, P);
      }
      const float sc = float(QS[orow * GPR + q]), bi = float(QB[orow * GPR + q]);
      u0 += fma(sc, P.thread_elements()[0], bi * sums[fn][q]);
      u1 += fma(sc, P.thread_elements()[1], bi * sums[fn + 1][q]);
    }
    for (int e = 0; e < 2; e++) {
      const int r = fn + e, row = min(rb + r, R - 1);
      const float normed = float(bfloat((float(HN[size_t(row) * W + orow]) * rinv[r * S + s]) * NW[orow]));
      prod[s][8 * tt + fm][r] = float(bfloat(bsig(float(bfloat(e ? u1 : u0))) * normed));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(t); i < DT * 8; i += NT) {
    const int d = i / 8, r = i % 8;
    float total = 0.0f;
    for (int ss = 0; ss < S; ss++) total += prod[ss][d][r];
    if (r < nr) MIXED[size_t(rb + r) * D + d0 + d] = bfloat(total / float(S));
  }
}

// qa_hc_down_split with qa_hc_up2's split sums and gates: threadgroup (b, r) on outputs 4 b .. 4 b + 3, split k
[[kernel]] void qa_hc_down_act(const device bfloat* HN [[buffer(0)]], const device float* SSP [[buffer(1)]],
    const device float* NW [[buffer(2)]], const device uint32_t* QW [[buffer(3)]],
    const device bfloat* QS [[buffer(4)]],
    const device bfloat* QB [[buffer(5)]], const constant float* eps [[buffer(6)]],
    const constant int* rows [[buffer(7)]],
    device float* ACT [[buffer(8)]], device bfloat* INJOUT [[buffer(9)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr int S = FZ_S, D = FZ_D, ND = FZ_ND, LOW = FZ_LOW, KS = FZ_KS, BITS = FZ_DB, GS = FZ_DG, OPS = 4;
  const uint lane = thread_index_in_simdgroup;
  const int k = int(simdgroup_index_in_threadgroup);
  const int R = rows[0];
  constexpr int W = S * D;
  constexpr int GROUPS = W / 32;
  const int b = int(threadgroup_position_in_grid.x), r = int(threadgroup_position_in_grid.y);
  if (r >= R) return; // a gated window's unverified rows, the whole threadgroup
  threadgroup float P[KS][OPS];
  const int grp = k * 32 + int(lane), e0 = grp * 32; // split k's groups 32 k .., a lane normalises its own 32 once
  const float rv = stream_rinv(SSP, r, e0 / D, D / 256, S, D, eps[0]);
  float x[32];
  for (int n = 0; n < 32; n++) x[n] = float(bfloat((float(HN[r * W + e0 + n]) * rv) * NW[e0 + n]));
  for (int j = 0; j < OPS; j++) {
    const int o = b * OPS + j;
    if (o >= ND) break;
#if FZ_Q4
    float acc = FZ_QGROUP(QW + (size_t(o) * GROUPS + grp) * 4, float(QS[o * GROUPS + grp]), float(QB[o * GROUPS + grp]), x);
#else
    float acc = qchunk_dot<BITS>(QW + size_t(o) * (W * BITS / 32) + grp * BITS, float(QS[o * (W / GS) + grp * 32 / GS]),
                                 float(QB[o * (W / GS) + grp * 32 / GS]), x);
#endif
    acc = simd_sum(acc);
    if (lane == 0) P[k][j] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int t = int(thread_index_in_threadgroup), o = b * OPS + t;
  if (t >= OPS || o >= ND) return;
  float v = 0.0f; // the splits in order, then qa_hc_up2's activation or inject gate
  for (int kk = 0; kk < KS; kk++) v += P[kk][t];
  const float v4 = float(bfloat(float(bfloat(v)) / float(S)));
  if (o < LOW) ACT[r * LOW + o] = bsilu(v4);
  else INJOUT[r * S + (o - LOW)] = bfloat(2.0f * bsig(v4));
}

// qa_hc_up2 from qa_hc_down_act's ACT [rows, LOW], no staging: lane 4 j + s of simdgroup g on up row s D + 8 g + j
[[kernel]] void qa_hc_up_act(const device bfloat* HN [[buffer(0)]], const device float* SSP [[buffer(1)]],
    const device float* NW [[buffer(2)]], const device float* ACT [[buffer(3)]],
    const device uint32_t* QW [[buffer(4)]],
    const device bfloat* QS [[buffer(5)]], const device bfloat* QB [[buffer(6)]],
    const constant float* eps [[buffer(7)]],
    const constant int* rows [[buffer(8)]], device bfloat* MIXED [[buffer(9)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr int S = FZ_S, D = FZ_D, LOW = FZ_LOW, BITS = FZ_UB, GS = FZ_UG;
  constexpr int W = S * D;
  constexpr int GPR = LOW / 32;
  const int R = rows[0];
  const int r = int(threadgroup_position_in_grid.y);
  if (r >= R) return; // a gated window's unverified rows
  const uint lane = thread_index_in_simdgroup;
  const int g = int(threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup);
  const int s = int(lane) % 4, d = g * 8 + int(lane) / 4;
  const int row = s * D + d;
  const device float* act = ACT + r * LOW;
  float u = 0.0f; // the row's group dots summed in order
  for (int q = 0; q < GPR; q++) {
    float x[32];
    for (int n = 0; n < 32; n++) x[n] = act[q * 32 + n];
#if FZ_Q4
    u += FZ_QGROUP(QW + (size_t(row) * GPR + q) * 4, float(QS[row * GPR + q]), float(QB[row * GPR + q]), x);
#else
    u += qchunk_dot<BITS>(QW + size_t(row) * (LOW * BITS / 32) + q * BITS, float(QS[row * (LOW / GS) + q * 32 / GS]),
                          float(QB[row * (LOW / GS) + q * 32 / GS]), x);
#endif
  }
  const float rv = stream_rinv(SSP, r, s, D / 256, S, D, eps[0]);
  const float normed = float(bfloat((float(HN[r * W + row]) * rv) * NW[row]));
  const float prod = float(bfloat(bsig(float(bfloat(u))) * normed));
  const float p1 = simd_shuffle_down(prod, 1), p2 = simd_shuffle_down(prod, 2), p3 = simd_shuffle_down(prod, 3);
  if (s != 0) return;
  float total = 0.0f; // dim d's four streams in order
  total += prod;
  total += p1;
  total += p2;
  total += p3;
  MIXED[r * D + d] = bfloat(total / float(S));
}
