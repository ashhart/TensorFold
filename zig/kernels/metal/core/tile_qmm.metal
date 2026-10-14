// Prompt matmuls before the M5: 32x32 output tiles, exact codes and inputs in simdgroup MMAs, fp32 group and row sums.
#include <metal_stdlib>
using namespace metal;
#ifndef TQ_BM
#define TQ_BM 32
#define TQ_BN 32
#endif

// K inputs, N outputs, R rows; one is 1.0 read at run time so the compiler can't fold or reorder the sums.
struct TfTqArgs { int K, N, R; float one; };

constant constexpr int TQ = TQ_BM;  // rows a tile (the work list's row tiles)
constant constexpr int TQN = TQ_BN;  // columns a tile
constant constexpr int TP = 40;  // bf16 a staged row: a stage's 32 values, pad (bank spread)

// bf16 bits of a small exact integer (no rounding below 256), at compile time.
constexpr uint tf_tq_bf(uint q) {
  uint e = 0;
  while ((2u << e) <= q) ++e;
  return q == 0 ? 0u : (((127u + e) << 7) | ((q - (1u << e)) << (7 - e)));
}

// A byte's two 4-bit codes (low first) as a bf16 pair.
struct TfTqPairs { uint v[256]; };
constexpr TfTqPairs tf_tq_pairs_make() {
  TfTqPairs t{};
  for (uint b = 0; b < 256; ++b) t.v[b] = tf_tq_bf(b & 15u) | (tf_tq_bf(b >> 4) << 16);
  return t;
}
constant TfTqPairs tf_tq_pairs = tf_tq_pairs_make();

// 8 codes from BITS / 2 halfwords of a little-endian bit stream as 4 words of bf16 pairs.
template <int BITS>
inline uint4 tf_tq_codes(const thread ushort* h) {
  const ulong v = ulong(h[0]) | (ulong(h[1]) << 16) | (BITS > 4 ? ulong(h[BITS > 4 ? 2 : 0]) << 32 : 0ul) |
                  (BITS > 6 ? ulong(h[BITS > 6 ? 3 : 0]) << 48 : 0ul);
  uint out[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint pair = uint(v >> (2 * i * BITS)) & ((1u << (2 * BITS)) - 1u);
    if (BITS == 4) out[i] = tf_tq_pairs.v[pair];
    else {
      const uint lo = pair & ((1u << BITS) - 1u), hi = pair >> BITS;
      out[i] = (as_type<uint>(float(lo)) >> 16) | (as_type<uint>(float(hi)) & 0xFFFF0000u);
    }
  }
  return uint4(out[0], out[1], out[2], out[3]);
}

// mlx-lm's compiled swiglu on bf16 inputs: silu(g) * u, rounded to bf16 at each step.
inline bfloat tf_tq_swiglu(bfloat gate, bfloat up) {
  const float g = gate;
  const bfloat s = static_cast<bfloat>(1.0f / (1.0f + metal::precise::exp(-g)));
  const bfloat silu = static_cast<bfloat>(g * float(s));
  return static_cast<bfloat>(float(silu) * float(up));
}

// A thread's stage share: 8 inputs of a row, 8 codes of a column; a group's first stage adds its scale, bias and sum.
template <int BITS>
struct TfTqAhead {
  uint4 x;
  ushort w[BITS / 2];
  float2 sb;
  float xs;
};

// GLU: the tile's halves are columns nb.. of gate (W, SC, BI) and of up (W2, SC2, BI2); else its BN columns of W.
template <int BITS, int BM, int BN, bool GLU>
inline void tf_tq_fetch(thread TfTqAhead<BITS>& a, const device bfloat* X, const device float* XS,
                        const device uint8_t* W, const device bfloat* SC, const device bfloat* BI,
                        const device uint8_t* W2, const device bfloat* SC2, const device bfloat* BI2, uint xr, uint sr,
                        int K, int nb, int s, int tid) {
  const int G = K / 64, g = s / 2, k0 = g * 64 + s % 2 * 32;
  // The thread's input row is X's xr and its sum row XS's sr (resolved once a tile).
  a.x = *((const device uint4*)(X + size_t(xr) * K + k0) + tid % 4);
  const bool up = GLU && tid >= 64;
  const int col = nb + (GLU ? tid / 4 % (BN / 2) : tid / 4);
  const device ushort* src =
      (const device ushort*)((up ? W2 : W) + size_t(col) * (K * BITS / 8) + (k0 + tid % 4 * 8) * BITS / 8);
  for (int i = 0; i < BITS / 2; ++i) a.w[i] = src[i];
  if (s % 2 == 0) {
    if (tid < BN) {
      const bool up2 = GLU && tid >= BN / 2;
      const size_t n = size_t(nb + (GLU ? tid % (BN / 2) : tid)) * G + g;
      a.sb = float2(float((up2 ? SC2 : SC)[n]), float((up2 ? BI2 : BI)[n]));
    }
    if (tid < BM) a.xs = XS[size_t(sr) * G + g];
  }
}

template <int BITS, int BM, int BN>
inline void tf_tq_stage(const thread TfTqAhead<BITS>& a, threadgroup bfloat* xs, threadgroup bfloat* ws,
                        threadgroup float2* sb, threadgroup float* xsum, int s, int tid) {
  *(threadgroup uint4*)(xs + tid / 4 * TP + tid % 4 * 8) = a.x;
  *(threadgroup uint4*)(ws + tid / 4 * TP + tid % 4 * 8) = tf_tq_codes<BITS>(a.w);
  if (s % 2 == 0) {  // a group's scales, biases and sums by its parity (read after its second stage)
    if (tid < BN) sb[s / 2 % 2 * BN + tid] = a.sb;
    if (tid < BM) xsum[s / 2 % 2 * BM + tid] = a.xs;
  }
}

// A simdgroup's column block j's staged row: its 16 columns in order, or (GLU) 8 gate columns then the same up columns.
template <bool GLU>
inline int tf_tq_col(int sc, int j) {
  return GLU ? j * 16 + sc / 2 : sc + 8 * j;
}

// A stage's 4 k-steps of a simdgroup's TI x TJ blocks on staged inputs xb and codes wb.
template <int TI, int TJ, bool GLU>
inline void tf_tq_mmas(thread simdgroup_matrix<float, 8, 8> (*P)[TJ], const threadgroup bfloat* xb,
                       const threadgroup bfloat* wb, int sr, int sc, int fm, int fn) {
  #pragma unroll
  for (int kk = 0; kk < 32; kk += 8) {
    simdgroup_matrix<bfloat, 8, 8> a[TI], b[TJ];
    // Each lane's two elements (row fm, columns fn and fn + 1) read directly: A along k, B down its two columns.
    for (int i = 0; i < TI; ++i) {
      const threadgroup bfloat* ap = xb + (sr + 8 * i + fm) * TP + kk + fn;
      a[i].thread_elements()[0] = ap[0];
      a[i].thread_elements()[1] = ap[1];
    }
    for (int j = 0; j < TJ; ++j) {
      const threadgroup bfloat* bp = wb + (tf_tq_col<GLU>(sc, j) + fn) * TP + kk + fm;
      b[j].thread_elements()[0] = bp[0];
      b[j].thread_elements()[1] = bp[TP];
    }
    for (int i = 0; i < TI; ++i)
      for (int j = 0; j < TJ; ++j) simdgroup_multiply_accumulate(P[i][j], a[i], b[j], P[i][j]);
  }
}

// One 32x32 tile of Y = X W^T: four simdgroups of 16x16, each 64-input group summed by bf16 MMAs into fp32 in k order.
template <int BITS, int BM, int BN, bool PARTIAL, bool GLU, bool MAP>
inline void tf_tq_tile(const device bfloat* X, const device float* XS, const device uint8_t* W, const device bfloat* SC,
                       const device bfloat* BI, const device uint8_t* W2, const device bfloat* SC2,
                       const device bfloat* BI2, const device uint* map, int K, int N, int R, device bfloat* OUT,
                       int rb, int nb, uint tid, uint sg, uint lane, threadgroup bfloat* xs, threadgroup bfloat* ws,
                       threadgroup float2* sb, threadgroup float* xsum) {
  // Then acc = fma(bias, xsum, fma(scale, group, acc)) per output, XS [R, K / 64] from tf_tile_xsum.
  static_assert(BM == 32 && BN == 32, "128 threads stage 32 rows and 32 columns, 8 values each");
  constexpr int TI = BM / 16, TJ = BN / 16;  // 8x8 blocks a simdgroup (2x2 simdgroups)
  const int G = K / 64, S = 2 * G;
  const int sr = int(sg) / 2 * (BM / 2), sc = int(sg) % 2 * (BN / 2);
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const bool live = !PARTIAL || rb + sr < R;  // an expert's partial tile: a simdgroup past its rows skips the MMAs
  // MAP: tile row r's inputs and sums are X's and XS's row map[r] (rows by expert reading their tokens).
  const int xi = min(rb + int(tid) / 4, R - 1), si = min(rb + int(tid), R - 1);
  const uint xr = MAP ? map[xi] : uint(xi), srw = MAP ? map[si] : uint(si);
  // GLU: 16 gate and the same 16 up columns, OUT [R, N] = swiglu of their bf16 rows (the twin's tf_tile_qmv_glu).
  const threadgroup bfloat* xb1 = xs + BM * TP;
  const threadgroup bfloat* wb1 = ws + BN * TP;
  float acc[TI][TJ][2];
  for (int i = 0; i < TI; ++i)
    for (int j = 0; j < TJ; ++j) acc[i][j][0] = acc[i][j][1] = 0.0f;
  // Two 32-input stages a group in two buffers: loads two stages ahead, stores one; a stage's barrier waits on neither.
  TfTqAhead<BITS> even, odd;  // the next even and odd stages' shares
  tf_tq_fetch<BITS, BM, BN, GLU>(even, X, XS, W, SC, BI, W2, SC2, BI2, xr, srw, K, nb, 0, int(tid));
  tf_tq_fetch<BITS, BM, BN, GLU>(odd, X, XS, W, SC, BI, W2, SC2, BI2, xr, srw, K, nb, 1, int(tid));
  tf_tq_stage<BITS, BM, BN>(even, xs, ws, sb, xsum, 0, int(tid));
  for (int g = 0; g < G; ++g) {
    const int s = 2 * g;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tf_tq_stage<BITS, BM, BN>(odd, xs + BM * TP, ws + BN * TP, sb, xsum, s + 1, int(tid));
    if (s + 2 < S) tf_tq_fetch<BITS, BM, BN, GLU>(even, X, XS, W, SC, BI, W2, SC2, BI2, xr, srw, K, nb, s + 2,
                                                  int(tid));
    simdgroup_matrix<float, 8, 8> P[TI][TJ];
    for (int i = 0; i < TI; ++i)
      for (int j = 0; j < TJ; ++j) P[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    if (live) tf_tq_mmas<TI, TJ, GLU>(P, xs, ws, sr, sc, fm, fn);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s + 2 < S) tf_tq_stage<BITS, BM, BN>(even, xs, ws, sb, xsum, s + 2, int(tid));
    if (s + 3 < S) tf_tq_fetch<BITS, BM, BN, GLU>(odd, X, XS, W, SC, BI, W2, SC2, BI2, xr, srw, K, nb, s + 3, int(tid));
    if (live) tf_tq_mmas<TI, TJ, GLU>(P, xb1, wb1, sr, sc, fm, fn);
    const threadgroup float2* sbg = sb + g % 2 * BN;
    const threadgroup float* xsg = xsum + g % 2 * BM;
    float xsr[TI];
    for (int i = 0; i < TI; ++i) xsr[i] = xsg[sr + 8 * i + fm];
    for (int j = 0; j < TJ; ++j)
      for (int e = 0; e < 2; ++e) {
        const float2 v = sbg[tf_tq_col<GLU>(sc, j) + fn + e];
        for (int i = 0; i < TI; ++i) acc[i][j][e] = fma(v.y, xsr[i],
                                                        fma(v.x, P[i][j].thread_elements()[e], acc[i][j][e]));
      }
  }
  for (int i = 0; i < TI; ++i)
    for (int j = 0; j < (GLU ? 1 : TJ); ++j)
      for (int e = 0; e < 2; ++e) {
        const int row = rb + sr + 8 * i + fm, n = nb + tf_tq_col<GLU>(sc, j) + fn + e;
        if (row >= R) continue;
        if (GLU) OUT[size_t(row) * N + n] = tf_tq_swiglu(bfloat(acc[i][0][e]), bfloat(acc[i][1][e]));
        else OUT[size_t(row) * N + n] = bfloat(acc[i][j][e]);
      }
}

// Y [R, N] = X [R, K] W^T, a 32x32 tile a threadgroup of 128 (tg.x columns, tg.y rows); XS at buffer 6.
template <int BITS>
[[kernel]] void tf_tile_qmm(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                            const device bfloat* SC [[buffer(2)]], const device bfloat* BI [[buffer(3)]],
                            constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                            const device float* XS [[buffer(6)]], uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat xs[2 * TQ * TP], ws[2 * TQN * TP];
  threadgroup float2 sb[2 * TQN];
  threadgroup float xsum[2 * TQ];
  tf_tq_tile<BITS, TQ, TQN, false, false, false>(X, XS, W, SC, BI, W, SC, BI, nullptr, a.K, a.N, a.R, OUT,
                                                 int(tg.y) * TQ, int(tg.x) * TQN, tid, sg, lane, xs, ws, sb, xsum);
}

// The tiles over rows sorted by expert: item tg.y of WORK[1..WORK[0]] = (expert << 16) | row tile, rows from STARTS.
template <int BITS, bool GLU>
[[kernel]] void tf_tile_qmm_grouped(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                                    const device bfloat* SC [[buffer(2)]], const device bfloat* BI [[buffer(3)]],
                                    constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                                    const device uint* starts [[buffer(6)]], const device uint* work [[buffer(7)]],
                                    const device float* XS [[buffer(8)]], const device uint8_t* W2 [[buffer(9)]],
                                    const device bfloat* SC2 [[buffer(10)]], const device bfloat* BI2 [[buffer(11)]],
                                    const device uint* map [[buffer(12)]], uint tid [[thread_index_in_threadgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat xs[2 * TQ * TP], ws[2 * TQN * TP];
  threadgroup float2 sb[2 * TQN];
  threadgroup float xsum[2 * TQ];
  if (tg.y >= work[0]) return;
  // Sorted row j reads X's and XS's row MAP[j], and writes OUT's row j.
  const uint item = work[1 + tg.y], e = item >> 16;
  const uint first = starts[e], rows = starts[e + 1] - first;
  // GLU: W2/SC2/BI2 are the up experts (the same shape), OUT [rows, N] their swiglu with W's gate, 16 columns a tile.
  const size_t K = size_t(a.K), N = size_t(a.N), at = e * N * (K * BITS / 8), sat = e * N * (K / 64);
  tf_tq_tile<BITS, TQ, TQN, true, GLU, true>(X, XS, W + at, SC + sat, BI + sat, W2 + at, SC2 + sat, BI2 + sat,
                                             map + first, a.K, a.N, int(rows), OUT + first * N,
                                             int(item & 0xFFFFu) * TQ, int(tg.x) * (GLU ? TQN / 2 : TQN), tid, sg, lane,
                                             xs, ws, sb, xsum);
}

// The grouped tiles' work list from STARTS [E + 1]: count, then (expert << 16) | tile for each expert's 32-row tiles.
[[kernel]] void tf_tile_qmm_work(const device uint* starts [[buffer(0)]], constant uint& experts [[buffer(1)]],
                                 device uint* work [[buffer(2)]]) {
  uint n = 0;
  for (uint e = 0; e < experts; ++e) {
    const uint tiles = (starts[e + 1] - starts[e] + TQ - 1) / TQ;
    for (uint i = 0; i < tiles; ++i) work[1 + n++] = (e << 16) | i;
  }
  work[0] = n;
}

// Each row's group sums for the twin [rows, K / 64]: x0 then + x1 ... + x63 in order, as the tiles' xsum.
[[kernel]] void tf_tile_xsum(const device bfloat* X [[buffer(0)]], constant TfTqArgs& a [[buffer(1)]],
                             device float* XS [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
  const int G = a.K / 64;
  if (int(at.x) >= G) return;
  const device uint4* x = (const device uint4*)(X + size_t(at.y) * a.K + at.x * 64);
  float t = 0.0f;
  for (int kb = 0; kb < 8; ++kb) {
    const uint4 v = x[kb];
    const uint w[4] = {v.x, v.y, v.z, v.w};
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float f = as_type<float>((j % 2) ? (w[j / 2] & 0xFFFF0000u) : (w[j / 2] << 16));
      t = kb == 0 && j == 0 ? f : fma(f, a.one, t);
    }
  }
  XS[size_t(at.y) * G + at.x] = t;
}

// Code k of a group's little-endian bit stream (words w) as a float; k is a constant after unrolling.
template <int BITS>
inline float tf_tq_code(const thread uint* w, int k) {
  const int bit = k * BITS, at = bit / 32, sh = bit % 32;
  uint v = w[at] >> sh;
  if (sh + BITS > 32) v |= w[at + 1] << (32 - sh);
  return float(v & ((1u << BITS) - 1u));
}

// The twin's columns a simdgroup and simdgroups a threadgroup.
template <int BITS> struct TfTqCols { static constant constexpr int value = 4; };
constant constexpr int TQ_SG = 2;

// Group g's 64-step fma chains (the MMA order) and input sum; at 4 bits a code in place meets x * 16^-j (same product).
template <int BITS, int NC>
inline float tf_tq_group(const device bfloat* x, const device uint8_t* w, size_t stride, int g, thread float* p,
                         float one) {
  constexpr int GW = 2 * BITS;  // words a group of 64 codes
  float t = 0.0f;
  for (int c = 0; c < NC; ++c) p[c] = 0.0f;
  uint wq[BITS == 4 ? NC : 1][8];  // 4 bits: each column's group, two 16-byte loads
  if (BITS == 4)
    #pragma unroll
    for (int c = 0; c < NC; ++c) {
      const device uint4* src = (const device uint4*)(w + c * stride + size_t(g) * (GW * 4));
      const uint4 a = src[0], b = src[1];
      wq[c][0] = a.x; wq[c][1] = a.y; wq[c][2] = a.z; wq[c][3] = a.w;
      wq[c][4] = b.x; wq[c][5] = b.y; wq[c][6] = b.z; wq[c][7] = b.w;
    }
  #pragma unroll
  for (int kb = 0; kb < 8; ++kb) {
    const uint4 xv = ((const device uint4*)(x + size_t(g) * 64))[kb];
    const uint xw[4] = {xv.x, xv.y, xv.z, xv.w};
    float xf[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float v = as_type<float>((j % 2) ? (xw[j / 2] & 0xFFFF0000u) : (xw[j / 2] << 16));
      t = kb == 0 && j == 0 ? v : fma(v, one, t);
      xf[j] = BITS == 4 ? v * as_type<float>(uint(127 - 4 * j) << 23) : v;
    }
    if (BITS == 4) {
      #pragma unroll
      for (int c = 0; c < NC; ++c) {
        const uint word = wq[c][kb];
        #pragma unroll
        for (int j = 0; j < 8; ++j) p[c] = fma(xf[j], float(word & (0xFu << (4 * j))), p[c]);
      }
    } else {
      // Codes 8 kb.. of this group sit in the two words from bit 8 kb BITS (48 or 64 bits).
      const int bit = kb * 8 * BITS, at = bit / 32, sh = bit % 32;
      #pragma unroll
      for (int c = 0; c < NC; ++c) {
        const device uint* wc = (const device uint*)(w + c * stride + size_t(g) * (GW * 4)) + at;
        const ulong v = ((sh + 8 * BITS > 64 ? ulong(wc[2]) << 32 : 0ul) | 0ul);
        const ulong lo = (ulong(wc[1]) << 32) | wc[0];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int o = sh + BITS * j;
          const uint q = o + BITS <= 64 ? uint(lo >> o) : uint((lo >> o) | (v << (64 - o)));
          p[c] = fma(xf[j], float(q & ((1u << BITS) - 1u)), p[c]);
        }
      }
    }
  }
  return t;
}

// Rows a 4-bit twin threadgroup takes over one load of its columns' codes (windows: each weight read once).
constant constexpr int TQ_RT = 4;

// tf_tq_group at 4 bits for `rows` (<= TQ_RT) rows K apart: the codes stay in registers, each row its own chains.
template <int NC>
inline void tf_tq_group_rows(const device bfloat* x, int K, int rows, const device uint8_t* w, size_t stride, int g,
                             thread float (*p)[NC], thread float* t, float one) {
  uint wq[NC][8];
  #pragma unroll
  for (int c = 0; c < NC; ++c) {
    const device uint4* src = (const device uint4*)(w + c * stride + size_t(g) * 32);
    const uint4 a = src[0], b = src[1];
    wq[c][0] = a.x; wq[c][1] = a.y; wq[c][2] = a.z; wq[c][3] = a.w;
    wq[c][4] = b.x; wq[c][5] = b.y; wq[c][6] = b.z; wq[c][7] = b.w;
  }
  for (int r = 0; r < TQ_RT; ++r) {
    t[r] = 0.0f;
    for (int c = 0; c < NC; ++c) p[r][c] = 0.0f;
  }
  for (int r = 0; r < rows; ++r) {
    const device uint4* xr = (const device uint4*)(x + size_t(r) * K + size_t(g) * 64);
    #pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      const uint4 xv = xr[kb];
      const uint xw[4] = {xv.x, xv.y, xv.z, xv.w};
      float xf[8];
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float v = as_type<float>((j % 2) ? (xw[j / 2] & 0xFFFF0000u) : (xw[j / 2] << 16));
        t[r] = kb == 0 && j == 0 ? v : fma(v, one, t[r]);
        xf[j] = v * as_type<float>(uint(127 - 4 * j) << 23);
      }
      #pragma unroll
      for (int c = 0; c < NC; ++c)
        #pragma unroll
        for (int j = 0; j < 8; ++j) p[r][c] = fma(xf[j], float(wq[c][kb] & (0xFu << (4 * j))), p[r][c]);
    }
  }
}

// NC columns of up to TQ_RT rows at 4 bits, 32 groups a round: a group a lane, then lane r * NC + c chains row r's c.
template <int NC>
inline void tf_tq_columns_rows(const device bfloat* x, int rows, float one, const device uint8_t* w,
                               const device bfloat* sc, const device bfloat* bi, int K, int N, device bfloat* out,
                               uint lane, threadgroup float* part) {
  const int G = K / 64;
  const size_t stride = size_t(K) / 2;
  const int r = int(lane) / NC, c = int(lane) % NC;
  float acc = 0.0f;
  for (int g0 = 0; g0 < G; g0 += 32) {  // a round's partials and sums staged in part [rows][NC + 1][32]
    const int g = g0 + int(lane);
    if (g < G) {
      float p[TQ_RT][NC], t[TQ_RT];
      tf_tq_group_rows<NC>(x, K, rows, w, stride, g, p, t, one);
      for (int i = 0; i < rows; ++i) {
        part[(i * (NC + 1) + NC) * 32 + lane] = t[i];
        for (int j = 0; j < NC; ++j) part[(i * (NC + 1) + j) * 32 + lane] = p[i][j];
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (r < rows) {
      const device bfloat* s = sc + c * G + g0;
      const device bfloat* b = bi + c * G + g0;
      const threadgroup float* pc = part + (r * (NC + 1) + c) * 32;
      const threadgroup float* sums = part + (r * (NC + 1) + NC) * 32;
      const int n = min(32, G - g0);
      #pragma unroll(8)
      for (int j = 0; j < n; ++j) acc = fma(float(b[j]), sums[j], fma(float(s[j]), pc[j], acc));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (r < rows) out[size_t(r) * N + c] = bfloat(acc);
}

// NC columns of a row at the tiles' bits, 32 groups a round: a group a lane, then lane c chains column c in order.
template <int BITS, int NC>
inline float tf_tq_columns(const device bfloat* x, float one, const device uint8_t* w, const device bfloat* sc,
                           const device bfloat* bi, int K, uint lane, threadgroup float* part) {
  const int G = K / 64;
  const size_t stride = size_t(K) * BITS / 8;
  threadgroup float* sums = part + NC * 32;
  const int c = min(int(lane), NC - 1);
  float acc = 0.0f;
  for (int g0 = 0; g0 < G; g0 += 32) {
    const int g = g0 + int(lane);
    if (g < G) {
      float p[NC];
      sums[lane] = tf_tq_group<BITS, NC>(x, w, stride, g, p, one);
      for (int j = 0; j < NC; ++j) part[j * 32 + lane] = p[j];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const device bfloat* s = sc + c * G + g0;
    const device bfloat* b = bi + c * G + g0;
    const threadgroup float* pc = part + c * 32;
    const int n = min(32, G - g0);
    #pragma unroll(8)
    for (int j = 0; j < n; ++j) acc = fma(float(b[j]), sums[j], fma(float(s[j]), pc[j], acc));
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  return acc;
}

// Y = X W^T at the tiles' bits for decode and windows: tg.y takes TQ_RT rows a code load at 4 bits, one at 6 and 8.
template <int BITS>
[[kernel]] void tf_tile_qmv(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                            const device bfloat* SC [[buffer(2)]], const device bfloat* BI [[buffer(3)]],
                            constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                            threadgroup float* part [[threadgroup(0)]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NC = TfTqCols<BITS>::value;
  const int n = (int(tg.x) * TQ_SG + int(sg)) * NC, K = a.K, G = K / 64;
  if (n >= a.N) return;
  if (BITS == 4 && a.R > 1) {
    const int r0 = int(tg.y) * TQ_RT, rows = min(TQ_RT, a.R - r0);
    tf_tq_columns_rows<NC>(X + size_t(r0) * K, rows, a.one, W + size_t(n) * (K / 2), SC + size_t(n) * G,
                           BI + size_t(n) * G, K, a.N, OUT + size_t(r0) * a.N + n, lane,
                           part + sg * rows * (NC + 1) * 32);
    return;
  }
  const int r = int(tg.y);
  const float y = tf_tq_columns<BITS, NC>(X + size_t(r) * K, a.one, W + size_t(n) * (K * BITS / 8), SC + size_t(n) * G,
                                         BI + size_t(n) * G, K, lane, part + sg * (NC + 1) * 32);
  if (int(lane) < NC) OUT[size_t(r) * a.N + n + lane] = bfloat(y);
}

// The twin by slot (tg.y): expert ids[slot]'s matrix on x row slot / a.R, out row slot.
template <int BITS>
[[kernel]] void tf_tile_qmv_indexed(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                                    const device bfloat* SC [[buffer(2)]], const device bfloat* BI [[buffer(3)]],
                                    constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                                    const device uint* ids [[buffer(6)]], threadgroup float* part [[threadgroup(0)]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NC = TfTqCols<BITS>::value;
  const int n = (int(tg.x) * TQ_SG + int(sg)) * NC, K = a.K, G = K / 64;
  if (n >= a.N) return;
  const size_t e = ids[tg.y], row = tg.y / uint(a.R), m = e * size_t(a.N) + n;
  const float y = tf_tq_columns<BITS, NC>(X + row * K, a.one, W + m * (K * BITS / 8), SC + m * G, BI + m * G, K, lane,
                                         part + sg * (NC + 1) * 32);
  if (int(lane) < NC) OUT[size_t(tg.y) * a.N + n + lane] = bfloat(y);
}

// Gate and up by slot at the tiles' bits (W/SC/BI and W2/SC2/BI2, the same shape), out = swiglu of their bf16 rows.
template <int BITS>
[[kernel]] void tf_tile_qmv_glu(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                                const device bfloat* SC [[buffer(2)]], const device bfloat* BI [[buffer(3)]],
                                constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                                const device uint* ids [[buffer(6)]], const device uint8_t* W2 [[buffer(7)]],
                                const device bfloat* SC2 [[buffer(8)]], const device bfloat* BI2 [[buffer(9)]],
                                threadgroup float* part [[threadgroup(0)]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NC = TfTqCols<BITS>::value;
  const int n = (int(tg.x) * TQ_SG + int(sg)) * NC, K = a.K, G = K / 64;
  if (n >= a.N) return;
  const size_t e = ids[tg.y], row = tg.y / uint(a.R), m = e * size_t(a.N) + n;
  threadgroup float* own = part + sg * (NC + 1) * 32;
  const float g = tf_tq_columns<BITS, NC>(X + row * K, a.one, W + m * (K * BITS / 8), SC + m * G, BI + m * G, K, lane,
                                          own);
  const float u = tf_tq_columns<BITS, NC>(X + row * K, a.one, W2 + m * (K * BITS / 8), SC2 + m * G, BI2 + m * G, K,
                                          lane, own);
  if (int(lane) < NC) OUT[size_t(tg.y) * a.N + n + lane] = tf_tq_swiglu(bfloat(g), bfloat(u));
}

template [[host_name("tf_tile_qmm_q4")]] [[kernel]] decltype(tf_tile_qmm<4>) tf_tile_qmm<4>;
template [[host_name("tf_tile_qmm_q6")]] [[kernel]] decltype(tf_tile_qmm<6>) tf_tile_qmm<6>;
template [[host_name("tf_tile_qmm_q8")]] [[kernel]] decltype(tf_tile_qmm<8>) tf_tile_qmm<8>;
template [[host_name("tf_tile_qmm_grouped_q4")]] [[kernel]] decltype(tf_tile_qmm_grouped<4, false>)
    tf_tile_qmm_grouped<4, false>;
template [[host_name("tf_tile_qmm_grouped_glu_q4")]] [[kernel]] decltype(tf_tile_qmm_grouped<4, true>)
    tf_tile_qmm_grouped<4, true>;
template [[host_name("tf_tile_qmm_grouped_q6")]] [[kernel]] decltype(tf_tile_qmm_grouped<6, false>)
    tf_tile_qmm_grouped<6, false>;
template [[host_name("tf_tile_qmm_grouped_glu_q6")]] [[kernel]] decltype(tf_tile_qmm_grouped<6, true>)
    tf_tile_qmm_grouped<6, true>;
template [[host_name("tf_tile_qmm_grouped_q8")]] [[kernel]] decltype(tf_tile_qmm_grouped<8, false>)
    tf_tile_qmm_grouped<8, false>;
template [[host_name("tf_tile_qmm_grouped_glu_q8")]] [[kernel]] decltype(tf_tile_qmm_grouped<8, true>)
    tf_tile_qmm_grouped<8, true>;
template [[host_name("tf_tile_qmv_q4")]] [[kernel]] decltype(tf_tile_qmv<4>) tf_tile_qmv<4>;
template [[host_name("tf_tile_qmv_q6")]] [[kernel]] decltype(tf_tile_qmv<6>) tf_tile_qmv<6>;
template [[host_name("tf_tile_qmv_q8")]] [[kernel]] decltype(tf_tile_qmv<8>) tf_tile_qmv<8>;
template [[host_name("tf_tile_qmv_indexed_q4")]] [[kernel]] decltype(tf_tile_qmv_indexed<4>) tf_tile_qmv_indexed<4>;
template [[host_name("tf_tile_qmv_indexed_q6")]] [[kernel]] decltype(tf_tile_qmv_indexed<6>) tf_tile_qmv_indexed<6>;
template [[host_name("tf_tile_qmv_indexed_q8")]] [[kernel]] decltype(tf_tile_qmv_indexed<8>) tf_tile_qmv_indexed<8>;
template [[host_name("tf_tile_qmv_glu_q4")]] [[kernel]] decltype(tf_tile_qmv_glu<4>) tf_tile_qmv_glu<4>;
template [[host_name("tf_tile_qmv_glu_q6")]] [[kernel]] decltype(tf_tile_qmv_glu<6>) tf_tile_qmv_glu<6>;
template [[host_name("tf_tile_qmv_glu_q8")]] [[kernel]] decltype(tf_tile_qmv_glu<8>) tf_tile_qmv_glu<8>;
