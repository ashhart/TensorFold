// Prompt matmuls before the M5: 32x32 tiles, exact codes and inputs in simdgroup MMAs, fp32 group and row sums.
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
inline void tf_tq_fetch(thread TfTqAhead<BITS>& a, const device bfloat* X, const device float* XS, const device uint8_t* W,
                        const device bfloat* SC, const device bfloat* BI, const device uint8_t* W2, const device bfloat* SC2,
                        const device bfloat* BI2, uint xr, uint sr, int K, int nb, int s, int tid) {
  const int G = K / 64, g = s / 2, k0 = g * 64 + s % 2 * 32;
  // The thread's input row is X's xr and its sum row XS's sr (resolved once a tile).
  a.x = *((const device uint4*)(X + size_t(xr) * K + k0) + tid % 4);
  const bool up = GLU && tid >= 64;
  const int col = nb + (GLU ? tid / 4 % (BN / 2) : tid / 4);
  const device ushort* src = (const device ushort*)((up ? W2 : W) + size_t(col) * (K * BITS / 8) + (k0 + tid % 4 * 8) * BITS / 8);
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
inline void tf_tq_stage(const thread TfTqAhead<BITS>& a, threadgroup bfloat* xs, threadgroup bfloat* ws, threadgroup float2* sb,
                        threadgroup float* xsum, int s, int tid) {
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
inline void tf_tq_mmas(thread simdgroup_matrix<float, 8, 8> (*P)[TJ], const threadgroup bfloat* xb, const threadgroup bfloat* wb,
                       int sr, int sc, int fm, int fn) {
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
                       const device bfloat* BI, const device uint8_t* W2, const device bfloat* SC2, const device bfloat* BI2,
                       const device uint* map, int K, int N, int R, device bfloat* OUT, int rb, int nb, uint tid, uint sg, uint lane,
                       threadgroup bfloat* xs, threadgroup bfloat* ws, threadgroup float2* sb, threadgroup float* xsum) {
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
    if (s + 2 < S) tf_tq_fetch<BITS, BM, BN, GLU>(even, X, XS, W, SC, BI, W2, SC2, BI2, xr, srw, K, nb, s + 2, int(tid));
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
        for (int i = 0; i < TI; ++i) acc[i][j][e] = fma(v.y, xsr[i], fma(v.x, P[i][j].thread_elements()[e], acc[i][j][e]));
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
                            constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]], const device float* XS [[buffer(6)]],
                            uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat xs[2 * TQ * TP], ws[2 * TQN * TP];
  threadgroup float2 sb[2 * TQN];
  threadgroup float xsum[2 * TQ];
  tf_tq_tile<BITS, TQ, TQN, false, false, false>(X, XS, W, SC, BI, W, SC, BI, nullptr, a.K, a.N, a.R, OUT, int(tg.y) * TQ,
                                                 int(tg.x) * TQN, tid, sg, lane, xs, ws, sb, xsum);
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
                                    uint sg [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat xs[2 * TQ * TP], ws[2 * TQN * TP];
  threadgroup float2 sb[2 * TQN];
  threadgroup float xsum[2 * TQ];
  if (tg.y >= work[0]) return;
  // Sorted row j reads X's and XS's row MAP[j], and writes OUT's row j.
  const uint item = work[1 + tg.y], e = item >> 16;
  const uint first = starts[e], rows = starts[e + 1] - first;
  // GLU: W2/SC2/BI2 are the up experts (the same shape), OUT [rows, N] their swiglu with W's gate, 16 columns a tile.
  const size_t K = size_t(a.K), N = size_t(a.N), at = e * N * (K * BITS / 8), sat = e * N * (K / 64);
  tf_tq_tile<BITS, TQ, TQN, true, GLU, true>(X, XS, W + at, SC + sat, BI + sat, W2 + at, SC2 + sat, BI2 + sat, map + first, a.K, a.N,
                                             int(rows), OUT + first * N, int(item & 0xFFFFu) * TQ, int(tg.x) * (GLU ? TQN / 2 : TQN),
                                             tid, sg, lane, xs, ws, sb, xsum);
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
[[kernel]] void tf_tile_xsum(const device bfloat* X [[buffer(0)]], constant TfTqArgs& a [[buffer(1)]], device float* XS [[buffer(2)]],
                             uint2 at [[thread_position_in_grid]]) {
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

template [[host_name("tf_tile_qmm_q4")]] [[kernel]] decltype(tf_tile_qmm<4>) tf_tile_qmm<4>;
template [[host_name("tf_tile_qmm_q6")]] [[kernel]] decltype(tf_tile_qmm<6>) tf_tile_qmm<6>;
template [[host_name("tf_tile_qmm_q8")]] [[kernel]] decltype(tf_tile_qmm<8>) tf_tile_qmm<8>;
template [[host_name("tf_tile_qmm_grouped_q4")]] [[kernel]] decltype(tf_tile_qmm_grouped<4, false>) tf_tile_qmm_grouped<4, false>;
template [[host_name("tf_tile_qmm_grouped_glu_q4")]] [[kernel]] decltype(tf_tile_qmm_grouped<4, true>) tf_tile_qmm_grouped<4, true>;
template [[host_name("tf_tile_qmm_grouped_q6")]] [[kernel]] decltype(tf_tile_qmm_grouped<6, false>) tf_tile_qmm_grouped<6, false>;
template [[host_name("tf_tile_qmm_grouped_glu_q6")]] [[kernel]] decltype(tf_tile_qmm_grouped<6, true>) tf_tile_qmm_grouped<6, true>;
template [[host_name("tf_tile_qmm_grouped_q8")]] [[kernel]] decltype(tf_tile_qmm_grouped<8, false>) tf_tile_qmm_grouped<8, false>;
template [[host_name("tf_tile_qmm_grouped_glu_q8")]] [[kernel]] decltype(tf_tile_qmm_grouped<8, true>) tf_tile_qmm_grouped<8, true>;
