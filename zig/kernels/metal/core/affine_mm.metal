// x W^T for MLX affine W on the tensor units: a step's weights dequantized to bf16 in the tile, one rounding each.
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"
using namespace tfp;

// Defines: TF_BITS (2-6 or 8), TF_GROUP (32 or 64), TF_OUT_T (bfloat, or float for partial sums another pass adds).
#ifndef TF_OUT_T
#define TF_OUT_T bfloat
#endif
constant constexpr int STEP_BYTES = 8 * TF_BITS; // a weight row's bytes in one 64-deep step
constant constexpr int HALF_BYTES = 4 * TF_BITS; // a thread's half of them (32 codes)
constant constexpr int NG = 64 / TF_GROUP;       // groups a step
constant constexpr int PAD = 64 + 8;              // the weight tile's row pitch (bf16)

struct MmArgs {
  int rows, n, k, experts; // rows (sorted by expert for a gather), outputs a row, depth, experts (gather)
};

// A dense matmul's pitches, and each batch's start in x, y (elements) and the weights (rows).
struct MmStrides {
  int x_row, y_row, x_batch, y_batch, w_batch, pad;
};

// A thread's 32 codes of one weight row in a step (16-byte loads; 8 bytes at 2 bits; words at 3, 5 and 6 bits).
#if TF_BITS == 2
struct Codes {
  uint2 v;
};

inline Codes am_load(const device uchar* w) {
  return Codes{*(const device uint2*)w};
}
#elif TF_BITS % 4 != 0
struct Codes {
  uint v[TF_BITS];
};

inline Codes am_load(const device uchar* w) {
  Codes c;
  TF_UNROLL
  for (short i = 0; i < TF_BITS; i++) {
    c.v[i] = ((const device uint*)w)[i];
  }
  return c;
}
#else
struct Codes {
  uint4 v[TF_BITS / 4];
};

inline Codes am_load(const device uchar* w) {
  Codes c;
  TF_UNROLL
  for (short i = 0; i < TF_BITS / 4; i++) {
    c.v[i] = ((const device uint4*)w)[i];
  }
  return c;
}
#endif

// Code i (0..31) of a thread's step: MLX's little-endian bit stream at every width.
inline float am_code(thread const Codes& c, short i) {
#if TF_BITS == 2
  return float((c.v[i / 16] >> (2 * (i % 16))) & 3u);
#elif TF_BITS % 4 != 0
  const short at = TF_BITS * i, word = at / 32, shift = at % 32;
  uint v = c.v[word] >> shift;
  if (shift + TF_BITS > 32) v |= c.v[word + 1] << (32 - shift);
  return float(v & ((1u << TF_BITS) - 1u));
#elif TF_BITS == 4
  return float((c.v[0][i / 8] >> (4 * (i % 8))) & 15u);
#else
  return float((c.v[i / 16][(i / 4) % 4] >> (8 * (i % 4))) & 255u);
#endif
}

// The thread's 32 weights, scale * code + bias in fp32 rounded once to bf16 (as MLX dequantizes), into the tile row.
inline void am_weights(thread const Codes& c, float s, float b, threadgroup bfloat* out) {
  threadgroup uint4* o = (threadgroup uint4*)out;
  TF_UNROLL
  for (short j = 0; j < 4; j++) { // 16-byte stores: codes 8j .. 8j + 7
    uint4 v;
    TF_UNROLL
    for (short m = 0; m < 4; m++) {
      const bfloat w0 = bfloat(fma(s, am_code(c, 8 * j + 2 * m), b));
      const bfloat w1 = bfloat(fma(s, am_code(c, 8 * j + 2 * m + 1), b));
      v[m] = uint(as_type<ushort>(w0)) | (uint(as_type<ushort>(w1)) << 16);
    }
    o[j] = v;
  }
}


// A lane's two rows (home.y and home.y + 8) of each of its TM fragments: their starts in x.
template <int TM>
struct AmRows {
  const device bfloat* x[TM][2];
};

// The fragment of x at column c from a lane's two row starts (8-byte loads on M5).
inline frag<bfloat> am_frag(const device bfloat* r0, const device bfloat* r1, int c, short2 home) {
#ifndef TF_SIMD_FRAGS
  return frag<bfloat>(*(const device bfloat4*)(r0 + c + home.x), *(const device bfloat4*)(r1 + c + home.x));
#else
  frag<bfloat> f;
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    f[e] = (e < 4 ? r0 : r1)[c + home.x + TF_COL(e)];
  }
  return f;
#endif
}

// The tile's 16 x 16 block at (r, c) as a transposed right operand (on M5 a lane's four columns of a row in one load).
inline frag<bfloat> am_tile_frag(const threadgroup bfloat* tile, int r, int c, short2 home) {
#ifndef TF_SIMD_FRAGS
  const threadgroup bfloat* p = tile + (r + home.y) * PAD + c + home.x;
  return frag<bfloat>(*(const threadgroup bfloat4*)p, *(const threadgroup bfloat4*)(p + 8 * PAD));
#else
  frag<bfloat> f;
  frag_get_t(f, tile, PAD, r, c, home);
  return f;
#endif
}

// One step's MMAs for the first TL of TM row fragments (TL < TM: a tile's tail, the rest dead).
template <int TM, int TL>
inline void am_step(thread frag<float> (&acc)[TM][2], thread const AmRows<TM>& rows, int k,
                    const threadgroup bfloat* tile, int tn, short2 home) {
  TF_UNROLL
  for (short q = 0; q < 4; q++) { // the step's four 16-deep slices, every row fragment's chain interleaved
    frag<bfloat> b0, b1, a[TL];
    b0 = am_tile_frag(tile, tn, 16 * q, home);
    b1 = am_tile_frag(tile, tn + 16, 16 * q, home);
    TF_UNROLL
    for (short m = 0; m < TL; m++) {
      a[m] = am_frag(rows.x[m][0], rows.x[m][1], k + 16 * q, home);
    }
    TF_UNROLL
    for (short m = 0; m < TL; m++) {
      mma_16x32<false, true>(acc[m][0], acc[m][1], a[m], b0, b1);
    }
  }
}

// TM 16-row fragments (`live` rows; a dead row's start is a live one's) times a [64, K] code block.
template <int TM>
inline void am_k_loop(thread frag<float> (&acc)[TM][2], thread const AmRows<TM>& rows, int K, int live,
                      const device uchar* wq, const device bfloat* scales, const device bfloat* biases,
                      threadgroup bfloat* tile, int tn, uint t, short2 home) {
  threadgroup bfloat* mine = tile + (t / 2) * PAD + 32 * (t % 2);
  const int g_mine = TF_GROUP == 32 ? int(t % 2) : 0; // at g32 each thread of a pair dequantizes its own group
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    acc[i][0] = frag<float>(0);
    acc[i][1] = frag<float>(0);
  }
  Codes next = am_load(wq); // a step's codes, scale and bias fetched a step ahead: the loads' latency off the barriers
  float s_next = float(scales[g_mine]), b_next = float(biases[g_mine]);
  for (int k = 0; k < K; k += 64) {
    const Codes codes = next;
    const float s_now = s_next, b_now = b_next;
    if (k + 64 < K) {
      next = am_load(wq + STEP_BYTES);
      s_next = float(scales[NG + g_mine]);
      b_next = float(biases[NG + g_mine]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    am_weights(codes, s_now, b_now, mine);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (live > 16 * (TM - 1)) {
      am_step<TM, TM>(acc, rows, k, tile, tn, home);
    } else if (TM > 1 && live > 0) { // a tail with only the first fragment live
      am_step<TM, 1>(acc, rows, k, tile, tn, home);
    }
    wq += STEP_BYTES;
    scales += NG;
    biases += NG;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// A simdgroup's TM x 2 fragments to rows dst[i] of y (ld N), rows below live only: rows put back in another order.
template <int TM>
inline void am_scatter(thread const frag<float> (&acc)[TM][2], device TF_OUT_T* y, const device int32_t* dst, int N,
                       int col, int live, short2 home) {
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const int r = 16 * i + home.y + (e >> 2) * 8;
      if (r < live) {
        device TF_OUT_T* yr = y + long(dst[r]) * N + col + home.x + TF_COL(e);
        yr[0] = TF_OUT_T(acc[i][0][e]);
        yr[16] = TF_OUT_T(acc[i][1][e]);
      }
    }
  }
}

template <int TM>
inline void am_store(thread const frag<float> (&acc)[TM][2], device TF_OUT_T* out, int N, int live, short2 home) {
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      if (live == 16 * TM) {
        frag_put(acc[i][j], out, N, 16 * i, 16 * j, home);
      } else {
        frag_put_in(acc[i][j], out, N, 16 * i, 16 * j, home, live, 32);
      }
    }
  }
}

// The tile'th BM-row tile over experts in order: its expert, first row and row count (false past the last tile).
template <int BM>
inline bool am_tile(const device int32_t* offsets, int experts, int total, int tile, uint lane, thread int& expert,
                    thread int& row, thread int& rows) {
  int before = 0;
  for (int base = 0; base < experts; base += 32) {
    const int e = base + int(lane);
    const int first = e < experts ? offsets[e] : total, last = e + 1 < experts ? offsets[e + 1] : total;
    const int count = (last - first + BM - 1) / BM;
    const int upto = before + simd_prefix_inclusive_sum(count);
    const int owner = simd_sum(int(upto <= tile));
    if (owner < 32) {
      expert = base + owner;
      row = simd_shuffle(first, ushort(owner)) + (tile - simd_shuffle(upto - count, ushort(owner))) * BM;
      rows = min(BM, simd_shuffle(last, ushort(owner)) - row);
      return true;
    }
    before = simd_shuffle(upto, ushort(31));
  }
  return false;
}

// A simdgroup's row starts: rows [row, row + rows) of x, or with `order` the sorted rows' token rows (order / topk).
template <int TM>
inline AmRows<TM> am_rows(const device bfloat* x, int ldx, int row, int rows, int tm, short2 home,
                          const device int32_t* order, int topk) {
  AmRows<TM> r;
  TF_UNROLL
  for (short m = 0; m < TM; m++) {
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      const int i = tm + 16 * m + home.y + 8 * h;
      const int at = row + (i < rows ? i : 0); // a dead row reads a live one; its products never reach y
      const long src = order != nullptr ? long(order[at] / topk) : long(at);
      r.x[m][h] = x + src * ldx;
    }
  }
  return r;
}

// Rows [row, row + rows) of x times weight rows [wrow0, wrow0 + 64) of a [*, K] matrix into y (ld N) at column col.
template <int BM>
inline void am_block(const device bfloat* x, int ldx, int row, int rows, int K, int ldy, int col, long wrow0, const device uint32_t* w, const device bfloat* scales,
                     const device bfloat* biases, device TF_OUT_T* y, threadgroup bfloat* tile, uint sg, uint lane,
                     const device int32_t* dst = nullptr, const device int32_t* order = nullptr,
                     int topk = 1) {
  constexpr int SM = BM / 2;
  constexpr int TM = SM / 16;
  const int t = int(sg) * 32 + int(lane);
  const int tm = SM * int(sg / 2), tn = 32 * int(sg % 2), live = clamp(rows - tm, 0, SM);
  const long wrow = wrow0 + t / 2;
  const short2 home = frag_home(ushort(lane));
  const AmRows<TM> starts = am_rows<TM>(x, ldx, row, rows, tm, home, order, topk);
  frag<float> acc[TM][2];
  am_k_loop<TM>(acc, starts, K, live, (const device uchar*)w + wrow * (K * TF_BITS / 8) + HALF_BYTES * (t % 2),
                scales + wrow * (K / TF_GROUP), biases + wrow * (K / TF_GROUP), tile, tn, uint(t), home);
  if (dst != nullptr) {
    if (live > 0) am_scatter<TM>(acc, y, dst + row + tm, ldy, col + tn, live, home);
  } else {
    am_store<TM>(acc, y + long(row + tm) * ldy + col + tn, ldy, live, home);
  }
}

// y [rows, n] = x [rows, k] W^T, n a multiple of 64: 64x64 tiles, a batch along tg.z (MmStrides).
[[kernel]] void tf_affine_mm(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                             const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                             constant MmArgs& a [[buffer(5)]], device TF_OUT_T* Y [[buffer(6)]],
                             constant MmStrides& st [[buffer(7)]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                             uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat tile[64 * PAD];
  const int row = int(tg.y) * 64, col = int(tg.x) * 64, b = int(tg.z);
  am_block<64>(X + long(b) * st.x_batch, st.x_row, row, min(64, a.rows - row), a.k, st.y_row, col,
               long(b) * st.w_batch + col, W, S, B, Y + long(b) * st.y_batch, tile, sg, lane);
}

#define TF_GATHER(BM)                                                                                                   \
  [[kernel]] void tf_affine_gather_##BM(                                                                                \
      const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],                                    \
      const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],                                      \
      constant MmArgs& a [[buffer(5)]], device TF_OUT_T* Y [[buffer(6)]],                                           \
      const device int32_t* OFFS [[buffer(7)]], uint sg [[simdgroup_index_in_threadgroup]],                           \
      uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {                           \
    threadgroup bfloat tile[64 * PAD];                                                                                 \
    int expert, row, rows;                                                                                             \
    if (!am_tile<BM>(OFFS, a.experts, a.rows, int(tg.y), lane, expert, row, rows)) return;                            \
    const int col = int(tg.x) * 64;                                                                                    \
    am_block<BM>(X, a.k, row, rows, a.k, a.n, col, long(expert) * a.n + col, W, S, B, Y,                           \
                 tile, sg, lane);                                                                                      \
  }
TF_GATHER(32)
TF_GATHER(64)

// The gathers with each sorted row's output put back at row DST[row] (e.g. the (row, slot) pairs' own order).
#define TF_GATHER_SCATTER(BM)                                                                                           \
  [[kernel]] void tf_affine_gather_scatter_##BM(                                                                        \
      const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],                                    \
      const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],                                      \
      constant MmArgs& a [[buffer(5)]], device TF_OUT_T* Y [[buffer(6)]],                                           \
      const device int32_t* OFFS [[buffer(7)]], const device int32_t* DST [[buffer(8)]],                              \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                            \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                                     \
    threadgroup bfloat tile[64 * PAD];                                                                                 \
    int expert, row, rows;                                                                                             \
    if (!am_tile<BM>(OFFS, a.experts, a.rows, int(tg.y), lane, expert, row, rows)) return;                            \
    const int col = int(tg.x) * 64;                                                                                    \
    am_block<BM>(X, a.k, row, rows, a.k, a.n, col, long(expert) * a.n + col, W, S, B, Y,                           \
                 tile, sg, lane, DST);                                                                                 \
  }
TF_GATHER_SCATTER(32)
TF_GATHER_SCATTER(64)

#ifdef TF_GLU_PLAIN
// mlx-lm's compiled SwiGLU on bf16 rows (LFM2): silu(g) * u, rounded to bf16 at each step; `limit` unused.
inline bfloat am_swiglu(float g, float u, float limit) {
  const float gt = float(bfloat(g));
  const bfloat s = static_cast<bfloat>(1.0f / (1.0f + metal::precise::exp(-gt)));
  const bfloat silu = static_cast<bfloat>(gt * float(s));
  return static_cast<bfloat>(float(silu) * float(bfloat(u)));
}
#else
// The glue's SwiGLU in bf16 (glm_act2): MLX's bf16 exp and abs, nn.silu's x * sigmoid(x), gate and up clipped at `limit`.
inline bfloat am_sigmoid(bfloat x) {
  const bfloat e = bfloat(__metal_exp(float(bfloat(__metal_fabs(float(x), __METAL_MAYBE_FAST_MATH__))), __METAL_MAYBE_FAST_MATH__));
  const bfloat y = bfloat(1) / (bfloat(1) + e);
  return (x < 0) ? y : (bfloat(1) - y);
}

inline bfloat am_swiglu(float g, float u, float limit) {
  const float lim = float(bfloat(limit));
  const bfloat gt = bfloat(metal::min(float(bfloat(g)), lim));
  const bfloat up = bfloat(metal::min(metal::max(float(bfloat(u)), -lim), lim));
  const bfloat sl = gt * am_sigmoid(gt);
  return sl * up;
}
#endif

// A simdgroup's gate (j 0) and up (j 1) fragments as SwiGLU outputs into y (ld N), rows below live only.
template <int TM>
inline void am_store_glu(thread const frag<float> (&acc)[TM][2], device bfloat* y, int N, int live, short2 home,
                         float limit) {
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const int r = 16 * i + home.y + (e >> 2) * 8;
      if (r < live) y[long(r) * N + home.x + TF_COL(e)] = am_swiglu(acc[i][0][e], acc[i][1][e], limit);
    }
  }
}

// The routed experts' gate and up over token rows read through ORDER (pair / topk), SwiGLU'd into act [rows, n]: 32 columns a tile.
#define TF_GATHER_GLU(BM)                                                                                               \
  [[kernel]] void tf_affine_gather_glu_##BM(                                                                            \
      const device bfloat* X [[buffer(0)]], const device uint32_t* WG [[buffer(1)]],                                   \
      const device bfloat* SG [[buffer(2)]], const device bfloat* BG [[buffer(3)]],                                    \
      constant MmArgs& a [[buffer(5)]], device bfloat* Y [[buffer(6)]],                                              \
      const device int32_t* OFFS [[buffer(7)]], const device int32_t* ORDER [[buffer(8)]],                            \
      const device uint32_t* WU [[buffer(9)]], const device bfloat* SU [[buffer(10)]],                                 \
      const device bfloat* BU [[buffer(11)]], constant float2& glu [[buffer(12)]],                                     \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                            \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                                     \
    constexpr int SM = BM / 2;                                                                                         \
    constexpr int TM = SM / 16;                                                                                        \
    threadgroup bfloat tile[64 * PAD];                                                                                 \
    int expert, row, rows;                                                                                             \
    if (!am_tile<BM>(OFFS, a.experts, a.rows, int(tg.y), lane, expert, row, rows)) return;                            \
    const int c0 = int(tg.x) * 32, t = int(sg) * 32 + int(lane), tr = t / 2;                                         \
    const bool up = ((tr >> 4) & 1) != 0; /* tile rows by 16: gate, up, gate, up for columns c0.., c0 + 16.. */        \
    const long wrow = long(expert) * a.n + c0 + (tr >> 5) * 16 + (tr & 15);                                           \
    const int tm = SM * int(sg / 2), tn = 32 * int(sg % 2), live = clamp(rows - tm, 0, SM);                           \
    const short2 home = frag_home(ushort(lane));                                                                       \
    const AmRows<TM> starts = am_rows<TM>(X, a.k, row, rows, tm, home, ORDER, int(glu.x));                           \
    frag<float> acc[TM][2];                                                                                            \
    am_k_loop<TM>(acc, starts, a.k, live,                                                                              \
                  (const device uchar*)(up ? WU : WG) + wrow * (a.k * TF_BITS / 8) + HALF_BYTES * (t % 2),            \
                  (up ? SU : SG) + wrow * (a.k / TF_GROUP), (up ? BU : BG) + wrow * (a.k / TF_GROUP), tile, tn, uint(t), \
                  home);                                                                                               \
    am_store_glu<TM>(acc, Y + long(row + tm) * a.n + c0 + tn / 2, a.n, live, home, glu.y);                            \
  }
TF_GATHER_GLU(32)
TF_GATHER_GLU(64)
