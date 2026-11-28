
#ifdef TF_GLU_PLAIN
// mlx-lm's compiled SwiGLU on bf16 rows (LFM2): silu(g) * u, rounded to bf16 at each step; `limit` unused.
inline bfloat am_swiglu(float g, float u, float limit) {
  const float gt = float(bfloat(g));
  const bfloat s = static_cast<bfloat>(1.0f / (1.0f + metal::precise::exp(-gt)));
  const bfloat silu = static_cast<bfloat>(gt * float(s));
  return static_cast<bfloat>(float(silu) * float(bfloat(u)));
}
#else
// The glue's SwiGLU in bf16 (glm_act2): MLX's bf16 exp and abs, nn.silu's x*sigmoid(x), gate and up clipped at `limit`.
inline bfloat am_sigmoid(bfloat x) {
  const bfloat e = bfloat(__metal_exp(float(bfloat(__metal_fabs(float(x), __METAL_MAYBE_FAST_MATH__))), __METAL_MAYBE_FAST_MATH__));
  const bfloat y = bfloat(1) / (bfloat(1) + e);
  return (x < 0) ? y : (bfloat(1) - y);
}

inline bfloat am_swiglu(float g, float u, float limit) {
  const float lim = float(bfloat(limit));
  const bfloat gt = bfloat(metal::min(float(bfloat(g)), lim));
  const bfloat up = bfloat(metal::min(metal::max(float(bfloat(u)), -lim), lim));
#ifdef TF_GLU_ROUND_SILU
  const bfloat sl = bfloat(precise::divide(float(gt), 1.0f + precise::exp(-float(gt))));
#else
  const bfloat sl = gt * am_sigmoid(gt);
#endif
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

// The routed experts' gate and up over ORDER's token rows (pair / topk), SwiGLU'd to act [rows, n]: 32 columns a tile.
#define TF_GATHER_GLU(BM)                                                                                               \
  [[kernel]] void tf_affine_gather_glu_##BM(                                                                            \
      const device bfloat* X [[buffer(0)]], const device uint32_t* WG [[buffer(1)]],                                   \
      const device bfloat* SG [[buffer(2)]], const device bfloat* BG [[buffer(3)]],                                    \
      const device float* XS [[buffer(4)]], constant MmArgs& a [[buffer(5)]], device bfloat* Y [[buffer(6)]],          \
      const device int32_t* OFFS [[buffer(7)]], const device int32_t* ORDER [[buffer(8)]],                            \
      const device uint32_t* WU [[buffer(9)]], const device bfloat* SU [[buffer(10)]],                                 \
      const device bfloat* BU [[buffer(11)]], constant float2& glu [[buffer(12)]], TF_AM_MAP_ARG                      \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                            \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                                     \
    constexpr int SM = BM / 2;                                                                                         \
    constexpr int TM = SM / 16;                                                                                        \
    threadgroup bfloat tile[64 * PAD];                                                                                 \
    threadgroup float sb[NG * 128];                                                                                    \
    int expert, row, rows;                                                                                             \
    if (!am_tile<BM>(OFFS, a.experts, a.rows, int(tg.y), lane, expert, row, rows)) return;                            \
    const int c0 = int(tg.x) * 32, t = int(sg) * 32 + int(lane), tr = t / 2;                                         \
    const bool up = ((tr >> 4) & 1) != 0;                                                                             \
    const long wrow = long(TF_AM_EXPERT(expert)) * a.n + c0 + (tr >> 5) * 16 + (tr & 15);                             \
    const int tm = SM * int(sg / 2), tn = 32 * int(sg % 2), live = clamp(rows - tm, 0, SM);                           \
    const short2 home = frag_home(ushort(lane));                                                                       \
    const AmRows<TM> starts = am_rows<TM>(X, a.k, XS, a.k / TF_GROUP, row, rows, tm, home, ORDER, int(glu.x));       \
    frag<float> acc[TM][2];                                                                                            \
    am_k_loop<TM>(acc, starts, a.k, live,                                                                              \
                  (const device uchar*)(up ? WU : WG) + wrow * (a.k * TF_BITS / 8) + HALF_BYTES * (t % 2),            \
                  (up ? SU : SG) + wrow * (a.k / TF_GROUP), (up ? BU : BG) + wrow * (a.k / TF_GROUP), tile, sb, tn,     \
                  uint(t), home);                                                                                      \
    am_store_glu<TM>(acc, Y + long(row + tm) * a.n + c0 + tn / 2, a.n, live, home, glu.y);                            \
  }
// Tile rows by 16: gate, up, gate, up for columns c0.., c0 + 16..
TF_GATHER_GLU(32)
TF_GATHER_GLU(64)
