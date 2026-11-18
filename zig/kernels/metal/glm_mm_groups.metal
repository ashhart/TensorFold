// Prompt products in column groups, core/affine_mm.metal's tiles, sums and bits (appended to it at 4 bits, g64).
#if TF_GROUP != 64 || TF_BITS != 4
#error "the column-group products read 4-bit codes in groups of 64"
#endif

constant constexpr int GT_GROUP = 32; // column tiles a group

// Launch id's row tile and column tile (`col`): groups of GT_GROUP columns, row tiles in order within a group.
template <int BM>
inline int gt_order(uint id, constant MmArgs& a, thread int& col) {
  const int T = min(a.rows, (a.rows + BM - 1) / BM + a.experts - 1), L = int(id); // the one-launch gathers' tile count
  const int g = L / (T * GT_GROUP), rem = L - g * T * GT_GROUP;
  col = g * GT_GROUP + rem % GT_GROUP;
  return rem / GT_GROUP;
}

// tf_affine_gather_glu_BM's gate, up and SwiGLU at the launch's row and column tiles.
template <int BM>
[[kernel]] void glm_gather_glu(const device bfloat* X [[buffer(0)]], const device uint32_t* WG [[buffer(1)]],
                               const device bfloat* SG [[buffer(2)]], const device bfloat* BG [[buffer(3)]],
                               const device float* XS [[buffer(4)]], constant MmArgs& a [[buffer(5)]],
                               device bfloat* Y [[buffer(6)]], const device int32_t* OFFS [[buffer(7)]],
                               const device int32_t* ORDER [[buffer(8)]], const device uint32_t* WU [[buffer(9)]],
                               const device bfloat* SU [[buffer(10)]], const device bfloat* BU [[buffer(11)]],
                               constant float2& glu [[buffer(12)]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int SM = BM / 2, TM = SM / 16;
  threadgroup bfloat tile[64 * PAD];
  threadgroup float sb[NG * 128];
  int expert, row, rows, ct;
  const int rt = gt_order<BM>(tg.x, a, ct);
  if (!am_tile<BM>(OFFS, a.experts, a.rows, rt, lane, expert, row, rows)) return;
  const int c0 = ct * 32, t = int(sg) * 32 + int(lane), tr = t / 2;
  const bool up = ((tr >> 4) & 1) != 0;
  const long wrow = long(expert) * a.n + c0 + (tr >> 5) * 16 + (tr & 15);
  const int tm = SM * int(sg / 2), tn = 32 * int(sg % 2), live = clamp(rows - tm, 0, SM);
  const short2 home = frag_home(ushort(lane));
  const AmRows<TM> starts = am_rows<TM>(X, a.k, XS, a.k / TF_GROUP, row, rows, tm, home, ORDER, int(glu.x));
  frag<float> acc[TM][2];
  am_k_loop<TM>(acc, starts, a.k, live, (const device uchar*)(up ? WU : WG) + wrow * (a.k * TF_BITS / 8) + HALF_BYTES * (t % 2),
                (up ? SU : SG) + wrow * (a.k / TF_GROUP), (up ? BU : BG) + wrow * (a.k / TF_GROUP), tile, sb, tn, uint(t), home);
  am_store_glu<TM>(acc, Y + long(row + tm) * a.n + c0 + tn / 2, a.n, live, home, glu.y);
}

// tf_affine_gather_scatter_BM at the launch's row and column tiles: sorted row i's outputs to row DST[i] of Y.
template <int BM>
[[kernel]] void glm_gather_scatter(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                                   const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                                   const device float* XS [[buffer(4)]], constant MmArgs& a [[buffer(5)]],
                                   device TF_OUT_T* Y [[buffer(6)]], const device int32_t* OFFS [[buffer(7)]],
                                   const device int32_t* DST [[buffer(8)]], uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat tile[64 * PAD];
  threadgroup float sb[NG * 128];
  int expert, row, rows, ct;
  const int rt = gt_order<BM>(tg.x, a, ct);
  if (!am_tile<BM>(OFFS, a.experts, a.rows, rt, lane, expert, row, rows)) return;
  const int col = ct * 64;
  am_block<BM>(X, a.k, XS, a.k / TF_GROUP, row, rows, a.k, a.n, col, long(expert) * a.n + col, W, S, B, Y, tile, sb, sg,
               lane, DST);
}

template [[host_name("glm_gather_glu_32")]] [[kernel]] decltype(glm_gather_glu<32>) glm_gather_glu<32>;
template [[host_name("glm_gather_glu_64")]] [[kernel]] decltype(glm_gather_glu<64>) glm_gather_glu<64>;
template [[host_name("glm_gather_scatter_32")]] [[kernel]] decltype(glm_gather_scatter<32>) glm_gather_scatter<32>;
template [[host_name("glm_gather_scatter_64")]] [[kernel]] decltype(glm_gather_scatter<64>) glm_gather_scatter<64>;

// tf_affine_mm (64 x 64 tiles, a batch along z) in groups of GT_GROUP column tiles, the last group the rest.
[[kernel]] void glm_mm_grouped(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                               const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                               const device float* XS [[buffer(4)]], constant MmArgs& a [[buffer(5)]],
                               device TF_OUT_T* Y [[buffer(6)]], constant MmStrides& st [[buffer(7)]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat tile[64 * PAD];
  threadgroup float sb[NG * 128];
  const int T = (a.rows + 63) / 64, NC = a.n / 64, L = int(tg.x);
  const int g = L / (T * GT_GROUP), gl = min(GT_GROUP, NC - g * GT_GROUP), rem = L - g * T * GT_GROUP;
  const int row = (rem / gl) * 64, col = (g * GT_GROUP + rem % gl) * 64, b = int(tg.z);
  am_block<64>(X + long(b) * st.x_batch, st.x_row, XS + long(b) * st.sums_batch * (a.k / TF_GROUP),
               st.sums_row * (a.k / TF_GROUP), row, min(64, a.rows - row), a.k, st.y_row, col,
               long(b) * st.w_batch + col, W, S, B, Y + long(b) * st.y_batch, tile, sb, sg, lane);
}
