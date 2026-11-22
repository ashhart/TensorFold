// Inkling 2 Mini's BF16 dense projections on the tensor units across all widths, after nax.h and units.metal.

struct Ink2DenseArgs {
  uint rows, n, k, ldy;
};

struct Ink2Dense6Args {
  uint rows, k, nq, nk, nv, nr, ns, ng;
};

// Threadgroup (tg_col, tg_row): 16*BANDS rows by 32 columns; 8 simdgroups split K by 8 and add partials in order.
template <uint BANDS, bool DROP_STEP>
inline void ink2_dense_core(device const bfloat* X, device const bfloat* W, device bfloat* Y, uint rows, uint n,
                            uint k, uint ldy, uint tg_row, uint tg_col, uint sg, uint lane,
                            threadgroup float* part) {
  const uint r_base = 16u * BANDS * tg_row;
  const uint col0 = 32u * tg_col;
  if (r_base >= rows || col0 >= n) return;
  const uint steps = k / 128u;
  if (steps == 0u) return;

  const uint rows_rem = min(rows - r_base, 16u * BANDS);
  const uint n_rem = min(n - col0, 32u);
  const uint n_bands = (rows_rem + 15u) / 16u;
  const bool full_n = n_rem == 32u;
  const bool full_n0 = n_rem >= 16u;

  const short2 home = tfp::frag_home(ushort(lane));
  device const bfloat* Xr = X + ulong(r_base) * k;
  device const bfloat* Wc = W + ulong(col0) * k;
  device bfloat* Yr = Y + ulong(r_base) * ldy + col0;

  tfp::frag<float> lo[BANDS], hi[BANDS];
  TF_UNROLL
  for (uint bb = 0u; bb < BANDS; ++bb) {
    lo[bb] = tfp::frag<float>(0.0f);
    hi[bb] = tfp::frag<float>(0.0f);
  }

  const uint b_end = (sg + 1u) * steps - (DROP_STEP ? 1u : 0u);
  for (uint b = sg * steps; b < b_end; ++b) {
    const int k_off = int(16u * b);
    tfp::frag<bfloat> b0, b1;
    if (full_n) {
      tfp::frag_get(b0, Wc, int(k), 0, k_off, home);
      tfp::frag_get(b1, Wc, int(k), 16, k_off, home);
    } else if (full_n0) {
      tfp::frag_get(b0, Wc, int(k), 0, k_off, home);
      tfp::frag_get_in(b1, Wc, int(k), 16, k_off, home, int(n_rem), int(k));
    } else {
      tfp::frag_get_in(b0, Wc, int(k), 0, k_off, home, int(n_rem), int(k));
      b1 = tfp::frag<bfloat>(bfloat(0.0f));
    }
    TF_UNROLL
    for (uint bb = 0u; bb < BANDS; ++bb) {
      if (bb < n_bands) {
        const int r_off = int(16u * bb);
        tfp::frag<bfloat> fa;
        if (16u * (bb + 1u) <= rows_rem) {
          tfp::frag_get(fa, Xr, int(k), r_off, k_off, home);
        } else {
          tfp::frag_get_in(fa, Xr, int(k), r_off, k_off, home, int(rows_rem), int(k));
        }
        ink2_mma(lo[bb], hi[bb], fa, b0, b1);
      }
    }
  }

  TF_UNROLL
  for (uint bb = 0u; bb < BANDS; ++bb) {
    if (bb < n_bands) {
      const int r_off = int(16u * bb);
      ink2_part_put(part, lo[bb], hi[bb], sg, lane);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0u) {
        ink2_part_sum(part, lo[bb], hi[bb], lane);
        if (full_n && 16u * (bb + 1u) <= rows_rem) {
          tfp::frag_put(lo[bb], Yr, int(ldy), r_off, 0, home);
          tfp::frag_put(hi[bb], Yr, int(ldy), r_off, 16, home);
        } else {
          tfp::frag_put_in(lo[bb], Yr, int(ldy), r_off, 0, home, int(rows_rem), int(n_rem));
          tfp::frag_put_in(hi[bb], Yr, int(ldy), r_off, 16, home, int(rows_rem), int(n_rem));
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

// Fused attention input projections (wq, wk, wv, wr, sink, gate) sharing the exact same reduction core.
template <uint BANDS, bool DROP_STEP>
inline void ink2_dense6(device const bfloat* xq, device const bfloat* xk, device const bfloat* xv,
                        device const bfloat* wq, device const bfloat* wk, device const bfloat* wv,
                        device const bfloat* wr, device const bfloat* ws, device const bfloat* wg,
                        device bfloat* yq, device bfloat* yk, device bfloat* yv,
                        device bfloat* yr, device bfloat* ys, device bfloat* yg,
                        constant Ink2Dense6Args& a, uint2 tg, uint sg, uint lane, threadgroup float* part) {
  const uint tq = (a.nq + 31u) / 32u;
  const uint tk = (a.nk + 31u) / 32u;
  const uint tv = (a.nv + 31u) / 32u;
  const uint tr = (a.nr + 31u) / 32u;
  const uint ts = (a.ns + 31u) / 32u;
  const uint tg_n = (a.ng + 31u) / 32u;
  uint tx = tg.x;
  device const bfloat* px;
  device const bfloat* pw;
  device bfloat* py;
  uint pn;
  if (tx < tq) {
    px = xq;
    pw = wq;
    py = yq;
    pn = a.nq;
  } else if ((tx -= tq) < tk) {
    px = xk;
    pw = wk;
    py = yk;
    pn = a.nk;
  } else if ((tx -= tk) < tv) {
    px = xv;
    pw = wv;
    py = yv;
    pn = a.nv;
  } else if ((tx -= tv) < tr) {
    px = xq;
    pw = wr;
    py = yr;
    pn = a.nr;
  } else if ((tx -= tr) < ts) {
    px = xq;
    pw = ws;
    py = ys;
    pn = a.ns;
  } else if ((tx -= ts) < tg_n) {
    px = xq;
    pw = wg;
    py = yg;
    pn = a.ng;
  } else {
    return;
  }
  ink2_dense_core<BANDS, DROP_STEP>(px, pw, py, a.rows, pn, a.k, pn, tg.y, tx, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_dense_units_16(
    device const bfloat* X [[buffer(0)]], device const bfloat* W [[buffer(1)]], device bfloat* Y [[buffer(2)]],
    constant Ink2DenseArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4096];
  ink2_dense_core<1u, false>(X, W, Y, a.rows, a.n, a.k, a.ldy, tg.y, tg.x, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_dense_units_32(
    device const bfloat* X [[buffer(0)]], device const bfloat* W [[buffer(1)]], device bfloat* Y [[buffer(2)]],
    constant Ink2DenseArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4096];
  ink2_dense_core<2u, false>(X, W, Y, a.rows, a.n, a.k, a.ldy, tg.y, tg.x, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_dense_units_64(
    device const bfloat* X [[buffer(0)]], device const bfloat* W [[buffer(1)]], device bfloat* Y [[buffer(2)]],
    constant Ink2DenseArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4096];
  ink2_dense_core<4u, false>(X, W, Y, a.rows, a.n, a.k, a.ldy, tg.y, tg.x, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_dense6_units_16(
    device const bfloat* xq [[buffer(0)]], device const bfloat* xk [[buffer(1)]],
    device const bfloat* xv [[buffer(2)]], device const bfloat* wq [[buffer(3)]],
    device const bfloat* wk [[buffer(4)]], device const bfloat* wv [[buffer(5)]],
    device const bfloat* wr [[buffer(6)]], device const bfloat* ws [[buffer(7)]],
    device const bfloat* wg [[buffer(8)]], device bfloat* yq [[buffer(9)]], device bfloat* yk [[buffer(10)]],
    device bfloat* yv [[buffer(11)]], device bfloat* yr [[buffer(12)]], device bfloat* ys [[buffer(13)]],
    device bfloat* yg [[buffer(14)]], constant Ink2Dense6Args& a [[buffer(15)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4096];
  ink2_dense6<1u, false>(xq, xk, xv, wq, wk, wv, wr, ws, wg, yq, yk, yv, yr, ys, yg, a, tg, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_dense6_units_32(
    device const bfloat* xq [[buffer(0)]], device const bfloat* xk [[buffer(1)]],
    device const bfloat* xv [[buffer(2)]], device const bfloat* wq [[buffer(3)]],
    device const bfloat* wk [[buffer(4)]], device const bfloat* wv [[buffer(5)]],
    device const bfloat* wr [[buffer(6)]], device const bfloat* ws [[buffer(7)]],
    device const bfloat* wg [[buffer(8)]], device bfloat* yq [[buffer(9)]], device bfloat* yk [[buffer(10)]],
    device bfloat* yv [[buffer(11)]], device bfloat* yr [[buffer(12)]], device bfloat* ys [[buffer(13)]],
    device bfloat* yg [[buffer(14)]], constant Ink2Dense6Args& a [[buffer(15)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4096];
  ink2_dense6<2u, false>(xq, xk, xv, wq, wk, wv, wr, ws, wg, yq, yk, yv, yr, ys, yg, a, tg, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_dense6_units_64(
    device const bfloat* xq [[buffer(0)]], device const bfloat* xk [[buffer(1)]],
    device const bfloat* xv [[buffer(2)]], device const bfloat* wq [[buffer(3)]],
    device const bfloat* wk [[buffer(4)]], device const bfloat* wv [[buffer(5)]],
    device const bfloat* wr [[buffer(6)]], device const bfloat* ws [[buffer(7)]],
    device const bfloat* wg [[buffer(8)]], device bfloat* yq [[buffer(9)]], device bfloat* yk [[buffer(10)]],
    device bfloat* yv [[buffer(11)]], device bfloat* yr [[buffer(12)]], device bfloat* ys [[buffer(13)]],
    device bfloat* yg [[buffer(14)]], constant Ink2Dense6Args& a [[buffer(15)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4096];
  ink2_dense6<4u, false>(xq, xk, xv, wq, wk, wv, wr, ws, wg, yq, yk, yv, yr, ys, yg, a, tg, sg, lane, part);
}
