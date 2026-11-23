// Inkling 2 Mini's BF16 dense projections on the tensor units across all widths, after nax.h and units.metal.

struct Ink2DenseArgs {
  uint rows, n, k, ldy;
};

struct Ink2Dense6Args {
  uint rows, k, nq, nk, nv, nr, ns, ng;
};

// Vectorized 8-byte bfloat4 fragment loads (k is always a multiple of 128, home.x is a multiple of 4).
inline void ink2_frag_get_bf4(thread tfp::frag<bfloat>& f, device const bfloat* p, int ld, int r, int c, short2 home) {
  device const bfloat* q = p + (r + home.y) * ld + (c + home.x);
  f = tfp::frag<bfloat>(*(device const bfloat4*)q, *(device const bfloat4*)(q + 8 * ld));
}

inline void ink2_frag_get_rows_bf4(thread tfp::frag<bfloat>& f, device const bfloat* p, int ld, int r, int c,
                                   short2 home, int nr) {
  const bfloat z1 = bfloat(0.0f);
  const bfloat4 z = bfloat4(z1, z1, z1, z1);
  const int r0 = r + home.y, r1 = r0 + 8;
  const bfloat4 v0 = r0 < nr ? *(device const bfloat4*)(p + r0 * ld + c + home.x) : z;
  const bfloat4 v1 = r1 < nr ? *(device const bfloat4*)(p + r1 * ld + c + home.x) : z;
  f = tfp::frag<bfloat>(v0, v1);
}

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
      ink2_frag_get_bf4(b0, Wc, int(k), 0, k_off, home);
      ink2_frag_get_bf4(b1, Wc, int(k), 16, k_off, home);
    } else if (full_n0) {
      ink2_frag_get_bf4(b0, Wc, int(k), 0, k_off, home);
      ink2_frag_get_rows_bf4(b1, Wc, int(k), 16, k_off, home, int(n_rem));
    } else {
      ink2_frag_get_rows_bf4(b0, Wc, int(k), 0, k_off, home, int(n_rem));
      b1 = tfp::frag<bfloat>(bfloat(0.0f));
    }
    TF_UNROLL
    for (uint bb = 0u; bb < BANDS; ++bb) {
      if (bb < n_bands) {
        const int r_off = int(16u * bb);
        tfp::frag<bfloat> fa;
        if (16u * (bb + 1u) <= rows_rem) {
          ink2_frag_get_bf4(fa, Xr, int(k), r_off, k_off, home);
        } else {
          ink2_frag_get_rows_bf4(fa, Xr, int(k), r_off, k_off, home, int(rows_rem));
        }
        ink2_mma(lo[bb], hi[bb], fa, b0, b1);
      }
    }
  }

  TF_UNROLL
  for (uint bb = 0u; bb < BANDS; ++bb) {
    if (bb < n_bands) {
      const int r_off = int(16u * bb);
      const bool has_q1 = rows_rem > 16u * bb + 8u;
      for (uint q = 0u; q < (has_q1 ? 2u : 1u); ++q) {
        ink2_part_put_half(part, lo[bb], hi[bb], q, sg, lane);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
          ink2_part_sum_half(part, lo[bb], hi[bb], q, lane);
          const int rr = r_off + int(home.y) + int(q) * 8;
          if (rr < int(rows_rem)) {
            const uint off = 4u * q;
            for (uint side = 0u; side < 2u; ++side) {
              const int cc = int(16u * side) + int(home.x);
              const float4 v = side == 0u ? float4(lo[bb][off], lo[bb][off + 1u], lo[bb][off + 2u], lo[bb][off + 3u])
                                          : float4(hi[bb][off], hi[bb][off + 1u], hi[bb][off + 2u], hi[bb][off + 3u]);
              for (int j = 0; j < 4; ++j)
                if (cc + j < int(n_rem)) Yr[rr * int(ldy) + cc + j] = bfloat(v[j]);
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
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
  threadgroup float part[2048];
  ink2_dense_core<1u, false>(X, W, Y, a.rows, a.n, a.k, a.ldy, tg.y, tg.x, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_dense_units_32(
    device const bfloat* X [[buffer(0)]], device const bfloat* W [[buffer(1)]], device bfloat* Y [[buffer(2)]],
    constant Ink2DenseArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[2048];
  ink2_dense_core<2u, false>(X, W, Y, a.rows, a.n, a.k, a.ldy, tg.y, tg.x, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_dense_units_64(
    device const bfloat* X [[buffer(0)]], device const bfloat* W [[buffer(1)]], device bfloat* Y [[buffer(2)]],
    constant Ink2DenseArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[2048];
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
  threadgroup float part[2048];
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
  threadgroup float part[2048];
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
  threadgroup float part[2048];
  ink2_dense6<4u, false>(xq, xk, xv, wq, wk, wv, wr, ws, wg, yq, yk, yv, yr, ys, yg, a, tg, sg, lane, part);
}
