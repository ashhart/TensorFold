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
              if (full_n) {
                *(device bfloat4*)(Yr + rr * int(ldy) + cc) = bfloat4(bfloat(v[0]), bfloat(v[1]), bfloat(v[2]), bfloat(v[3]));
              } else {
                for (int j = 0; j < 4; ++j)
                  if (cc + j < int(n_rem)) Yr[rr * int(ldy) + cc + j] = bfloat(v[j]);
              }
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

// Exact non-transposed 16x32x16 BF16->FP32 tensor-unit op without relaxed precision.
inline void ink2_mma_nt(thread tfp::frag<float>& lo, thread tfp::frag<float>& hi, thread const tfp::frag<bfloat>& a,
                        thread const tfp::frag<bfloat>& b0, thread const tfp::frag<bfloat>& b1) {
  using namespace mpp::tensor_ops;
  constexpr auto shape = matmul2d_descriptor(16, 32, 16, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<shape, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto right = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
                                                            metal::remove_addrspace_t<decltype(right)>, float>();
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    left[e] = a[e];
    right[e] = b0[e];
    right[8 + e] = b1[e];
    acc[e] = lo[e];
    acc[8 + e] = hi[e];
  }
  op.run(left, right, acc);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    lo[e] = acc[e];
    hi[e] = acc[8 + e];
  }
}

// Router scores [rows, 2304] x [128, 2304]^T -> FP32 [rows, 128] via BANDS*16 x 32 M5 NAX tiles, 8 KiB scratch.
template <uint BANDS, bool DROP_STEP>
inline void ink2_route_core(device const bfloat* X, device const bfloat* W, device float* Y,
                            constant Ink2DenseArgs& a, uint2 tg, uint sg, uint lane, threadgroup float* part) {
  const uint r_base = 16u * BANDS * tg.y, col0 = 32u * tg.x;
  if (r_base >= a.rows || col0 >= a.n) return;
  const uint steps = a.k / 128u;
  const uint rows_rem = min(a.rows - r_base, 16u * BANDS);
  const uint n_bands = (rows_rem + 15u) / 16u;
  const short2 home = tfp::frag_home(ushort(lane));
  device const bfloat* Xr = X + ulong(r_base) * a.k;
  device const bfloat* Wc = W + ulong(col0) * a.k;
  device float* Yr = Y + ulong(r_base) * a.ldy + col0;
  tfp::frag<float> lo[BANDS], hi[BANDS];
  TF_UNROLL
  for (uint bb = 0u; bb < BANDS; ++bb) { lo[bb] = tfp::frag<float>(0.0f); hi[bb] = tfp::frag<float>(0.0f); }
  const uint b_end = (sg + 1u) * steps - (DROP_STEP ? 1u : 0u);
  for (uint b = sg * steps; b < b_end; ++b) {
    const int k_off = int(16u * b);
    tfp::frag<bfloat> b0, b1;
    ink2_frag_get_bf4(b0, Wc, int(a.k), 0, k_off, home);
    ink2_frag_get_bf4(b1, Wc, int(a.k), 16, k_off, home);
    TF_UNROLL
    for (uint bb = 0u; bb < BANDS; ++bb) if (bb < n_bands) {
      tfp::frag<bfloat> fa;
      if (16u * (bb + 1u) <= rows_rem) ink2_frag_get_bf4(fa, Xr, int(a.k), int(16u * bb), k_off, home);
      else ink2_frag_get_rows_bf4(fa, Xr, int(a.k), int(16u * bb), k_off, home, int(rows_rem));
      ink2_mma(lo[bb], hi[bb], fa, b0, b1);
    }
  }
  TF_UNROLL
  for (uint bb = 0u; bb < BANDS; ++bb) if (bb < n_bands) {
    const bool has_q1 = rows_rem > 16u * bb + 8u;
    for (uint q = 0u; q < (has_q1 ? 2u : 1u); ++q) {
      ink2_part_put_half(part, lo[bb], hi[bb], q, sg, lane);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0u) {
        ink2_part_sum_half(part, lo[bb], hi[bb], q, lane);
        const int rr = int(16u * bb) + int(home.y) + int(q) * 8;
        if (rr < int(rows_rem)) {
          const uint off = 4u * q;
          *(device float4*)(Yr + rr * int(a.ldy) + int(home.x)) =
              float4(lo[bb][off], lo[bb][off + 1u], lo[bb][off + 2u], lo[bb][off + 3u]);
          *(device float4*)(Yr + rr * int(a.ldy) + 16 + int(home.x)) =
              float4(hi[bb][off], hi[bb][off + 1u], hi[bb][off + 2u], hi[bb][off + 3u]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_route_scores_units_16(
    device const bfloat* X [[buffer(0)]], device const bfloat* W [[buffer(1)]], device float* Y [[buffer(2)]],
    constant Ink2DenseArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[2048];
  ink2_route_core<1u, false>(X, W, Y, a, tg, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_route_scores_units_64(
    device const bfloat* X [[buffer(0)]], device const bfloat* W [[buffer(1)]], device float* Y [[buffer(2)]],
    constant Ink2DenseArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[2048];
  ink2_route_core<4u, false>(X, W, Y, a, tg, sg, lane, part);
}

struct Ink2LoopBlendArgs { uint rows, stride; };

// Loop gate blend [rows, 64] x [2304, 64]^T for forget+gain plus exact four-step FP32 blend, 0 B TG memory.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_loop_blend_units(
    device const float* h [[buffer(0)]], device const float* m [[buffer(1)]],
    device const bfloat* zeta [[buffer(2)]], device const bfloat* forget [[buffer(3)]],
    device const bfloat* gain [[buffer(4)]], device float* out [[buffer(5)]],
    constant Ink2LoopBlendArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint r_base = 16u * tg.y, col0 = 32u * tg.x;
  if (r_base >= a.rows || col0 >= 2304u) return;
  const uint rows_rem = min(a.rows - r_base, 16u);
  const short2 home = tfp::frag_home(ushort(lane));
  device const bfloat* zr = zeta + ulong(r_base) * 64u;
  device const bfloat* fc = forget + ulong(col0) * 64u;
  device const bfloat* gc = gain + ulong(col0) * 64u;
  tfp::frag<float> flo = tfp::frag<float>(0.0f), fhi = tfp::frag<float>(0.0f);
  tfp::frag<float> glo = tfp::frag<float>(0.0f), ghi = tfp::frag<float>(0.0f);
  TF_UNROLL
  for (uint b = 0u; b < 4u; ++b) {
    const int k_off = int(16u * b);
    tfp::frag<bfloat> zf, w0, w1;
    if (rows_rem == 16u) ink2_frag_get_bf4(zf, zr, 64, 0, k_off, home);
    else ink2_frag_get_rows_bf4(zf, zr, 64, 0, k_off, home, int(rows_rem));
    ink2_frag_get_bf4(w0, fc, 64, 0, k_off, home);
    ink2_frag_get_bf4(w1, fc, 64, 16, k_off, home);
    ink2_mma(flo, fhi, zf, w0, w1);
    ink2_frag_get_bf4(w0, gc, 64, 0, k_off, home);
    ink2_frag_get_bf4(w1, gc, 64, 16, k_off, home);
    ink2_mma(glo, ghi, zf, w0, w1);
  }
  for (uint q = 0u; q < 2u; ++q) {
    const uint rr = r_base + uint(home.y) + 8u * q;
    if (rr >= a.rows) continue;
    const uint off = 4u * q;
    for (uint side = 0u; side < 2u; ++side) {
      const uint cc = col0 + 16u * side + uint(home.x);
      const float4 fv = side == 0u ? float4(flo[off], flo[off + 1u], flo[off + 2u], flo[off + 3u])
                                   : float4(fhi[off], fhi[off + 1u], fhi[off + 2u], fhi[off + 3u]);
      const float4 gv = side == 0u ? float4(glo[off], glo[off + 1u], glo[off + 2u], glo[off + 3u])
                                   : float4(ghi[off], ghi[off + 1u], ghi[off + 2u], ghi[off + 3u]);
      const float4 hv = *(device const float4*)(h + ulong(rr) * 2304u + cc);
      const float4 mv = *(device const float4*)(m + ulong(rr) * a.stride + cc);
      float4 ov;
      for (uint j = 0u; j < 4u; ++j) {
        const float f = precise::tanh(fv[j]);
        const float g = precise::tanh(gv[j]);
        const float keep = 1.0f - f;
        const float left = keep * hv[j];
        const float right = g * mv[j];
        ov[j] = left + right;
      }
      *(device float4*)(out + ulong(rr) * 2304u + cc) = ov;
    }
  }
}

struct Ink2RelTableArgs { uint rows; };

// Relative bias table [rows, 16] x [16, 1024] -> BF16 [rows, 1024] via single-op M5 NAX tiles, 0 B TG memory.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_rel_table_units(
    device const bfloat* rel [[buffer(0)]], device const bfloat* proj [[buffer(1)]],
    device bfloat* tab [[buffer(2)]], constant Ink2RelTableArgs& a [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint r_base = 16u * tg.y, col_base = 256u * tg.x;
  if (r_base >= a.rows || col_base >= 1024u) return;
  const uint rows_rem = min(a.rows - r_base, 16u);
  const short2 home = tfp::frag_home(ushort(lane));
  tfp::frag<bfloat> rf;
  if (rows_rem == 16u) ink2_frag_get_bf4(rf, rel + ulong(r_base) * 16u, 16, 0, 0, home);
  else ink2_frag_get_rows_bf4(rf, rel + ulong(r_base) * 16u, 16, 0, 0, home, int(rows_rem));
  const uint r0 = r_base + uint(home.y), r1 = r0 + 8u;
  const bool live0 = r0 < a.rows, live1 = r1 < a.rows;
  for (uint t = 0u; t < 8u; ++t) {
    const uint col0 = col_base + 32u * t;
    tfp::frag<bfloat> p0, p1;
    ink2_frag_get_bf4(p0, proj, 1024, 0, int(col0), home);
    ink2_frag_get_bf4(p1, proj, 1024, 0, int(col0 + 16u), home);
    tfp::frag<float> lo = tfp::frag<float>(0.0f), hi = tfp::frag<float>(0.0f);
    ink2_mma_nt(lo, hi, rf, p0, p1);
    if (live0) {
      *(device bfloat4*)(tab + ulong(r0) * 1024u + col0 + uint(home.x)) =
          bfloat4(bfloat(lo[0]), bfloat(lo[1]), bfloat(lo[2]), bfloat(lo[3]));
      *(device bfloat4*)(tab + ulong(r0) * 1024u + col0 + 16u + uint(home.x)) =
          bfloat4(bfloat(hi[0]), bfloat(hi[1]), bfloat(hi[2]), bfloat(hi[3]));
    }
    if (live1) {
      *(device bfloat4*)(tab + ulong(r1) * 1024u + col0 + uint(home.x)) =
          bfloat4(bfloat(lo[4]), bfloat(lo[5]), bfloat(lo[6]), bfloat(lo[7]));
      *(device bfloat4*)(tab + ulong(r1) * 1024u + col0 + 16u + uint(home.x)) =
          bfloat4(bfloat(hi[4]), bfloat(hi[5]), bfloat(hi[6]), bfloat(hi[7]));
    }
  }
}
