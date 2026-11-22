// Grouped expert GEMMs on NAX / simdgroup MMA (rows >= 12): GPU bucketing + indirect 32x64x32 tiled GEMM.
#define I2X_BM 32u
#define I2X_BN 64u
#define I2X_BK 32u
// 36 bfloats = 72 bytes = 18 banks/row: avoids 4-way bank conflicts across home.y = 0..7 with 8-byte alignment.
#define I2X_BK_PAD 36u
#define I2X_BUCKET_THREADS 256u
#define I2X_GROUPED_THREADS 128u

struct Inkling2ExpBucketArgs {
  uint pairs, experts, gate_up_n_tiles, down_n_tiles;
};

inline bfloat4 i2x_to_bf4(float4 v) {
  return bfloat4(bfloat(v.x), bfloat(v.y), bfloat(v.z), bfloat(v.w));
}

// Bucket active pairs by expert_id; emit indirect args at tile_meta[0..1] and (expert, start, nr, 0) at tile_meta[2+t].
[[kernel, max_total_threads_per_threadgroup(I2X_BUCKET_THREADS)]]
void inkling2_exp_bucket(const device uint* ids [[buffer(0)]], device uint* sorted_pairs [[buffer(1)]],
                         device uint4* tile_meta [[buffer(2)]], constant Inkling2ExpBucketArgs& a [[buffer(3)]],
                         uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint tg_count[128];
  threadgroup atomic_uint tg_cursor[128];
  threadgroup uint sg_pair_totals[8];
  threadgroup uint sg_tile_totals[8];

  if (tid < 128u) {
    atomic_store_explicit(&tg_count[tid], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint p = tid; p < a.pairs; p += I2X_BUCKET_THREADS) {
    const uint e = ids[p];
    if (e < a.experts) {
      atomic_fetch_add_explicit(&tg_count[e], 1u, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint count = (tid < a.experts) ? atomic_load_explicit(&tg_count[tid], memory_order_relaxed) : 0u;
  const uint ntiles = (count + I2X_BM - 1u) / I2X_BM;
  const uint p_exc = simd_prefix_exclusive_sum(count);
  const uint t_exc = simd_prefix_exclusive_sum(ntiles);
  if (lane == 31u) {
    sg_pair_totals[sg] = p_exc + count;
    sg_tile_totals[sg] = t_exc + ntiles;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < 128u) {
    uint p_off = 0u;
    uint t_off = 0u;
    for (uint s = 0u; s < sg; ++s) {
      p_off += sg_pair_totals[s];
      t_off += sg_tile_totals[s];
    }
    const uint start = p_off + p_exc;
    const uint tile_start = t_off + t_exc;
    atomic_store_explicit(&tg_cursor[tid], start, memory_order_relaxed);
    if (tid < a.experts) {
      for (uint r = 0u; r < count; r += I2X_BM) {
        tile_meta[2u + tile_start + (r / I2X_BM)] = uint4(tid, start + r, min(I2X_BM, count - r), 0u);
      }
    }
  }
  if (tid == 0u) {
    const uint total_tiles = sg_tile_totals[0] + sg_tile_totals[1] + sg_tile_totals[2] + sg_tile_totals[3];
    tile_meta[0] = uint4(a.gate_up_n_tiles, total_tiles, 1u, total_tiles);
    tile_meta[1] = uint4(a.down_n_tiles, total_tiles, 1u, total_tiles);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint p = tid; p < a.pairs; p += I2X_BUCKET_THREADS) {
    const uint e = ids[p];
    if (e < a.experts) {
      const uint dst = atomic_fetch_add_explicit(&tg_cursor[e], 1u, memory_order_relaxed);
      sorted_pairs[dst] = p;
    }
  }
}

// Grouped 32x64x32 GEMM (2x2 SGs, 16x32 each): GATE_UP fuses BF16 SwiGLU in regs; DOWN scatters BF16 by pair_id.
template <bool NV, bool GATE_UP>
inline void i2x_grouped_gemm(const device uchar* weight, const device uchar* w_scales,
                             const device float* w_factor, const device bfloat* x,
                             const device uint* sorted_pairs, const device uint4* tile_meta,
                             device bfloat* out, constant Inkling2ExpDotArgs& a,
                             threadgroup bfloat (&tg_A)[I2X_BM][I2X_BK_PAD],
                             threadgroup bfloat (&tg_B)[2][32][I2X_BK_PAD],
                             uint2 tg, uint lane, uint sg) {
  if (tg.y >= tile_meta[0].y) return;

  const uint4 tile = tile_meta[2u + tg.y];
  const uint expert = tile.x;
  const uint first = tile.y;
  const uint nr = tile.z;
  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const uint sg_m0 = sg_m * 16u;
  const uint n0 = tg.x * I2X_BN;
  const uint sg_n0 = n0 + sg_n * 32u;

  // 128 threads collaboratively load 32x32 tg_A (4 threads/row, 8 bfloats/thread).
  const uint tid = sg * 32u + lane;
  const uint a_r = tid >> 2u;
  const uint a_c = (tid & 3u) * 8u;
  const bool a_valid = a_r < nr;
  const uint a_src_row = a_valid ? (GATE_UP ? (sorted_pairs[first + a_r] / a.src_div) : (first + a_r)) : 0u;
  const device bfloat* a_ptr = x + ulong(a_src_row) * a.k + a_c;

  const uint pitch = NV ? (a.k >> 1u) : a.k;
  const uint blocks = a.k / (NV ? 16u : 32u);
  const ulong expert_w_base = ulong(expert) * a.n;
  const float factor = NV ? w_factor[expert] : 1.0f;

  // 2 lanes per weight row (16 weights/lane), 1 pass across all 64 rows (sg_m selects 0..15 vs 16..31 of tg_B[sg_n]).
  const uint r_lane = lane >> 1u;
  const uint c_lane = lane & 1u;
  const uint b_col = c_lane * 16u;
  const uint lr = sg_m0 + r_lane;
  const uint w_row = sg_n0 + lr;
  const bool w_valid = w_row < a.n;
  const ulong row_idx = expert_w_base + w_row;
  const device uchar* w_ptr = weight + row_idx * pitch + (NV ? c_lane * 8u : c_lane * 16u);
  const device uchar* s_ptr = w_scales + row_idx * blocks + (NV ? c_lane : 0u);
  threadgroup bfloat* dst = &tg_B[sg_n][lr][b_col];

  const short2 home = tfp::frag_home(ushort(lane));
  tfp::frag<float> acc_lo = tfp::frag<float>(0.0f);
  tfp::frag<float> acc_hi = tfp::frag<float>(0.0f);

  for (uint kk = 0u; kk < a.k; kk += I2X_BK) {
    if (a_valid) {
      const device bfloat* xp = a_ptr + kk;
      *(threadgroup bfloat4*)(&tg_A[a_r][a_c]) = bfloat4(xp[0], xp[1], xp[2], xp[3]);
      *(threadgroup bfloat4*)(&tg_A[a_r][a_c + 4u]) = bfloat4(xp[4], xp[5], xp[6], xp[7]);
    } else {
      const bfloat4 z = bfloat4(bfloat(0.0f));
      *(threadgroup bfloat4*)(&tg_A[a_r][a_c]) = z;
      *(threadgroup bfloat4*)(&tg_A[a_r][a_c + 4u]) = z;
    }

    if (w_valid) {
      if (NV) {
        const float sc = i2x_e4m3_lut(s_ptr[0]);
        const device uchar4* wp4 = (const device uchar4*)w_ptr;
        const uchar4 p0 = wp4[0];
        const uchar4 p1 = wp4[1];
        const float4 lo0 = float4(kI2xE2m1[p0.x & 15u], kI2xE2m1[p0.x >> 4u],
                                  kI2xE2m1[p0.y & 15u], kI2xE2m1[p0.y >> 4u]);
        const float4 lo1 = float4(kI2xE2m1[p0.z & 15u], kI2xE2m1[p0.z >> 4u],
                                  kI2xE2m1[p0.w & 15u], kI2xE2m1[p0.w >> 4u]);
        const float4 lo2 = float4(kI2xE2m1[p1.x & 15u], kI2xE2m1[p1.x >> 4u],
                                  kI2xE2m1[p1.y & 15u], kI2xE2m1[p1.y >> 4u]);
        const float4 lo3 = float4(kI2xE2m1[p1.z & 15u], kI2xE2m1[p1.z >> 4u],
                                  kI2xE2m1[p1.w & 15u], kI2xE2m1[p1.w >> 4u]);
        *(threadgroup bfloat4*)(dst + 0u) = i2x_to_bf4((lo0 * sc) * factor);
        *(threadgroup bfloat4*)(dst + 4u) = i2x_to_bf4((lo1 * sc) * factor);
        *(threadgroup bfloat4*)(dst + 8u) = i2x_to_bf4((lo2 * sc) * factor);
        *(threadgroup bfloat4*)(dst + 12u) = i2x_to_bf4((lo3 * sc) * factor);
      } else {
        const uchar sb = s_ptr[0];
        const float sc = i2x_mx_scale(sb);
        const device uchar4* wp4 = (const device uchar4*)w_ptr;
        const uchar4 p0 = wp4[0];
        const uchar4 p1 = wp4[1];
        const uchar4 p2 = wp4[2];
        const uchar4 p3 = wp4[3];
        float4 w0 = float4(i2x_e4m3_lut(p0.x), i2x_e4m3_lut(p0.y), i2x_e4m3_lut(p0.z), i2x_e4m3_lut(p0.w));
        float4 w1 = float4(i2x_e4m3_lut(p1.x), i2x_e4m3_lut(p1.y), i2x_e4m3_lut(p1.z), i2x_e4m3_lut(p1.w));
        float4 w2 = float4(i2x_e4m3_lut(p2.x), i2x_e4m3_lut(p2.y), i2x_e4m3_lut(p2.z), i2x_e4m3_lut(p2.w));
        float4 w3 = float4(i2x_e4m3_lut(p3.x), i2x_e4m3_lut(p3.y), i2x_e4m3_lut(p3.z), i2x_e4m3_lut(p3.w));
        if (sb == 0u) {
          w0 *= 0.5f;
          w1 *= 0.5f;
          w2 *= 0.5f;
          w3 *= 0.5f;
        }
        *(threadgroup bfloat4*)(dst + 0u) = i2x_to_bf4(w0 * sc);
        *(threadgroup bfloat4*)(dst + 4u) = i2x_to_bf4(w1 * sc);
        *(threadgroup bfloat4*)(dst + 8u) = i2x_to_bf4(w2 * sc);
        *(threadgroup bfloat4*)(dst + 12u) = i2x_to_bf4(w3 * sc);
      }
    } else {
      const bfloat4 z = bfloat4(bfloat(0.0f));
      *(threadgroup bfloat4*)(dst + 0u) = z;
      *(threadgroup bfloat4*)(dst + 4u) = z;
      *(threadgroup bfloat4*)(dst + 8u) = z;
      *(threadgroup bfloat4*)(dst + 12u) = z;
    }

    w_ptr += NV ? 16u : 32u;
    s_ptr += NV ? 2u : 1u;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    TF_UNROLL
    for (int ks = 0; ks < 32; ks += 16) {
      tfp::frag<bfloat> fa, fb0, fb1;
#ifndef TF_SIMD_FRAGS
      tfp::frag_get_tg(fa, &tg_A[0][0], I2X_BK_PAD, int(sg_m0), ks, home);
      tfp::frag_get_tg(fb0, &tg_B[sg_n][0][0], I2X_BK_PAD, 0, ks, home);
      tfp::frag_get_tg(fb1, &tg_B[sg_n][0][0], I2X_BK_PAD, 16, ks, home);
#else
      tfp::frag_get(fa, &tg_A[0][0], I2X_BK_PAD, int(sg_m0), ks, home);
      tfp::frag_get_t(fb0, &tg_B[sg_n][0][0], I2X_BK_PAD, 0, ks, home);
      tfp::frag_get_t(fb1, &tg_B[sg_n][0][0], I2X_BK_PAD, 16, ks, home);
#endif
      tfp::mma_16x32<false, true>(acc_lo, acc_hi, fa, fb0, fb1);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (GATE_UP) {
    // Both M5 and TF_SIMD_FRAGS place adjacent even/odd columns (gate, up) at acc[2k], acc[2k+1] in the same lane.
    const uint out_col_base = sg_n0 >> 1u;
    TF_UNROLL
    for (short half_idx = 0; half_idx < 2; ++half_idx) {
      const tfp::frag<float> acc = (half_idx == 0) ? acc_lo : acc_hi;
      const uint col_off = out_col_base + uint(half_idx) * 8u;
      TF_UNROLL
      for (short k = 0; k < 4; ++k) {
        const short e = k * 2;
        const uint rr = sg_m0 + uint(home.y) + uint(e >> 2) * 8u;
        const uint cc = col_off + (uint(home.x + TF_COL(e)) >> 1u);
        if (rr < nr && cc < a.out_width) {
          const float g = float(bfloat(acc[e]));
          const float u = float(bfloat(acc[e + 1]));
          out[ulong(first + rr) * a.out_width + cc] = bfloat(i2x_swiglu(g, u, a.gate_cap, a.up_cap));
        }
      }
    }
  } else {
    const uint r0 = sg_m0 + uint(home.y);
    const uint r1 = r0 + 8u;
    const uint p0 = (r0 < nr) ? sorted_pairs[first + r0] : 0u;
    const uint p1 = (r1 < nr) ? sorted_pairs[first + r1] : 0u;
    TF_UNROLL
    for (short half_idx = 0; half_idx < 2; ++half_idx) {
      const tfp::frag<float> acc = (half_idx == 0) ? acc_lo : acc_hi;
      const uint col_off = sg_n0 + uint(half_idx) * 16u;
      TF_UNROLL
      for (short e = 0; e < 8; ++e) {
        const bool hi_row = (e >> 2) != 0;
        const uint rr = hi_row ? r1 : r0;
        const uint cc = col_off + uint(home.x + TF_COL(e));
        if (rr < nr && cc < a.out_width) {
          const uint pid = hi_row ? p1 : p0;
          out[ulong(pid) * a.out_width + cc] = bfloat(acc[e]);
        }
      }
    }
  }
}

#define I2X_GROUPED(symbol, nv, gate_up) \
  [[kernel, max_total_threads_per_threadgroup(I2X_GROUPED_THREADS)]] \
  void symbol(const device uchar* weight [[buffer(0)]], const device uchar* w_scales [[buffer(1)]], \
              const device float* w_factor [[buffer(2)]], const device bfloat* x [[buffer(3)]], \
              const device uint* sorted_pairs [[buffer(4)]], const device uint4* tile_meta [[buffer(5)]], \
              device bfloat* out [[buffer(6)]], constant Inkling2ExpDotArgs& a [[buffer(7)]], \
              uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
              uint sg [[simdgroup_index_in_threadgroup]]) { \
    alignas(16) threadgroup bfloat tg_A[I2X_BM][I2X_BK_PAD]; \
    alignas(16) threadgroup bfloat tg_B[2][32][I2X_BK_PAD]; \
    i2x_grouped_gemm<nv, gate_up>(weight, w_scales, w_factor, x, sorted_pairs, tile_meta, out, a, tg_A, tg_B, tg, lane, sg); \
  }

I2X_GROUPED(inkling2_exp_grouped_gate_up_nvfp4, true, true)
I2X_GROUPED(inkling2_exp_grouped_gate_up_mxfp8, false, true)
I2X_GROUPED(inkling2_exp_grouped_down_nvfp4, true, false)
I2X_GROUPED(inkling2_exp_grouped_down_mxfp8, false, false)
#undef I2X_GROUPED
