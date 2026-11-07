inline void tf_fp8_expand4(threadgroup half* wx, uint tid, uint u) {
  const half2 a = as_type<half2>(((u & 0x007F007Fu) << 7) | ((u << 8) & 0x80008000u));
  const half2 b = as_type<half2>(((u >> 1) & 0x3F803F80u) | (u & 0x80008000u));
  const half2 k256 = half2(256.0h, 256.0h);
  threadgroup half2* o = (threadgroup half2*)(wx + 4u * tid);
  o[0] = half2(a.x, b.x) * k256;
  o[1] = half2(a.y, b.y) * k256;
}

inline void tf_fp8_expand8(threadgroup half* wx, uint tid, uint2 q) {
  const half2 k256 = half2(256.0h, 256.0h);
  const uint u0 = q.x, u1 = q.y;
  const half2 a0 = as_type<half2>(((u0 & 0x007F007Fu) << 7) | ((u0 << 8) & 0x80008000u));
  const half2 b0 = as_type<half2>(((u0 >> 1) & 0x3F803F80u) | (u0 & 0x80008000u));
  const half2 a1 = as_type<half2>(((u1 & 0x007F007Fu) << 7) | ((u1 << 8) & 0x80008000u));
  const half2 b1 = as_type<half2>(((u1 >> 1) & 0x3F803F80u) | (u1 & 0x80008000u));
  threadgroup half2* o = (threadgroup half2*)(wx + 8u * tid);
  o[0] = half2(a0.x, b0.x) * k256;
  o[1] = half2(a0.y, b0.y) * k256;
  o[2] = half2(a1.x, b1.x) * k256;
  o[3] = half2(a1.y, b1.y) * k256;
}

inline void tf_i8_expand4(threadgroup half* wx, uint tid, uint u) {
  const int b0 = int(u << 24) >> 24, b1 = int(u << 16) >> 24, b2 = int(u << 8) >> 24, b3 = int(u) >> 24;
  threadgroup half2* o = (threadgroup half2*)(wx + 4u * tid);
  o[0] = half2(half(float(b0)), half(float(b1)));
  o[1] = half2(half(float(b2)), half(float(b3)));
}

inline void tf_i8_expand8(threadgroup half* wx, uint tid, uint2 q) {
  const uint u0 = q.x, u1 = q.y;
  const int a0 = int(u0 << 24) >> 24, a1 = int(u0 << 16) >> 24, a2 = int(u0 << 8) >> 24, a3 = int(u0) >> 24;
  const int b0 = int(u1 << 24) >> 24, b1 = int(u1 << 16) >> 24, b2 = int(u1 << 8) >> 24, b3 = int(u1) >> 24;
  threadgroup half2* o = (threadgroup half2*)(wx + 8u * tid);
  o[0] = half2(half(float(a0)), half(float(a1)));
  o[1] = half2(half(float(a2)), half(float(a3)));
  o[2] = half2(half(float(b0)), half(float(b1)));
  o[3] = half2(half(float(b2)), half(float(b3)));
}

template <bool WIDE, typename O>
inline void tf_exl3_prefill_body_fp8(threadgroup half* wx, const device half* x, const device uchar* w8,
                                     const device float* scales, device O* sums, constant TfExl3Args& a,
                                     uint lane, uint sg, uint tid, uint3 tg) {
  constexpr uint RG = 8u, CG = 1u;
  const uint trows = RG * 16u;
  const uint rb = tg.y * trows, nb = tg.x * 64u;
  if (rb >= a.rows) return;
  const uint rows_here = min(trows, a.rows - rb);
  const uint cols_here = min(64u, a.n - nb);
  const short2 home = frag_home(ushort(lane));
  const uint nb64 = a.n >> 6, block = nb >> 6;
  const uint sgr = sg % RG, sgc = sg / RG;
  frag<float> lo0, hi0, lo1, hi1;
  lo0 = frag<float>(0); hi0 = frag<float>(0); lo1 = frag<float>(0); hi1 = frag<float>(0);
  if (WIDE) {
    if (tid < 128u) {
      const uint2 q = *reinterpret_cast<const device uint2*>(w8 + ((block) << 10) + 8u * tid);
      if (a.halves == 0u) tf_i8_expand8(wx, tid, q); else tf_fp8_expand8(wx, tid, q);
    }
  } else if (tid < 256u) {
    const uint u = *reinterpret_cast<const device uint*>(w8 + ((block) << 10) + 4u * tid);
    if (a.halves == 0u) tf_i8_expand4(wx, tid, u); else tf_fp8_expand4(wx, tid, u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint cur_buf = 0;
  for (uint kb = 0; kb < a.k; kb += 16u) {
    const uint nkb = kb + 16u;
    uint nu = 0u;
    uint2 nq = uint2(0u);
    if (nkb < a.k) {
      if (WIDE) {
        if (tid < 128u) nq = *reinterpret_cast<const device uint2*>(w8 + (((nkb >> 4) * nb64 + block) << 10) + 8u * tid);
      } else if (tid < 256u) {
        nu = *reinterpret_cast<const device uint*>(w8 + (((nkb >> 4) * nb64 + block) << 10) + 4u * tid);
      }
    }
    frag<half> av;
    tf_frag_get_dev4(av, x, int(a.x_stride), int(rb + sgr * 16u), int(kb), home, int(rb + rows_here));
    threadgroup half* cur = wx + cur_buf * 1024u;
    frag<half> b0, b1;
    const int wc = int(sgc * 64u);
    tf_frag_get_tg4(b0, cur, 64, 0, wc, home);
    tf_frag_get_tg4(b1, cur, 64, 0, wc + 16, home);
    mma_16x32<false, false>(lo0, hi0, av, b0, b1);
    tf_frag_get_tg4(b0, cur, 64, 0, wc + 32, home);
    tf_frag_get_tg4(b1, cur, 64, 0, wc + 48, home);
    mma_16x32<false, false>(lo1, hi1, av, b0, b1);
    if (nkb < a.k) {
      if (WIDE) {
        if (tid < 128u) {
          if (a.halves == 0u) tf_i8_expand8(wx + (cur_buf ^ 1u) * 1024u, tid, nq);
          else tf_fp8_expand8(wx + (cur_buf ^ 1u) * 1024u, tid, nq);
        }
      } else if (tid < 256u) {
        if (a.halves == 0u) tf_i8_expand4(wx + (cur_buf ^ 1u) * 1024u, tid, nu);
        else tf_fp8_expand4(wx + (cur_buf ^ 1u) * 1024u, tid, nu);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cur_buf ^= 1u;
  }
  const int c0 = int(nb + sgc * 64u);
  const int r0 = int(rb + sgr * 16u);
  // A bf16 destination is the caller's final output (the pre-rotated pack has no epilogue transform), so it carries
  // the call's y stride; an f32 destination is the internal sums tile and is contiguous at n. sizeof(float) == 4
  // separates the two instantiations.
  const int out_ld = (sizeof(O) == 4) ? int(a.n) : int(a.y_stride);
  if (scales != nullptr) {
    // The per-output-row E4M3 scale. One multiply an output, applied to the finished accumulator, so the MMA ran on
    // the codes alone and the row scale stays exact.
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      lo0[e] *= scales[c0 + home.x + (e & 3)];
      hi0[e] *= scales[c0 + 16 + home.x + (e & 3)];
      lo1[e] *= scales[c0 + 32 + home.x + (e & 3)];
      hi1[e] *= scales[c0 + 48 + home.x + (e & 3)];
    }
  }
  frag_put_in(lo0, sums, out_ld, r0, c0, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi0, sums, out_ld, r0, c0 + 16, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo1, sums, out_ld, r0, c0 + 32, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi1, sums, out_ld, r0, c0 + 48, home, int(rb + rows_here), int(nb + cols_here));
}

// THE BLOCK-SCALED INT8 BODY for the pre-rotated pack. W_target's columns are quantized with one int8 grid per 128-k
// block (scales[kb * n + col], f32). The block scale is folded into the half weight at the expand step (the exact
// half product of the int8 code and the half scale), so the MMA loop keeps the shipped body's single accumulator and
// geometry.
inline void tf_i8_expand4s(threadgroup half* wx, uint tid, uint u, half2 s01, half2 s23) {
  const int b0 = int(u << 24) >> 24, b1 = int(u << 16) >> 24, b2 = int(u << 8) >> 24, b3 = int(u) >> 24;
  threadgroup half2* o = (threadgroup half2*)(wx + 4u * tid);
  o[0] = half2(half(float(b0)), half(float(b1))) * s01;
  o[1] = half2(half(float(b2)), half(float(b3))) * s23;
}

inline void tf_i8_expand8s(threadgroup half* wx, uint tid, uint2 q, half2 s01, half2 s23, half2 s45, half2 s67) {
  const uint u0 = q.x, u1 = q.y;
  const int a0 = int(u0 << 24) >> 24, a1 = int(u0 << 16) >> 24, a2 = int(u0 << 8) >> 24, a3 = int(u0) >> 24;
  const int b0 = int(u1 << 24) >> 24, b1 = int(u1 << 16) >> 24, b2 = int(u1 << 8) >> 24, b3 = int(u1) >> 24;
  threadgroup half2* o = (threadgroup half2*)(wx + 8u * tid);
  o[0] = half2(half(float(a0)), half(float(a1))) * s01;
  o[1] = half2(half(float(a2)), half(float(a3))) * s23;
  o[2] = half2(half(float(b0)), half(float(b1))) * s45;
  o[3] = half2(half(float(b2)), half(float(b3))) * s67;
}

template <bool WIDE>
inline void tf_exl3_prefill_body_i8b(threadgroup half* wx, const device half* x, const device uchar* w8,
                                     const device float* bscales, device float* sums, constant TfExl3Args& a,
                                     uint lane, uint sg, uint tid, uint3 tg) {
  constexpr uint RG = 8u;
  const uint trows = RG * 16u;
  const uint rb = tg.y * trows, nb = tg.x * 64u;
  if (rb >= a.rows) return;
  const uint rows_here = min(trows, a.rows - rb);
  const uint cols_here = min(64u, a.n - nb);
  const short2 home = frag_home(ushort(lane));
  const uint nb64 = a.n >> 6, block = nb >> 6;
  const uint sgr = sg % RG, sgc = sg / RG;
  // The four (narrow) or eight (wide) output columns this thread's packed bytes cover, constant across k.
  const uint n0 = WIDE ? ((tid & 7u) * 8u) : ((tid & 15u) * 4u);
  half2 s01, s23, s45, s67;
  const device float* bs0 = bscales + nb;
  s01 = half2(half(bs0[n0]), half(bs0[n0 + 1u]));
  s23 = half2(half(bs0[n0 + 2u]), half(bs0[n0 + 3u]));
  s45 = half2(half(bs0[n0 + 4u]), half(bs0[n0 + 5u]));
  s67 = half2(half(bs0[n0 + 6u]), half(bs0[n0 + 7u]));
  frag<float> lo0, hi0, lo1, hi1;
  lo0 = frag<float>(0); hi0 = frag<float>(0); lo1 = frag<float>(0); hi1 = frag<float>(0);
  if (WIDE) {
    if (tid < 128u) {
      const uint2 q = *reinterpret_cast<const device uint2*>(w8 + ((block) << 10) + 8u * tid);
      tf_i8_expand8s(wx, tid, q, s01, s23, s45, s67);
    }
  } else if (tid < 256u) {
    const uint u = *reinterpret_cast<const device uint*>(w8 + ((block) << 10) + 4u * tid);
    tf_i8_expand4s(wx, tid, u, s01, s23);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint cur_buf = 0;
  for (uint kb = 0; kb < a.k; kb += 16u) {
    const uint nkb = kb + 16u;
    uint nu = 0u;
    uint2 nq = uint2(0u);
    if (nkb < a.k) {
      if (WIDE) {
        if (tid < 128u) nq = *reinterpret_cast<const device uint2*>(w8 + (((nkb >> 4) * nb64 + block) << 10) + 8u * tid);
      } else if (tid < 256u) {
        nu = *reinterpret_cast<const device uint*>(w8 + (((nkb >> 4) * nb64 + block) << 10) + 4u * tid);
      }
    }
    frag<half> av;
    tf_frag_get_dev4(av, x, int(a.k), int(rb + sgr * 16u), int(kb), home, int(rb + rows_here));
    threadgroup half* cur = wx + cur_buf * 1024u;
    frag<half> b0, b1;
    const int wc = int(sgc * 64u);
    tf_frag_get_tg4(b0, cur, 64, 0, wc, home);
    tf_frag_get_tg4(b1, cur, 64, 0, wc + 16, home);
    mma_16x32<false, false>(lo0, hi0, av, b0, b1);
    tf_frag_get_tg4(b0, cur, 64, 0, wc + 32, home);
    tf_frag_get_tg4(b1, cur, 64, 0, wc + 48, home);
    mma_16x32<false, false>(lo1, hi1, av, b0, b1);
    if (nkb < a.k) {
      // One scale load per 128-k block, reused by its eight 16-k expands.
      if ((nkb & 127u) == 0u) {
        const device float* bs = bscales + ulong(nkb >> 7) * a.n + nb;
        s01 = half2(half(bs[n0]), half(bs[n0 + 1u]));
        s23 = half2(half(bs[n0 + 2u]), half(bs[n0 + 3u]));
        s45 = half2(half(bs[n0 + 4u]), half(bs[n0 + 5u]));
        s67 = half2(half(bs[n0 + 6u]), half(bs[n0 + 7u]));
      }
      if (WIDE) {
        if (tid < 128u) tf_i8_expand8s(wx + (cur_buf ^ 1u) * 1024u, tid, nq, s01, s23, s45, s67);
      } else if (tid < 256u) {
        tf_i8_expand4s(wx + (cur_buf ^ 1u) * 1024u, tid, nu, s01, s23);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cur_buf ^= 1u;
  }
  const int c0 = int(nb + sgc * 64u);
  const int r0 = int(rb + sgr * 16u);
  frag_put_in(lo0, sums, int(a.n), r0, c0, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi0, sums, int(a.n), r0, c0 + 16, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(lo1, sums, int(a.n), r0, c0 + 32, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(hi1, sums, int(a.n), r0, c0 + 48, home, int(rb + rows_here), int(nb + cols_here));
}

kernel void tf_exl3_matmul_prefill_i8b(const device half* const x [[buffer(0)]],
                                       const device uchar* const w8 [[buffer(1)]],
                                       device float* const sums [[buffer(2)]],
                                       constant TfExl3Args& a [[buffer(3)]],
                                       const device float* const bscales [[buffer(4)]],
                                       uint lane [[thread_index_in_simdgroup]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_i8b<false>(wx, x, w8, bscales, sums, a, lane, sg, tid, tg);
}

kernel void tf_exl3_matmul_prefill_i8bw(const device half* const x [[buffer(0)]],
                                        const device uchar* const w8 [[buffer(1)]],
                                        device float* const sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device float* const bscales [[buffer(4)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_i8b<true>(wx, x, w8, bscales, sums, a, lane, sg, tid, tg);
}

// (v1, exact f32 accumulate): the block-scaled int8 body for the pre-rotated pack. The accumulate order is the same
// 16-k MMA order; each block's finished fragment is scaled and added to the running f32 accumulator, so the result is
// sum_kb scale_kb * sum_{k in kb} code[k,n]*x[k] -- the exact linear map of the stored W_hat.
template <bool WIDE>
inline void tf_exl3_prefill_body_i8a(threadgroup half* wx, const device half* x, const device uchar* w8,
                                     const device float* bscales, device float* sums, constant TfExl3Args& a,
                                     uint lane, uint sg, uint tid, uint3 tg) {
  constexpr uint RG = 8u;
  const uint trows = RG * 16u;
  const uint rb = tg.y * trows, nb = tg.x * 64u;
  if (rb >= a.rows) return;
  const uint rows_here = min(trows, a.rows - rb);
  const uint cols_here = min(64u, a.n - nb);
  const short2 home = frag_home(ushort(lane));
  const uint nb64 = a.n >> 6, block = nb >> 6;
  const uint sgr = sg % RG, sgc = sg / RG;
  frag<float> lo0, hi0, lo1, hi1;      // the current 128-k block
  frag<float> acc0, acc1, acc2, acc3;  // the scaled running sum
  lo0 = frag<float>(0); hi0 = frag<float>(0); lo1 = frag<float>(0); hi1 = frag<float>(0);
  acc0 = frag<float>(0); acc1 = frag<float>(0); acc2 = frag<float>(0); acc3 = frag<float>(0);
  const int c0 = int(nb + sgc * 64u);
  if (WIDE) {
    if (tid < 128u) {
      const uint2 q = *reinterpret_cast<const device uint2*>(w8 + ((block) << 10) + 8u * tid);
      tf_i8_expand8(wx, tid, q);
    }
  } else if (tid < 256u) {
    const uint u = *reinterpret_cast<const device uint*>(w8 + ((block) << 10) + 4u * tid);
    tf_i8_expand4(wx, tid, u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint cur_buf = 0;
  for (uint kb = 0; kb < a.k; kb += 16u) {
    const uint nkb = kb + 16u;
    uint nu = 0u;
    uint2 nq = uint2(0u);
    if (nkb < a.k) {
      if (WIDE) {
        if (tid < 128u) nq = *reinterpret_cast<const device uint2*>(w8 + (((nkb >> 4) * nb64 + block) << 10) + 8u * tid);
      } else if (tid < 256u) {
        nu = *reinterpret_cast<const device uint*>(w8 + (((nkb >> 4) * nb64 + block) << 10) + 4u * tid);
      }
    }
    if ((kb & 127u) == 0u) {
      lo0 = frag<float>(0); hi0 = frag<float>(0); lo1 = frag<float>(0); hi1 = frag<float>(0);
    }
    frag<half> av;
    tf_frag_get_dev4(av, x, int(a.k), int(rb + sgr * 16u), int(kb), home, int(rb + rows_here));
    threadgroup half* cur = wx + cur_buf * 1024u;
    frag<half> b0, b1;
    const int wc = int(sgc * 64u);
    tf_frag_get_tg4(b0, cur, 64, 0, wc, home);
    tf_frag_get_tg4(b1, cur, 64, 0, wc + 16, home);
    mma_16x32<false, false>(lo0, hi0, av, b0, b1);
    tf_frag_get_tg4(b0, cur, 64, 0, wc + 32, home);
    tf_frag_get_tg4(b1, cur, 64, 0, wc + 48, home);
    mma_16x32<false, false>(lo1, hi1, av, b0, b1);
    if (nkb < a.k) {
      if (WIDE) {
        if (tid < 128u) tf_i8_expand8(wx + (cur_buf ^ 1u) * 1024u, tid, nq);
      } else if (tid < 256u) {
        tf_i8_expand4(wx + (cur_buf ^ 1u) * 1024u, tid, nu);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cur_buf ^= 1u;
    // Nkb is the first k of the NEXT block: at every 128 boundary the finished fragment is scaled and folded.
    if ((nkb & 127u) == 0u) {
      const uint bi = kb >> 7;
      const device float* bs = bscales + ulong(bi) * a.n;
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int cA = c0 + home.x + (e & 3);
        acc0[e] = fma(lo0[e], bs[cA], acc0[e]);
        acc1[e] = fma(hi0[e], bs[cA + 16], acc1[e]);
        acc2[e] = fma(lo1[e], bs[cA + 32], acc2[e]);
        acc3[e] = fma(hi1[e], bs[cA + 48], acc3[e]);
      }
    }
  }
  const int r0 = int(rb + sgr * 16u);
  frag_put_in(acc0, sums, int(a.n), r0, c0, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(acc1, sums, int(a.n), r0, c0 + 16, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(acc2, sums, int(a.n), r0, c0 + 32, home, int(rb + rows_here), int(nb + cols_here));
  frag_put_in(acc3, sums, int(a.n), r0, c0 + 48, home, int(rb + rows_here), int(nb + cols_here));
}


kernel void tf_exl3_matmul_prefill_i8a(const device half* const x [[buffer(0)]],
                                       const device uchar* const w8 [[buffer(1)]],
                                       device float* const sums [[buffer(2)]],
                                       constant TfExl3Args& a [[buffer(3)]],
                                       const device float* const bscales [[buffer(4)]],
                                       uint lane [[thread_index_in_simdgroup]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_i8a<false>(wx, x, w8, bscales, sums, a, lane, sg, tid, tg);
}

kernel void tf_exl3_matmul_prefill_fp8(const device half* const x [[buffer(0)]],
                                       const device uchar* const w8 [[buffer(1)]],
                                       device float* const sums [[buffer(2)]],
                                       constant TfExl3Args& a [[buffer(3)]],
                                       const device float* const scales [[buffer(4)]],
                                       uint lane [[thread_index_in_simdgroup]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_fp8<false, float>(wx, x, w8, scales, sums, a, lane, sg, tid, tg);
}

kernel void tf_exl3_matmul_prefill_fp8w(const device half* const x [[buffer(0)]],
                                        const device uchar* const w8 [[buffer(1)]],
                                        device float* const sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device float* const scales [[buffer(4)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_fp8<true, float>(wx, x, w8, scales, sums, a, lane, sg, tid, tg);
}

