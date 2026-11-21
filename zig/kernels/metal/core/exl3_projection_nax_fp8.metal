inline void tf_fp8_expand4(threadgroup half* wx, uint tid, uint u) {
  const half2 a = as_type<half2>(((u & 0x007F007Fu) << 7) | ((u << 8) & 0x80008000u));
  const half2 b = as_type<half2>(((u >> 1) & 0x3F803F80u) | (u & 0x80008000u));
  const half2 k256 = half2(256.0h, 256.0h);
  threadgroup half2* o = (threadgroup half2*)(wx + 4u * tid);
  o[0] = half2(a.x, b.x) * k256;
  o[1] = half2(a.y, b.y) * k256;
}

inline void tf_i8_expand4(threadgroup half* wx, uint tid, uint u) {
  const int b0 = int(u << 24) >> 24, b1 = int(u << 16) >> 24, b2 = int(u << 8) >> 24, b3 = int(u) >> 24;
  threadgroup half2* o = (threadgroup half2*)(wx + 4u * tid);
  o[0] = half2(half(float(b0)), half(float(b1)));
  o[1] = half2(half(float(b2)), half(float(b3)));
}

template <typename O>
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
  if (tid < 256u) {
    const uint u = *reinterpret_cast<const device uint*>(w8 + ((block) << 10) + 4u * tid);
    if (a.halves == 0u) tf_i8_expand4(wx, tid, u); else tf_fp8_expand4(wx, tid, u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint cur_buf = 0;
  for (uint kb = 0; kb < a.k; kb += 16u) {
    const uint nkb = kb + 16u;
    uint nu = 0u;
    if (nkb < a.k) {
      if (tid < 256u) {
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
      if (tid < 256u) {
        if (a.halves == 0u) tf_i8_expand4(wx + (cur_buf ^ 1u) * 1024u, tid, nu);
        else tf_fp8_expand4(wx + (cur_buf ^ 1u) * 1024u, tid, nu);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cur_buf ^= 1u;
  }
  const int c0 = int(nb + sgc * 64u);
  const int r0 = int(rb + sgr * 16u);
  // A bf16 destination is the caller's final output (the pre-rotated pack has no epilogue transform).
  const int out_ld = (sizeof(O) == 4) ? int(a.n) : int(a.y_stride);
  if (scales != nullptr) {
    // The per-output-row E4M3 scale.
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
  tf_exl3_prefill_body_fp8<float>(wx, x, w8, scales, sums, a, lane, sg, tid, tg);
}
