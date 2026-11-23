// DeepSeek V4.1 opt-in FP64-certified BF16 prompt experts on M5 Ultra Metal 4.1 tensor units (Pass 2 patch).

// Exact 32-element double-float dot product used in Pass 2 on flagged unsafe or abnormal-scale groups.
template <bool FP8>
inline XuDD xu_exact_sum_dd(device const uchar* wcol, device const half* xh, device const bfloat* xb, float s) {
  const uint4 wq = *(device const uint4*)wcol;
  uint4 xq[4];
  _Pragma("unroll") for (int i = 0; i < 4; ++i)
    xq[i] = FP8 ? ((device const uint4*)xh)[i] : ((device const uint4*)xb)[i];
  float hi = 0.0f, lo = 0.0f;
  _Pragma("unroll") for (uint k = 0; k < 32; ++k) {
    const float2 f = k3_fp4_pair((wq[k / 8] >> (8 * ((k / 2) % 4))) & 0xFFu);
    float x;
    if (FP8) {
      const uint at = xu_pos(k);
      x = float(as_type<half2>(xq[at / 8][(at / 2) % 4])[at % 2]);
    } else {
      x = float(as_type<bfloat2>(xq[k / 8][(k / 2) % 4])[k % 2]);
      if (TF_CTL == 1 && !isfinite(x)) x = 0.0f;
    }
    xu_neumaier(hi, lo, x * ((k & 1) ? f.y : f.x));
  }
  if (TF_CTL == 3) {
    hi = float(half(hi));
    lo = 0.0f;
  }
  const float scale = FP8 ? fabs(s) : 1.0f;
  if (!isfinite(hi) || !isfinite(scale)) return {hi * scale, 0.0f};
  return xu_two_sum(hi * scale, lo * scale);
}

// Pass 2 for xu_up: patches flagged unsafe/abnormal groups onto Pass 1 exact 48-bit partial sums.
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_dsv41_xu_patch_up(
    device const half* xc [[buffer(0)]], device const float* xs [[buffer(1)]],
    device const ExpertPtrs* table [[buffer(2)]], device const uint* pairs [[buffer(3)]],
    device const float* wts [[buffer(4)]], device const uint4* tiles [[buffer(5)]],
    device const uint4* patch_meta [[buffer(6)]], device const float4* patch_vals [[buffer(7)]],
    device const uint* patch_cnt [[buffer(8)]], device bfloat* act [[buffer(9)]],
    constant XmArgs& a [[buffer(10)]], constant XuPatchArgs& pa [[buffer(11)]],
    device const uint* unsafe_mask [[buffer(12)]],
    uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint ntg [[threadgroups_per_grid]]) {
  const uint nd = patch_cnt[0];
  if (nd == 0u) return;
  const uint warp_g = tg * 4u + sg;
  const uint stride_w = ntg * 4u;
  const uint K = a.K, G = K >> 5, Ge = (TF_CTL == 5) ? (G - 1u) : G;

  if (nd <= pa.patch_cap) {
    for (uint wi = warp_g; wi < nd; wi += stride_w) {
      const uint4 meta = patch_meta[wi];
      const uint pair_pos = meta.x, ex = meta.y, n0 = meta.z;
      const bool has_abn = (meta.w != 0u);
      const uint col = n0 + lane;
      const uint p_idx = pairs[pair_pos], row = p_idx / a.topk;
      const ExpertPtrs w = table[ex];
      device const half* xr = xc + ulong(row) * K;
      device const float* sr = xs + ulong(row) * G;
      device const uchar* w1p = w.w1p + ulong(col) * (K >> 1);
      device const uchar* w1s = w.w1s + ulong(col) * G;
      device const uchar* w3p = w.w3p + ulong(col) * (K >> 1);
      device const uchar* w3s = w.w3s + ulong(col) * G;
      const float4 pv = patch_vals[ulong(wi) * 32u + lane];
      XuDD a0 = {pv.x, pv.y}, a1 = {pv.z, pv.w};
      if (TF_CTL != 6 && !has_abn && row < pa.ug_cap) {
        const uint nw = (Ge + 31u) >> 5;
        device const uint* um = unsafe_mask + ulong(row) * 8u;
        for (uint wi_m = 0u; wi_m < nw; ++wi_m) {
          uint mask = um[wi_m];
          while (mask != 0u) {
            const uint g = (wi_m << 5) + ctz(mask);
            mask &= mask - 1u;
            if (g >= Ge) break;
            const float s = sr[g];
            const uchar e1 = w1s[g], e3 = w3s[g];
            a0 = k3_scaled_dd(xu_exact_sum_dd<true>(w1p + (g << 4), xr + (g << 5), nullptr, s), e1, a0);
            a1 = k3_scaled_dd(xu_exact_sum_dd<true>(w3p + (g << 4), xr + (g << 5), nullptr, s), e3, a1);
          }
        }
      } else {
        for (uint g = 0u; g < Ge; ++g) {
          const float s = sr[g];
          const bool us = (TF_CTL != 6) && (signbit(s) || !isfinite(s));
          if (!us && !has_abn) continue;
          const uchar e1 = w1s[g], e3 = w3s[g];
          if (us || e1 > 240u) {
            const XuDD d = xu_exact_sum_dd<true>(w1p + (g << 4), xr + (g << 5), nullptr, s);
            a0 = k3_scaled_dd(d, e1, a0);
          }
          if (us || e3 > 240u) {
            const XuDD d = xu_exact_sum_dd<true>(w3p + (g << 4), xr + (g << 5), nullptr, s);
            a1 = k3_scaled_dd(d, e3, a1);
          }
        }
      }
      const float g_val = ds_clip(float(xu_dd_to_bf16(a0)), -INFINITY, a.limit);
      const float u_val = ds_clip(float(xu_dd_to_bf16(a1)), -a.limit, a.limit);
      const float v = g_val / (1.0f + precise::exp(-g_val)) * u_val * wts[p_idx];
      const float vv = (TF_CTL == 2) ? v : float(bfloat(v));
      const float amax = simd_max(fabs(vv));
      const bool bad = simd_any(!isfinite(vv));
      const int s_exp = ds_exp(amax);
      const float q = ds_val(ds_code(min(fabs(vv) * ds_pow2(-s_exp), 448.0f), signbit(vv))) * ds_pow2(s_exp);
      act[ulong(pair_pos) * a.N + col] = bfloat(bad ? NAN : q);
    }
  } else {
    // Overflow fallback: exact full recompute over all active (tile, lr, n0) blocks.
    const uint nt = tiles[0].x, nb = a.N >> 5;
    const uint total = nt * XU_RT * nb;
    for (uint wi = warp_g; wi < total; wi += stride_w) {
      const uint bi = wi % nb, tmp = wi / nb, lr = tmp % XU_RT, ti = tmp / XU_RT;
      const uint4 tile = tiles[1u + ti];
      if (lr >= tile.z) continue;
      const uint pair_pos = tile.y + lr, ex = tile.x, n0 = bi << 5, col = n0 + lane;
      const uint p_idx = pairs[pair_pos], row = p_idx / a.topk;
      const ExpertPtrs w = table[ex];
      device const half* xr = xc + ulong(row) * K;
      device const float* sr = xs + ulong(row) * G;
      device const uchar* w1p = w.w1p + ulong(col) * (K >> 1);
      device const uchar* w1s = w.w1s + ulong(col) * G;
      device const uchar* w3p = w.w3p + ulong(col) * (K >> 1);
      device const uchar* w3s = w.w3s + ulong(col) * G;
      XuDD a0 = {0.0f, 0.0f}, a1 = {0.0f, 0.0f};
      for (uint g = 0u; g < Ge; ++g) {
        a0 = k3_scaled_dd(xu_exact_sum_dd<true>(w1p + (g << 4), xr + (g << 5), nullptr, sr[g]), w1s[g], a0);
        a1 = k3_scaled_dd(xu_exact_sum_dd<true>(w3p + (g << 4), xr + (g << 5), nullptr, sr[g]), w3s[g], a1);
      }
      const float g_val = ds_clip(float(xu_dd_to_bf16(a0)), -INFINITY, a.limit);
      const float u_val = ds_clip(float(xu_dd_to_bf16(a1)), -a.limit, a.limit);
      const float v = g_val / (1.0f + precise::exp(-g_val)) * u_val * wts[p_idx];
      const float vv = (TF_CTL == 2) ? v : float(bfloat(v));
      const float amax = simd_max(fabs(vv));
      const bool bad = simd_any(!isfinite(vv));
      const int s_exp = ds_exp(amax);
      const float q = ds_val(ds_code(min(fabs(vv) * ds_pow2(-s_exp), 448.0f), signbit(vv))) * ds_pow2(s_exp);
      act[ulong(pair_pos) * a.N + col] = bfloat(bad ? NAN : q);
    }
  }
}

// Pass 2 for xu_down: patches flagged unsafe/abnormal groups onto Pass 1 exact 48-bit partial sums.
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_dsv41_xu_patch_down(
    device const bfloat* act [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
    device const uint4* tiles [[buffer(2)]], device const float* ds [[buffer(3)]],
    device const uint4* patch_meta [[buffer(4)]], device const float2* patch_vals [[buffer(5)]],
    device const uint* patch_cnt [[buffer(6)]], device bfloat* out [[buffer(7)]],
    constant XmArgs& a [[buffer(8)]], constant XuPatchArgs& pa [[buffer(9)]],
    device const uint* unsafe_mask [[buffer(10)]],
    uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint ntg [[threadgroups_per_grid]]) {
  const uint nd = patch_cnt[0];
  if (nd == 0u) return;
  const uint warp_g = tg * 4u + sg;
  const uint stride_w = ntg * 4u;
  const uint K = a.K, G = K >> 5, Ge = (TF_CTL == 5) ? (G - 1u) : G;

  if (nd <= pa.patch_cap) {
    for (uint wi = warp_g; wi < nd; wi += stride_w) {
      const uint4 meta = patch_meta[wi];
      const uint pair_pos = meta.x, ex = meta.y, n0 = meta.z;
      const bool has_abn = (meta.w != 0u);
      const uint col = n0 + lane;
      const ExpertPtrs w = table[ex];
      device const bfloat* ar = act + ulong(pair_pos) * K;
      device const float* sr = ds + ulong(pair_pos) * G;
      device const uchar* wp = w.w2p + ulong(col) * (K >> 1);
      device const uchar* ws = w.w2s + ulong(col) * G;
      const float2 pv = patch_vals[ulong(wi) * 32u + lane];
      XuDD acc = {pv.x, pv.y};
      if (TF_CTL != 6 && !has_abn && pair_pos < pa.ug_cap) {
        const uint nw = (Ge + 31u) >> 5;
        device const uint* um = unsafe_mask + ulong(pair_pos) * 8u;
        for (uint wi_m = 0u; wi_m < nw; ++wi_m) {
          uint mask = um[wi_m];
          while (mask != 0u) {
            const uint g = (wi_m << 5) + ctz(mask);
            mask &= mask - 1u;
            if (g >= Ge) break;
            acc = k3_scaled_dd(xu_exact_sum_dd<false>(wp + (g << 4), nullptr, ar + (g << 5), 1.0f), ws[g], acc);
          }
        }
      } else {
        for (uint g = 0u; g < Ge; ++g) {
          const float s = sr[g];
          const bool us = (TF_CTL != 6) && (signbit(s) || !isfinite(s));
          if (!us && !has_abn) continue;
          const uchar e2 = ws[g];
          if (us || e2 > 240u) {
            const XuDD d = xu_exact_sum_dd<false>(wp + (g << 4), nullptr, ar + (g << 5), 1.0f);
            acc = k3_scaled_dd(d, e2, acc);
          }
        }
      }
      out[ulong(pair_pos) * a.N + col] = xu_dd_to_bf16(acc);
    }
  } else {
    const uint nt = tiles[0].x, nb = a.N >> 5;
    const uint total = nt * XU_RT * nb;
    for (uint wi = warp_g; wi < total; wi += stride_w) {
      const uint bi = wi % nb, tmp = wi / nb, lr = tmp % XU_RT, ti = tmp / XU_RT;
      const uint4 tile = tiles[1u + ti];
      if (lr >= tile.z) continue;
      const uint pair_pos = tile.y + lr, ex = tile.x, n0 = bi << 5, col = n0 + lane;
      const ExpertPtrs w = table[ex];
      device const bfloat* ar = act + ulong(pair_pos) * K;
      device const uchar* wp = w.w2p + ulong(col) * (K >> 1);
      device const uchar* ws = w.w2s + ulong(col) * G;
      XuDD acc = {0.0f, 0.0f};
      for (uint g = 0u; g < Ge; ++g)
        acc = k3_scaled_dd(xu_exact_sum_dd<false>(wp + (g << 4), nullptr, ar + (g << 5), 1.0f), ws[g], acc);
      out[ulong(pair_pos) * a.N + col] = xu_dd_to_bf16(acc);
    }
  }
}

// Count==1 exact dd up kernel reading tf_dsv41_xu_quant_ug output directly (un-permutes xu_pos quad order).
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_xp_upc_one_xu(
    device const half* XH [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device const uint* nslots [[buffer(3)]],
    device const uint* pairs [[buffer(4)]], device bfloat* act [[buffer(5)]], constant DsArgs& a [[buffer(6)]],
    device const float* wts [[buffer(7)]], device const float* XS [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half4 xh[DS_KMAX / 32 * DS_GS / 4];
  threadgroup float xs[DS_KMAX / 32], vals[1][32];
  if (tg.y >= nslots[0] || slots[3 * tg.y + 2] != 1) return;
  const uint e = slots[3 * tg.y], first = slots[3 * tg.y + 1];
  const ExpertPtrs w = table[e];
  const uint nl = sg * 4 + (lane >> 3), n = tg.x * 32 + nl, c = lane & 7, rowb = a.K / 2, groups = a.K / 32;
  device const uchar* const wp[2] = {w.w1p + n * rowb, w.w3p + n * rowb};
  device const uchar* const ws[2] = {w.w1s + n * groups, w.w3s + n * groups};
  const uint row = pairs[first] / a.topk;
  for (uint i = t; i < a.K / 4; i += 256) {
    const uint pq = i & 7u;
    const uint lq = ((pq & 1u) << 2) | (pq >> 1);
    xh[(i >> 3) * (DS_GS / 4) + lq] = ((device const half4*)(XH + ulong(row) * a.K))[i];
  }
  for (uint i = t; i < groups; i += 256) xs[i] = XS[ulong(row) * groups + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  XuDD acc[2][1] = {{{0.0f, 0.0f}}, {{0.0f, 0.0f}}};
  ds_upc_one_body(acc, xh, xs, wp, ws, groups, c);
  ds_up_out_dd<1>(acc, vals, act, wts, pairs, a, first, 0, 1, nl, n, c, nullptr);
}
