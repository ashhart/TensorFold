// DeepSeek-V4.1 DSpark top-K GPU candidate extraction + single-TG resident Markov chain (topk in 1..256).

inline uint tf_mk_tg_scan256(uint val, uint lane, uint sg, threadgroup uint* sg_sum, threadgroup uint* sg_off,
                             threadgroup uint& sg_tot_all, thread uint& tot_out) {
  const uint sg_ex = simd_prefix_exclusive_sum(val);
  const uint sg_tot = simd_sum(val);
  if (lane == 0u) sg_sum[sg] = sg_tot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    const uint x = lane < 8u ? sg_sum[lane] : 0u;
    const uint off = simd_prefix_exclusive_sum(x);
    if (lane < 8u) sg_off[lane] = off;
    if (lane == 7u) sg_tot_all = off + x;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tot_out = sg_tot_all;
  return sg_off[sg] + sg_ex;
}

// Stage 1: Grid (16, total_rows, 1), TG (256, 1, 1). Exact top-K per slice via 4-pass 8-bit radix select.
[[max_total_threads_per_threadgroup(256)]] kernel void tf_dsv41_markov_topk_part(
    constant TfMarkovArgs& a [[buffer(0)]], device const float* logits [[buffer(1)]],
    device float* part_val [[buffer(2)]], device uint* part_id [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint sg_sum[8];
  threadgroup uint sg_off[8];
  threadgroup uint sg_tot_all;
  threadgroup uint sh_bucket;
  threadgroup uint sh_above;

  const uint s = tg.x;
  const uint r = tg.y;
  const uint K = a.topk;
  const uint chunk = (a.vocab + TF_MK_SLICES - 1u) / TF_MK_SLICES;
  const uint v_begin = min(s * chunk, a.vocab);
  const uint v_end = min(v_begin + chunk, a.vocab);
  const uint len = v_end - v_begin;
  const uint out_base = (r * TF_MK_SLICES + s) * K;
  device const float* row = logits + ulong(r) * a.vocab;

  if (len <= K) {
    if (tid < len) {
      part_val[out_base + tid] = row[v_begin + tid];
      part_id[out_base + tid] = v_begin + tid;
    } else if (tid < K) {
      part_val[out_base + tid] = -INFINITY;
      part_id[out_base + tid] = TF_MK_NONE;
    }
    return;
  }

  uint prefix = 0u, mask = 0u, rem = K;
  for (uint pass = 0u; pass < 4u; ++pass) {
    const uint shift = 24u - 8u * pass;
    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint v = v_begin + tid; v < v_end; v += 256u) {
      const uint fk = tf_mk_fkey(row[v]);
      if ((fk & mask) == prefix) atomic_fetch_add_explicit(&hist[(fk >> shift) & 0xFFu], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint idx = 255u - tid;
    const uint cnt = atomic_load_explicit(&hist[idx], memory_order_relaxed);
    uint tot_unused = 0u;
    const uint above = tf_mk_tg_scan256(cnt, lane, sg, sg_sum, sg_off, sg_tot_all, tot_unused);
    if (above < rem && above + cnt >= rem) {
      sh_bucket = idx;
      sh_above = above;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= (sh_bucket << shift);
    mask |= (0xFFu << shift);
    rem -= sh_above;
  }

  const uint thresh = prefix;
  const uint gt_lim = K - rem;
  uint gt_taken = 0u, eq_taken = 0u;
  for (uint base = v_begin; base < v_end; base += 256u) {
    const uint v = base + tid;
    bool is_gt = false, is_eq = false;
    float val = -INFINITY;
    if (v < v_end) {
      val = row[v];
      const uint fk = tf_mk_fkey(val);
      is_gt = (fk > thresh);
      is_eq = (fk == thresh);
    }
    uint tot = 0u;
    const uint ex = tf_mk_tg_scan256((uint(is_gt) << 16) | uint(is_eq), lane, sg, sg_sum, sg_off, sg_tot_all, tot);
    const uint gt_rank = ex >> 16, eq_rank = ex & 0xFFFFu;
    const uint gt_tot = tot >> 16, eq_tot = tot & 0xFFFFu;
    const uint rem_eq = (rem > eq_taken) ? (rem - eq_taken) : 0u;
    if (is_gt) {
      const uint dst = gt_taken + gt_rank;
      part_val[out_base + dst] = val;
      part_id[out_base + dst] = v;
    } else if (is_eq && eq_rank < rem_eq) {
      const uint dst = gt_lim + eq_taken + eq_rank;
      part_val[out_base + dst] = val;
      part_id[out_base + dst] = v;
    }
    gt_taken += gt_tot;
    eq_taken += min(eq_tot, rem_eq);
    if (gt_taken == gt_lim && eq_taken == rem) break;
  }
}

// Stage 2: Grid (1, total_rows, 1), TG (256, 1, 1). Merges 16 * K winners and bitonic-sorts into cand/base.
[[max_total_threads_per_threadgroup(256)]] kernel void tf_dsv41_markov_topk_merge(
    constant TfMarkovArgs& a [[buffer(0)]], device const float* part_val [[buffer(1)]],
    device const uint* part_id [[buffer(2)]], device int* cand [[buffer(3)]], device float* base_out [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint sg_sum[8];
  threadgroup uint sg_off[8];
  threadgroup uint sg_tot_all;
  threadgroup uint sh_bucket;
  threadgroup uint sh_above;
  threadgroup float sv[256];
  threadgroup uint si[256];

  const uint r = tg.y;
  const uint K = a.topk;
  const uint total = TF_MK_SLICES * K;
  const uint in_base = r * total;
  const uint eff_k = min(K, a.vocab);
  uint S = 1u;
  while (S < K) S <<= 1u;

  sv[tid] = -INFINITY;
  si[tid] = TF_MK_NONE;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (a.vocab <= K) {
    uint taken = 0u;
    for (uint base = 0u; base < total; base += 256u) {
      const uint idx = base + tid;
      const uint id = idx < total ? part_id[in_base + idx] : TF_MK_NONE;
      const bool live = (id != TF_MK_NONE);
      uint tot = 0u;
      const uint rank = tf_mk_tg_scan256(uint(live), lane, sg, sg_sum, sg_off, sg_tot_all, tot);
      if (live) {
        sv[taken + rank] = part_val[in_base + idx];
        si[taken + rank] = id;
      }
      taken += tot;
    }
  } else {
    uint prefix = 0u, mask = 0u, rem = eff_k;
    for (uint pass = 0u; pass < 4u; ++pass) {
      const uint shift = 24u - 8u * pass;
      atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint idx = tid; idx < total; idx += 256u) {
        const uint id = part_id[in_base + idx];
        if (id != TF_MK_NONE) {
          const uint fk = tf_mk_fkey(part_val[in_base + idx]);
          if ((fk & mask) == prefix) atomic_fetch_add_explicit(&hist[(fk >> shift) & 0xFFu], 1u, memory_order_relaxed);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint bidx = 255u - tid;
      const uint cnt = atomic_load_explicit(&hist[bidx], memory_order_relaxed);
      uint tot_unused = 0u;
      const uint above = tf_mk_tg_scan256(cnt, lane, sg, sg_sum, sg_off, sg_tot_all, tot_unused);
      if (above < rem && above + cnt >= rem) {
        sh_bucket = bidx;
        sh_above = above;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      prefix |= (sh_bucket << shift);
      mask |= (0xFFu << shift);
      rem -= sh_above;
    }

    const uint thresh = prefix;
    const uint gt_lim = eff_k - rem;
    uint gt_taken = 0u, eq_taken = 0u;
    for (uint base = 0u; base < total; base += 256u) {
      const uint idx = base + tid;
      const uint id = idx < total ? part_id[in_base + idx] : TF_MK_NONE;
      bool is_gt = false, is_eq = false;
      float val = -INFINITY;
      if (id != TF_MK_NONE) {
        val = part_val[in_base + idx];
        const uint fk = tf_mk_fkey(val);
        is_gt = (fk > thresh);
        is_eq = (fk == thresh);
      }
      uint tot = 0u;
      const uint ex = tf_mk_tg_scan256((uint(is_gt) << 16) | uint(is_eq), lane, sg, sg_sum, sg_off, sg_tot_all, tot);
      const uint gt_rank = ex >> 16, eq_rank = ex & 0xFFFFu;
      const uint gt_tot = tot >> 16, eq_tot = tot & 0xFFFFu;
      const uint rem_eq = (rem > eq_taken) ? (rem - eq_taken) : 0u;
      if (is_gt) {
        const uint dst = gt_taken + gt_rank;
        sv[dst] = val;
        si[dst] = id;
      } else if (is_eq && eq_rank < rem_eq) {
        const uint dst = gt_lim + eq_taken + eq_rank;
        sv[dst] = val;
        si[dst] = id;
      }
      gt_taken += gt_tot;
      eq_taken += min(eq_tot, rem_eq);
      if (gt_taken == gt_lim && eq_taken == rem) break;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint size = 2u; size <= S; size <<= 1u) {
    for (uint stride = size >> 1u; stride > 0u; stride >>= 1u) {
      if (tid < S) {
        const uint ixj = tid ^ stride;
        if (ixj > tid) {
          const bool asc = ((tid & size) == 0u);
          const float va = sv[tid], vb = sv[ixj];
          const uint ia = si[tid], ib = si[ixj];
          const bool do_swap = asc ? tf_mk_better(vb, ib, va, ia) : tf_mk_better(va, ia, vb, ib);
          if (do_swap) {
            sv[tid] = vb; si[tid] = ib;
            sv[ixj] = va; si[ixj] = ia;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  if (tid < K) {
    const uint out_idx = r * K + tid;
    const uint id = si[tid];
    cand[out_idx] = (id == TF_MK_NONE) ? -1 : int(id);
    base_out[out_idx] = (id == TF_MK_NONE) ? -INFINITY : sv[tid];
  }
}

// Stage 3: Grid (asks, 1, 1), TG (256, 1, 1). All B Markov steps resident in one TG (greedy, raced, filtered).
template <uint PER>
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_markov_chain_topk(
    constant TfMarkovArgs& a [[buffer(0)]], device const int* cand [[buffer(1)]], device const float* base_in [[buffer(2)]],
    device const ushort* w1 [[buffer(3)]], device const ushort* w2 [[buffer(4)]], device const float* cw [[buffer(5)]],
    device const float* hid [[buffer(6)]], device const uint* ids [[buffer(7)]], device uint* drafts [[buffer(8)]],
    device float* conf [[buffer(9)]], device const TfMarkovAsk* asks [[buffer(10)]],
    uint j [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint RANK = 32u * PER;
  threadgroup float e_buf[RANK];
  threadgroup float sv[256];
  threadgroup uint si[256];
  threadgroup float sprob[256];
  threadgroup float ts[8];
  threadgroup float tz[8];
  threadgroup uint ti[8];
  threadgroup uint sh_keep;
  threadgroup uint sh_next;

  const TfMarkovAsk k = asks[j];
  const uint K = a.topk;
  const uint live_lim = min(K, a.vocab);
  uint S = 1u;
  while (S < K) S <<= 1u;
  uint prev = ids[j * a.rows];

  for (uint step = 0u; step < a.rows; ++step) {
    if (sg == 0u) {
      float e[PER];
      tf_mk_row<PER>(w1, prev, lane, e);
      float cs = 0.0f;
      for (uint i = 0u; i < PER; ++i) {
        e_buf[lane * PER + i] = e[i];
        cs = fma(e[i], cw[a.dim + lane * PER + i], cs);
      }
      cs = simd_sum(cs);
      if (lane == 0u) conf[j * a.rows + step] = 1.0f / (1.0f + precise::exp(-(hid[j * a.rows + step] + cs)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row_idx = j * a.rows + step;
    if (tid < K) {
      const int cid = cand[row_idx * K + tid];
      if (cid >= 0) {
        device const ushort* wp = w2 + ulong(uint(cid)) * RANK;
        float acc = 0.0f;
        for (uint w = 0u; w < RANK / 8u; ++w) {
          const uint4 q = ((device const uint4*)wp)[w];
          for (uint i = 0u; i < 4u; ++i) {
            acc = fma(e_buf[8u * w + 2u * i], tf_mk_bf16(q[i] & 0xFFFFu), acc);
            acc = fma(e_buf[8u * w + 2u * i + 1u], tf_mk_bf16(q[i] >> 16), acc);
          }
        }
        sv[tid] = base_in[row_idx * K + tid] + acc;
        si[tid] = uint(cid);
      } else {
        sv[tid] = -INFINITY;
        si[tid] = TF_MK_NONE;
      }
    } else {
      sv[tid] = -INFINITY;
      si[tid] = TF_MK_NONE;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (k.sampled == 0u) {
      float bv = tid < K ? sv[tid] : -INFINITY;
      uint bi = tid < K ? si[tid] : TF_MK_NONE;
      for (ushort off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(bv, off);
        const uint oi = simd_shuffle_down(bi, off);
        if (tf_mk_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
      }
      if (lane == 0u) { tz[sg] = bv; ti[sg] = bi; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0u) {
        bv = lane < 8u ? tz[lane] : -INFINITY;
        bi = lane < 8u ? ti[lane] : TF_MK_NONE;
        for (ushort off = 16; off > 0; off >>= 1) {
          const float ov = simd_shuffle_down(bv, off);
          const uint oi = simd_shuffle_down(bi, off);
          if (tf_mk_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0u) {
          const uint chosen = (bi == TF_MK_NONE) ? prev : bi;
          drafts[row_idx] = chosen;
          sh_next = chosen;
        }
      }
    } else {
      for (uint size = 2u; size <= S; size <<= 1u) {
        for (uint stride = size >> 1u; stride > 0u; stride >>= 1u) {
          if (tid < S) {
            const uint ixj = tid ^ stride;
            if (ixj > tid) {
              const bool asc = ((tid & size) == 0u);
              const float va = sv[tid], vb = sv[ixj];
              const uint ia = si[tid], ib = si[ixj];
              const bool do_swap = asc ? tf_mk_better(vb, ib, va, ia) : tf_mk_better(va, ia, vb, ib);
              if (do_swap) {
                sv[tid] = vb; si[tid] = ib;
                sv[ixj] = va; si[ixj] = ia;
              }
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }

      uint lim = live_lim;
      if (k.top_k > 0u && k.top_k < lim) lim = k.top_k;
      const float x0 = sv[0] / k.t;
      if (k.top_p > 0.0f && k.top_p < 1.0f && tid < lim) {
        sprob[tid] = precise::exp(sv[tid] / k.t - x0);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (tid == 0u) {
        uint keep = lim;
        while (keep > 0u && si[keep - 1u] == TF_MK_NONE) keep -= 1u;
        if (keep > 0u && k.top_p > 0.0f && k.top_p < 1.0f) {
          float total = 0.0f;
          for (uint c = 0u; c < keep; ++c) total += sprob[c];
          float cum = 0.0f;
          uint below = 0u;
          for (uint c = 0u; c < keep; ++c) {
            cum += sprob[c] / total;
            if (cum < k.top_p) below += 1u;
          }
          keep = min(keep, below + 1u);
        }
        if (keep > 0u && k.min_p > 0.0f) {
          const float floor_v = x0 + precise::log(k.min_p);
          uint n_keep = 0u;
          for (uint c = 0u; c < keep; ++c) {
            if (sv[c] / k.t >= floor_v) n_keep += 1u;
          }
          keep = max(1u, n_keep);
        }
        sh_keep = keep;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const uint keep = sh_keep;
      const ulong gbase = tf_key_base(k.seed, k.pos0 + step);
      float bs = -INFINITY, bz = -INFINITY;
      uint bi = TF_MK_NONE;
      if (tid < keep && si[tid] != TF_MK_NONE) {
        const float z = sv[tid];
        const uint id = si[tid];
        bs = z / k.t + tf_key_gumbel(tf_key_mix(gbase ^ ulong(id)));
        bz = z;
        bi = id;
      }
      for (ushort off = 16; off > 0; off >>= 1)
        tf_key_beats(bs, bz, bi, simd_shuffle_down(bs, off), simd_shuffle_down(bz, off), simd_shuffle_down(bi, off));
      if (lane == 0u) { ts[sg] = bs; tz[sg] = bz; ti[sg] = bi; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0u) {
        bs = lane < 8u ? ts[lane] : -INFINITY;
        bz = lane < 8u ? tz[lane] : -INFINITY;
        bi = lane < 8u ? ti[lane] : TF_MK_NONE;
        for (ushort off = 16; off > 0; off >>= 1)
          tf_key_beats(bs, bz, bi, simd_shuffle_down(bs, off), simd_shuffle_down(bz, off), simd_shuffle_down(bi, off));
        if (lane == 0u) {
          const uint chosen = (bi == TF_MK_NONE) ? prev : bi;
          drafts[row_idx] = chosen;
          sh_next = chosen;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prev = sh_next;
  }
}

#define TF_MK(PER) \
  template [[host_name("tf_dsv41_markov_z_" #PER)]] [[kernel]] decltype(tf_dsv41_markov_z<PER>) tf_dsv41_markov_z<PER>; \
  template [[host_name("tf_dsv41_markov_pick_" #PER)]] [[kernel]] decltype(tf_dsv41_markov_pick<PER>) tf_dsv41_markov_pick<PER>; \
  template [[host_name("tf_dsv41_markov_chain_topk_" #PER)]] [[kernel]] decltype(tf_dsv41_markov_chain_topk<PER>) tf_dsv41_markov_chain_topk<PER>;
TF_MK(1)
TF_MK(2)
TF_MK(4)
TF_MK(8)
