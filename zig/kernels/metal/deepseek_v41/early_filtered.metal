// DeepSeek-V4.1 behind-the-verify picks (greedy, raced, min_p, top_k <= 256, nucleus) and multi-slot accept.
#include <metal_stdlib>
using namespace metal;

#define TF_TG 1024u
#define TF_NONE 0xffffffffu
#define TF_MAX_ROWS 16u
#define TF_SLOT_PLAN_INTS 68u

struct TfEarlyPickArgs {
  uint vocab; // logits a row (the row stride)
};

// The better of two (value, id) pairs: the larger value, ties to the lower id (no NaN reaches here).
inline void tf_early_better(thread float& v, thread uint& i, float ov, uint oi) {
  if (ov > v || (ov == v && oi < i)) {
    v = ov;
    i = oi;
  }
}

// Order-preserving uint key for IEEE-754 float: NaN -> 0, -inf -> 0x007FFFFF, -0 == +0 -> 0x80000000.
inline uint tf_early_fkey(float v) {
  uint u = as_type<uint>(v);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return 0u;
  if (u == 0x80000000u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

inline bool tf_early_ord_better(float va, uint ia, float vb, uint ib) {
  if (ia == TF_NONE) return false;
  if (ib == TF_NONE) return true;
  const uint ka = tf_early_fkey(va);
  const uint kb = tf_early_fkey(vb);
  return (ka > kb) || (ka == kb && ia < ib);
}

// 1024-thread exclusive prefix sum + total across 32 simdgroups.
inline uint tf_early_tg_scan1024(uint val, uint lane, uint sg, threadgroup uint* sg_sum, threadgroup uint* sg_off,
                                 threadgroup uint& sg_tot_all, thread uint& tot_out) {
  const uint sg_ex = simd_prefix_exclusive_sum(val);
  const uint sg_tot = simd_sum(val);
  if (lane == 0u) sg_sum[sg] = sg_tot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    const uint x = sg_sum[lane];
    const uint off = simd_prefix_exclusive_sum(x);
    sg_off[lane] = off;
    if (lane == 31u) sg_tot_all = off + x;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tot_out = sg_tot_all;
  return sg_off[sg] + sg_ex;
}

// Per-row sampling parameters for mixed-slot / filtered behind-the-verify picks (32 bytes, 8-byte aligned).
struct TfEarlyRow {
  ulong base;
  float t;
  uint sampled;
  uint top_k;
  float top_p;
  float min_p;
  uint pad;
};

// Unified per-row GPU picker: greedy, raced, min_p, top_k <= 256, and pure nucleus (exact full-vocab Z).
[[max_total_threads_per_threadgroup(TF_TG)]] kernel void tf_dsv41_early_filtered(
    constant TfEarlyPickArgs& args [[buffer(0)]],
    device const float* logits [[buffer(1)]],
    device uint* picks [[buffer(2)]],
    device const TfEarlyRow* rows [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part_s[32];
  threadgroup float part_v[32];
  threadgroup uint part_i[32];
  threadgroup uint part_bad[32];
  threadgroup atomic_uint hist[256];
  threadgroup uint sg_sum[32];
  threadgroup uint sg_off[32];
  threadgroup uint sg_tot_all;
  threadgroup uint sh_bucket;
  threadgroup uint sh_above;
  threadgroup float sh_vmax;
  threadgroup float sh_zfull;
  threadgroup uint sh_bad;
  threadgroup uint sh_greedy;
  threadgroup uint sh_keep;
  threadgroup float sv[256];
  threadgroup uint si[256];
  threadgroup float sprob[256];

  device const float* x = logits + ulong(row) * args.vocab;
  const TfEarlyRow r = rows[row];

  // Case 1: Greedy row.
  if (r.sampled == 0u) {
    float bv = -INFINITY;
    uint bi = TF_NONE;
    for (uint i = tid; i < args.vocab; i += TF_TG) {
      const float v = x[i];
      if (!isnan(v)) tf_early_better(bv, bi, v, i);
    }
    for (ushort off = 16; off > 0; off >>= 1) tf_early_better(bv, bi, simd_shuffle_down(bv, off), simd_shuffle_down(bi, off));
    if (lane == 0u) { part_v[simd] = bv; part_i[simd] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd != 0u) return;
    bv = part_v[lane];
    bi = part_i[lane];
    for (ushort off = 16; off > 0; off >>= 1) tf_early_better(bv, bi, simd_shuffle_down(bv, off), simd_shuffle_down(bi, off));
    if (lane == 0u) picks[row] = (isnan(x[0]) || bi == TF_NONE) ? 0u : bi;
    return;
  }

  const bool has_topk = (r.top_k > 0u && r.top_k < args.vocab);
  const bool has_topp = (r.top_p > 0.0f && r.top_p < 1.0f);

  // Case 2: Raced or min_p-only over full vocabulary (no sort needed, exact).
  if (!has_topk && !has_topp) {
    float floor_v = -INFINITY;
    if (r.min_p > 0.0f) {
      float gv = -INFINITY;
      uint gi = TF_NONE, bad = 0u;
      for (uint i = tid; i < args.vocab; i += TF_TG) {
        const float v = x[i];
        const float xv = v / r.t;
        if (!tf_key_finite(v) || !tf_key_finite(xv)) { bad = 1u; continue; }
        tf_early_better(gv, gi, v, i);
      }
      for (ushort off = 16; off > 0; off >>= 1) {
        tf_early_better(gv, gi, simd_shuffle_down(gv, off), simd_shuffle_down(gi, off));
        bad |= simd_shuffle_down(bad, off);
      }
      if (lane == 0u) { part_v[simd] = gv; part_i[simd] = gi; part_bad[simd] = bad; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (simd == 0u) {
        gv = part_v[lane];
        gi = part_i[lane];
        bad = part_bad[lane];
        for (ushort off = 16; off > 0; off >>= 1) {
          tf_early_better(gv, gi, simd_shuffle_down(gv, off), simd_shuffle_down(gi, off));
          bad |= simd_shuffle_down(bad, off);
        }
        if (lane == 0u) { sh_vmax = gv; sh_greedy = gi; sh_bad = bad; }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sh_bad != 0u) { if (tid == 0u) picks[row] = TF_NONE; return; }
      floor_v = sh_vmax / r.t + precise::log(r.min_p);
    }

    float bs = -INFINITY, bv = -INFINITY;
    uint bi = TF_NONE, bad = 0u;
    for (uint i = tid; i < args.vocab; i += TF_TG) {
      const float v = x[i];
      const float xv = v / r.t;
      if (!tf_key_finite(v) || !tf_key_finite(xv)) { bad = 1u; continue; }
      if (!(xv >= floor_v)) continue;
      tf_key_beats(bs, bv, bi, xv + tf_key_gumbel(tf_key_mix(r.base ^ ulong(i))), v, i);
    }
    for (ushort off = 16; off > 0; off >>= 1) {
      tf_key_beats(bs, bv, bi, simd_shuffle_down(bs, off), simd_shuffle_down(bv, off), simd_shuffle_down(bi, off));
      bad |= simd_shuffle_down(bad, off);
    }
    if (lane == 0u) { part_s[simd] = bs; part_v[simd] = bv; part_i[simd] = bi; part_bad[simd] = bad; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd != 0u) return;
    bs = part_s[lane]; bv = part_v[lane]; bi = part_i[lane]; bad = part_bad[lane];
    for (ushort off = 16; off > 0; off >>= 1) {
      tf_key_beats(bs, bv, bi, simd_shuffle_down(bs, off), simd_shuffle_down(bv, off), simd_shuffle_down(bi, off));
      bad |= simd_shuffle_down(bad, off);
    }
    if (lane == 0u) picks[row] = bad != 0u ? TF_NONE : (bi == TF_NONE ? sh_greedy : bi);
    return;
  }

  // Case 3: top_k <= 256 and/or pure nucleus top_p (0 < top_p < 1).
  const bool pure_nucleus = !has_topk;
  const uint M = pure_nucleus ? min(256u, args.vocab) : min(r.top_k, args.vocab);
  if (M == 0u || M > 256u) {
    if (tid == 0u) picks[row] = TF_NONE;
    return;
  }

  // Check finiteness + find max logit across full vocab.
  float vmax = -INFINITY;
  uint bad = 0u;
  for (uint i = tid; i < args.vocab; i += TF_TG) {
    const float v = x[i];
    const float xv = v / r.t;
    if (!tf_key_finite(v) || !tf_key_finite(xv)) bad = 1u;
    else vmax = max(vmax, v);
  }
  vmax = simd_max(vmax);
  bad = simd_any(bad != 0u) ? 1u : 0u;
  if (lane == 0u) { part_v[simd] = vmax; part_bad[simd] = bad; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0u) {
    vmax = simd_max(part_v[lane]);
    bad = simd_any(part_bad[lane] != 0u) ? 1u : 0u;
    if (lane == 0u) { sh_vmax = vmax; sh_bad = bad; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sh_bad != 0u) { if (tid == 0u) picks[row] = TF_NONE; return; }

  // Pure nucleus needs exact full-vocab softmax denominator Z_full.
  if (pure_nucleus) {
    const float x0_full = sh_vmax / r.t;
    float zp = 0.0f;
    for (uint i = tid; i < args.vocab; i += TF_TG) zp += precise::exp(x[i] / r.t - x0_full);
    zp = simd_sum(zp);
    if (lane == 0u) part_s[simd] = zp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
      zp = simd_sum(part_s[lane]);
      if (lane == 0u) sh_zfull = zp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tf_key_finite(sh_zfull) || sh_zfull <= 0.0f) {
      if (tid == 0u) picks[row] = TF_NONE;
      return;
    }
  }

  uint S = 1u;
  while (S < M) S <<= 1u;
  if (tid < 256u) { sv[tid] = -INFINITY; si[tid] = TF_NONE; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (args.vocab <= M) {
    if (tid < args.vocab) { sv[tid] = x[tid]; si[tid] = tid; }
  } else {
    uint prefix = 0u, mask = 0u, rem = M;
    for (uint pass = 0u; pass < 4u; ++pass) {
      const uint shift = 24u - 8u * pass;
      if (tid < 256u) atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = tid; i < args.vocab; i += TF_TG) {
        const uint fk = tf_early_fkey(x[i]);
        if ((fk & mask) == prefix) atomic_fetch_add_explicit(&hist[(fk >> shift) & 0xFFu], 1u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint idx = (tid < 256u) ? (255u - tid) : 0u;
      const uint cnt = (tid < 256u) ? atomic_load_explicit(&hist[idx], memory_order_relaxed) : 0u;
      uint tot_unused = 0u;
      const uint above = tf_early_tg_scan1024(cnt, lane, simd, sg_sum, sg_off, sg_tot_all, tot_unused);
      if (tid < 256u && above < rem && above + cnt >= rem) {
        sh_bucket = idx;
        sh_above = above;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      prefix |= (sh_bucket << shift);
      mask |= (0xFFu << shift);
      rem -= sh_above;
    }

    const uint thresh = prefix;
    const uint gt_lim = M - rem;
    uint gt_taken = 0u, eq_taken = 0u;
    for (uint base = 0u; base < args.vocab; base += TF_TG) {
      const uint v = base + tid;
      bool is_gt = false, is_eq = false;
      float val = -INFINITY;
      if (v < args.vocab) {
        val = x[v];
        const uint fk = tf_early_fkey(val);
        is_gt = (fk > thresh);
        is_eq = (fk == thresh);
      }
      uint tot = 0u;
      const uint ex = tf_early_tg_scan1024((uint(is_gt) << 16) | uint(is_eq), lane, simd, sg_sum, sg_off, sg_tot_all, tot);
      const uint gt_rank = ex >> 16, eq_rank = ex & 0xFFFFu;
      const uint gt_tot = tot >> 16, eq_tot = tot & 0xFFFFu;
      const uint rem_eq = (rem > eq_taken) ? (rem - eq_taken) : 0u;
      if (is_gt) {
        const uint dst = gt_taken + gt_rank;
        sv[dst] = val;
        si[dst] = v;
      } else if (is_eq && eq_rank < rem_eq) {
        const uint dst = gt_lim + eq_taken + eq_rank;
        sv[dst] = val;
        si[dst] = v;
      }
      gt_taken += gt_tot;
      eq_taken += min(eq_tot, rem_eq);
      if (gt_taken == gt_lim && eq_taken == rem) break;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Bitonic sort 0..S-1 by (value desc, id asc).
  for (uint size = 2u; size <= S; size <<= 1u) {
    for (uint stride = size >> 1u; stride > 0u; stride >>= 1u) {
      if (tid < S) {
        const uint ixj = tid ^ stride;
        if (ixj > tid) {
          const bool asc = ((tid & size) == 0u);
          const float va = sv[tid], vb = sv[ixj];
          const uint ia = si[tid], ib = si[ixj];
          const bool do_swap = asc ? tf_early_ord_better(vb, ib, va, ia) : tf_early_ord_better(va, ia, vb, ib);
          if (do_swap) {
            sv[tid] = vb; si[tid] = ib;
            sv[ixj] = va; si[ixj] = ia;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  const float x0 = sv[0] / r.t;
  if (has_topp && tid < M) sprob[tid] = precise::exp(sv[tid] / r.t - x0);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {
    uint keep = M;
    if (has_topp) {
      float total = pure_nucleus ? sh_zfull : 0.0f;
      if (!pure_nucleus) for (uint c = 0u; c < M; ++c) total += sprob[c];
      float cum = 0.0f;
      uint below = 0u;
      for (uint c = 0u; c < M; ++c) {
        cum += sprob[c] / total;
        if (cum < r.top_p) below += 1u;
      }
      if (pure_nucleus && below == M && M < args.vocab) {
        keep = 0u; // nucleus tail exceeds 256 tokens -> clean host fallback
      } else {
        keep = min(M, below + 1u);
      }
    }
    if (keep > 0u && r.min_p > 0.0f) {
      const float floor_v = x0 + precise::log(r.min_p);
      uint n_keep = 0u;
      for (uint c = 0u; c < keep; ++c) {
        if (sv[c] / r.t >= floor_v) n_keep += 1u;
      }
      keep = max(1u, n_keep);
    }
    sh_keep = keep;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint keep = sh_keep;
  if (keep == 0u) {
    if (tid == 0u) picks[row] = TF_NONE;
    return;
  }

  float bs = -INFINITY, bv = -INFINITY;
  uint bi = TF_NONE;
  if (tid < keep && si[tid] != TF_NONE) {
    const float v = sv[tid];
    const uint id = si[tid];
    bs = v / r.t + tf_key_gumbel(tf_key_mix(r.base ^ ulong(id)));
    bv = v;
    bi = id;
  }
  for (ushort off = 16; off > 0; off >>= 1)
    tf_key_beats(bs, bv, bi, simd_shuffle_down(bs, off), simd_shuffle_down(bv, off), simd_shuffle_down(bi, off));
  if (lane == 0u) { part_s[simd] = bs; part_v[simd] = bv; part_i[simd] = bi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd != 0u) return;
  bs = lane < 8u ? part_s[lane] : -INFINITY;
  bv = lane < 8u ? part_v[lane] : -INFINITY;
  bi = lane < 8u ? part_i[lane] : TF_NONE;
  for (ushort off = 16; off > 0; off >>= 1)
    tf_key_beats(bs, bv, bi, simd_shuffle_down(bs, off), simd_shuffle_down(bv, off), simd_shuffle_down(bi, off));
  if (lane == 0u) picks[row] = bi;
}

// Per-slot accept descriptor for multi-slot behind-the-verify acceptance (24 bytes).
struct TfEarlySlot {
  uint r0;     // first row in tokens / picks / ipos / idest
  uint n;      // window rows for this slot
  uint block;  // draft block width B (uniform across the group)
  uint skew;   // test hook: 1 swaps acc 0 and 1
  uint pos_lo; // (start + 2) low 32 bits
  uint pos_hi; // (start + 2) high 32 bits
};

struct TfEarlyAcceptMultiArgs {
  uint slots;       // number of slots G (1..16)
  uint has_mk;      // 1 if mk_asks buffer is bound and pos0 should be written
  uint ask_stride;  // byte stride of TfMarkovAsk (40)
  uint pos_off;     // byte offset of pos0 inside TfMarkovAsk (8)
  uint asks_off;    // int offset of Plan.asks inside plan buffer
};

// Grid (G, 1, 1), TG (64, 1, 1): threadgroup j resolves slot j's accepted count and copies its SlotPlan into plan.
[[max_total_threads_per_threadgroup(64)]] kernel void tf_dsv41_early_accept_multi(
    constant TfEarlyAcceptMultiArgs& args [[buffer(0)]],
    device const TfEarlySlot* slot_desc [[buffer(1)]],
    device const uint* tokens [[buffer(2)]],
    device const uint* picks [[buffer(3)]],
    device const int* plans [[buffer(4)]],
    device int* plan [[buffer(5)]],
    device uint* ids [[buffer(6)]],
    device uint* result [[buffer(7)]],
    device uchar* mk_asks [[buffer(8)]],
    uint j [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (j >= args.slots) return;
  const TfEarlySlot s = slot_desc[j];
  uint acc = 0u;
  while (acc + 1u < s.n && tokens[s.r0 + acc + 1u] == picks[s.r0 + acc]) acc++;
  if (s.skew != 0u && s.n > 1u && acc < 2u) acc = 1u - acc;
  device const int* from = plans + ulong(j * TF_MAX_ROWS + acc) * TF_SLOT_PLAN_INTS;
  if (tid < s.n) {
    plan[2u * TF_MAX_ROWS + s.r0 + tid] = from[tid];
    plan[3u * TF_MAX_ROWS + s.r0 + tid] = from[TF_MAX_ROWS + tid];
  }
  if (tid < s.block) {
    plan[j * s.block + tid] = from[2u * TF_MAX_ROWS + tid];
    plan[TF_MAX_ROWS + j * s.block + tid] = from[3u * TF_MAX_ROWS + tid];
  }
  if (tid < 4u) {
    plan[args.asks_off + 4u * j + tid] = from[4u * TF_MAX_ROWS + tid];
  }
  if (tid == 0u) {
    const uint bonus = picks[s.r0 + acc];
    ids[j * s.block] = bonus;
    result[2u * j] = acc;
    result[2u * j + 1u] = bonus;
    if (args.has_mk != 0u) {
      *(device ulong*)(mk_asks + ulong(j) * args.ask_stride + args.pos_off) = (ulong(s.pos_hi) << 32 | s.pos_lo) + acc;
    }
  }
}
