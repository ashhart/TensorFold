// TP-sharded GLM-5.3-DFlash2 head (Q4-MMA & BF16 top-16), CandidateSelector (tau=1.5), GPU accept, and ANE bridge.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

inline float df2h_sum32(float v) {
  for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d);
  return v;
}

inline float4 df2h_u2_f4(uint2 w) {
  return float4(as_type<float>(w.x << 16), as_type<float>(w.x & 0xFFFF0000u),
                as_type<float>(w.y << 16), as_type<float>(w.y & 0xFFFF0000u));
}

inline bool df2h_better(float va, uint ia, float vb, uint ib) { return va > vb || (va == vb && ia < ib); }

struct Df2AcceptArgs { uint rows, base_pos, mask_tok, pad0; };

// Compute exact greedy kept count, anchor token, and anchor_pos on GPU and populate DFlash2 input token block.
kernel void df2_accept_setup(const device uint* win_ids [[buffer(0)]], const device uint* win_picks [[buffer(1)]],
                             constant Df2AcceptArgs& a [[buffer(2)]], device uint* meta [[buffer(3)]],
                             device uint* ids [[buffer(4)]], uint t [[thread_index_in_threadgroup]]) {
  if (t != 0u) return;
  uint kept = 1u;
  while (kept < a.rows && win_ids[kept] == win_picks[kept - 1u]) ++kept;
  const uint anchor = win_picks[kept - 1u];
  meta[0] = kept;
  meta[1] = anchor;
  meta[2] = a.base_pos + kept;
  meta[3] = a.base_pos;
  ids[0] = anchor;
  for (uint j = 1u; j < 8u; ++j) ids[j] = a.mask_tok;
}

struct Df2TreeAcceptArgs { uint rows, base_pos, mask_tok, k_loc; int parents[8]; };

// Single-TG GPU greedy tree accept for DFlash2: populates d.meta[0..13], d.ids, compact_disp[0..7], and hist_idx.
kernel void df2_tree_accept(const device uint* win_ids [[buffer(0)]], const device uint* win_picks [[buffer(1)]],
                            constant Df2TreeAcceptArgs& a [[buffer(2)]], device uint* meta [[buffer(3)]],
                            device uint* ids [[buffer(4)]], device uint* compact_disp [[buffer(5)]],
                            device uint* hist_idx [[buffer(6)]], uint t [[thread_index_in_threadgroup]]) {
  if (t != 0u) return;
  uint path[8], cur = 0u, kept = 1u;
  path[0] = 0u;
  while (kept < a.rows) {
    const uint want = win_picks[cur];
    uint nxt = 0xFFFFFFFFu;
    for (uint r = cur + 1u; r < a.rows; ++r) {
      if (a.parents[r] == int(cur) && win_ids[r] == want) { nxt = r; break; }
    }
    if (nxt == 0xFFFFFFFFu) break;
    path[kept++] = nxt;
    cur = nxt;
  }
  uint need_c = 0u;
  for (uint k = 0u; k < kept; ++k) if (path[k] != k) need_c = 1u;
  const uint anchor = win_picks[path[kept - 1u]];
  meta[0] = kept; meta[1] = anchor; meta[2] = a.base_pos + kept; meta[3] = a.base_pos; meta[4] = need_c; meta[5] = 0u;
  for (uint k = 0u; k < 8u; ++k) meta[6u + k] = (k < kept) ? path[k] : 0u;
  ids[0] = anchor;
  for (uint j = 1u; j < 8u; ++j) ids[j] = a.mask_tok;
  compact_disp[0] = need_c; compact_disp[1] = need_c; compact_disp[2] = need_c; compact_disp[3] = 0u;
  const uint tgs = (a.k_loc + 255u) >> 8;
  compact_disp[4] = need_c ? tgs : 0u; compact_disp[5] = need_c; compact_disp[6] = need_c; compact_disp[7] = 0u;
  for (uint j = 0u; j < a.rows; ++j) hist_idx[j] = (j < kept) ? path[j] : 0u;
}

struct Df2LmArgs { uint vocab_loc, K, vocab_lo, rows, n_tgs, pad0, pad1, pad2; };

// Fast INT4 g64 hardware simdgroup_matrix<half,8,8> lm_head + exact top-16/64 (128t, 0.357 ms vs 1.000 ms BF16!).
kernel void df2_lm_q4_mma_7(const device float4* X4 [[buffer(0)]], const device uint4* W4 [[buffer(1)]],
                            const device half* S [[buffer(2)]], constant Df2LmArgs& a [[buffer(3)]],
                            device float* out_v [[buffer(4)]], device uint* out_i [[buffer(5)]],
                            uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint LD = 72u;
  threadgroup half xs[8u * LD];
  threadgroup half ws[4u * 16u * LD];
  threadgroup half* w16 = ws + sg * 16u * LD;
  const uint V = a.vocab_loc, K = a.K, rows = min(a.rows, 7u), n_tgs = a.n_tgs;
  const uint o0 = tg * 64u + sg * 16u;
  const uint k32 = K >> 5, k4 = K >> 2, g_cols = K >> 6, k_steps = K >> 6;
  const uint lr = lane >> 1, lc = (lane & 1u) << 5;
  const bool sg_act = o0 < V;
  simdgroup_matrix<float, 8, 8> acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_matrix<float, 8, 8> acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint kb = 0u; kb < k_steps; ++kb) {
    const uint k_base = kb * 64u;
    if (tid < 112u) {
      const uint r = tid >> 4, c4 = tid & 15u;
      *reinterpret_cast<threadgroup half4*>(&xs[r * LD + c4 * 4u]) =
          (r < rows) ? half4(clamp(X4[ulong(r) * k4 + (k_base >> 2) + c4], -65504.0f, 65504.0f)) : half4(0.0h);
    }
    if (sg_act && (o0 + lr) < V) {
      const half sc = S[ulong(o0 + lr) * g_cols + kb];
      const uint4 qw = W4[ulong(o0 + lr) * k32 + kb * 2u + (lane & 1u)];
      const uint w[4] = {qw.x, qw.y, qw.z, qw.w};
      threadgroup half4* d4 = reinterpret_cast<threadgroup half4*>(w16 + lr * LD + lc);
      #pragma clang loop unroll(full)
      for (uint k = 0u; k < 4u; ++k) {
        const uint v = w[k];
        const half2 a0 = (as_type<half2>(((v & 0x000Fu) | ((v & 0x00F0u) << 12)) | 0x64006400u) - 1032.0h) * sc;
        const half2 a1 = (as_type<half2>((((v >> 8) & 0x000Fu) | (((v >> 8) & 0x00F0u) << 12)) | 0x64006400u) - 1032.0h) * sc;
        const half2 b0 = (as_type<half2>((((v >> 16) & 0x000Fu) | (((v >> 16) & 0x00F0u) << 12)) | 0x64006400u) - 1032.0h) * sc;
        const half2 b1 = (as_type<half2>((((v >> 24) & 0x000Fu) | (((v >> 24) & 0x00F0u) << 12)) | 0x64006400u) - 1032.0h) * sc;
        d4[2u * k] = half4(a0, a1);
        d4[2u * k + 1u] = half4(b0, b1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_act) {
      #pragma clang loop unroll(full)
      for (uint ks = 0u; ks < 8u; ++ks) {
        simdgroup_matrix<half, 8, 8> ma, mb0, mb1;
        simdgroup_load(ma, xs + ks * 8u, LD, ulong2(0, 0), false);
        simdgroup_load(mb0, w16 + ks * 8u, LD, ulong2(0, 0), true);
        simdgroup_load(mb1, w16 + 8u * LD + ks * 8u, LD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc0, ma, mb0, acc0);
        simdgroup_multiply_accumulate(acc1, ma, mb1, acc1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup float* tv = reinterpret_cast<threadgroup float*>(ws);
  if (sg_act) {
    threadgroup float* stg = tv + 7u * 64u + sg * 128u;
    simdgroup_store(acc0, stg, 16u, ulong2(0, 0), false);
    simdgroup_store(acc1, stg + 8u, 16u, ulong2(0, 0), false);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lane; idx < rows * 16u; idx += 32u) {
      const uint r = idx >> 4, c = idx & 15u;
      tv[r * 64u + sg * 16u + c] = (o0 + c < V) ? stg[idx] : -INFINITY;
    }
  } else {
    for (uint idx = lane; idx < rows * 16u; idx += 32u) tv[(idx >> 4) * 64u + sg * 16u + (idx & 15u)] = -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = sg; r < rows; r += 4u) {
    const uint c0 = tg * 64u + lane, c1 = c0 + 32u;
    float v0 = (c0 < V) ? tv[r * 64u + lane] : -INFINITY;
    uint i0 = (c0 < V) ? (a.vocab_lo + c0) : 0xFFFFFFFFu;
    float v1 = (c1 < V) ? tv[r * 64u + 32u + lane] : -INFINITY;
    uint i1 = (c1 < V) ? (a.vocab_lo + c1) : 0xFFFFFFFFu;
    for (uint k = 2u; k <= 32u; k <<= 1) for (uint j = k >> 1; j > 0u; j >>= 1) {
      const uint l = lane ^ j;
      const bool down = (lane & k) == 0u;
      const float lv0 = simd_shuffle(v0, ushort(l)); const uint li0 = simd_shuffle(i0, ushort(l));
      if ((l > lane) ? (down ? df2h_better(lv0, li0, v0, i0) : df2h_better(v0, i0, lv0, li0))
                     : (down ? df2h_better(v0, i0, lv0, li0) : df2h_better(lv0, li0, v0, i0))) { v0 = lv0; i0 = li0; }
      const float lv1 = simd_shuffle(v1, ushort(l)); const uint li1 = simd_shuffle(i1, ushort(l));
      if ((l > lane) ? (down ? df2h_better(lv1, li1, v1, i1) : df2h_better(v1, i1, lv1, li1))
                     : (down ? df2h_better(v1, i1, lv1, li1) : df2h_better(lv1, li1, v1, i1))) { v1 = lv1; i1 = li1; }
    }
    float vm = (lane < 16u) ? v0 : simd_shuffle(v1, ushort(lane - 16u));
    uint im = (lane < 16u) ? i0 : simd_shuffle(i1, ushort(lane - 16u));
    for (uint k = 2u; k <= 32u; k <<= 1) for (uint j = k >> 1; j > 0u; j >>= 1) {
      const uint l = lane ^ j;
      const bool down = (lane & k) == 0u;
      const float lv = simd_shuffle(vm, ushort(l)); const uint li = simd_shuffle(im, ushort(l));
      if ((l > lane) ? (down ? df2h_better(lv, li, vm, im) : df2h_better(vm, im, lv, li))
                     : (down ? df2h_better(vm, im, lv, li) : df2h_better(lv, li, vm, im))) { vm = lv; im = li; }
    }
    if (lane < 16u) {
      const ulong dst = (ulong(r) * n_tgs + tg) * 16u + lane;
      out_v[dst] = vm; out_i[dst] = im;
    }
  }
}

// Fallback exact BF16 lm_head (rows 1..7) + per-TG top-16 out of 32 vocab cols.
kernel void df2_lm_bf16_topk16(const device float4* X4 [[buffer(0)]], const device uint4* W4 [[buffer(1)]],
                               constant Df2LmArgs& a [[buffer(2)]], device float* out_v [[buffer(3)]],
                               device uint* out_i [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float4 xs[7 * 64];
  threadgroup float tv[7][32];
  threadgroup uint ti[7][32];
  const uint rows = min(a.rows, 7u), k8 = a.K >> 3, k4 = a.K >> 2, c256 = a.K >> 8;
  const uint o_base = tg * 32u + sg * 4u;
  float acc[4][7];
  #pragma clang loop unroll(full)
  for (uint q = 0u; q < 4u; ++q)
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 7u; ++r) acc[q][r] = 0.0f;
  for (uint c = 0u; c < c256; ++c) {
    const uint xb = c * 64u;
    for (uint idx = tid; idx < rows * 64u; idx += 256u) xs[idx] = X4[ulong(idx >> 6) * k4 + xb + (idx & 63u)];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint q = 0u; q < 4u; ++q) {
      const uint o = o_base + q;
      if (o < a.vocab_loc) {
        const uint4 qw = W4[ulong(o) * k8 + c * 32u + lane];
        const float4 wa = df2h_u2_f4(uint2(qw.x, qw.y)), wb = df2h_u2_f4(uint2(qw.z, qw.w));
        for (uint r = 0u; r < rows; ++r) acc[q][r] += dot(wa, xs[r * 64u + lane * 2u]) + dot(wb, xs[r * 64u + lane * 2u + 1u]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  #pragma clang loop unroll(full)
  for (uint q = 0u; q < 4u; ++q) {
    const uint o = o_base + q, slot = sg * 4u + q;
    for (uint r = 0u; r < rows; ++r) {
      const float v = (o < a.vocab_loc) ? df2h_sum32(acc[q][r]) : -INFINITY;
      if (lane == 0u) { tv[r][slot] = v; ti[r][slot] = (o < a.vocab_loc) ? (a.vocab_lo + o) : 0xFFFFFFFFu; }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg < rows) {
    const uint r = sg;
    float v = tv[r][lane];
    uint id = ti[r][lane];
    for (uint k = 2u; k <= 32u; k <<= 1)
      for (uint j = k >> 1; j > 0u; j >>= 1) {
        const uint l = lane ^ j;
        const float lv = simd_shuffle(v, ushort(l));
        const uint li = simd_shuffle(id, ushort(l));
        const bool down = (lane & k) == 0u;
        const bool swap = (l > lane) ? (down ? df2h_better(lv, li, v, id) : df2h_better(v, id, lv, li))
                                     : (down ? df2h_better(v, id, lv, li) : df2h_better(lv, li, v, id));
        if (swap) { v = lv; id = li; }
      }
    if (lane < 16u) {
      const ulong dst = (ulong(r) * a.n_tgs + tg) * 16u + lane;
      out_v[dst] = v; out_i[dst] = id;
    }
  }
}

// Race-free reduce of n_tgs * 16 per-TG candidates into 16 top-k pairs [16 val_u32, 16 id_u32] in send_slot.
kernel void df2_topk16_reduce(const device float* in_v [[buffer(0)]], const device uint* in_i [[buffer(1)]],
                              constant Df2LmArgs& a [[buffer(2)]], device uint* out_u [[buffer(3)]],
                              uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sbv[8];
  threadgroup uint sbi[8];
  const uint total = a.n_tgs * 16u;
  const ulong base = ulong(r) * total;
  float best_v[16];
  uint best_i[16];
  for (uint k = 0u; k < 16u; ++k) { best_v[k] = -INFINITY; best_i[k] = 0xFFFFFFFFu; }
  for (uint idx = t; idx < total; idx += 256u) {
    const float v = in_v[base + idx];
    const uint id = in_i[base + idx];
    if (df2h_better(v, id, best_v[15], best_i[15])) {
      int pos = 15;
      while (pos > 0 && df2h_better(v, id, best_v[pos - 1], best_i[pos - 1])) {
        best_v[pos] = best_v[pos - 1]; best_i[pos] = best_i[pos - 1]; --pos;
      }
      best_v[pos] = v; best_i[pos] = id;
    }
  }
  uint cur = 0u;
  for (uint k = 0u; k < 16u; ++k) {
    float mv = best_v[cur];
    uint mi = best_i[cur];
    for (ushort d = 16; d > 0; d >>= 1) {
      const float ov = simd_shuffle_xor(mv, d);
      const uint oi = simd_shuffle_xor(mi, d);
      if (df2h_better(ov, oi, mv, mi)) { mv = ov; mi = oi; }
    }
    if (lane == 0u) { sbv[sg] = mv; sbi[sg] = mi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = -INFINITY; uint gi = 0xFFFFFFFFu;
    for (uint s = 0u; s < 8u; ++s) if (df2h_better(sbv[s], sbi[s], gm, gi)) { gm = sbv[s]; gi = sbi[s]; }
    if (t == 0u) { out_u[r * 32u + k] = as_type<uint>(gm); out_u[r * 32u + 16u + k] = gi; }
    if (best_i[cur] == gi && best_v[cur] == gm) {
      if (cur < 15u) ++cur;
      else { best_v[cur] = -INFINITY; best_i[cur] = 0xFFFFFFFFu; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

struct Df2WalkArgs {
  uint slots, rank, top, vocab;
  uint tp_rank, tp_size, slot_u32, anchor;
  float edge_scale;
  uint pad0, pad1, pad2;
};

// Merge TP peers' top-16 and grow exact 7-node best-first tree ([1..7]=tok, [8..14]=reach, [16..22]=par, [24..30]=own).
kernel void df2_merge_walk(const device uint* mine [[buffer(0)]], const device uint* recv [[buffer(1)]],
                           const device float* H [[buffer(2)]], const device bfloat* A [[buffer(3)]],
                           const device bfloat* B [[buffer(4)]], constant Df2WalkArgs& a [[buffer(5)]],
                           device uint* seq [[buffer(6)]], const device uint* meta [[buffer(7)]],
                           uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float uv[7][16], edge_s[16], open_p[32], open_own[32], cur_p;
  threadgroup uint ui[7][16], open_tok[32], open_row[32], n_open, cur_pred, cur_row;
  threadgroup int open_par[32], cur_par;
  threadgroup float4 ah4[64];
  if (t < a.slots) {
    const uint e = t;
    float mv[16]; uint mi[16];
    for (uint k = 0u; k < 16u; ++k) { mv[k] = -INFINITY; mi[k] = 0xFFFFFFFFu; }
    for (uint p = 0u; p < a.tp_size; ++p) {
      const device uint* src = (p == a.tp_rank) ? mine : (recv + ulong(p) * a.slot_u32);
      for (uint k = 0u; k < 16u; ++k) {
        const float v = as_type<float>(src[e * 32u + k]);
        const uint id = src[e * 32u + 16u + k];
        if (df2h_better(v, id, mv[15], mi[15])) {
          int pos = 15;
          while (pos > 0 && df2h_better(v, id, mv[pos - 1], mi[pos - 1])) {
            mv[pos] = mv[pos - 1]; mi[pos] = mi[pos - 1]; --pos;
          }
          mv[pos] = v; mi[pos] = id;
        }
      }
    }
    for (uint k = 0u; k < 16u; ++k) { uv[e][k] = mv[k]; ui[e][k] = mi[k]; }
  }
  if (t == 0u) {
    const uint anc = (a.anchor == 0xFFFFFFFFu) ? meta[1] : a.anchor;
    cur_pred = anc; cur_par = -1; cur_row = 0u; cur_p = 1.0f; n_open = 0u;
    seq[0] = anc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const device uint2* A2 = reinterpret_cast<const device uint2*>(A);
  const device uint2* B2 = reinterpret_cast<const device uint2*>(B);
  const device float4* H4 = reinterpret_cast<const device float4*>(H);
  constexpr float INV_TAU = 1.0f / 1.5f;
  const uint b_max = (a.pad0 != 0u) ? 3u : 1u;
  for (uint step = 0u; step < a.slots; ++step) {
    if (cur_row < a.slots && cur_pred < a.vocab) {
      if (t < 64u) ah4[t] = df2h_u2_f4(A2[ulong(cur_pred) * 64u + t]) * H4[ulong(cur_row) * 64u + t];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint k = 0u; k < 2u; ++k) {
        const uint c = sg * 2u + k;
        if (c < a.top) {
          const ulong bo = ulong(min(ui[cur_row][c], a.vocab - 1u)) * 64u;
          const float s = df2h_sum32(dot(ah4[lane], df2h_u2_f4(B2[bo + lane])) +
                                     dot(ah4[lane + 32u], df2h_u2_f4(B2[bo + lane + 32u])));
          if (lane == 0u) edge_s[c] = s;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (t == 0u) {
        float sc[16], max_sc = -INFINITY;
        for (uint c = 0u; c < a.top; ++c) {
          const float s = (ui[cur_row][c] < a.vocab) ? (uv[cur_row][c] + a.edge_scale * edge_s[c]) : -INFINITY;
          sc[c] = s;
          if (s > max_sc) max_sc = s;
        }
        float z = 0.0f;
        for (uint c = 0u; c < a.top; ++c) if (sc[c] > -INFINITY) z += exp((sc[c] - max_sc) * INV_TAU);
        const float inv_z = (z > 0.0f) ? (1.0f / z) : 0.0f;
        bool used[16];
        for (uint c = 0u; c < 16u; ++c) used[c] = false;
        for (uint b = 0u; b < b_max && n_open < 32u; ++b) {
          uint best_c = 16u; float best_s = -INFINITY;
          for (uint c = 0u; c < a.top; ++c) if (!used[c] && sc[c] > best_s) { best_s = sc[c]; best_c = c; }
          if (best_c >= a.top) break;
          used[best_c] = true;
          const float alt_cal = (b == 0u) ? 1.0f : 0.88f;
          const float ow = clamp(alt_cal * exp((best_s - max_sc) * INV_TAU) * inv_z, 0.0f, 0.99f);
          const float rp = clamp(cur_p * ow, 0.0f, 0.99f);
          open_p[n_open] = rp; open_own[n_open] = ow;
          open_tok[n_open] = ui[cur_row][best_c];
          open_par[n_open] = cur_par; open_row[n_open] = cur_row;
          ++n_open;
        }
      }
    }
    if (t == 0u) {
      if (n_open == 0u) {
        seq[1u + step] = 0xFFFFFFFFu; seq[8u + step] = 0u;
        seq[16u + step] = 0xFFFFFFFFu; seq[24u + step] = 0u;
        cur_row = a.slots;
      } else {
        uint bi = 0u;
        for (uint i = 1u; i < n_open; ++i) {
          if (open_p[i] > open_p[bi] ||
              (open_p[i] == open_p[bi] && (open_par[i] < open_par[bi] ||
               (open_par[i] == open_par[bi] && open_tok[i] < open_tok[bi])))) bi = i;
        }
        const uint tok = open_tok[bi], rw = open_row[bi];
        const int par = open_par[bi];
        const float rp = open_p[bi], ow = open_own[bi];
        --n_open;
        open_p[bi] = open_p[n_open]; open_own[bi] = open_own[n_open];
        open_tok[bi] = open_tok[n_open]; open_par[bi] = open_par[n_open]; open_row[bi] = open_row[n_open];
        seq[1u + step] = tok;
        seq[8u + step] = as_type<uint>(rp);
        seq[16u + step] = as_type<uint>(par);
        seq[24u + step] = as_type<uint>(ow);
        cur_pred = tok; cur_par = int(step); cur_row = rw + 1u; cur_p = rp;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

struct Df2CompactArgs { uint kept, k_loc, pad0, pad1; uint path[8]; };

// In-place compaction of tapped target hidden rows `taps[path[k]] -> taps[k]` after a branching tree accept (<0.01 ms).
kernel void df2_compact_taps(device bfloat* taps [[buffer(0)]], constant Df2CompactArgs& a [[buffer(1)]],
                             uint gid [[thread_position_in_grid]]) {
  if (gid >= a.k_loc) return;
  for (uint k = 1u; k < a.kept; ++k) {
    const uint src = a.path[k];
    if (src != k) taps[ulong(k) * a.k_loc + gid] = taps[ulong(src) * a.k_loc + gid];
  }
}

// Indirect-dispatched tap compaction reading kept & path from d.meta (0 TGs when main branch wins!).
kernel void df2_compact_taps_indirect(device bfloat* taps [[buffer(0)]], const device uint* meta [[buffer(1)]],
                                      constant uint& k_loc [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= k_loc) return;
  const uint kept = meta[0];
  for (uint k = 1u; k < kept; ++k) {
    const uint src = meta[6u + k];
    if (src != k) taps[ulong(k) * k_loc + gid] = taps[ulong(src) * k_loc + gid];
  }
}

// Fast INT8 g64 hardware simdgroup_matrix<half,8,8> lm_head + exact top-16/64 (128t, 18x lower error than Q4!).
kernel void df2_lm_q8_mma_7(const device float4* X4 [[buffer(0)]], const device uint4* W4 [[buffer(1)]],
                            const device half* S [[buffer(2)]], constant Df2LmArgs& a [[buffer(3)]],
                            device float* out_v [[buffer(4)]], device uint* out_i [[buffer(5)]],
                            uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint LD = 72u;
  threadgroup half xs[8u * LD];
  threadgroup half ws[4u * 16u * LD];
  threadgroup half* w16 = ws + sg * 16u * LD;
  const uint V = a.vocab_loc, K = a.K, rows = min(a.rows, 7u), n_tgs = a.n_tgs, o0 = tg * 64u + sg * 16u;
  const uint k16 = K >> 4, k4 = K >> 2, g_cols = K >> 6, k_steps = K >> 6, lr = lane >> 1, lc = (lane & 1u) << 5;
  const bool sg_act = o0 < V;
  simdgroup_matrix<float, 8, 8> acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_matrix<float, 8, 8> acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint kb = 0u; kb < k_steps; ++kb) {
    const uint k_base = kb * 64u;
    if (tid < 112u) {
      const uint r = tid >> 4, c4 = tid & 15u;
      *reinterpret_cast<threadgroup half4*>(&xs[r * LD + c4 * 4u]) =
          (r < rows) ? half4(clamp(X4[ulong(r) * k4 + (k_base >> 2) + c4], -65504.0f, 65504.0f)) : half4(0.0h);
    }
    if (sg_act && (o0 + lr) < V) {
      const half sc = S[ulong(o0 + lr) * g_cols + kb];
      const ulong wb = ulong(o0 + lr) * k16 + kb * 4u + (lane & 1u) * 2u;
      const uint4 qa = W4[wb], qb = W4[wb + 1u];
      threadgroup half4* d4 = reinterpret_cast<threadgroup half4*>(w16 + lr * LD + lc);
      d4[0] = half4(as_type<char4>(qa.x)) * sc; d4[1] = half4(as_type<char4>(qa.y)) * sc;
      d4[2] = half4(as_type<char4>(qa.z)) * sc; d4[3] = half4(as_type<char4>(qa.w)) * sc;
      d4[4] = half4(as_type<char4>(qb.x)) * sc; d4[5] = half4(as_type<char4>(qb.y)) * sc;
      d4[6] = half4(as_type<char4>(qb.z)) * sc; d4[7] = half4(as_type<char4>(qb.w)) * sc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_act) {
      #pragma clang loop unroll(full)
      for (uint ks = 0u; ks < 8u; ++ks) {
        simdgroup_matrix<half, 8, 8> ma, mb0, mb1;
        simdgroup_load(ma, xs + ks * 8u, LD, ulong2(0, 0), false);
        simdgroup_load(mb0, w16 + ks * 8u, LD, ulong2(0, 0), true);
        simdgroup_load(mb1, w16 + 8u * LD + ks * 8u, LD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc0, ma, mb0, acc0);
        simdgroup_multiply_accumulate(acc1, ma, mb1, acc1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup float* tv = reinterpret_cast<threadgroup float*>(ws);
  if (sg_act) {
    threadgroup float* stg = tv + 7u * 64u + sg * 128u;
    simdgroup_store(acc0, stg, 16u, ulong2(0, 0), false);
    simdgroup_store(acc1, stg + 8u, 16u, ulong2(0, 0), false);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lane; idx < rows * 16u; idx += 32u) {
      const uint r = idx >> 4, c = idx & 15u;
      tv[r * 64u + sg * 16u + c] = (o0 + c < V) ? stg[idx] : -INFINITY;
    }
  } else {
    for (uint idx = lane; idx < rows * 16u; idx += 32u) tv[(idx >> 4) * 64u + sg * 16u + (idx & 15u)] = -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = sg; r < rows; r += 4u) {
    const uint c0 = tg * 64u + lane, c1 = c0 + 32u;
    float v0 = (c0 < V) ? tv[r * 64u + lane] : -INFINITY, v1 = (c1 < V) ? tv[r * 64u + 32u + lane] : -INFINITY;
    uint i0 = (c0 < V) ? (a.vocab_lo + c0) : 0xFFFFFFFFu, i1 = (c1 < V) ? (a.vocab_lo + c1) : 0xFFFFFFFFu;
    for (uint k = 2u; k <= 32u; k <<= 1) for (uint j = k >> 1; j > 0u; j >>= 1) {
      const uint l = lane ^ j; const bool down = (lane & k) == 0u;
      const float lv0 = simd_shuffle(v0, ushort(l)); const uint li0 = simd_shuffle(i0, ushort(l));
      if ((l > lane) ? (down ? df2h_better(lv0, li0, v0, i0) : df2h_better(v0, i0, lv0, li0))
                     : (down ? df2h_better(v0, i0, lv0, li0) : df2h_better(lv0, li0, v0, i0))) { v0 = lv0; i0 = li0; }
      const float lv1 = simd_shuffle(v1, ushort(l)); const uint li1 = simd_shuffle(i1, ushort(l));
      if ((l > lane) ? (down ? df2h_better(lv1, li1, v1, i1) : df2h_better(v1, i1, lv1, li1))
                     : (down ? df2h_better(v1, i1, lv1, li1) : df2h_better(lv1, li1, v1, i1))) { v1 = lv1; i1 = li1; }
    }
    float vm = (lane < 16u) ? v0 : simd_shuffle(v1, ushort(lane - 16u));
    uint im = (lane < 16u) ? i0 : simd_shuffle(i1, ushort(lane - 16u));
    for (uint k = 2u; k <= 32u; k <<= 1) for (uint j = k >> 1; j > 0u; j >>= 1) {
      const uint l = lane ^ j; const bool down = (lane & k) == 0u;
      const float lv = simd_shuffle(vm, ushort(l)); const uint li = simd_shuffle(im, ushort(l));
      if ((l > lane) ? (down ? df2h_better(lv, li, vm, im) : df2h_better(vm, im, lv, li))
                     : (down ? df2h_better(vm, im, lv, li) : df2h_better(lv, li, vm, im))) { vm = lv; im = li; }
    }
    if (lane < 16u) { const ulong dst = (ulong(r) * n_tgs + tg) * 16u + lane; out_v[dst] = vm; out_i[dst] = im; }
  }
}

struct Df2AneArgs { uint rows, channels, s_ane, row0, src_stride, src_off, add, pad0; };

// Transpose taps_loc [rows, src_stride] (bf16) * 2^-6 -> ANE planar [1, channels, 1, s_ane] (fp16, overflow-safe).
kernel void df2_ane_pack_in(const device bfloat* src [[buffer(0)]], device half* dst [[buffer(1)]],
                            constant Df2AneArgs& a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.s_ane || p.y >= a.channels) return;
  const uint r = a.row0 + p.x;
  const float v = (r < a.rows) ? (float(src[ulong(r) * a.src_stride + a.src_off + p.y]) * 0.015625f) : 0.0f;
  dst[ulong(p.y) * a.s_ane + p.x] = half(clamp(v, -65504.0f, 65504.0f));
}

// Transpose ANE planar [1, channels, 1, s_ane] (fp16) * 2^6 -> pair.part() [rows, channels] (fp32), optional add.
kernel void df2_ane_unpack_out(const device half* src [[buffer(0)]], device float* dst [[buffer(1)]],
                               constant Df2AneArgs& a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.channels || p.y >= a.s_ane) return;
  const uint r = a.row0 + p.y;
  if (r < a.rows) {
    const float v = float(src[ulong(p.x) * a.s_ane + p.y]) * 64.0f;
    if (a.add != 0u) dst[ulong(r) * a.channels + p.x] += v;
    else dst[ulong(r) * a.channels + p.x] = v;
  }
}
