// A MoE layer's route in two launches for 1-16 rows: router logits (each 4-expert block's weights staged once, every
// row's sums in the one-row kernel's order), then the top-k and the expert groups this Mac computes.
// Defines: TF_K (hidden), TF_E (experts, a multiple of 4), TF_TOPK, TF_MAXR (rows at most, <= 16).
#include <metal_stdlib>
using namespace metal;

#ifndef TF_OUT_T
#define TF_OUT_T float
#endif
constant constexpr int ITERS = TF_K / 32;
constant constexpr int PER = (TF_E + 31) / 32;

// Logits [rows, E] = x W^T. A threadgroup takes a 4-expert block q (tg.x) for four rows (tg.y): the block's packed bf16
// weights and the rows' x staged a quarter of K at a time. Lane thrM * 4 + c of simdgroup 0 takes row 4 tg.y + c: its
// sums over k = 32 it + 4 thrM + tm in order, then the tree over thrM (the one-row kernel's order).
[[kernel]] void tf_route_logits(const device bfloat* X [[buffer(0)]], const device uint4* RP [[buffer(1)]],
                                constant int& rows [[buffer(2)]], device TF_OUT_T* OUT [[buffer(3)]],
                                uint2 tg [[threadgroup_position_in_grid]], uint2 tpos [[thread_position_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]], uint s [[simdgroup_index_in_threadgroup]]) {
  constexpr int QI = ITERS / 4; // iterations a quarter
  const uint t = tpos.x;
  threadgroup uint4 w[8 * QI * 2];
  threadgroup bfloat xs[4][32 * QI];
  const int q = int(tg.x), r0 = int(tg.y) * 4;
  const device uint4* src = RP + size_t(q) * 8 * ITERS * 2;
  const int thrM = int(lane) / 4, c = int(lane) % 4;
  const int r = r0 + c;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int part = 0; part < 4; part++) {
    if (part > 0) threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(t); i < 8 * QI * 2; i += 256) { // (m, it in this quarter, h) of the block's (m, it, h) layout
      const int m = i / (QI * 2), rest = i % (QI * 2);
      w[i] = src[(m * ITERS + part * QI) * 2 + rest];
    }
    for (int i = int(t); i < 4 * 32 * QI / 4; i += 256) { // four bf16 of a row's quarter a thread
      const int rr = i / (8 * QI), k = (i % (8 * QI)) * 4;
      const device bfloat* xr = X + size_t(min(r0 + rr, rows - 1)) * TF_K + part * 32 * QI + k;
      xs[rr][k] = xr[0]; xs[rr][k + 1] = xr[1]; xs[rr][k + 2] = xr[2]; xs[rr][k + 3] = xr[3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s != 0) continue;
    for (int hi = 0; hi < QI; hi++) {
      float inter[4][4];
      for (int h = 0; h < 2; h++) {
        const uint4 v = w[(thrM * QI + hi) * 2 + h];
        const uint words[4] = {v.x, v.y, v.z, v.w};
        for (int j = 0; j < 4; j++) {
          const int e = h * 8 + j * 2;
          inter[e / 4][e % 4] = as_type<float>(words[j] << 16);
          inter[(e + 1) / 4][(e + 1) % 4] = as_type<float>(words[j] & 0xffff0000u);
        }
      }
      const int bm = 4 * thrM + 32 * hi;
      float vc[4];
      for (int tm = 0; tm < 4; tm++) vc[tm] = float(xs[c][bm + tm]);
      for (int tm = 0; tm < 4; tm++)
        for (int tn = 0; tn < 4; tn++) acc[tn] += vc[tm] * inter[tm][tn];
    }
  }
  if (s != 0) return;
  for (int tn = 0; tn < 4; tn++) {
    float v = acc[tn];
    for (ushort sm = 4; sm >= 1; sm >>= 1) v += simd_shuffle_down(v, 4 * sm);
    if (thrM == 0 && r < rows) OUT[size_t(r) * TF_E + 4 * q + tn] = static_cast<TF_OUT_T>(v);
  }
}

inline float tf_sigmoid_precise(float x) {
  float e = metal::precise::exp(metal::abs(x));
  float y = 1.0f / (1.0f + e);
  return (x < 0) ? y : (1.0f - y);
}

struct RouteArgs {
  int rows;
  int lo, hi; // the experts this Mac computes: [lo, hi)
  float scale;
};

// One threadgroup of 512: each row's top TF_TOPK experts by sigmoid + bias (ties to the lower id) and their normalized,
// scaled weights; then the experts in [lo, hi) the window picked (ascending, ids from lo) with their picks, and the picks
// this Mac and the peer compute (ascending), with this Mac's count also in `cnt` (the link's word).
[[kernel]] void tf_route_select(const device float* LOGITS [[buffer(0)]], const device float* BIAS [[buffer(1)]],
                                constant RouteArgs& a [[buffer(2)]], device int* PICK [[buffer(3)]],
                                device float* WTS [[buffer(4)]], device int* LIDS [[buffer(5)]],
                                device int* LMEM [[buffer(6)]], device int* LCOUNT [[buffer(7)]],
                                device int* MINE [[buffer(8)]], device int* THEIRS [[buffer(9)]],
                                device int* COUNTS [[buffer(10)]], device atomic_uint* CNT [[buffer(11)]],
                                uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                uint g [[simdgroup_index_in_threadgroup]]) {
  const int R = a.rows;
  const int e = int(t);
  threadgroup int picks[TF_MAXR * TF_TOPK];
  threadgroup int offs[3][16];
  if (int(g) < R) {
    const int r = int(g);
    float c[PER], sc[PER];
    for (int j = 0; j < PER; j++) {
      const int id = j * 32 + int(lane);
      if (id < TF_E) {
        sc[j] = tf_sigmoid_precise(LOGITS[r * TF_E + id]);
        c[j] = sc[j] + BIAS[id];
      } else {
        sc[j] = 0.0f; c[j] = -INFINITY;
      }
    }
    float w[TF_TOPK];
    for (int k = 0; k < TF_TOPK; k++) {
      float best = -INFINITY, bsc = 0.0f;
      int bid = TF_E;
      for (int j = 0; j < PER; j++) {
        const int id = j * 32 + int(lane);
        if (id < TF_E && (c[j] > best || (c[j] == best && id < bid))) { best = c[j]; bid = id; bsc = sc[j]; }
      }
      for (int off = 16; off > 0; off /= 2) {
        const float ob = simd_shuffle_xor(best, off);
        const int oi = simd_shuffle_xor(bid, off);
        const float os = simd_shuffle_xor(bsc, off);
        if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; bsc = os; }
      }
      w[k] = bsc;
      if (int(lane) == bid % 32) c[bid / 32] = -INFINITY;
      if (lane == 0) { picks[r * TF_TOPK + k] = bid; PICK[r * TF_TOPK + k] = bid; }
    }
    if (lane == 0) {
      float total = w[0];
      for (int k = 1; k < TF_TOPK; k++) total = total + w[k];
      for (int k = 0; k < TF_TOPK; k++) WTS[r * TF_TOPK + k] = (w[k] / total) * a.scale;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // the held experts the window picked, ascending, with their member picks in pick order
  int members[TF_MAXR];
  int count = 0;
  const bool held_id = e >= a.lo && e < a.hi;
  if (held_id)
    for (int p = 0; p < R * TF_TOPK; p++)
      if (picks[p] == e) members[count++] = p;
  const int held = count > 0 ? 1 : 0;
  // each pick: computed here or by the peer
  const int pe = e < R * TF_TOPK ? picks[e] : -1;
  const int mine = pe >= a.lo && pe < a.hi ? 1 : 0;
  const int theirs = pe >= 0 && mine == 0 ? 1 : 0;
  const int b0 = simd_prefix_exclusive_sum(held), b1 = simd_prefix_exclusive_sum(mine), b2 = simd_prefix_exclusive_sum(theirs);
  if (lane == 31) { offs[0][g] = b0 + held; offs[1][g] = b1 + mine; offs[2][g] = b2 + theirs; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int o0 = 0, o1 = 0, o2 = 0, n0 = 0, n1 = 0, n2 = 0;
  for (uint k = 0; k < 16; k++) {
    if (k < g) { o0 += offs[0][k]; o1 += offs[1][k]; o2 += offs[2][k]; }
    n0 += offs[0][k]; n1 += offs[1][k]; n2 += offs[2][k];
  }
  if (held) {
    const int u = o0 + b0;
    LIDS[u] = e - a.lo;
    for (int j = 0; j < TF_MAXR; j++) LMEM[u * TF_MAXR + j] = j < count ? members[j] : -1;
  }
  if (mine) MINE[o1 + b1] = e;
  if (theirs) THEIRS[o2 + b2] = e;
  if (t == 0) {
    LCOUNT[0] = n0;
    COUNTS[0] = n1;
    COUNTS[1] = n2;
    atomic_store_explicit(CNT, uint(n1), memory_order_relaxed);
  }
}
