// A MoE layer's route for 1-16 rows: router logits in the one-row kernel's order, then the top-k and this Mac's expert groups.
#include <metal_stdlib>
using namespace metal;

#ifndef TF_OUT_T
#define TF_OUT_T float
#endif
constant constexpr int ITERS = TF_K / 32;
constant constexpr int PER = (TF_E + 31) / 32;

// Logits [rows, E] = x W^T: a 4-expert block (tg.x) for four rows (tg.y); simdgroup s sums k = 32 i + 4 s + tm, lane l taking i = 4 l..4 l + 3, then the eight partials in order.
[[kernel]] void tf_route_logits(const device bfloat* X [[buffer(0)]], const device uint4* RP [[buffer(1)]],
                                constant int& rows [[buffer(2)]], device TF_OUT_T* OUT [[buffer(3)]],
                                uint2 tg [[threadgroup_position_in_grid]], uint2 tpos [[thread_position_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]], uint s [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[8][16];
  const int q = int(tg.x), r0 = int(tg.y) * 4;
  const device uint4* w = RP + ((size_t(q) * 8 + s) * ITERS + 4 * lane) * 2;
  float acc[4][4]; // [row][expert]
  for (int c = 0; c < 4; c++)
    for (int tn = 0; tn < 4; tn++) acc[c][tn] = 0.0f;
  for (int ii = 0; ii < 4; ii++) {
    float inter[4][4];
    for (int h = 0; h < 2; h++) {
      const uint4 v = w[ii * 2 + h];
      const uint words[4] = {v.x, v.y, v.z, v.w};
      for (int j = 0; j < 4; j++) {
        const int e = h * 8 + j * 2;
        inter[e / 4][e % 4] = as_type<float>(words[j] << 16);
        inter[(e + 1) / 4][(e + 1) % 4] = as_type<float>(words[j] & 0xffff0000u);
      }
    }
    const int k = 32 * (4 * int(lane) + ii) + 4 * int(s);
    for (int c = 0; c < 4; c++) {
      const device bfloat* x = X + size_t(min(r0 + c, rows - 1)) * TF_K + k;
      for (int tm = 0; tm < 4; tm++) {
        const float xv = float(x[tm]);
        for (int tn = 0; tn < 4; tn++) acc[c][tn] += xv * inter[tm][tn];
      }
    }
  }
  for (int c = 0; c < 4; c++)
    for (int tn = 0; tn < 4; tn++) {
      const float v = simd_sum(acc[c][tn]);
      if (lane == 0) part[s][c * 4 + tn] = v;
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int o = int(tpos.x);
  if (o < 16 && r0 + o / 4 < rows) {
    float v = part[0][o];
    for (int k = 1; k < 8; k++) v += part[k][o];
    OUT[size_t(r0 + o / 4) * TF_E + 4 * q + o % 4] = static_cast<TF_OUT_T>(v);
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

// One threadgroup of 512: each row's top-k by sigmoid + bias (ties to the lower id) and weights, then this Mac's groups and pick lists.
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
