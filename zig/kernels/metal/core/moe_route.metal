// A MoE layer's route, 1-16 rows: router logits in the one-row kernel's order, then top-k and this Mac's expert groups.
#include <metal_stdlib>
using namespace metal;

#ifndef TF_OUT_T
#define TF_OUT_T float
#endif
constant constexpr int ITERS = TF_K / 32;
constant constexpr int PERL = ITERS / 32; // the 32-wide slices of k each lane takes
constant constexpr int PER = (TF_E + 31) / 32;

// Logits [rows, E] = x W^T, tg 4 experts x 4 rows; simdgroup s: k=32i+4s+tm, lane l: PERL i's from PERL*l; 8 partials.
[[kernel]] void tf_route_logits(const device bfloat* X [[buffer(0)]], const device uint4* RP [[buffer(1)]],
                                constant int& rows [[buffer(2)]], device TF_OUT_T* OUT [[buffer(3)]],
                                uint2 tg [[threadgroup_position_in_grid]], uint2 tpos [[thread_position_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]], uint s [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[8][16];
  const int q = int(tg.x), r0 = int(tg.y) * 4;
  const device uint4* w = RP + ((size_t(q) * 8 + s) * ITERS + PERL * lane) * 2;
  float acc[4][4]; // [row][expert]
  for (int c = 0; c < 4; c++)
    for (int tn = 0; tn < 4; tn++) acc[c][tn] = 0.0f;
  for (int ii = 0; ii < PERL; ii++) {
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
    const int k = 32 * (PERL * int(lane) + ii) + 4 * int(s);
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
#ifdef TF_ROUND_LOGITS
    v = float(bfloat(v));
#endif
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

// Row r's first n picks by sigmoid + bias in every lane: the largest score (simd_max), its lowest id (simd_min).
inline void tf_route_picks(const device float* LOGITS, const device float* BIAS, int r, uint lane, int n, thread int* pick,
                           thread float* w) {
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
  for (int k = 0; k < TF_TOPK; k++) {
    if (k >= n) continue;
    float best = -INFINITY, bsc = 0.0f;
    int bid = TF_E;
    for (int j = 0; j < PER; j++) {
      const int id = j * 32 + int(lane);
      if (id < TF_E && (c[j] > best || (c[j] == best && id < bid))) { best = c[j]; bid = id; bsc = sc[j]; }
    }
    const float top = simd_max(best); // no lane's best is NaN: a NaN score never passes the compares
    bid = simd_min(best == top ? bid : TF_E);
    bsc = simd_shuffle(bsc, ushort(bid % 32));
    pick[k] = bid;
    w[k] = bid < TF_E ? bsc : 0.0f;
    for (int j = 0; j < PER; j++)
      if (j * 32 + int(lane) == bid) c[j] = -INFINITY;
  }
}

// The picks' weights: normalized in pick order, then scaled.
inline void tf_route_weights(thread const float* w, float scale, device float* out) {
  float total = w[0];
  for (int k = 1; k < TF_TOPK; k++) total = total + w[k];
  for (int k = 0; k < TF_TOPK; k++) out[k] = (w[k] / total) * scale;
}

// A one-row MoE kernel's pick u, the select's picks replayed; `write`: lane 0 writes the picks and weights too.
inline int tf_route_pick_at(const device float* LOGITS, const device float* BIAS, int u, uint lane, bool write,
                            device int* PICK, device float* WTS, float scale) {
  int p[TF_TOPK];
  float w[TF_TOPK];
  tf_route_picks(LOGITS, BIAS, 0, lane, write ? TF_TOPK : u + 1, p, w);
  if (write && lane == 0) {
    for (int k = 0; k < TF_TOPK; k++) PICK[k] = p[k];
    tf_route_weights(w, scale, WTS);
  }
  int e = p[0];
  for (int k = 1; k < TF_TOPK; k++) e = k == u ? p[k] : e;
  return e;
}

// 512 threads: each row's top-k by sigmoid + bias (tf_route_picks), weights, then this Mac's groups and pick lists.
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
  threadgroup uchar slot[TF_MAXR][TF_E]; // row r's pick k of expert e, 0xff unpicked
  threadgroup int offs[3][16];
  if (int(g) < R) {
    const int r = int(g);
    for (int id = int(lane); id < TF_E; id += 32) slot[r][id] = 0xff;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    int p[TF_TOPK];
    float w[TF_TOPK];
    tf_route_picks(LOGITS, BIAS, r, lane, TF_TOPK, p, w);
    if (lane == 0) {
      for (int k = 0; k < TF_TOPK; k++) {
        picks[r * TF_TOPK + k] = p[k];
        PICK[r * TF_TOPK + k] = p[k];
        if (p[k] < TF_E) slot[r][p[k]] = uchar(k);
      }
      tf_route_weights(w, a.scale, WTS + r * TF_TOPK);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // the held experts the window picked, ascending, with their member picks in pick order (a row picks an expert once)
  int count = 0;
  const bool held_id = e >= a.lo && e < a.hi && e < TF_E;
  if (held_id)
    for (int r = 0; r < R; r++) count += slot[r][e] != 0xff ? 1 : 0;
  const int held = count > 0 ? 1 : 0;
  // each pick: computed here or by the peer
  const int pe = e < R * TF_TOPK ? picks[e] : -1;
  const int mine = pe >= a.lo && pe < a.hi ? 1 : 0;
  const int theirs = pe >= 0 && mine == 0 ? 1 : 0;
  const int b0 = simd_prefix_exclusive_sum(held), b1 = simd_prefix_exclusive_sum(mine), b2 = simd_prefix_exclusive_sum(theirs);
  if (lane == 31) { offs[0][g] = b0 + held; offs[1][g] = b1 + mine; offs[2][g] = b2 + theirs; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // simdgroup g's offsets: the counts of the groups before it (integer sums, any order), and the totals
  const int c0 = lane < 16 ? offs[0][lane] : 0, c1 = lane < 16 ? offs[1][lane] : 0, c2 = lane < 16 ? offs[2][lane] : 0;
  const int o0 = simd_shuffle(simd_prefix_exclusive_sum(c0), ushort(g)), n0 = simd_sum(c0);
  const int o1 = simd_shuffle(simd_prefix_exclusive_sum(c1), ushort(g)), n1 = simd_sum(c1);
  const int o2 = simd_shuffle(simd_prefix_exclusive_sum(c2), ushort(g)), n2 = simd_sum(c2);
  if (held) {
    const int u = o0 + b0;
    LIDS[u] = e - a.lo;
    int j = 0;
    for (int r = 0; r < R; r++) {
      const int s = slot[r][e];
      if (s != 0xff) LMEM[u * TF_MAXR + j++] = r * TF_TOPK + s;
    }
    for (; j < TF_MAXR; j++) LMEM[u * TF_MAXR + j] = -1;
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
