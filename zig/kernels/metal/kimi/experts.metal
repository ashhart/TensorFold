// Routed MXFP4 experts: a plan grouping a round's (row, rank) pairs by expert, then each expert's weights read once.
#include "kimi_common.h"

// An expert's six checkpoint tensors by GPU address (zero-copy shard buffers or our stacks).
struct ExpertPtrs {
  device const uchar* w1p;
  device const uchar* w1s;
  device const uchar* w3p;
  device const uchar* w3s;
  device const uchar* w2p;
  device const uchar* w2s;
};

struct PlanArgs {
  uint pairs, experts, first, last;
};

constant constexpr uint NOWHERE = 0xFFFFFFFFu;

// Local pairs (first <= expert < last) a slot per active expert in expert order; order inside a slot is storage only.
kernel void k3_route_plan(device const uint* ids [[buffer(0)]], device uint* slots [[buffer(1)]],
                          device uint* nslots [[buffer(2)]], device uint* pairs [[buffer(3)]],
                          device uint* where [[buffer(4)]], constant PlanArgs& a [[buffer(5)]],
                          uint t [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup atomic_uint count[1024], cursor[1024];
  threadgroup uint start[1024], part_n[32], part_s[32];
  for (uint e = t; e < 1024; e += nt) {
    atomic_store_explicit(&count[e], 0u, memory_order_relaxed);
    atomic_store_explicit(&cursor[e], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint p = t; p < a.pairs; p += nt) {
    const uint e = ids[p];
    if (e >= a.first && e < a.last) atomic_fetch_add_explicit(&count[e], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n = t < a.experts ? atomic_load_explicit(&count[t], memory_order_relaxed) : 0u;
  const uint before_n = simd_prefix_exclusive_sum(n);
  const uint before_s = simd_prefix_exclusive_sum(n > 0 ? 1u : 0u);
  if (lane == 31) part_n[sg] = before_n + n, part_s[sg] = before_s + (n > 0 ? 1u : 0u);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint base_n = 0, base_s = 0;
  for (uint g = 0; g < sg; ++g) base_n += part_n[g], base_s += part_s[g];
  if (t < a.experts) start[t] = base_n + before_n;
  if (t < a.experts && n > 0) {
    slots[3 * (base_s + before_s)] = t;
    slots[3 * (base_s + before_s) + 1] = base_n + before_n;
    slots[3 * (base_s + before_s) + 2] = n;
  }
  if (t == nt - 1) {
    uint total = 0;
    for (uint g = 0; g < nt / 32; ++g) total += part_s[g];
    nslots[0] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint p = t; p < a.pairs; p += nt) {
    const uint e = ids[p];
    uint pos = NOWHERE;
    if (e >= a.first && e < a.last) {
      pos = start[e] + atomic_fetch_add_explicit(&cursor[e], 1u, memory_order_relaxed);
      pairs[pos] = p;
    }
    where[p] = pos;
  }
}

struct XpArgs {
  uint K, N, topk, rows_max;
  float beta, lin;
};

// The slot's activation rows (pairs' rows) staged in threadgroup memory, RP rows at a time.
template <int RP>
inline uint k3_stage(threadgroup bfloat* xs, device const bfloat* X, device const uint* pairs, uint first,
                     uint count, uint r0, uint K, uint topk, bool by_row, uint t, uint nt) {
  const uint rn = min(uint(RP), count - r0);
  for (uint r = 0; r < rn; ++r) {
    const uint p = pairs[first + r0 + r];
    device const bfloat* src = X + (by_row ? p / topk : first + r0 + r) * K;
    for (uint i = t; i < K / 4; i += nt) ((threadgroup uint2*)(xs + r * K))[i] = ((device const uint2*)src)[i];
  }
  return rn;
}

// Lane 8o + c runs chain c (groups c, c+8, ...) of output o: per group 32 codes in order from zero, then fma by its scale.
template <int RP, int NW>
inline void k3_xp_chains(thread float (&acc)[NW][RP], threadgroup const bfloat* xs, uint K, uint rn,
                         device const uchar* const thread (&wp)[NW], device const uchar* const thread (&ws)[NW],
                         uint c) {
  for (int v = 0; v < NW; ++v)
    for (int r = 0; r < RP; ++r) acc[v][r] = 0.0f;
  for (uint g = c; g < K / 32; g += 8) {
    float p[NW][RP];
    for (int v = 0; v < NW; ++v)
      for (int r = 0; r < RP; ++r) p[v][r] = 0.0f;
    for (int h = 0; h < 2; ++h) {
      float w[NW][16];
      for (int v = 0; v < NW; ++v) {
        const uint2 q = *(device const uint2*)(wp[v] + 16 * g + 8 * h);
        for (int b = 0; b < 8; ++b) {
          const float2 f = k3_fp4_pair(((b < 4 ? q.x : q.y) >> (8 * (b & 3))) & 0xFFu);
          w[v][2 * b] = f.x;
          w[v][2 * b + 1] = f.y;
        }
      }
      for (int r = 0; r < RP; ++r) {
        if (uint(r) >= rn) break;
        threadgroup const bfloat* x = xs + r * K + 32 * g + 16 * h;
        for (int j = 0; j < 16; j += 2) {
          const float2 x2 = float2(*(threadgroup const bfloat2*)(x + j));
          for (int v = 0; v < NW; ++v) {
            p[v][r] = fma(x2.x, w[v][j], p[v][r]);
            p[v][r] = fma(x2.y, w[v][j + 1], p[v][r]);
          }
        }
      }
    }
    for (int v = 0; v < NW; ++v) {
      const float sc = k3_e8m0(ws[v][g]);
      for (int r = 0; r < RP; ++r) acc[v][r] = fma(p[v][r], sc, acc[v][r]);
    }
  }
}

inline float k3_xtree8(float v) {
  v += simd_shuffle_xor(v, 1);
  v += simd_shuffle_xor(v, 2);
  v += simd_shuffle_xor(v, 4);
  return v;
}

// Gate and up (w1, w3) for every pair of the slot's expert, SiTU, bf16 activations by pair position.
template <int RP>
kernel void k3_xp_up(device const bfloat* X [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
                     device const uint* slots [[buffer(2)]], device const uint* nslots [[buffer(3)]],
                     device const uint* pairs [[buffer(4)]], device bfloat* act [[buffer(5)]],
                     constant XpArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat xs[RP * 3584];
  if (tg.y >= nslots[0]) return;
  const uint e = slots[3 * tg.y], first = slots[3 * tg.y + 1], count = slots[3 * tg.y + 2];
  const ExpertPtrs w = table[e];
  const uint n = (tg.x * nsg + sg) * 4 + (lane >> 3), c = lane & 7, rowb = a.K / 2, groups = a.K / 32;
  device const uchar* const wp[2] = {w.w1p + n * rowb, w.w3p + n * rowb};
  device const uchar* const ws[2] = {w.w1s + n * groups, w.w3s + n * groups};
  for (uint r0 = 0; r0 < count; r0 += RP) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint rn = k3_stage<RP>(xs, X, pairs, first, count, r0, a.K, a.topk, true, t, nsg * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[2][RP];
    k3_xp_chains<RP, 2>(acc, xs, a.K, rn, wp, ws, c);
    for (int r = 0; r < RP; ++r) {
      const float gv = float(bfloat(k3_xtree8(acc[0][r]))), uv = float(bfloat(k3_xtree8(acc[1][r])));
      if (c == 0 && uint(r) < rn) act[(first + r0 + r) * a.N + n] = bfloat(k3_situ(gv, uv, a.beta, a.lin));
    }
  }
}

// Down (w2) for every pair of the slot's expert: bf16 outputs by pair position.
template <int RP>
kernel void k3_xp_down(device const bfloat* act [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
                       device const uint* slots [[buffer(2)]], device const uint* nslots [[buffer(3)]],
                       device const uint* pairs [[buffer(4)]], device bfloat* out [[buffer(5)]],
                       constant XpArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat xs[RP * 3072];
  if (tg.y >= nslots[0]) return;
  const uint e = slots[3 * tg.y], first = slots[3 * tg.y + 1], count = slots[3 * tg.y + 2];
  const ExpertPtrs w = table[e];
  const uint n = (tg.x * nsg + sg) * 4 + (lane >> 3), c = lane & 7, rowb = a.K / 2, groups = a.K / 32;
  device const uchar* const wp[1] = {w.w2p + n * rowb};
  device const uchar* const ws[1] = {w.w2s + n * groups};
  for (uint r0 = 0; r0 < count; r0 += RP) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint rn = k3_stage<RP>(xs, act, pairs, first, count, r0, a.K, a.topk, false, t, nsg * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[1][RP];
    k3_xp_chains<RP, 1>(acc, xs, a.K, rn, wp, ws, c);
    for (int r = 0; r < RP; ++r) {
      const float v = k3_xtree8(acc[0][r]);
      if (c == 0 && uint(r) < rn) out[(first + r0 + r) * a.N + n] = bfloat(v);
    }
  }
}

// The routed sum in the cluster's order: per expert group fmas from zero, then the tree over groups [g.y, g.y + g.z).
template <typename OutT>
kernel void k3_xp_combine(device const bfloat* out [[buffer(0)]], device const uint* where [[buffer(1)]],
                          device const float* weights [[buffer(2)]], device OutT* y [[buffer(3)]],
                          device const uint* ids [[buffer(4)]], constant XpArgs& a [[buffer(5)]],
                          constant uint3& g [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= a.N) return;
  float part[8];
  for (uint i = 0; i < 8; ++i) part[i] = 0.0f;
  for (uint k = 0; k < a.topk; ++k) {
    const uint p = where[gid.y * a.topk + k];
    if (p == NOWHERE) continue;
    const uint grp = ids[gid.y * a.topk + k] / g.x - g.y;
    part[grp] = fma(weights[gid.y * a.topk + k], float(out[p * a.N + gid.x]), part[grp]);
  }
  for (uint w = g.z; w > 1; w /= 2)
    for (uint i = 0; i < w / 2; ++i) part[i] = part[2 * i] + part[2 * i + 1];
  y[gid.y * a.N + gid.x] = OutT(part[0]);
}

template [[host_name("k3_xp_combine")]] kernel void k3_xp_combine<bfloat>(device const bfloat*, device const uint*,
    device const float*, device bfloat*, device const uint*, constant XpArgs&, constant uint3&, uint2);
template [[host_name("k3_xp_combine_f32")]] kernel void k3_xp_combine<float>(device const bfloat*, device const uint*,
    device const float*, device float*, device const uint*, constant XpArgs&, constant uint3&, uint2);

#define K3_XP(KIND, RP) \
  template [[host_name("k3_xp_" #KIND "_r" #RP)]] kernel void k3_xp_##KIND<RP>(device const bfloat*, \
      device const ExpertPtrs*, device const uint*, device const uint*, device const uint*, device bfloat*, \
      constant XpArgs&, uint2, uint, uint, uint, uint);
K3_XP(up, 1)
K3_XP(up, 4)
K3_XP(down, 1)
K3_XP(down, 4)
