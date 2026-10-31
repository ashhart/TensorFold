// DeepSeek V4.1's copy of Kimi's expert plan, chains and combine: the E8M0 scale over its full range.
#include <metal_stdlib>
using namespace metal;

// Two E2M1 codes of a byte as half bits scaled by 2^-14 (0.5 is a half subnormal), low nibble first.
inline float2 k3_fp4_pair(uint b) {
  const uint mags = ((b & 0x77u) * 0x00200200u) & 0x0E000E00u;
  const uint signs = ((b & 0x88u) * 0x01001000u) & 0x80008000u;
  return float2(as_type<half2>(mags | signs));
}

// E8M0 scale 2^(e-127) times 2^14 (undoing k3_fp4_pair's scale): exact for e in 0..240.
inline float k3_e8m0(uchar e) {
  return as_type<float>((uint(e) + 14u) << 23);
}

// fma(p * 2^14 * 2^(e-127), 1, acc) with the scale split in two normal floats: exact for every E8M0 byte but 255 (NaN).
inline float k3_scaled_fma(float p, uchar e, float acc) {
  if (e - 1u < 240u) return fma(p, k3_e8m0(e), acc);
  if (e == 255u) return NAN;
  const int d = int(e) - 127, d1 = d / 2;
  return fma(p * 16384.0f * as_type<float>(uint(d1 + 127) << 23), as_type<float>(uint(d - d1 + 127) << 23), acc);
}

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
kernel void tf_dsv41_route_plan(device const uint* ids [[buffer(0)]], device uint* slots [[buffer(1)]],
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

// Lane 8o + c runs chain c (groups c, c+8, ...) of output o: per group 32 codes from zero, then fma by its scale.
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
      for (int r = 0; r < RP; ++r) acc[v][r] = k3_scaled_fma(p[v][r], ws[v][g], acc[v][r]);
    }
  }
}

inline float k3_xtree8(float v) {
  v += simd_shuffle_xor(v, 1);
  v += simd_shuffle_xor(v, 2);
  v += simd_shuffle_xor(v, 4);
  return v;
}

// The routed sum in the cluster's order: per expert group fmas from zero, then the tree over groups [g.y, g.y + g.z).
template <typename OutT>
kernel void tf_dsv41_xp_combine(device const bfloat* out [[buffer(0)]], device const uint* where [[buffer(1)]],
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

template [[host_name("tf_dsv41_xp_combine")]] kernel void tf_dsv41_xp_combine<bfloat>(device const bfloat*, device const uint*,
    device const float*, device bfloat*, device const uint*, constant XpArgs&, constant uint3&, uint2);
template [[host_name("tf_dsv41_xp_combine_f32")]] kernel void tf_dsv41_xp_combine<float>(device const bfloat*, device const uint*,
    device const float*, device float*, device const uint*, constant XpArgs&, constant uint3&, uint2);
