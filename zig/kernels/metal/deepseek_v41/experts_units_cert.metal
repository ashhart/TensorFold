// DeepSeek V4.1 opt-in FP64-certified BF16 prompt experts on M5 Ultra Metal 4.1 tensor units (Pass 1).
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#define XU_RT 32u

struct XmArgs {
  uint K, N, topk, pairs;
  float limit;
};

struct XuPatchArgs {
  uint patch_cap;
  uint ug_cap;
};

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_xu_tiles(
    device const uint* slots [[buffer(0)]], device const uint* nslots [[buffer(1)]], device uint4* tiles [[buffer(2)]],
    uint t [[thread_index_in_threadgroup]]) {
  if (t != 0) return;
  uint n = 0;
  for (uint s = 0; s < nslots[0]; ++s)
    for (uint r = 0; r < slots[3 * s + 2]; r += XU_RT)
      tiles[1 + n++] = uint4(slots[3 * s], slots[3 * s + 1] + r, min(XU_RT, slots[3 * s + 2] - r), 0u);
  tiles[0] = uint4(n, 0u, 0u, 0u);
}

// Hybrid decode splitter: count < min_xu into one_slots, count >= min_xu into tiles/multi_pairs, resets patch_cnt.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_xu_tiles_hybrid(
    device const uint* slots [[buffer(0)]], device const uint* nslots [[buffer(1)]], device uint4* tiles [[buffer(2)]],
    device uint* one_slots [[buffer(3)]], device uint* multi_pairs [[buffer(4)]], device uint* ind [[buffer(5)]],
    device uint* patch_cnt [[buffer(6)]], constant uint4& shape [[buffer(7)]],
    uint t [[thread_index_in_threadgroup]]) {
  if (t != 0u) return;
  const uint n_slots = nslots[0];
  const uint min_xu = shape.z;
  uint n_one = 0u, n_tiles = 0u, n_mp = 0u;
  for (uint s = 0u; s < n_slots; ++s) {
    const uint e = slots[3u * s], first = slots[3u * s + 1u], count = slots[3u * s + 2u];
    if (count < min_xu) {
      one_slots[3u * n_one] = e;
      one_slots[3u * n_one + 1u] = first;
      one_slots[3u * n_one + 2u] = count;
      n_one++;
    } else {
      for (uint r = 0u; r < count; r += XU_RT)
        tiles[1u + n_tiles++] = uint4(e, first + r, min(XU_RT, count - r), 0u);
      for (uint r = 0u; r < count; ++r) multi_pairs[n_mp++] = first + r;
    }
  }
  tiles[0] = uint4(n_tiles, 0u, 0u, 0u);
  ind[0] = shape.x; ind[1] = n_one; ind[2] = 1u; ind[3] = 0u;
  ind[4] = shape.y; ind[5] = n_one; ind[6] = 1u; ind[7] = 0u;
  ind[8] = shape.x; ind[9] = n_tiles; ind[10] = 1u; ind[11] = 0u;
  ind[12] = shape.y; ind[13] = n_tiles; ind[14] = 1u; ind[15] = 0u;
  ind[16] = 1u; ind[17] = n_mp; ind[18] = 1u; ind[19] = 0u;
  ind[28] = n_one;
  ind[29] = n_mp;
  ind[30] = n_tiles;
  patch_cnt[0] = 0u; patch_cnt[4] = 0u; patch_cnt[5] = 1u; patch_cnt[6] = 1u;
  patch_cnt[8] = 0u; patch_cnt[12] = 0u; patch_cnt[13] = 1u; patch_cnt[14] = 1u;
}

inline half xu_h(uint c) {
  return as_type<half>(ushort(((c & 0x7Fu) << 7) | ((c & 0x80u) << 8))) * 256.0h;
}

inline uint xu_int(uint c) {
  const uint e = (c >> 3) & 15u, m = c & 7u;
  return e == 0 ? m : (8u + m) << (e - 1);
}

inline bool xu_exact(ulong sum, ulong low) {
  return low == 0 || 12 * sum < (ulong(1) << 24) * low;
}

inline uint xu_pos(uint k) {
  return 2 * (k & 12u) + (k & 3u) + ((k & 16u) >> 2);
}

// Quantize BF16 X to swizzled exact half + scales and build a per-row 256-bit unsafe-group bitmask.
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_dsv41_xu_quant_ug(
    device const bfloat* X [[buffer(0)]], device half* xc [[buffer(1)]], device float* xs [[buffer(2)]],
    constant uint3& a [[buffer(3)]], device uint* unsafe_mask [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint K = a.x, rows = a.y, ug_cap = a.z, G = K >> 5, row = tg.y;
  if (row >= rows) return;
  threadgroup atomic_uint tg_mask[8];
  if (tid < 8u) atomic_store_explicit(&tg_mask[tid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint g = tid; g < G; g += 128u) {
    device const uint4* src = (device const uint4*)(X + ulong(row) * K + (g << 5));
    float v[32], amax = 0.0f;
    bool bad = false;
    for (uint q = 0; q < 4; ++q)
      for (uint m = 0; m < 4; ++m) {
        const float2 f = float2(as_type<bfloat2>(src[q][m]));
        v[8 * q + 2 * m] = f.x, v[8 * q + 2 * m + 1] = f.y;
      }
    for (uint j = 0; j < 32; ++j) {
      if (TF_CTL == 1 && !isfinite(v[j])) v[j] = 0.0f;
      bad |= !isfinite(v[j]);
      amax = max(amax, fabs(v[j]));
    }
    const int e = ds_exp(amax);
    uint sum = 0, bits = 0;
    half h[32];
    for (uint j = 0; j < 32; ++j) {
      const uint c = ds_code(min(fabs(v[j]) * ds_pow2(-e), 448.0f), signbit(v[j]));
      sum += xu_int(c), bits |= xu_int(c);
      h[xu_pos(j)] = xu_h(c);
    }
    device half4* dst = (device half4*)(xc + ulong(row) * K + (g << 5));
    for (uint q = 0; q < 8; ++q) dst[q] = half4(h[4 * q], h[4 * q + 1], h[4 * q + 2], h[4 * q + 3]);
    const bool safe = !bad && xu_exact(sum, bits & (~bits + 1u));
    xs[ulong(row) * G + g] = bad ? -NAN : (safe ? 1.0f : -1.0f) * ds_pow2(e);
    if (!safe) atomic_fetch_or_explicit(&tg_mask[g >> 5], 1u << (g & 31u), memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 8u && row < ug_cap) unsafe_mask[ulong(row) * 8u + tid] = atomic_load_explicit(&tg_mask[tid], memory_order_relaxed);
}

inline float xu_max8(float x) {
  for (ushort d = 1; d < 8; d <<= 1) x = max(x, simd_shuffle_xor(x, d));
  return x;
}

inline int xu_min8(int x) {
  for (ushort d = 1; d < 8; d <<= 1) x = min(x, simd_shuffle_xor(x, d));
  return x;
}

inline uint xu_sum8(uint x) {
  for (ushort d = 1; d < 8; d <<= 1) x += simd_shuffle_xor(x, d);
  return x;
}

// Prepare swizzled half activations for xu_down and build a per-pair 256-bit unsafe-group bitmask.
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_xu_dprep_ug(
    device const bfloat* act [[buffer(0)]], device half* dh [[buffer(1)]], device float* ds [[buffer(2)]],
    constant uint3& a [[buffer(3)]], device uint* unsafe_mask [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint K = a.x, rows = a.y, ug_cap = a.z, G = K >> 5, row = tg.y;
  if (row >= rows) return;
  threadgroup atomic_uint tg_mask[8];
  if (tid < 8u) atomic_store_explicit(&tg_mask[tid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint sub = tid & 7u;
  for (uint g_base = 0u; g_base < G; g_base += 32u) {
    const uint g = g_base + (tid >> 3);
    if (g < G) {
      const uint k = (g << 5) + (sub << 2);
      const uint2 raw = *(device const uint2*)(act + ulong(row) * K + k);
      float v[4];
      float top = 0.0f, bad = 0.0f;
      for (int i = 0; i < 4; ++i) {
        v[i] = as_type<float>(((raw[i / 2] >> (16 * (i % 2))) & 0xFFFFu) << 16);
        if (TF_CTL == 1 && !isfinite(v[i])) v[i] = 0.0f;
        top = max(top, fabs(v[i]));
        bad = isfinite(v[i]) ? bad : 1.0f;
      }
      const float amax = xu_max8(top);
      const int e = amax > 0.0f ? ilogb(amax) : 0;
      int lb[4], low = 1 << 20;
      for (int i = 0; i < 4; ++i) {
        const uint b = as_type<uint>(v[i]) & 0x7FFFFFFFu;
        lb[i] = b != 0 ? int(b >> 23) - 150 + int(ctz(b | 0x800000u)) : 1 << 20;
        low = min(low, lb[i]);
      }
      low = xu_min8(low);
      uint sum = 0;
      for (int i = 0; i < 4; ++i) {
        const uint b = as_type<uint>(v[i]) & 0x7FFFFFFFu, odd = ((b & 0x7FFFFFu) | 0x800000u) >> ctz(b | 0x800000u);
        const int sh = lb[i] - low;
        sum += b == 0 ? 0u : (sh >= 21 || odd >= (1u << (21 - sh)) ? 1u << 21 : odd << sh);
      }
      const bool exact = amax == 0.0f || (amax >= 0x1p-60f && 12u * xu_sum8(sum) < (1u << 24));
      const bool safe = xu_max8(bad) == 0.0f && exact;
      const float f = ds_pow2(9 - e);
      const half4 h = safe ? half4(half(v[0] * f), half(v[1] * f), half(v[2] * f), half(v[3] * f)) : half4(0.0h);
      *(device half4*)(dh + ulong(row) * K + (g << 5) + xu_pos(sub << 2)) = h;
      if (sub == 0u) {
        ds[ulong(row) * G + g] = (safe ? 1.0f : -1.0f) * ds_pow2(e - 9);
        if (!safe) atomic_fetch_or_explicit(&tg_mask[g >> 5], 1u << (g & 31u), memory_order_relaxed);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 8u && row < ug_cap) unsafe_mask[ulong(row) * 8u + tid] = atomic_load_explicit(&tg_mask[tid], memory_order_relaxed);
}

// Hybrid dprep: runs only on count >= 2 pairs listed in multi_pairs via indirect grid { 1, n_mp, 1 }.
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_xu_dprep_ug_hybrid(
    device const bfloat* act [[buffer(0)]], device half* dh [[buffer(1)]], device float* ds [[buffer(2)]],
    constant uint3& a [[buffer(3)]], device uint* unsafe_mask [[buffer(4)]],
    device const uint* multi_pairs [[buffer(5)]], device const uint* n_multi [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint K = a.x, ug_cap = a.z, G = K >> 5;
  if (tg.y >= n_multi[0]) return;
  const uint row = multi_pairs[tg.y];
  threadgroup atomic_uint tg_mask[8];
  if (tid < 8u) atomic_store_explicit(&tg_mask[tid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint sub = tid & 7u;
  for (uint g_base = 0u; g_base < G; g_base += 32u) {
    const uint g = g_base + (tid >> 3);
    if (g < G) {
      const uint k = (g << 5) + (sub << 2);
      const uint2 raw = *(device const uint2*)(act + ulong(row) * K + k);
      float v[4];
      float top = 0.0f, bad = 0.0f;
      for (int i = 0; i < 4; ++i) {
        v[i] = as_type<float>(((raw[i / 2] >> (16 * (i % 2))) & 0xFFFFu) << 16);
        if (TF_CTL == 1 && !isfinite(v[i])) v[i] = 0.0f;
        top = max(top, fabs(v[i]));
        bad = isfinite(v[i]) ? bad : 1.0f;
      }
      const float amax = xu_max8(top);
      const int e = amax > 0.0f ? ilogb(amax) : 0;
      int lb[4], low = 1 << 20;
      for (int i = 0; i < 4; ++i) {
        const uint b = as_type<uint>(v[i]) & 0x7FFFFFFFu;
        lb[i] = b != 0 ? int(b >> 23) - 150 + int(ctz(b | 0x800000u)) : 1 << 20;
        low = min(low, lb[i]);
      }
      low = xu_min8(low);
      uint sum = 0;
      for (int i = 0; i < 4; ++i) {
        const uint b = as_type<uint>(v[i]) & 0x7FFFFFFFu, odd = ((b & 0x7FFFFFu) | 0x800000u) >> ctz(b | 0x800000u);
        const int sh = lb[i] - low;
        sum += b == 0 ? 0u : (sh >= 21 || odd >= (1u << (21 - sh)) ? 1u << 21 : odd << sh);
      }
      const bool exact = amax == 0.0f || (amax >= 0x1p-60f && 12u * xu_sum8(sum) < (1u << 24));
      const bool safe = xu_max8(bad) == 0.0f && exact;
      const float f = ds_pow2(9 - e);
      const half4 h = safe ? half4(half(v[0] * f), half(v[1] * f), half(v[2] * f), half(v[3] * f)) : half4(0.0h);
      *(device half4*)(dh + ulong(row) * K + (g << 5) + xu_pos(sub << 2)) = h;
      if (sub == 0u) {
        ds[ulong(row) * G + g] = (safe ? 1.0f : -1.0f) * ds_pow2(e - 9);
        if (!safe) atomic_fetch_or_explicit(&tg_mask[g >> 5], 1u << (g & 31u), memory_order_relaxed);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 8u && row < ug_cap) unsafe_mask[ulong(row) * 8u + tid] = atomic_load_explicit(&tg_mask[tid], memory_order_relaxed);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_xu_zero_cnt(
    device uint* cnt [[buffer(0)]], uint tid [[thread_index_in_threadgroup]]) {
  if (tid == 0u) {
    cnt[0] = 0u;
    cnt[4] = 0u;
    cnt[5] = 1u;
    cnt[6] = 1u;
  }
}

using namespace mpp::tensor_ops;
constexpr constant auto xu_shape = matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply);
using xu_wt = tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline>;

// Zero-spill Pass-1 chain: accumulates safe groups in 48-bit dd and flags unsafe/abnormal rows for Pass 2.
inline uint xu_chain_fast(thread float (&cur)[16], thread float (&comp)[16], device const uchar* wp, device const uchar* ws,
                          device const half* xb, device const float* sb, const thread uint (&rb)[2],
                          uint hx, uint hy, uint K) {
  const uint G = K / 32;
  matmul2d<xu_shape, execution_simdgroup> op;
  xu_wt wt((device uchar*)wp, dextents<int32_t, 2>(K, 32));
  const uint Ge = TF_CTL == 5 ? G - 1 : G;
  _Pragma("unroll") for (int i = 0; i < 16; ++i) {
    cur[i] = 0.0f;
    comp[i] = 0.0f;
  }
  bool bad0 = false, bad1 = false, any_abn = false;
  for (uint g = 0; g < Ge; ++g) {
    const uint4 x0 = *(device const uint4*)(xb + rb[0] + 32 * g + 2 * hx);
    const uint4 x1 = *(device const uint4*)(xb + rb[1] + 32 * g + 2 * hx);
    auto wg = wt.slice<32, 32>(32 * g, 0);
    auto left = op.get_left_input_cooperative_tensor<half, metal_fp4_e2m1_format, float>();
    auto acc = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
                                                     metal::remove_addrspace_t<decltype(wg)>, float>();
    _Pragma("unroll") for (int i = 0; i < 16; ++i) {
      const uint j = (i & 3) + 4 * (i >> 3);
      left[i] = as_type<half2>(((i >> 2) & 1 ? x1 : x0)[j >> 1])[j & 1];
    }
    op.run(left, wg, acc);
    const float s0 = sb[rb[0] / 32 + g], s1 = sb[rb[1] / 32 + g];
    const bool u0 = (TF_CTL != 6) && (signbit(s0) || !isfinite(s0));
    const bool u1 = (TF_CTL != 6) && (signbit(s1) || !isfinite(s1));
    bad0 |= u0;
    bad1 |= u1;
    const float sf0 = u0 ? 0.0f : (s0 * 0x1p-14f);
    const float sf1 = u1 ? 0.0f : (s1 * 0x1p-14f);
    float ks[8];
    bool abn = false;
    _Pragma("unroll") for (int j = 0; j < 8; ++j) {
      const uchar e = ws[(hx + (j & 3) + 16 * (j >> 2)) * G + g];
      const bool ok = (e <= 240u);
      abn |= !ok;
      ks[j] = ok ? k3_e8m0(e) : 0.0f;
    }
    if (abn) {
      bad0 = true;
      bad1 = true;
      any_abn = true;
    }
    _Pragma("unroll") for (int i = 0; i < 16; ++i) {
      const bool skip = ((i >> 2) & 1) ? u1 : u0;
      const float ksc = ks[(i & 3) + 4 * (i >> 3)];
      float term = (skip || ksc == 0.0f) ? 0.0f : (acc[i] * ((i >> 2) & 1 ? sf1 : sf0)) * ksc;
      if (TF_CTL == 3) term = float(half(term));
      xu_neumaier_fast(cur[i], comp[i], term);
    }
  }
  return simd_or((bad0 ? (1u << hy) : 0u) | (bad1 ? (256u << hy) : 0u) | (any_abn ? (1u << 16) : 0u));
}

inline short2 xu_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

// 32-row Pass 1 for xu_up: 4 simdgroups (128T) = 2 x 16-row chunks x (W1, W3), 16 KiB TG memory.
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_dsv41_xu_up(
    device const half* xc [[buffer(0)]], device const float* xs [[buffer(1)]], device const ExpertPtrs* table [[buffer(2)]],
    device const uint4* tiles [[buffer(3)]], device const uint* pairs [[buffer(4)]], device bfloat* act [[buffer(5)]],
    constant XmArgs& a [[buffer(6)]], device const float* wts [[buffer(7)]],
    device uint4* patch_meta [[buffer(8)]], device float4* patch_vals [[buffer(9)]],
    device atomic_uint* patch_cnt [[buffer(10)]], constant XuPatchArgs& pa [[buffer(11)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float gv_hi[32][32];
  threadgroup float gv_lo[32][32];
  threadgroup float u_hi[32][32];
  threadgroup float u_lo[32][32];
  threadgroup uint bm_sm[4];
  if (tg.y >= tiles[0].x) return;
  const uint4 tile = tiles[1 + tg.y];
  const ExpertPtrs w = table[tile.x];
  const short2 home = xu_home(lane);
  const uint hx = home.x, hy = home.y;
  const uint K = a.K, G = K / 32, n0 = tg.x * 32, first = tile.y, nr = tile.z;
  const uint chunk = sg >> 1, mat = sg & 1u;
  const uint r_base = chunk * 16u;
  const uint r_cnt = (nr > r_base) ? min(16u, nr - r_base) : 0u;
  float cur[16], comp[16];
  uint bm_local = 0u;
  if (r_cnt > 0u) {
    uint rb[2];
    for (int h = 0; h < 2; ++h) rb[h] = (pairs[first + r_base + min(hy + 8 * h, r_cnt - 1)] / a.topk) * K;
    device const uchar* wp = (mat == 0u ? w.w1p : w.w3p) + n0 * (K / 2);
    device const uchar* ws = (mat == 0u ? w.w1s : w.w3s) + n0 * G;
    bm_local = xu_chain_fast(cur, comp, wp, ws, xc, xs, rb, hx, hy, K);
    if (mat == 0u) {
      for (int i = 0; i < 16; ++i) {
        const uint lr = r_base + hy + 8 * ((i >> 2) & 1), col = hx + (i & 3) + 16 * (i >> 3);
        gv_hi[lr][col] = cur[i];
        gv_lo[lr][col] = comp[i];
      }
    } else {
      for (int i = 0; i < 16; ++i) {
        const uint lr = r_base + hy + 8 * ((i >> 2) & 1), col = hx + (i & 3) + 16 * (i >> 3);
        u_hi[lr][col] = cur[i];
        u_lo[lr][col] = comp[i];
      }
    }
  }
  if (lane == 0u) bm_sm[sg] = bm_local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (r_cnt > 0u) {
    const uint bm = bm_sm[2u * chunk] | bm_sm[2u * chunk + 1u];
    for (uint r = mat; r < r_cnt; r += 2u) {
      const uint lr = r_base + r;
      const uint pair_pos = first + lr;
      if ((bm & (1u << r)) == 0u) {
        const float g = ds_clip(float(xu_dd_to_bf16({gv_hi[lr][lane], gv_lo[lr][lane]})), -INFINITY, a.limit);
        const float u = ds_clip(float(xu_dd_to_bf16({u_hi[lr][lane], u_lo[lr][lane]})), -a.limit, a.limit);
        const float v = g / (1.0f + precise::exp(-g)) * u * wts[pairs[pair_pos]];
        const float vv = TF_CTL == 2 ? v : float(bfloat(v));
        const float amax = simd_max(fabs(vv));
        const bool bad = simd_any(!isfinite(vv));
        const int s = ds_exp(amax);
        const float q = ds_val(ds_code(min(fabs(vv) * ds_pow2(-s), 448.0f), signbit(vv))) * ds_pow2(s);
        act[ulong(pair_pos) * a.N + n0 + lane] = bfloat(bad ? NAN : q);
      } else {
        uint slot = 0u;
        if (lane == 0u) {
          slot = atomic_fetch_add_explicit(patch_cnt, 1u, memory_order_relaxed);
          atomic_fetch_max_explicit(patch_cnt + 4u, min(2048u, (slot + 4u) >> 2), memory_order_relaxed);
          if (slot < pa.patch_cap) patch_meta[slot] = uint4(pair_pos, tile.x, n0, bm >> 16);
        }
        slot = simd_shuffle(slot, 0);
        if (slot < pa.patch_cap)
          patch_vals[ulong(slot) * 32u + lane] = float4(gv_hi[lr][lane], gv_lo[lr][lane], u_hi[lr][lane], u_lo[lr][lane]);
      }
    }
  }
  if (TF_CTL == 4 && tg.x == 0 && tg.y == 0 && sg == 0 && lane == 0) act[a.pairs * a.N] = bfloat(1.0f);
}

// 32-row Pass 1 for xu_down: 2 simdgroups (64T) share 32 W2 columns via xu_chain_fast, 8 KiB TG memory.
[[kernel, max_total_threads_per_threadgroup(64)]] void tf_dsv41_xu_down(
    device const bfloat* act [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
    device const uint4* tiles [[buffer(2)]], device bfloat* out [[buffer(3)]], constant XmArgs& a [[buffer(4)]],
    device const half* dh [[buffer(5)]], device const float* ds [[buffer(6)]],
    device uint4* patch_meta [[buffer(7)]], device float2* patch_vals [[buffer(8)]],
    device atomic_uint* patch_cnt [[buffer(9)]], constant XuPatchArgs& pa [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float hi_sm[2][16][32];
  threadgroup float lo_sm[2][16][32];
  if (tg.y >= tiles[0].x) return;
  const uint4 tile = tiles[1 + tg.y];
  const ExpertPtrs w = table[tile.x];
  const short2 home = xu_home(lane);
  const uint hx = home.x, hy = home.y;
  const uint K = a.K, G = K >> 5, n0 = tg.x * 32u, first = tile.y, nr = tile.z;
  const uint r_base = sg * 16u;
  if (n0 < a.N && r_base < nr) {
    const uint r_cnt = min(16u, nr - r_base);
    uint rb[2];
    for (int h = 0; h < 2; ++h) rb[h] = (first + r_base + min(hy + 8u * h, r_cnt - 1u)) * K;
    float cur[16], comp[16];
    const uint bm = xu_chain_fast(cur, comp, w.w2p + ulong(n0) * (K >> 1), w.w2s + ulong(n0) * G, dh, ds, rb, hx, hy, K);
    for (int i = 0; i < 16; ++i) {
      const uint lr = hy + 8 * ((i >> 2) & 1), col = hx + (i & 3) + 16 * (i >> 3);
      hi_sm[sg][lr][col] = cur[i];
      lo_sm[sg][lr][col] = comp[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0u; r < r_cnt; ++r) {
      const uint pair_pos = first + r_base + r;
      if ((bm & (1u << r)) == 0u) {
        out[ulong(pair_pos) * a.N + n0 + lane] = xu_dd_to_bf16({hi_sm[sg][r][lane], lo_sm[sg][r][lane]});
      } else {
        uint slot = 0u;
        if (lane == 0u) {
          slot = atomic_fetch_add_explicit(patch_cnt, 1u, memory_order_relaxed);
          atomic_fetch_max_explicit(patch_cnt + 4u, min(2048u, (slot + 4u) >> 2), memory_order_relaxed);
          if (slot < pa.patch_cap) patch_meta[slot] = uint4(pair_pos, tile.x, n0, bm >> 16);
        }
        slot = simd_shuffle(slot, 0);
        if (slot < pa.patch_cap)
          patch_vals[ulong(slot) * 32u + lane] = float2(hi_sm[sg][r][lane], lo_sm[sg][r][lane]);
      }
    }
  }
  if (TF_CTL == 4 && tg.x == 0 && tg.y == 0 && sg == 0 && lane == 0) out[a.pairs * a.N] = bfloat(1.0f);
}
