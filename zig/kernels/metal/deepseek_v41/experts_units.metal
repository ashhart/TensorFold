// DeepSeek V4.1 prompt experts on M5 tensor units, bit-equal to decode: exact groups there, the rest by decode fmas.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// An E4M3 code's value as a half, and as an integer (value * 512).
inline half xu_h(uint c) { return as_type<half>(ushort(((c & 0x7Fu) << 7) | ((c & 0x80u) << 8))) * 256.0h; }
inline uint xu_int(uint c) {
  const uint e = (c >> 3) & 15u, m = c & 7u;
  return e == 0 ? m : (8u + m) << (e - 1);
}

// Every partial sum of 32 such values times fp4 weights is an fp32 integer multiple of `low` below 2^24 * low.
inline bool xu_exact(ulong sum, ulong low) { return low == 0 || 12 * sum < (ulong(1) << 24) * low; }

// Where k of a 32-group is stored: a lane's eight k (hx..hx+3, 16+hx..16+hx+3) side by side.
inline uint xu_pos(uint k) { return 2 * (k & 12u) + (k & 3u) + ((k & 16u) >> 2); }

// act_quant of every source row as halves of the codes (xu_pos order); xs is the block scale, negative when unsafe.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_xu_quant(
    device const bfloat* X [[buffer(0)]], device half* xc [[buffer(1)]], device float* xs [[buffer(2)]],
    constant uint2& a [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
  const uint K = a.x, rows = a.y;
  if (gid.x >= K / 32 || gid.y >= rows) return;
  device const uint4* src = (device const uint4*)(X + gid.y * K + 32 * gid.x);
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
  device half4* dst = (device half4*)(xc + gid.y * K + 32 * gid.x);
  for (uint q = 0; q < 8; ++q) dst[q] = half4(h[4 * q], h[4 * q + 1], h[4 * q + 2], h[4 * q + 3]);
  const bool safe = !bad && xu_exact(sum, bits & (~bits + 1u));
  xs[gid.y * (K / 32) + gid.x] = bad ? -NAN : (safe ? 1.0f : -1.0f) * ds_pow2(e);
}

// The reductions of dprep over a block's eight lanes (4 values a lane).
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

// Down's input as halves of act * 2^(9 - e) (xu_pos order), 4 values a lane; ds = 2^(e - 9), negative if unsafe.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_xu_dprep(
    device const bfloat* act [[buffer(0)]], device half* dh [[buffer(1)]], device float* ds [[buffer(2)]],
    constant uint2& a [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
  const uint K = a.x, rows = a.y, k = 4 * gid.x;
  if (k >= K || gid.y >= rows) return;
  const uint2 raw = *(device const uint2*)(act + gid.y * K + k);
  float v[4];
  float top = 0.0f, bad = 0.0f;
  for (int i = 0; i < 4; ++i) {
    v[i] = as_type<float>(((raw[i / 2] >> (16 * (i % 2))) & 0xFFFFu) << 16);
    top = max(top, fabs(v[i]));
    bad = isfinite(v[i]) ? bad : 1.0f;
  }
  const float amax = xu_max8(top);
  const int e = amax > 0.0f ? ilogb(amax) : 0;
  // each value over the block's lowest set bit, held at 2^21 (past 2^24 / 12 the block is unsafe anyway)
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
  const half4 h = half4(half(v[0] * f), half(v[1] * f), half(v[2] * f), half(v[3] * f));
  *(device half4*)(dh + gid.y * K + (k & ~31u) + xu_pos(k & 31u)) = h;
  if ((gid.x & 7) == 0) ds[gid.y * (K / 32) + k / 32] = (safe ? 1.0f : -1.0f) * ds_pow2(e - 9);
}

using namespace mpp::tensor_ops;
constexpr constant auto xu_shape = matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply);
using xu_wt = tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline>;

// One element's group sum as the decode kernels make it: 32 fmas in k order from zero.
template <bool FP8>
inline float xu_exact_sum(device const uchar* wcol, device const half* xh, device const bfloat* xb) {
  const uint4 wq = *(device const uint4*)wcol;
  uint4 xq[4];
  _Pragma("unroll") for (int i = 0; i < 4; ++i)
    xq[i] = FP8 ? ((device const uint4*)xh)[i] : ((device const uint4*)xb)[i];
  float p = 0.0f;
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
    p = fma(x, (k & 1) ? f.y : f.x, p);
  }
  return p;
}

// A group sum before the scale fma: up's times the row scale, down's as is (TF_CTL 3 rounds it to fp16 first).
template <bool FP8>
inline float xu_term(float p, float s) {
  if (TF_CTL == 3) p = float(half(p));
  return FP8 ? p * fabs(s) : p;
}

// Chain c (groups c, c + 8, ...) as decode runs it, safe groups on the tensor unit (raw fp4: sums times 2^14).
template <bool FP8>
inline void xu_chain(thread float (&cur)[16], device const uchar* wp, device const uchar* ws, device const half* xb,
                     device const float* sb, device const bfloat* bb, const thread uint (&rb)[2], uint hx, uint hy,
                     uint K, uint c, uint lane) {
  const uint G = K / 32;
  matmul2d<xu_shape, execution_simdgroup> op;
  xu_wt wt((device uchar*)wp, dextents<int32_t, 2>(K, 32));
  const uint Ge = TF_CTL == 5 ? G - 1 : G;
  _Pragma("unroll") for (int i = 0; i < 16; ++i) cur[i] = 0.0f;
  for (uint g = c; g < Ge; g += 8) {
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
    float q[16];
    _Pragma("unroll") for (int i = 0; i < 16; ++i) {
      const float s = (i >> 2) & 1 ? s1 : s0;
      const float p = acc[i] * (FP8 ? 0x1p-14f : fabs(s) * 0x1p-14f);
      q[i] = TF_CTL == 3 ? xu_term<FP8>(p, s) : acc[i] * (fabs(s) * 0x1p-14f);
    }
    const uint unsafe = simd_or((signbit(s0) ? 1u << hy : 0u) | (signbit(s1) ? 256u << hy : 0u));
    for (uint m = unsafe; m; m &= m - 1) {
      const uint r = ctz(m), src = ((r & 4) << 2) | ((r & 3) << 1);
      const uint at = simd_shuffle(r < 8 ? rb[0] : rb[1], ushort(src)) + 32 * g;
      const float s = simd_shuffle(r < 8 ? s0 : s1, ushort(src));
      const float e = xu_term<FP8>(xu_exact_sum<FP8>(wp + lane * (K / 2) + 16 * g, xb + at, bb + at), s);
      _Pragma("unroll") for (int i = 0; i < 16; ++i) {
        const float v = simd_shuffle(e, ushort(hx + (i & 3) + 16 * (i >> 3)));
        if (hy + 8 * ((i >> 2) & 1) == r) q[i] = v;
      }
    }
    bool normal = true;
    uchar sc[8];
    _Pragma("unroll") for (int j = 0; j < 8; ++j) {
      sc[j] = ws[(hx + (j & 3) + 16 * (j >> 2)) * G + g];
      normal = normal && (sc[j] - 1u < 240u);
    }
    if (simd_all(normal))
      _Pragma("unroll") for (int i = 0; i < 16; ++i) cur[i] = fma(q[i], k3_e8m0(sc[(i & 3) + 4 * (i >> 3)]), cur[i]);
    else
      _Pragma("unroll") for (int i = 0; i < 16; ++i)
        cur[i] = k3_scaled_fma(q[i], sc[(i & 3) + 4 * (i >> 3)], cur[i]);
  }
}

// The decode tree over eight chains (simdgroup c holds chain c) into chain 0; part: 4 x 16 x 32 floats, all must call.
inline void xu_tree(thread float (&cur)[16], uint c, threadgroup float* part, uint lane) {
  for (uint lv = 0; lv < 3; ++lv) {
    const uint step = 1u << lv, live = (c & (step - 1)) == 0, at = (c >> (lv + 1)) * 16;
    if (live && (c & step)) for (int i = 0; i < 16; ++i) part[(at + i) * 32 + lane] = cur[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (live && !(c & step)) for (int i = 0; i < 16; ++i) cur[i] = part[(at + i) * 32 + lane] + cur[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// A lane's tile cells: element i is column hx + (i & 3) + 16 (i >> 3), row hy + 8 ((i >> 2) & 1).
inline short2 xu_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

// Gate (simdgroups 0-7, a chain each) and up (8-15) of 16 pair positions, their trees, the V4.1 epilogue, act_quant.
[[kernel, max_total_threads_per_threadgroup(512)]] void tf_dsv41_xu_up(
    device const half* xc [[buffer(0)]], device const float* xs [[buffer(1)]], device const ExpertPtrs* table [[buffer(2)]],
    device const uint4* tiles [[buffer(3)]], device const uint* pairs [[buffer(4)]], device bfloat* act [[buffer(5)]],
    constant XmArgs& a [[buffer(6)]], device const float* wts [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[2][4 * 16 * 32];
  if (tg.y >= tiles[0].x) return;
  const uint4 tile = tiles[1 + tg.y];
  const ExpertPtrs w = table[tile.x];
  const short2 home = xu_home(lane);
  const uint hx = home.x, hy = home.y, m = sg >> 3, c = sg & 7;
  const uint K = a.K, G = K / 32, n0 = tg.x * 32, first = tile.y, nr = tile.z;
  uint rb[2];
  for (int h = 0; h < 2; ++h) rb[h] = (pairs[first + min(hy + 8 * h, nr - 1)] / a.topk) * K;
  device const uchar* wp = (m == 0 ? w.w1p : w.w3p) + n0 * (K / 2);
  device const uchar* ws = (m == 0 ? w.w1s : w.w3s) + n0 * G;
  float cur[16];
  xu_chain<true>(cur, wp, ws, xc, xs, nullptr, rb, hx, hy, K, c, lane);
  xu_tree(cur, c, part[m], lane);
  threadgroup float (*gv)[32] = (threadgroup float (*)[32])part[0];
  threadgroup float (*vals)[32] = (threadgroup float (*)[32])(part[0] + 16 * 32);
  if (sg == 0)
    for (int i = 0; i < 16; ++i) gv[hy + 8 * ((i >> 2) & 1)][hx + (i & 3) + 16 * (i >> 3)] = float(bfloat(cur[i]));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 8)
    for (int i = 0; i < 16; ++i) {
      const uint lr = hy + 8 * ((i >> 2) & 1), col = hx + (i & 3) + 16 * (i >> 3);
      const float g = ds_clip(gv[lr][col], -INFINITY, a.limit);
      const float u = ds_clip(float(bfloat(cur[i])), -a.limit, a.limit);
      const float v = g / (1.0f + precise::exp(-g)) * u * (lr < nr ? wts[pairs[first + lr]] : 0.0f);
      vals[lr][col] = TF_CTL == 2 ? v : float(bfloat(v));
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg < nr) {
    const float v = vals[sg][lane];
    const float amax = simd_max(fabs(v));
    const bool bad = simd_any(!isfinite(v));
    const int s = ds_exp(amax);
    const float q = ds_val(ds_code(min(fabs(v) * ds_pow2(-s), 448.0f), signbit(v))) * ds_pow2(s);
    act[(first + sg) * a.N + n0 + lane] = bfloat(bad ? NAN : q);
  }
  if (TF_CTL == 4 && tg.x == 0 && tg.y == 0 && sg == 0 && lane == 0) act[a.pairs * a.N] = bfloat(1.0f);
}

// Down of 16 pair positions over dprep's halves: simdgroup 8 b + c runs chain c of 32-column block b; bf16 outputs.
[[kernel, max_total_threads_per_threadgroup(512)]] void tf_dsv41_xu_down(
    device const bfloat* act [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
    device const uint4* tiles [[buffer(2)]], device bfloat* out [[buffer(3)]], constant XmArgs& a [[buffer(4)]],
    device const half* dh [[buffer(5)]], device const float* ds [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[2][4 * 16 * 32];
  if (tg.y >= tiles[0].x) return;
  const uint4 tile = tiles[1 + tg.y];
  const ExpertPtrs w = table[tile.x];
  const short2 home = xu_home(lane);
  const uint hx = home.x, hy = home.y, c = sg & 7;
  const uint K = a.K, G = K / 32, n0 = tg.x * 64 + 32 * (sg >> 3), first = tile.y, nr = tile.z;
  uint rb[2];
  for (int h = 0; h < 2; ++h) rb[h] = (first + min(hy + 8 * h, nr - 1)) * K;
  float cur[16] = {};
  if (n0 < a.N)
    xu_chain<false>(cur, w.w2p + n0 * (K / 2), w.w2s + n0 * G, dh, ds, act, rb, hx, hy, K, c, lane);
  xu_tree(cur, c, part[sg >> 3], lane);
  if (c == 0 && n0 < a.N)
    for (int i = 0; i < 16; ++i) {
      const uint lr = hy + 8 * ((i >> 2) & 1);
      if (lr < nr) out[(first + lr) * a.N + n0 + hx + (i & 3) + 16 * (i >> 3)] = bfloat(cur[i]);
    }
  if (TF_CTL == 4 && tg.x == 0 && tg.y == 0 && sg == 0 && lane == 0) out[a.pairs * a.N] = bfloat(1.0f);
}
