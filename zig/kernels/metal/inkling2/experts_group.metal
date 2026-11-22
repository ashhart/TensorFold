// Prompt chunks group the route pairs by expert so one weight decode feeds many pairs; outputs equal the one-row dot.
#define I2X_SGS 8u
#define I2X_PLAN_THREADS 1024u
// A threadgroup decodes RG groups of four packed rows per 32-block K slice; a pad word per lane avoids bank conflicts.
#define I2X_NV_RG 2u
#define I2X_MX_RG 1u
#define I2X_NV_WORDS (2u * I2X_NV_RG * 4u * 32u * 9u)
#define I2X_MX_WORDS (2u * I2X_MX_RG * 4u * 32u * 17u)

struct Inkling2ExpPlanArgs {
  uint pairs, experts, tile, gate_up_groups;
  uint down_groups, p0, p1, p2;
};

// Sort pair slots by expert (bucket `experts` is inactive) into tiles {bucket, first pair, pairs}; tiles[0].x counts.
[[kernel, max_total_threads_per_threadgroup(I2X_PLAN_THREADS)]]
void inkling2_exp_group_plan(const device uint* ids [[buffer(0)]], device uint* order [[buffer(1)]],
                             device uint4* tiles [[buffer(2)]], device uint* launch [[buffer(3)]],
                             constant Inkling2ExpPlanArgs& a [[buffer(4)]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint seen[130];
  threadgroup uint first[130];
  if (t < 130u) atomic_store_explicit(&seen[t], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint p = t; p < a.pairs; p += I2X_PLAN_THREADS)
    atomic_fetch_add_explicit(&seen[min(ids[p], a.experts)], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) {
    uint begin = 0u, n = 0u;
    for (uint e = 0u; e <= a.experts; ++e) {
      const uint count = atomic_load_explicit(&seen[e], memory_order_relaxed);
      first[e] = begin;
      for (uint at = 0u; at < count; at += a.tile) tiles[1u + n++] = uint4(e, begin + at, min(a.tile, count - at), 0u);
      begin += count;
      atomic_store_explicit(&seen[e], 0u, memory_order_relaxed);
    }
    tiles[0] = uint4(n, 0u, 0u, 0u);
    launch[0] = a.gate_up_groups; launch[1] = n; launch[2] = 1u;
    launch[4] = a.down_groups; launch[5] = n; launch[6] = 1u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint p = t; p < a.pairs; p += I2X_PLAN_THREADS) {
    const uint e = min(ids[p], a.experts);
    order[first[e] + atomic_fetch_add_explicit(&seen[e], 1u, memory_order_relaxed)] = p;
  }
}

// Grouped rows store each 32-block K slice as [granule][lane] (a granule is 8 BF16, gpb per block: 2 NVFP4, 4 MXFP8).
inline uint i2x_slot(uint granule, uint gpb) {
  const uint block = granule / gpb;
  return ((block >> 5u) * gpb + granule % gpb) * 32u + (block & 31u);
}

inline uint i2x_row_elems(uint k, uint gpb) {
  return ((k / (gpb * 8u) + 31u) / 32u) * gpb * 256u;
}

struct Inkling2ExpPackArgs {
  uint k, rows, gpb, pad;
};

[[kernel, max_total_threads_per_threadgroup(I2X_DOT_THREADS)]]
void inkling2_exp_group_pack(const device uint4* x [[buffer(0)]], device uint4* packed [[buffer(1)]],
                             constant Inkling2ExpPackArgs& a [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
  const uint granules = a.k / 8u;
  if (at.x >= granules || at.y >= a.rows) return;
  packed[ulong(at.y) * (i2x_row_elems(a.k, a.gpb) / 8u) + i2x_slot(at.x, a.gpb)] = x[ulong(at.y) * granules + at.x];
}

inline float2 i2x_wide(uint word) { return float2(as_type<bfloat2>(word)); }

// What one thread decodes of a K slice: a scale code and its packed weight bytes.
struct I2xRaw {
  uint code;
  uint4 q;
};

template <bool NV>
inline I2xRaw i2x_group_load(const device uchar* wp, const device uchar* sp, uint pitch, uint blocks, uint j, uint u) {
  I2xRaw raw = {0u, uint4(0u)};
  const uint row = u >> 6u, block = 32u * j + ((u >> 1u) & 31u), half_ = u & 1u;
  if (block >= blocks) return raw;
  raw.code = sp[row * blocks + block];
  if (NV) raw.q.x = *(const device uint*)(wp + row * pitch + block * 8u + half_ * 4u);
  else raw.q = *(const device uint4*)(wp + row * pitch + block * 32u + half_ * 16u);
  return raw;
}

// Decoded BF16 pairs of the slice's rows land at slice[(row * 32 + lane) * stride + word].
template <bool NV>
inline void i2x_group_decode(I2xRaw raw, uint blocks, float outer, uint j, uint u, threadgroup uint* slice) {
  const uint stride = NV ? 9u : 17u;
  const uint row = u >> 6u, lane = (u >> 1u) & 31u, half_ = u & 1u;
  if (32u * j + lane >= blocks) return;
  threadgroup uint* dst = slice + (row * 32u + lane) * stride;
  if (NV) {
    const float sc = i2x_e4m3(raw.code);
    for (uint b = 0u; b < 4u; ++b) {
      const uint byte = (raw.q.x >> (8u * b)) & 255u;
      const uint lo = as_type<ushort>(bfloat((kI2xE2m1[byte & 15u] * sc) * outer));
      const uint hi = as_type<ushort>(bfloat((kI2xE2m1[byte >> 4u] * sc) * outer));
      dst[half_ * 4u + b] = lo | (hi << 16u);
    }
  } else {
    const float sc = i2x_mx_scale(raw.code), mant = raw.code == 0u ? 0.5f : 1.0f;
    for (uint w = 0u; w < 4u; ++w) {
      const uint word = raw.q[w];
      const uint v0 = as_type<ushort>(bfloat((i2x_e4m3(word & 255u) * mant) * sc));
      const uint v1 = as_type<ushort>(bfloat((i2x_e4m3((word >> 8u) & 255u) * mant) * sc));
      const uint v2 = as_type<ushort>(bfloat((i2x_e4m3((word >> 16u) & 255u) * mant) * sc));
      const uint v3 = as_type<ushort>(bfloat((i2x_e4m3(word >> 24u) * mant) * sc));
      dst[half_ * 8u + 2u * w] = v0 | (v1 << 16u);
      dst[half_ * 8u + 2u * w + 1u] = v2 | (v3 << 16u);
    }
  }
}

// Element offset of an output column in a pair's row: gate/up rows are stored grouped, down rows plainly.
template <bool NV, bool GATE_UP>
inline ulong i2x_group_dst(const constant Inkling2ExpDotArgs& a, uint pair, uint col) {
  if (!GATE_UP) return ulong(pair) * a.out_width + col;
  const uint gpb = NV ? 2u : 4u;
  return ulong(pair) * i2x_row_elems(a.out_width, gpb) + i2x_slot(col / 8u, gpb) * 8u + (col & 7u);
}

// A tile holds up to 8 * RT pairs; each SIMDgroup owns RT of them and every lane runs the one-row dot's K chain.
template <bool NV, bool GATE_UP, uint RG>
inline void i2x_group_dot(const device uchar* weight, const device uchar* w_scales, const device float* w_factor,
                          const device bfloat* x, const device uint* order, const device uint4* tiles,
                          device bfloat* out, constant Inkling2ExpDotArgs& a, threadgroup uint* wt, uint2 at, uint t) {
  constexpr uint RT = 4u / RG;
  constexpr uint stride = NV ? 9u : 17u, words = NV ? 8u : 16u, gpb = NV ? 2u : 4u;
  constexpr uint slice_words = RG * 4u * 32u * stride;
  const uint lane = t & 31u, sg = t >> 5u;
  const uint col0 = at.x * RG * (GATE_UP ? 2u : 4u);
  if (col0 >= a.out_width) return;
  const uint4 tile = tiles[1u + at.y];
  const uint expert = tile.x, live = tile.z;
  const uint first = sg * RT;
  uint pair[RT], xo[RT];
  for (uint r = 0u; r < RT; ++r) {
    pair[r] = order[tile.y + min(first + r, live - 1u)];
    xo[r] = (pair[r] / a.src_div) * i2x_row_elems(a.k, gpb);
  }
  if (expert >= a.experts) {
    if (lane != 0u) return;
    for (uint r = 0u; r < RT && first + r < live; ++r) {
      for (uint g = 0u; g < RG; ++g) {
        device bfloat* zero = out + i2x_group_dst<NV, GATE_UP>(a, pair[r], col0 + g * (GATE_UP ? 2u : 4u));
        zero[0] = bfloat(0.0f);
        zero[1] = bfloat(0.0f);
        if (!GATE_UP) { zero[2] = bfloat(0.0f); zero[3] = bfloat(0.0f); }
      }
    }
    return;
  }
  const uint pitch = NV ? a.k / 2u : a.k;
  const uint blocks = a.k / (NV ? 16u : 32u);
  const ulong w_row = ulong(expert) * a.n + col0 * (GATE_UP ? 2u : 1u);
  const device uchar* wp = weight + w_row * pitch;
  const device uchar* sp = w_scales + w_row * blocks;
  const float outer = NV ? w_factor[expert] : 1.0f;
  const bool busy = first < live;

  float4 total[RG][RT];
  for (uint g = 0u; g < RG; ++g) for (uint r = 0u; r < RT; ++r) total[g][r] = 0.0f;
  I2xRaw raw[RG];
  for (uint g = 0u; g < RG; ++g) raw[g] = i2x_group_load<NV>(wp, sp, pitch, blocks, 0u, t + 256u * g);
  for (uint j = 0u; 32u * j < blocks; ++j) {
    threadgroup uint* slice = wt + (j & 1u) * slice_words;
    for (uint g = 0u; g < RG; ++g) i2x_group_decode<NV>(raw[g], blocks, outer, j, t + 256u * g, slice);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = 0u; g < RG; ++g) raw[g] = i2x_group_load<NV>(wp, sp, pitch, blocks, j + 1u, t + 256u * g);
    const uint block = 32u * j + lane;
    if (!busy || block >= blocks) continue;
    float4 run[RG][RT];
    for (uint g = 0u; g < RG; ++g) for (uint r = 0u; r < RT; ++r) run[g][r] = 0.0f;
    const threadgroup uint* mine = slice + lane * stride;
    for (uint h = 0u; h < words / 4u; ++h) {
      uint4 xw[RT];
      for (uint r = 0u; r < RT; ++r) xw[r] = ((const device uint4*)(x + xo[r]))[(j * gpb + h) * 32u + lane];
      for (uint i = 0u; i < 4u; ++i) {
        float2 v[RT];
        for (uint r = 0u; r < RT; ++r) v[r] = i2x_wide(xw[r][i]);
        for (uint g = 0u; g < RG; ++g) {
          const threadgroup uint* rows = mine + g * (128u * stride) + 4u * h + i;
          const float2 w0 = i2x_wide(rows[0]), w1 = i2x_wide(rows[32u * stride]);
          const float2 w2 = i2x_wide(rows[64u * stride]), w3 = i2x_wide(rows[96u * stride]);
          const float4 wl = float4(w0.x, w1.x, w2.x, w3.x), wh = float4(w0.y, w1.y, w2.y, w3.y);
          for (uint r = 0u; r < RT; ++r) {
            run[g][r] += wl * v[r].x;
            run[g][r] += wh * v[r].y;
          }
        }
      }
    }
    for (uint g = 0u; g < RG; ++g) for (uint r = 0u; r < RT; ++r) total[g][r] += run[g][r];
  }
  if (!busy) return;
  for (uint shift = 16u; shift != 0u; shift >>= 1u)
    for (uint g = 0u; g < RG; ++g) for (uint r = 0u; r < RT; ++r) total[g][r] += simd_shuffle_xor(total[g][r], shift);
  if (lane != 0u) return;
  for (uint r = 0u; r < RT && first + r < live; ++r) {
    for (uint g = 0u; g < RG; ++g) {
      device bfloat* dst = out + i2x_group_dst<NV, GATE_UP>(a, pair[r], col0 + g * (GATE_UP ? 2u : 4u));
      if (GATE_UP) {
        const float4 sums = i2x_bf4(total[g][r]);
        dst[0] = bfloat(i2x_swiglu(sums.x, sums.y, a.gate_cap, a.up_cap));
        dst[1] = bfloat(i2x_swiglu(sums.z, sums.w, a.gate_cap, a.up_cap));
      } else {
        dst[0] = bfloat(total[g][r].x);
        dst[1] = bfloat(total[g][r].y);
        dst[2] = bfloat(total[g][r].z);
        dst[3] = bfloat(total[g][r].w);
      }
    }
  }
}

#define I2X_GROUP_DOT(symbol, nv, gate_up, rg, words) \
  [[kernel, max_total_threads_per_threadgroup(I2X_DOT_THREADS)]] \
  void symbol(const device uchar* weight [[buffer(0)]], const device uchar* w_scales [[buffer(1)]], \
              const device float* w_factor [[buffer(2)]], const device bfloat* x [[buffer(3)]], \
              const device uint* order [[buffer(4)]], device bfloat* out [[buffer(5)]], \
              constant Inkling2ExpDotArgs& a [[buffer(6)]], const device uint4* tiles [[buffer(7)]], \
              uint2 at [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) { \
    threadgroup uint wt[words]; \
    i2x_group_dot<nv, gate_up, rg>(weight, w_scales, w_factor, x, order, tiles, out, a, wt, at, t); \
  }

I2X_GROUP_DOT(inkling2_exp_group_gate_up_nvfp4, true, true, I2X_NV_RG, I2X_NV_WORDS)
I2X_GROUP_DOT(inkling2_exp_group_gate_up_mxfp8, false, true, I2X_MX_RG, I2X_MX_WORDS)
I2X_GROUP_DOT(inkling2_exp_group_down_nvfp4, true, false, I2X_NV_RG, I2X_NV_WORDS)
I2X_GROUP_DOT(inkling2_exp_group_down_mxfp8, false, false, I2X_MX_RG, I2X_MX_WORDS)
#undef I2X_GROUP_DOT
