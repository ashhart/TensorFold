// Grouped prompt experts on simdgroup MMA, bit-equal to the one-row dot (an 8x8x8 MMA sums into C in k order).
#define I2M_THREADS 256u
#define I2M_STEPS 16u
#define I2M_WORDS (2u * I2M_STEPS * 32u * 4u)
#define I2M_XWORDS (2u * I2M_STEPS * 64u)

using i2m_frag = simdgroup_matrix<float, 8, 8>;

// What one thread decodes of a stage: a scale code and sixteen packed weights (NVFP4 one block, MXFP8 half of one).
struct I2mRaw {
  uint code;
  uint4 q;
};

// Pass r of the lane tree takes the blocks b = r (mod 4); stage s of a pass holds `per_stage` of them, four apart.
template <bool NV>
inline uint i2m_block(uint pass, uint stage, uint unit) {
  return pass + 4u * (stage * (NV ? 8u : 4u) + unit);
}

template <bool NV>
inline I2mRaw i2m_load(const device uchar* wp, const device uchar* sp, uint pitch, uint blocks, uint pass, uint stage,
                       uint t) {
  const uint row = t >> 3u, u = t & 7u;
  const uint block = i2m_block<NV>(pass, stage, NV ? u : (u >> 1u));
  I2mRaw raw = {0u, uint4(0u)};
  if (block >= blocks) return raw;
  raw.code = sp[row * blocks + block];
  if (NV) {
    const device uint* p = (const device uint*)(wp + row * pitch + block * 8u);
    raw.q = uint4(p[0], p[1], 0u, 0u);
  } else {
    raw.q = *(const device uint4*)(wp + row * pitch + block * 32u + (u & 1u) * 16u);
  }
  return raw;
}

// Decode as the one-row dot does (E2M1 via FP16 bits): one K octet (eight weights, four BF16 pairs) per uint4.
template <bool NV>
inline void i2m_decode(I2mRaw raw, float outer, uint t, threadgroup uint* tgw) {
  const uint row = t >> 3u, u = t & 7u;
  threadgroup uint* dst = tgw + (2u * u * 32u + row) * 4u;
  uint words[8];
  if (NV) {
    const float sc = i2x_e4m3(raw.code) * 16384.0f;
    for (uint w = 0u; w < 2u; ++w) {
      for (uint b = 0u; b < 4u; ++b) {
        const float2 e = i2x_e2m1_pair((raw.q[w] >> (8u * b)) & 255u);
        const uint lo = as_type<ushort>(bfloat((e.x * sc) * outer));
        const uint hi = as_type<ushort>(bfloat((e.y * sc) * outer));
        words[4u * w + b] = lo | (hi << 16u);
      }
    }
  } else {
    const float sc = i2x_mx_scale(raw.code), mant = select(1.0f, 0.5f, raw.code == 0u);
    for (uint w = 0u; w < 4u; ++w) {
      const uint word = raw.q[w];
      const uint v0 = as_type<ushort>(bfloat((i2x_e4m3(word & 255u) * mant) * sc));
      const uint v1 = as_type<ushort>(bfloat((i2x_e4m3((word >> 8u) & 255u) * mant) * sc));
      const uint v2 = as_type<ushort>(bfloat((i2x_e4m3((word >> 16u) & 255u) * mant) * sc));
      const uint v3 = as_type<ushort>(bfloat((i2x_e4m3(word >> 24u) * mant) * sc));
      words[2u * w] = v0 | (v1 << 16u);
      words[2u * w + 1u] = v2 | (v3 << 16u);
    }
  }
  *(threadgroup uint4*)dst = uint4(words[0], words[1], words[2], words[3]);
  *(threadgroup uint4*)(dst + 32u * 4u) = uint4(words[4], words[5], words[6], words[7]);
}

// Activations stay row-major per pair, [step][pair tile][pair][k]; the B fragment loads them transposed.
template <bool NV>
inline void i2m_stage_x(const device uint4* xrow, uint pass, uint stage, uint blocks, uint t, threadgroup uint* tgx) {
  constexpr uint gpb = NV ? 2u : 4u;
  const uint pair = t >> 4u, ss = t & 15u;
  const uint block = i2m_block<NV>(pass, stage, ss / gpb);
  uint4 q = uint4(0u);
  if (block < blocks) q = xrow[block * gpb + ss % gpb];
  *(threadgroup uint4*)(tgx + (((ss * 2u + (pair >> 3u)) * 8u + (pair & 7u)) * 4u)) = q;
}

template <bool NV, bool GATE_UP>
inline void i2m_group_dot(const device uchar* weight, const device uchar* w_scales, const device float* w_factor,
                          const device bfloat* x, const device uint* order, const device uint4* tiles,
                          device bfloat* out, constant Inkling2ExpDotArgs& a, threadgroup uint* tgw,
                          threadgroup uint* tgx, uint2 at, uint t) {
  constexpr uint gpb = NV ? 2u : 4u, bsteps = NV ? 2u : 4u, per_stage = NV ? 8u : 4u;
  constexpr uint cols = GATE_UP ? 16u : 32u;
  if (at.x * cols >= a.out_width) return;
  const uint lane = t & 31u, sg = t >> 5u, ot = sg >> 1u, pt = sg & 1u;
  const uint4 tile = tiles[1u + at.y];
  const uint expert = tile.x, live = tile.z;
  if (expert >= a.experts) {
    for (uint i = t; i < live * cols; i += I2M_THREADS) {
      const uint who = order[tile.y + i / cols];
      out[ulong(who) * a.out_width + at.x * cols + i % cols] = bfloat(0.0f);
    }
    return;
  }
  const uint pitch = NV ? a.k / 2u : a.k;
  const uint blocks = a.k / (NV ? 16u : 32u);
  const ulong w_row = ulong(expert) * a.n + ulong(at.x) * 32u;
  const device uchar* wp = weight + w_row * pitch;
  const device uchar* sp = w_scales + w_row * blocks;
  const float outer = NV ? w_factor[expert] : 1.0f;
  const uint xpair = order[tile.y + min(t >> 4u, live - 1u)];
  const device uint4* xrow = (const device uint4*)(x + ulong(xpair / a.src_div) * a.k);

  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) + ((lane >> 1u) & 3u);
  const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
  const bool busy = pt * 8u < live;
  float2 sums[4];
  uint buf = 0u;
  for (uint pass = 0u; pass < 4u; ++pass) {
    // Lane chains L = pass + 4m (m < 8) of one pass; each takes its blocks in order, one per round.
    i2m_frag acc[8];
    _Pragma("clang loop unroll(full)") for (uint m = 0u; m < 8u; ++m) acc[m] = i2m_frag(0.0f);
    const uint mine = pass < blocks ? (blocks - pass + 3u) / 4u : 0u;
    const uint rounds = (mine + 7u) / 8u;
    I2mRaw raw = i2m_load<NV>(wp, sp, pitch, blocks, pass, 0u, t);
    for (uint round = 0u; round < rounds; ++round) {
      _Pragma("clang loop unroll(full)") for (uint si = 0u; si < 8u / per_stage; ++si) {
        const uint stage = round * (8u / per_stage) + si;
        i2m_decode<NV>(raw, outer, t, tgw + buf * (I2M_STEPS * 32u * 4u));
        i2m_stage_x<NV>(xrow, pass, stage, blocks, t, tgx + buf * (I2M_STEPS * 64u));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        raw = i2m_load<NV>(wp, sp, pitch, blocks, pass, stage + 1u, t);
        const threadgroup uint* aw = tgw + buf * (I2M_STEPS * 32u * 4u);
        const threadgroup uint* bx = tgx + buf * (I2M_STEPS * 64u);
        buf ^= 1u;
        if (!busy) continue;
        _Pragma("clang loop unroll(full)") for (uint i = 0u; i < per_stage; ++i) {
          if (i2m_block<NV>(pass, stage, i) >= blocks) break;
          // One block's sequential FP32 sum from zero (the MMAs chain through `part`), then the lane's running sum.
          i2m_frag part = i2m_frag(0.0f);
          _Pragma("clang loop unroll(full)") for (uint s = 0u; s < bsteps; ++s) {
            const uint ss = i * bsteps + s;
            simdgroup_matrix<bfloat, 8, 8> fa, fb;
            simdgroup_load(fa, (const threadgroup bfloat*)(aw + (ss * 32u + ot * 8u) * 4u), 8);
            simdgroup_load(fb, (const threadgroup bfloat*)(bx + (ss * 2u + pt) * 32u), 8, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(part, fa, fb, part);
          }
          acc[si * per_stage + i].thread_elements()[0] += part.thread_elements()[0];
          acc[si * per_stage + i].thread_elements()[1] += part.thread_elements()[1];
        }
      }
    }
    // Tree inside the pass: lanes (m, m + 4), then (m, m + 2), then (m, m + 1).
    float v[2][8];
    _Pragma("clang loop unroll(full)") for (uint m = 0u; m < 8u; ++m) {
      v[0][m] = acc[m].thread_elements()[0];
      v[1][m] = acc[m].thread_elements()[1];
    }
    _Pragma("clang loop unroll(full)") for (uint s = 4u; s != 0u; s >>= 1u) {
      _Pragma("clang loop unroll(full)") for (uint m = 0u; m < s; ++m) {
        v[0][m] += v[0][m + s];
        v[1][m] += v[1][m + s];
      }
    }
    sums[pass] = float2(v[0][0], v[1][0]);
  }
  if (!busy) return;
  const float2 total = (sums[0] + sums[2]) + (sums[1] + sums[3]);
  float r0 = float(bfloat(total.x)), r1 = float(bfloat(total.y));
  if (GATE_UP) {
    const float up0 = simd_shuffle_xor(r0, 2u), up1 = simd_shuffle_xor(r1, 2u);
    r0 = i2x_swiglu(r0, up0, a.gate_cap, a.up_cap);
    r1 = i2x_swiglu(r1, up1, a.gate_cap, a.up_cap);
    if (lane & 2u) return;
  }
  const uint col = GATE_UP ? at.x * 16u + ot * 4u + (fm >> 1u) : at.x * 32u + ot * 8u + fm;
  const uint p0 = pt * 8u + fn;
  if (p0 < live) out[ulong(order[tile.y + p0]) * a.out_width + col] = bfloat(r0);
  if (p0 + 1u < live) out[ulong(order[tile.y + p0 + 1u]) * a.out_width + col] = bfloat(r1);
}

#define I2M_GROUP_DOT(symbol, nv, gate_up) \
  [[kernel, max_total_threads_per_threadgroup(I2M_THREADS)]] \
  void symbol(const device uchar* weight [[buffer(0)]], const device uchar* w_scales [[buffer(1)]], \
              const device float* w_factor [[buffer(2)]], const device bfloat* x [[buffer(3)]], \
              const device uint* order [[buffer(4)]], device bfloat* out [[buffer(5)]], \
              constant Inkling2ExpDotArgs& a [[buffer(6)]], const device uint4* tiles [[buffer(7)]], \
              uint2 at [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) { \
    threadgroup uint tgw[I2M_WORDS]; \
    threadgroup uint tgx[I2M_XWORDS]; \
    i2m_group_dot<nv, gate_up>(weight, w_scales, w_factor, x, order, tiles, out, a, tgw, tgx, at, t); \
  }

I2M_GROUP_DOT(inkling2_exp_mma_gate_up_nvfp4, true, true)
I2M_GROUP_DOT(inkling2_exp_mma_gate_up_mxfp8, false, true)
I2M_GROUP_DOT(inkling2_exp_mma_down_nvfp4, true, false)
I2M_GROUP_DOT(inkling2_exp_mma_down_mxfp8, false, false)
#undef I2M_GROUP_DOT
