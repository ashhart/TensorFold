// Grouped prompt experts on simdgroup MMA, bit-equal to the one-row dot (an 8x8x8 MMA sums into C in k order).
#define I2M_THREADS 256u
// A tile is 64 weight rows by up to 32 pairs; a stage puts STEPS K octets of both through threadgroup memory.
#define I2M_ROWS 64u
#define I2M_PAIRS 32u
#define I2M_STEPS(nv) ((nv) ? 4u : 8u)
// The 32 lane chains of a row split into NP passes of NC chains each: pass r takes the chains L = r (mod NP).
#define I2M_NP 8u
#define I2M_NC (32u / I2M_NP)
// Simdgroup sg takes row tile sg / PG and the pair tiles [pg * PT, pg * PT + PT) of its pair group.
#define I2M_OT (I2M_ROWS / 8u)
#define I2M_PG (8u / I2M_OT)
#define I2M_PT (I2M_PAIRS / (8u * I2M_PG))
#define I2M_WORDS(nv) (2u * I2M_STEPS(nv) * I2M_ROWS * 4u)
#define I2M_XWORDS(nv) (2u * I2M_STEPS(nv) * I2M_PAIRS * 4u)
// Decode units (two K octets of one row) and staged activation octets of a stage, per thread.
#define I2M_UNITS(nv) (I2M_ROWS * (I2M_STEPS(nv) / 2u))
#define I2M_DU(nv) ((I2M_UNITS(nv) + I2M_THREADS - 1u) / I2M_THREADS)
#define I2M_XU(nv) ((I2M_PAIRS * I2M_STEPS(nv) + I2M_THREADS - 1u) / I2M_THREADS)

using i2m_frag = simdgroup_matrix<float, 8, 8>;

// What one thread decodes of a unit: a scale code and sixteen packed weights (NVFP4 one block, MXFP8 half of one).
struct I2mRaw {
  uint code;
  uint4 q;
};

// Unit index `i` of a stage: eight consecutive lanes take eight rows of one unit.
template <bool NV>
inline uint i2m_row(uint i) {
  return (i & 7u) | ((i / (8u * (I2M_STEPS(NV) / 2u))) << 3u);
}
template <bool NV>
inline uint i2m_unit(uint i) {
  return (i >> 3u) % (I2M_STEPS(NV) / 2u);
}

// Pass r of the lane tree takes the blocks b = r (mod NP); stage s of a pass holds `per_stage` of them, NP apart.
template <bool NV>
inline uint i2m_block(uint pass, uint stage, uint unit) {
  return pass + I2M_NP * (stage * (NV ? I2M_STEPS(NV) / 2u : I2M_STEPS(NV) / 4u) + unit);
}

template <bool NV>
inline I2mRaw i2m_load(const device uchar* wp, const device uchar* sp, uint pitch, uint blocks, uint pass, uint stage,
                       uint i) {
  I2mRaw raw = {0u, uint4(0u)};
  if (i >= I2M_UNITS(NV)) return raw;
  const uint row = i2m_row<NV>(i), u = i2m_unit<NV>(i);
  const uint block = i2m_block<NV>(pass, stage, NV ? u : (u >> 1u));
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
inline void i2m_decode(I2mRaw raw, float outer, uint i, threadgroup uint* tgw) {
  if (i >= I2M_UNITS(NV)) return;
  const uint row = i2m_row<NV>(i), u = i2m_unit<NV>(i);
  threadgroup uint* dst = tgw + (2u * u * I2M_ROWS + row) * 4u;
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
  *(threadgroup uint4*)(dst + I2M_ROWS * 4u) = uint4(words[4], words[5], words[6], words[7]);
}

// Activations stay row-major per pair, [step][pair tile][pair][k]; the loads run a stage ahead of the LDS stores.
template <bool NV>
inline void i2m_load_x(const device uint4* xrow, bool live, uint pass, uint stage, uint blocks, uint t,
                       thread uint4 (&q)[I2M_XU(NV)]) {
  constexpr uint gpb = NV ? 2u : 4u;
  _Pragma("clang loop unroll(full)") for (uint h = 0u; h < I2M_XU(NV); ++h) {
    const uint ss = t / I2M_PAIRS + (I2M_THREADS / I2M_PAIRS) * h;
    const uint block = i2m_block<NV>(pass, stage, ss / gpb);
    q[h] = live && ss < I2M_STEPS(NV) && block < blocks ? xrow[block * gpb + ss % gpb] : uint4(0u);
  }
}

template <bool NV>
inline void i2m_store_x(const thread uint4 (&q)[I2M_XU(NV)], uint t, threadgroup uint* tgx) {
  const uint pair = t % I2M_PAIRS;
  _Pragma("clang loop unroll(full)") for (uint h = 0u; h < I2M_XU(NV); ++h) {
    const uint ss = t / I2M_PAIRS + (I2M_THREADS / I2M_PAIRS) * h;
    if (ss < I2M_STEPS(NV))
      *(threadgroup uint4*)(tgx + (((ss * (I2M_PAIRS / 8u) + (pair >> 3u)) * 8u + (pair & 7u)) * 4u)) = q[h];
  }
}

// One staged stage on the MMAs: each block's K octets sum through `part` from zero, then into its lane chain.
template <bool NV, bool CHECK, uint PTA>
inline void i2m_stage_mma(const threadgroup uint* aw, const threadgroup uint* bx,
                          thread i2m_frag (&acc)[I2M_PT][I2M_NC], uint si, uint ot, uint pg, uint blocks, uint pass,
                          uint stage) {
  constexpr uint bsteps = NV ? 2u : 4u, per_stage = I2M_STEPS(NV) / bsteps, PT = PTA;
  _Pragma("clang loop unroll(full)") for (uint i = 0u; i < per_stage; ++i) {
    if (CHECK && i2m_block<NV>(pass, stage, i) >= blocks) break;
    i2m_frag part[PT];
    _Pragma("clang loop unroll(full)") for (uint j = 0u; j < PT; ++j) part[j] = i2m_frag(0.0f);
    _Pragma("clang loop unroll(full)") for (uint s = 0u; s < bsteps; ++s) {
      const uint ss = i * bsteps + s;
      simdgroup_matrix<bfloat, 8, 8> fa, fb;
      simdgroup_load(fa, (const threadgroup bfloat*)(aw + (ss * I2M_ROWS + ot * 8u) * 4u), 8);
      _Pragma("clang loop unroll(full)") for (uint j = 0u; j < PT; ++j) {
        simdgroup_load(fb, (const threadgroup bfloat*)(bx + (ss * (I2M_PAIRS / 8u) + pg * I2M_PT + j) * 32u), 8,
                       ulong2(0, 0), true);
        simdgroup_multiply_accumulate(part[j], fa, fb, part[j]);
      }
    }
    _Pragma("clang loop unroll(full)") for (uint j = 0u; j < PT; ++j) {
      acc[j][si * per_stage + i].thread_elements()[0] += part[j].thread_elements()[0];
      acc[j][si * per_stage + i].thread_elements()[1] += part[j].thread_elements()[1];
    }
  }
}

// The pair tiles a simdgroup runs, 1..PT, are fixed per tile: skip the dead ones with a uniform switch.
template <bool NV, bool CHECK>
inline void i2m_stage_run(uint active, const threadgroup uint* aw, const threadgroup uint* bx,
                          thread i2m_frag (&acc)[I2M_PT][I2M_NC], uint si, uint ot, uint pg, uint blocks, uint pass,
                          uint stage) {
#define I2M_RUN(n) i2m_stage_mma<NV, CHECK, (n) < I2M_PT ? (n) : I2M_PT>(aw, bx, acc, si, ot, pg, blocks, pass, stage)
  switch (active) {
    case 1u: I2M_RUN(1u); break;
    case 2u: I2M_RUN(2u); break;
    case 3u: I2M_RUN(3u); break;
    case 4u: I2M_RUN(4u); break;
    case 5u: I2M_RUN(5u); break;
    case 6u: I2M_RUN(6u); break;
    case 7u: I2M_RUN(7u); break;
    default: I2M_RUN(I2M_PT); break;
  }
#undef I2M_RUN
}

// `at` is (column group, tile); the plan lays tiles along grid x so one expert's tiles run together.
template <bool NV, bool GATE_UP>
inline void i2m_group_dot(const device uchar* weight, const device uchar* w_scales, const device float* w_factor,
                          const device bfloat* x, const device uint* order, const device uint4* tiles,
                          device bfloat* out, constant Inkling2ExpDotArgs& a, threadgroup uint* tgw,
                          threadgroup uint* tgx, uint2 at, uint t) {
  constexpr uint bsteps = NV ? 2u : 4u, per_stage = I2M_STEPS(NV) / bsteps;
  constexpr uint cols = GATE_UP ? I2M_ROWS / 2u : I2M_ROWS, PT = I2M_PT;
  if (at.x * cols >= a.out_width) return;
  const uint lane = t & 31u, sg = t >> 5u, ot = sg / I2M_PG, pg = sg % I2M_PG;
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
  const ulong w_row = ulong(expert) * a.n + ulong(at.x) * I2M_ROWS;
  const device uchar* wp = weight + w_row * pitch;
  const device uchar* sp = w_scales + w_row * blocks;
  const float outer = NV ? w_factor[expert] : 1.0f;
  const uint xp = t % I2M_PAIRS;
  const device uint4* xrow = (const device uint4*)(x + ulong(order[tile.y + min(xp, live - 1u)] / a.src_div) * a.k);
  const bool xlive = xp < live;

  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) + ((lane >> 1u) & 3u);
  const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
  // Pair tiles of this simdgroup's group that hold live pairs.
  const uint tiles_live = (live + 7u) / 8u;
  const uint active = min(PT, tiles_live > pg * PT ? tiles_live - pg * PT : 0u);
  float2 sums[PT][I2M_NP];
  uint buf = 0u;
  for (uint pass = 0u; pass < I2M_NP; ++pass) {
    // Lane chains L = pass + NP * m (m < NC) of one pass; each takes its blocks in order, one per round.
    i2m_frag acc[PT][I2M_NC];
    _Pragma("clang loop unroll(full)") for (uint j = 0u; j < PT; ++j)
      _Pragma("clang loop unroll(full)") for (uint m = 0u; m < I2M_NC; ++m) acc[j][m] = i2m_frag(0.0f);
    const uint mine = pass < blocks ? (blocks - pass + I2M_NP - 1u) / I2M_NP : 0u;
    const uint rounds = (mine + I2M_NC - 1u) / I2M_NC;
    I2mRaw raw[I2M_DU(NV)];
    _Pragma("clang loop unroll(full)") for (uint h = 0u; h < I2M_DU(NV); ++h)
      raw[h] = i2m_load<NV>(wp, sp, pitch, blocks, pass, 0u, t + I2M_THREADS * h);
    uint4 xq[I2M_XU(NV)];
    i2m_load_x<NV>(xrow, xlive, pass, 0u, blocks, t, xq);
    for (uint round = 0u; round < rounds; ++round) {
      _Pragma("clang loop unroll(full)") for (uint si = 0u; si < I2M_NC / per_stage; ++si) {
        const uint stage = round * (I2M_NC / per_stage) + si;
        _Pragma("clang loop unroll(full)") for (uint h = 0u; h < I2M_DU(NV); ++h)
          i2m_decode<NV>(raw[h], outer, t + I2M_THREADS * h, tgw + buf * (I2M_STEPS(NV) * I2M_ROWS * 4u));
        i2m_store_x<NV>(xq, t, tgx + buf * (I2M_STEPS(NV) * I2M_PAIRS * 4u));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("clang loop unroll(full)") for (uint h = 0u; h < I2M_DU(NV); ++h)
          raw[h] = i2m_load<NV>(wp, sp, pitch, blocks, pass, stage + 1u, t + I2M_THREADS * h);
        i2m_load_x<NV>(xrow, xlive, pass, stage + 1u, blocks, t, xq);
        const threadgroup uint* aw = tgw + buf * (I2M_STEPS(NV) * I2M_ROWS * 4u);
        const threadgroup uint* bx = tgx + buf * (I2M_STEPS(NV) * I2M_PAIRS * 4u);
        buf ^= 1u;
        if (active == 0u) continue;
        // Only the last stage of a pass can run past the blocks of K.
        if (i2m_block<NV>(pass, stage, per_stage - 1u) < blocks)
          i2m_stage_run<NV, false>(active, aw, bx, acc, si, ot, pg, blocks, pass, stage);
        else
          i2m_stage_run<NV, true>(active, aw, bx, acc, si, ot, pg, blocks, pass, stage);
      }
    }
    // Tree inside the pass: lanes (m, m + NC / 2), then down to (m, m + 1).
    _Pragma("clang loop unroll(full)") for (uint j = 0u; j < PT; ++j) {
      float v[2][I2M_NC];
      _Pragma("clang loop unroll(full)") for (uint m = 0u; m < I2M_NC; ++m) {
        v[0][m] = acc[j][m].thread_elements()[0];
        v[1][m] = acc[j][m].thread_elements()[1];
      }
      _Pragma("clang loop unroll(full)") for (uint s = I2M_NC / 2u; s != 0u; s >>= 1u) {
        _Pragma("clang loop unroll(full)") for (uint m = 0u; m < s; ++m) {
          v[0][m] += v[0][m + s];
          v[1][m] += v[1][m + s];
        }
      }
      sums[j][pass] = float2(v[0][0], v[1][0]);
    }
  }
  _Pragma("clang loop unroll(full)") for (uint j = 0u; j < PT; ++j) {
    if (j >= active) break;
    // Then the passes: lanes (p, p + NP / 2), down to (p, p + 1).
    _Pragma("clang loop unroll(full)") for (uint s = I2M_NP / 2u; s != 0u; s >>= 1u)
      _Pragma("clang loop unroll(full)") for (uint p = 0u; p < s; ++p) sums[j][p] += sums[j][p + s];
    const float2 total = sums[j][0];
    float r0 = float(bfloat(total.x)), r1 = float(bfloat(total.y));
    if (GATE_UP) {
      const float up0 = simd_shuffle_xor(r0, 2u), up1 = simd_shuffle_xor(r1, 2u);
      r0 = i2x_swiglu(r0, up0, a.gate_cap, a.up_cap);
      r1 = i2x_swiglu(r1, up1, a.gate_cap, a.up_cap);
      if (lane & 2u) continue;
    }
    const uint col = GATE_UP ? at.x * cols + ot * 4u + (fm >> 1u) : at.x * cols + ot * 8u + fm;
    const uint p0 = (pg * PT + j) * 8u + fn;
    if (p0 < live) out[ulong(order[tile.y + p0]) * a.out_width + col] = bfloat(r0);
    if (p0 + 1u < live) out[ulong(order[tile.y + p0 + 1u]) * a.out_width + col] = bfloat(r1);
  }
}

#define I2M_GROUP_DOT(symbol, nv, gate_up) \
  [[kernel, max_total_threads_per_threadgroup(I2M_THREADS)]] \
  void symbol(const device uchar* weight [[buffer(0)]], const device uchar* w_scales [[buffer(1)]], \
              const device float* w_factor [[buffer(2)]], const device bfloat* x [[buffer(3)]], \
              const device uint* order [[buffer(4)]], device bfloat* out [[buffer(5)]], \
              constant Inkling2ExpDotArgs& a [[buffer(6)]], const device uint4* tiles [[buffer(7)]], \
              uint2 grid [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) { \
    threadgroup uint tgw[I2M_WORDS(nv)]; \
    threadgroup uint tgx[I2M_XWORDS(nv)]; \
    i2m_group_dot<nv, gate_up>(weight, w_scales, w_factor, x, order, tiles, out, a, tgw, tgx, grid.yx, t); \
  }

I2M_GROUP_DOT(inkling2_exp_mma_gate_up_nvfp4, true, true)
I2M_GROUP_DOT(inkling2_exp_mma_gate_up_mxfp8, false, true)
I2M_GROUP_DOT(inkling2_exp_mma_down_nvfp4, true, false)
I2M_GROUP_DOT(inkling2_exp_mma_down_mxfp8, false, false)
#undef I2M_GROUP_DOT
