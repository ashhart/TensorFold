// Routed experts on simdgroup matrices for GPUs without tensor units: the sums of the grouped kernels, bit for bit.
#define INK2_MMA_THREADS 256u
// A threadgroup's tile: 64 weight rows (one simdgroup 8) by a band of 32 of an expert's pairs.
#define INK2_MMA_ROWS 64u
#define INK2_MMA_PAIRS 32u
// A stage puts STEPS K octets of both through threadgroup memory: two blocks of a row.
#define INK2_MMA_STEPS(nv) ((nv) ? 4u : 8u)
// The 32 lane chains run in NP passes of NC adjacent chains; stage s of a round takes blocks 2s and 2s + 1 of them.
#define INK2_MMA_NP 8u
#define INK2_MMA_NC 4u
#define INK2_MMA_PT (INK2_MMA_PAIRS / 8u)
#define INK2_MMA_WORDS(nv) (2u * INK2_MMA_STEPS(nv) * INK2_MMA_ROWS * 4u)
#define INK2_MMA_XWORDS(nv) (2u * INK2_MMA_STEPS(nv) * INK2_MMA_PAIRS * 4u)
// Decode units (two K octets of one row) and staged activation octets of a stage, per thread.
#define INK2_MMA_UNITS(nv) (INK2_MMA_ROWS * (INK2_MMA_STEPS(nv) / 2u))
#define INK2_MMA_DU(nv) ((INK2_MMA_UNITS(nv) + INK2_MMA_THREADS - 1u) / INK2_MMA_THREADS)
#define INK2_MMA_XU(nv) ((INK2_MMA_PAIRS * INK2_MMA_STEPS(nv) + INK2_MMA_THREADS - 1u) / INK2_MMA_THREADS)

using ink2_frag = simdgroup_matrix<float, 8, 8>;

// E4M3FN as FP16 bits (same layout, bias 15) times 2^8; the sign sits in bit 15, and 0x7f / 0xff stay NaN.
inline float ink2_mma_e4m3(uint b) {
  const float v = float(as_type<half>(ushort(((b & 0x7fu) << 7) | ((b & 0x80u) << 8)))) * 256.0f;
  return (b & 0x7fu) == 0x7fu ? as_type<float>(((b & 0x80u) << 24) | 0x7fc00000u) : v;
}

// Two E2M1 codes of one byte as FP16 bit pairs; the product's 2^14 is carried by the NVFP4 block scale.
inline float2 ink2_mma_e2m1_pair(uint code) {
  return float2(as_type<half2>(((code * 0x00200200u) & 0x0e000e00u) | ((code * 0x01001000u) & 0x80008000u)));
}

// What one thread decodes of a unit: a scale code and sixteen packed weights (NVFP4 one block, MXFP8 half of one).
struct Ink2MmaRaw {
  uint code;
  uint4 q;
};

// Unit index `i` of a stage: eight consecutive lanes take eight rows of one unit.
template <bool NV>
inline uint ink2_mma_row(uint i) {
  return (i & 7u) | ((i / (8u * (INK2_MMA_STEPS(NV) / 2u))) << 3u);
}
template <bool NV>
inline uint ink2_mma_unit(uint i) {
  return (i >> 3u) % (INK2_MMA_STEPS(NV) / 2u);
}

// Block `bi` (0 or 1) of stage `stage` for the chains of `pass`: chain 4 * pass + m takes blocks m, m + 32, ...
inline uint ink2_mma_block(uint pass, uint stage, uint bi) {
  return 32u * (stage >> 1u) + 4u * pass + 2u * (stage & 1u) + bi;
}

template <bool NV>
inline Ink2MmaRaw ink2_mma_load(const device uchar* wp, const device uchar* sp, uint pitch, uint blocks, uint pass,
                                uint stage, uint i) {
  Ink2MmaRaw raw = {0u, uint4(0u)};
  if (i >= INK2_MMA_UNITS(NV)) return raw;
  const uint row = ink2_mma_row<NV>(i), u = ink2_mma_unit<NV>(i);
  const uint block = ink2_mma_block(pass, stage, NV ? u : (u >> 1u));
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

// One K octet (eight weights, four BF16 pairs) per uint4, each weight rounded as ink2_nv_block and ink2_mx_block do.
template <bool NV>
inline void ink2_mma_decode(Ink2MmaRaw raw, float phi, uint i, threadgroup uint* tgw) {
  if (i >= INK2_MMA_UNITS(NV)) return;
  const uint row = ink2_mma_row<NV>(i), u = ink2_mma_unit<NV>(i);
  threadgroup uint* dst = tgw + (2u * u * INK2_MMA_ROWS + row) * 4u;
  uint words[8];
  if (NV) {
    const float sc = ink2_mma_e4m3(raw.code) * 16384.0f;
    for (uint w = 0u; w < 2u; ++w) {
      for (uint b = 0u; b < 4u; ++b) {
        const float2 e = ink2_mma_e2m1_pair((raw.q[w] >> (8u * b)) & 255u);
        const uint lo = as_type<ushort>(bfloat((e.x * sc) * phi));
        const uint hi = as_type<ushort>(bfloat((e.y * sc) * phi));
        words[4u * w + b] = lo | (hi << 16u);
      }
    }
  } else {
    // 2^(s - 127) as the grouped kernel's ldexp gives it; the code 0 scale halves 2^-126.
    const float sc = as_type<float>(max(raw.code, 1u) << 23u), mant = select(1.0f, 0.5f, raw.code == 0u);
    for (uint w = 0u; w < 4u; ++w) {
      const uint word = raw.q[w];
      const uint v0 = as_type<ushort>(bfloat((ink2_mma_e4m3(word & 255u) * mant) * sc));
      const uint v1 = as_type<ushort>(bfloat((ink2_mma_e4m3((word >> 8u) & 255u) * mant) * sc));
      const uint v2 = as_type<ushort>(bfloat((ink2_mma_e4m3((word >> 16u) & 255u) * mant) * sc));
      const uint v3 = as_type<ushort>(bfloat((ink2_mma_e4m3(word >> 24u) * mant) * sc));
      words[2u * w] = v0 | (v1 << 16u);
      words[2u * w + 1u] = v2 | (v3 << 16u);
    }
  }
  *(threadgroup uint4*)dst = uint4(words[0], words[1], words[2], words[3]);
  *(threadgroup uint4*)(dst + INK2_MMA_ROWS * 4u) = uint4(words[4], words[5], words[6], words[7]);
}

// Activations stay row-major per pair, [step][pair tile][pair][k]; the loads run a stage ahead of the stores.
template <bool NV>
inline void ink2_mma_load_x(const device uint4* xrow, bool live, uint pass, uint stage, uint blocks, uint t,
                            thread uint4 (&q)[INK2_MMA_XU(NV)]) {
  constexpr uint gpb = NV ? 2u : 4u;
  _Pragma("clang loop unroll(full)") for (uint h = 0u; h < INK2_MMA_XU(NV); ++h) {
    const uint ss = t / INK2_MMA_PAIRS + (INK2_MMA_THREADS / INK2_MMA_PAIRS) * h;
    const uint block = ink2_mma_block(pass, stage, ss / gpb);
    q[h] = live && ss < INK2_MMA_STEPS(NV) && block < blocks ? xrow[block * gpb + ss % gpb] : uint4(0u);
  }
}

template <bool NV>
inline void ink2_mma_store_x(const thread uint4 (&q)[INK2_MMA_XU(NV)], uint t, threadgroup uint* tgx) {
  const uint pair = t % INK2_MMA_PAIRS;
  _Pragma("clang loop unroll(full)") for (uint h = 0u; h < INK2_MMA_XU(NV); ++h) {
    const uint ss = t / INK2_MMA_PAIRS + (INK2_MMA_THREADS / INK2_MMA_PAIRS) * h;
    if (ss < INK2_MMA_STEPS(NV))
      *(threadgroup uint4*)(tgx + (((ss * (INK2_MMA_PAIRS / 8u) + (pair >> 3u)) * 8u + (pair & 7u)) * 4u)) = q[h];
  }
}

// One staged stage on the MMAs: each block's K octets add into its lane chain's fragment, in order.
template <bool NV, bool CHECK, uint PT>
inline void ink2_mma_stage(const threadgroup uint* aw, const threadgroup uint* bx,
                           thread ink2_frag (&acc)[INK2_MMA_PT][INK2_MMA_NC], uint si, uint ot, uint blocks,
                           uint pass, uint stage) {
  constexpr uint bsteps = NV ? 2u : 4u;
  _Pragma("clang loop unroll(full)") for (uint i = 0u; i < 2u; ++i) {
    if (CHECK && ink2_mma_block(pass, stage, i) >= blocks) break;
    _Pragma("clang loop unroll(full)") for (uint s = 0u; s < bsteps; ++s) {
      const uint ss = i * bsteps + s;
      simdgroup_matrix<bfloat, 8, 8> fa, fb;
      simdgroup_load(fa, (const threadgroup bfloat*)(aw + (ss * INK2_MMA_ROWS + ot * 8u) * 4u), 8);
      _Pragma("clang loop unroll(full)") for (uint j = 0u; j < PT; ++j) {
        simdgroup_load(fb, (const threadgroup bfloat*)(bx + (ss * (INK2_MMA_PAIRS / 8u) + j) * 32u), 8, ulong2(0, 0),
                       true);
        simdgroup_multiply_accumulate(acc[j][si * 2u + i], fa, fb, acc[j][si * 2u + i]);
      }
    }
  }
}

// The pair tiles a band holds, 1 to 4, are fixed for it: skip the dead ones with a uniform switch.
template <bool NV, bool CHECK>
inline void ink2_mma_run(uint active, const threadgroup uint* aw, const threadgroup uint* bx,
                         thread ink2_frag (&acc)[INK2_MMA_PT][INK2_MMA_NC], uint si, uint ot, uint blocks, uint pass,
                         uint stage) {
  switch (active) {
    case 1u: ink2_mma_stage<NV, CHECK, 1u>(aw, bx, acc, si, ot, blocks, pass, stage); break;
    case 2u: ink2_mma_stage<NV, CHECK, 2u>(aw, bx, acc, si, ot, blocks, pass, stage); break;
    case 3u: ink2_mma_stage<NV, CHECK, 3u>(aw, bx, acc, si, ot, blocks, pass, stage); break;
    default: ink2_mma_stage<NV, CHECK, INK2_MMA_PT>(aw, bx, acc, si, ot, blocks, pass, stage); break;
  }
}

// Threadgroup (x, e): rows 64x .. 64x + 63 of expert e for each band of 32 of its pairs; GU is X by row, else by pair.
template <bool NV, bool GU>
inline void ink2_mma(device const bfloat* X, device const uchar* W, device const uchar* S, float phi,
                     device const uint* counts, device const uint* members, device bfloat* Y,
                     constant Ink2GroupedArgs& a, uint2 tg, uint t, threadgroup uint* tgw, threadgroup uint* tgx) {
  // A BF16 product is exact in FP32 and an MMA adds its eight into C in order: a lane's chain is one fragment.
  const uint expert = tg.y, count = counts[expert];
  if (count == 0u) return;
  constexpr uint cols = GU ? INK2_MMA_ROWS / 2u : INK2_MMA_ROWS, per_round = INK2_MMA_NC / 2u;
  const uint lane = t & 31u, ot = t >> 5u;
  const uint rows_w = GU ? 2u * a.n : a.n, pitch = NV ? a.k / 2u : a.k, blocks = a.k / (NV ? 16u : 32u);
  const ulong w_row = ulong(expert) * rows_w + ulong(tg.x) * INK2_MMA_ROWS;
  device const uchar* wp = W + w_row * pitch;
  device const uchar* sp = S + w_row * blocks;
  const uint xp = t % INK2_MMA_PAIRS;
  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) + ((lane >> 1u) & 3u);
  const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
  uint buf = 0u;
  for (uint band = 0u; band < count; band += INK2_MMA_PAIRS) {
    const uint live = min(INK2_MMA_PAIRS, count - band);
    const uint pair = members[expert * a.cap + band + min(xp, live - 1u)];
    const device uint4* xrow = (const device uint4*)(X + ulong(GU ? pair / a.topk : pair) * a.k);
    const bool xlive = xp < live;
    const uint active = min(INK2_MMA_PT, (live + 7u) / 8u);
    float2 sums[INK2_MMA_PT][INK2_MMA_NP];
    for (uint pass = 0u; pass < INK2_MMA_NP; ++pass) {
      ink2_frag acc[INK2_MMA_PT][INK2_MMA_NC];
      _Pragma("clang loop unroll(full)") for (uint j = 0u; j < INK2_MMA_PT; ++j)
        _Pragma("clang loop unroll(full)") for (uint m = 0u; m < INK2_MMA_NC; ++m) acc[j][m] = ink2_frag(0.0f);
      // Chain 4 * pass + m has a block in round r while 32 r + 4 pass + m < blocks.
      const uint rounds = blocks > 4u * pass ? (blocks - 4u * pass + 31u) / 32u : 0u;
      Ink2MmaRaw raw[INK2_MMA_DU(NV)];
      _Pragma("clang loop unroll(full)") for (uint h = 0u; h < INK2_MMA_DU(NV); ++h)
        raw[h] = ink2_mma_load<NV>(wp, sp, pitch, blocks, pass, 0u, t + INK2_MMA_THREADS * h);
      uint4 xq[INK2_MMA_XU(NV)];
      ink2_mma_load_x<NV>(xrow, xlive, pass, 0u, blocks, t, xq);
      for (uint round = 0u; round < rounds; ++round) {
        _Pragma("clang loop unroll(full)") for (uint si = 0u; si < per_round; ++si) {
          const uint stage = round * per_round + si;
          _Pragma("clang loop unroll(full)") for (uint h = 0u; h < INK2_MMA_DU(NV); ++h)
            ink2_mma_decode<NV>(raw[h], phi, t + INK2_MMA_THREADS * h, tgw + buf * (INK2_MMA_STEPS(NV) * INK2_MMA_ROWS * 4u));
          ink2_mma_store_x<NV>(xq, t, tgx + buf * (INK2_MMA_STEPS(NV) * INK2_MMA_PAIRS * 4u));
          threadgroup_barrier(mem_flags::mem_threadgroup);
          _Pragma("clang loop unroll(full)") for (uint h = 0u; h < INK2_MMA_DU(NV); ++h)
            raw[h] = ink2_mma_load<NV>(wp, sp, pitch, blocks, pass, stage + 1u, t + INK2_MMA_THREADS * h);
          ink2_mma_load_x<NV>(xrow, xlive, pass, stage + 1u, blocks, t, xq);
          const threadgroup uint* aw = tgw + buf * (INK2_MMA_STEPS(NV) * INK2_MMA_ROWS * 4u);
          const threadgroup uint* bx = tgx + buf * (INK2_MMA_STEPS(NV) * INK2_MMA_PAIRS * 4u);
          buf ^= 1u;
          // Only the last round of a pass can run past the blocks of K.
          if (ink2_mma_block(pass, stage, 1u) < blocks)
            ink2_mma_run<NV, false>(active, aw, bx, acc, si, ot, blocks, pass, stage);
          else
            ink2_mma_run<NV, true>(active, aw, bx, acc, si, ot, blocks, pass, stage);
        }
      }
      // The pass's four chains add as simd_sum's first two levels do: (0 + 1) + (2 + 3).
      _Pragma("clang loop unroll(full)") for (uint j = 0u; j < INK2_MMA_PT; ++j) {
        const float2 c0 = float2(acc[j][0].thread_elements()[0], acc[j][0].thread_elements()[1]);
        const float2 c1 = float2(acc[j][1].thread_elements()[0], acc[j][1].thread_elements()[1]);
        const float2 c2 = float2(acc[j][2].thread_elements()[0], acc[j][2].thread_elements()[1]);
        const float2 c3 = float2(acc[j][3].thread_elements()[0], acc[j][3].thread_elements()[1]);
        sums[j][pass] = (c0 + c1) + (c2 + c3);
      }
    }
    _Pragma("clang loop unroll(full)") for (uint j = 0u; j < INK2_MMA_PT; ++j) {
      if (j >= active) break;
      // Then the passes, adjacent first: (0 + 1) + (2 + 3) and (4 + 5) + (6 + 7), then the two.
      _Pragma("clang loop unroll(full)") for (uint s = 1u; s < INK2_MMA_NP; s <<= 1u)
        _Pragma("clang loop unroll(full)") for (uint p = 0u; p < INK2_MMA_NP; p += 2u * s) sums[j][p] += sums[j][p + s];
      const float2 total = sums[j][0];
      const uint p0 = j * 8u + fn;
      if (GU) {
        // A row's gate and up rows are neighbours: the lane with the odd row shares its sums.
        const float up0 = simd_shuffle_xor(total.x, 2u), up1 = simd_shuffle_xor(total.y, 2u);
        if (lane & 2u) continue;
        const uint col = tg.x * cols + ot * 4u + (fm >> 1u);
        if (p0 < live) Y[ulong(members[expert * a.cap + band + p0]) * a.n + col] = ink2_act(total.x, up0);
        if (p0 + 1u < live) Y[ulong(members[expert * a.cap + band + p0 + 1u]) * a.n + col] = ink2_act(total.y, up1);
      } else {
        const uint col = tg.x * cols + ot * 8u + fm;
        if (p0 < live) Y[ulong(members[expert * a.cap + band + p0]) * a.n + col] = bfloat(total.x);
        if (p0 + 1u < live) Y[ulong(members[expert * a.cap + band + p0 + 1u]) * a.n + col] = bfloat(total.y);
      }
    }
  }
}

#define INK2_MMA_NVFP4(symbol, gu)                                                                                 \
  [[kernel, max_total_threads_per_threadgroup(INK2_MMA_THREADS)]] void symbol(                                    \
      device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]], device const uchar* S [[buffer(2)]], \
      device const float* amax [[buffer(3)]], device const float* denominator [[buffer(4)]],                      \
      device const uint* counts [[buffer(5)]], device const uint* members [[buffer(6)]],                          \
      device bfloat* Y [[buffer(7)]], constant Ink2GroupedArgs& a [[buffer(8)]],                                  \
      uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {                        \
    threadgroup uint tgw[INK2_MMA_WORDS(true)];                                                                    \
    threadgroup uint tgx[INK2_MMA_XWORDS(true)];                                                                   \
    ink2_mma<true, gu>(X, W, S, ink2_phi(amax, denominator, tg.y), counts, members, Y, a, tg, t, tgw, tgx);       \
  }

#define INK2_MMA_MXFP8(symbol, gu)                                                                                 \
  [[kernel, max_total_threads_per_threadgroup(INK2_MMA_THREADS)]] void symbol(                                    \
      device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]], device const uchar* S [[buffer(2)]], \
      device const uint* counts [[buffer(3)]], device const uint* members [[buffer(4)]],                          \
      device bfloat* Y [[buffer(5)]], constant Ink2GroupedArgs& a [[buffer(6)]],                                  \
      uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {                        \
    threadgroup uint tgw[INK2_MMA_WORDS(false)];                                                                   \
    threadgroup uint tgx[INK2_MMA_XWORDS(false)];                                                                  \
    ink2_mma<false, gu>(X, W, S, 1.0f, counts, members, Y, a, tg, t, tgw, tgx);                                   \
  }

INK2_MMA_NVFP4(tf_ink2_nvfp4_gu_mma, true)
INK2_MMA_NVFP4(tf_ink2_nvfp4_down_mma, false)
INK2_MMA_MXFP8(tf_ink2_mxfp8_gu_mma, true)
INK2_MMA_MXFP8(tf_ink2_mxfp8_down_mma, false)
