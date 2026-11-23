// Routed experts on simdgroup matrices: the scalar kernels' sums, bit for bit.
#define INK2_MMA_THREADS 256u
// A threadgroup's tile: 64 weight rows (one simdgroup 8) by a band of 32 of an expert's pairs.
#define INK2_MMA_ROWS 64u
#define INK2_MMA_PAIRS 32u
#define INK2_MMA_PT (INK2_MMA_PAIRS / 8u)
// A stage puts eight K octets of both through threadgroup memory.
#define INK2_MMA_STEPS 8u
#define INK2_MMA_WORDS (INK2_MMA_STEPS * INK2_MMA_ROWS * 4u)
#define INK2_MMA_XWORDS (INK2_MMA_STEPS * INK2_MMA_PAIRS * 4u)

struct Ink2MmaGroupArgs { uint pairs, experts, cap; };
struct Ink2MmaArgs { uint k, n, topk, cap; };

using ink2_frag = simdgroup_matrix<float, 8, 8>;

// Expert e's route pairs in pair order: counts[e], then members[e][0 .. counts[e]).
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_ink2_mma_group(
    device const int* ids [[buffer(0)]], device uint* counts [[buffer(1)]],
    device uint* members [[buffer(2)]], constant Ink2MmaGroupArgs& a [[buffer(3)]],
    uint e [[thread_position_in_grid]]) {
  if (e >= a.experts) return;
  uint n = 0u;
  for (uint p = 0u; p < a.pairs; ++p)
    if (uint(ids[p]) == e) {
      if (n < a.cap) members[e * a.cap + n] = p;
      ++n;
    }
  counts[e] = min(n, a.cap);
}

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

// Unit `i` of a stage: eight consecutive lanes take eight rows of one unit, four units a row.
inline uint ink2_mma_row(uint i) { return (i & 7u) | ((i >> 5u) << 3u); }
inline uint ink2_mma_unit(uint i) { return (i >> 3u) & 3u; }

template <bool NV>
inline Ink2MmaRaw ink2_mma_load(const device uchar* wp, const device uchar* sp, uint pitch, uint blocks, uint stage,
                                uint i) {
  Ink2MmaRaw raw = {0u, uint4(0u)};
  const uint row = ink2_mma_row(i), u = ink2_mma_unit(i);
  if (NV) {
    const uint block = 4u * stage + u;
    raw.code = sp[row * blocks + block];
    const device uint* p = (const device uint*)(wp + row * pitch + block * 8u);
    raw.q = uint4(p[0], p[1], 0u, 0u);
  } else {
    raw.code = sp[row * blocks + 2u * stage + (u >> 1u)];
    raw.q = *(const device uint4*)(wp + row * pitch + stage * 64u + u * 16u);
  }
  return raw;
}

// One K octet (eight weights, four BF16 pairs) per uint4, each weight rounded as the scalar decode rounds it.
template <bool NV>
inline void ink2_mma_decode(Ink2MmaRaw raw, float phi, uint i, threadgroup uint* tgw) {
  const uint row = ink2_mma_row(i), u = ink2_mma_unit(i);
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
    // 2^(s - 127) as ldexp gives it; the code 0 scale halves 2^-126.
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

// One stage on the MMAs: a K octet adds its eight products into each output's one chain, in order.
template <uint PT>
inline void ink2_mma_stage(const threadgroup uint* aw, const threadgroup uint* bx, thread ink2_frag (&acc)[INK2_MMA_PT],
                           uint ot) {
  _Pragma("clang loop unroll(full)") for (uint s = 0u; s < INK2_MMA_STEPS; ++s) {
    simdgroup_matrix<bfloat, 8, 8> fa, fb;
    simdgroup_load(fa, (const threadgroup bfloat*)(aw + (s * INK2_MMA_ROWS + ot * 8u) * 4u), 8);
    _Pragma("clang loop unroll(full)") for (uint j = 0u; j < PT; ++j) {
      simdgroup_load(fb, (const threadgroup bfloat*)(bx + (s * INK2_MMA_PT + j) * 32u), 8, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(acc[j], fa, fb, acc[j]);
    }
  }
}

// The pair tiles a band holds, 1 to 4, are fixed for it: skip the dead ones with a uniform switch.
inline void ink2_mma_run(uint active, const threadgroup uint* aw, const threadgroup uint* bx,
                         thread ink2_frag (&acc)[INK2_MMA_PT], uint ot) {
  switch (active) {
    case 1u: ink2_mma_stage<1u>(aw, bx, acc, ot); break;
    case 2u: ink2_mma_stage<2u>(aw, bx, acc, ot); break;
    case 3u: ink2_mma_stage<3u>(aw, bx, acc, ot); break;
    default: ink2_mma_stage<INK2_MMA_PT>(aw, bx, acc, ot); break;
  }
}

// Threadgroup (x, e): rows 64x .. 64x + 63 of expert e for each band of 32 of its pairs; GU is X by row, else by pair.
template <bool NV, bool GU>
inline void ink2_mma(device const bfloat* X, device const uchar* W, device const uchar* S, float phi,
                     device const uint* counts, device const uint* members, device bfloat* Y,
                     constant Ink2MmaArgs& a, uint2 tg, uint t, threadgroup uint* tgw, threadgroup uint* tgx) {
  // A BF16 product is exact in FP32 and an MMA adds its eight into C in order: a scalar sum is one fragment.
  const uint expert = tg.y, count = counts[expert];
  if (count == 0u) return;
  constexpr uint cols = GU ? INK2_MMA_ROWS / 2u : INK2_MMA_ROWS;
  const uint lane = t & 31u, ot = t >> 5u;
  const uint rows_w = GU ? 2u * a.n : a.n, pitch = NV ? a.k / 2u : a.k, blocks = a.k / (NV ? 16u : 32u);
  const ulong w_row = ulong(expert) * rows_w + ulong(tg.x) * INK2_MMA_ROWS;
  device const uchar* wp = W + w_row * pitch;
  device const uchar* sp = S + w_row * blocks;
  const uint xp = t % INK2_MMA_PAIRS, xo = t / INK2_MMA_PAIRS;
  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) + ((lane >> 1u) & 3u);
  const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
  const uint stages = a.k / 64u;
  for (uint band = 0u; band < count; band += INK2_MMA_PAIRS) {
    const uint live = min(INK2_MMA_PAIRS, count - band);
    const uint pair = members[expert * a.cap + band + min(xp, live - 1u)];
    const device uint4* xrow = (const device uint4*)(X + ulong(GU ? pair / a.topk : pair) * a.k);
    const uint active = min(INK2_MMA_PT, (live + 7u) / 8u);
    ink2_frag acc[INK2_MMA_PT];
    _Pragma("clang loop unroll(full)") for (uint j = 0u; j < INK2_MMA_PT; ++j) acc[j] = ink2_frag(0.0f);
    Ink2MmaRaw raw = ink2_mma_load<NV>(wp, sp, pitch, blocks, 0u, t);
    uint4 xq = xrow[xo];
    uint buf = 0u;
    for (uint stage = 0u; stage < stages; ++stage) {
      ink2_mma_decode<NV>(raw, phi, t, tgw + buf * INK2_MMA_WORDS);
      *(threadgroup uint4*)(tgx + buf * INK2_MMA_XWORDS + ((xo * INK2_MMA_PT + (xp >> 3u)) * 8u + (xp & 7u)) * 4u) = xq;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (stage + 1u < stages) {
        raw = ink2_mma_load<NV>(wp, sp, pitch, blocks, stage + 1u, t);
        xq = xrow[(stage + 1u) * INK2_MMA_STEPS + xo];
      }
      ink2_mma_run(active, tgw + buf * INK2_MMA_WORDS, tgx + buf * INK2_MMA_XWORDS, acc, ot);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      buf ^= 1u;
    }
    _Pragma("clang loop unroll(full)") for (uint j = 0u; j < INK2_MMA_PT; ++j) {
      if (j >= active) break;
      const float2 total = float2(acc[j].thread_elements()[0], acc[j].thread_elements()[1]);
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
    // The next band restages from the first buffer.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define INK2_MMA_NVFP4(symbol, gu)                                                                                 \
  [[kernel, max_total_threads_per_threadgroup(INK2_MMA_THREADS)]] void symbol(                                    \
      device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]], device const uchar* S [[buffer(2)]], \
      device const float* amax [[buffer(3)]], device const float* denominator [[buffer(4)]],                      \
      device const uint* counts [[buffer(5)]], device const uint* members [[buffer(6)]],                          \
      device bfloat* Y [[buffer(7)]], constant Ink2MmaArgs& a [[buffer(8)]],                                      \
      uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {                        \
    threadgroup uint tgw[2u * INK2_MMA_WORDS];                                                                     \
    threadgroup uint tgx[2u * INK2_MMA_XWORDS];                                                                    \
    ink2_mma<true, gu>(X, W, S, ink2_phi(amax, denominator, tg.y), counts, members, Y, a, tg, t, tgw, tgx);       \
  }

#define INK2_MMA_MXFP8(symbol, gu)                                                                                 \
  [[kernel, max_total_threads_per_threadgroup(INK2_MMA_THREADS)]] void symbol(                                    \
      device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]], device const uchar* S [[buffer(2)]], \
      device const uint* counts [[buffer(3)]], device const uint* members [[buffer(4)]],                          \
      device bfloat* Y [[buffer(5)]], constant Ink2MmaArgs& a [[buffer(6)]],                                      \
      uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {                        \
    threadgroup uint tgw[2u * INK2_MMA_WORDS];                                                                     \
    threadgroup uint tgx[2u * INK2_MMA_XWORDS];                                                                    \
    ink2_mma<false, gu>(X, W, S, 1.0f, counts, members, Y, a, tg, t, tgw, tgx);                                   \
  }

INK2_MMA_NVFP4(tf_ink2_nvfp4_gu_mma, true)
INK2_MMA_NVFP4(tf_ink2_nvfp4_down_mma, false)
INK2_MMA_MXFP8(tf_ink2_mxfp8_gu_mma, true)
INK2_MMA_MXFP8(tf_ink2_mxfp8_down_mma, false)

inline uint ink2_simd_match(uint raw, bool valid, uint nbits) {
  const uint vmask = uint(ulong(simd_vote::vote_t(simd_ballot(valid))));
  uint match = valid ? vmask : 0u;
  for (uint b = 0u; b < nbits; ++b) {
    const uint bm = uint(ulong(simd_vote::vote_t(simd_ballot(valid && (((raw >> b) & 1u) != 0u)))));
    match &= ((raw >> b) & 1u) ? bm : ~bm;
  }
  return match;
}

// Parallel wide grouping for pairs > 256: 8 simdgroups slice pairs and prefix-sum into (counts, members).
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_group_wide(
    device const int* ids [[buffer(0)]], device uint* counts [[buffer(1)]],
    device uint* members [[buffer(2)]], constant Ink2MmaGroupArgs& a [[buffer(3)]],
    uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint sg_cnt[2048];
  const uint E = a.experts;
  for (uint i = t; i < 8u * E; i += 256u) sg_cnt[i] = 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint span = (a.pairs + 7u) / 8u;
  const uint lo = min(sg * span, a.pairs);
  const uint hi = min(lo + span, a.pairs);
  const uint below = (1u << lane) - 1u;
  const uint nbits = (E <= 128u) ? 7u : 8u;

  for (uint i0 = lo; i0 < hi; i0 += 32u) {
    const uint i = i0 + lane;
    const uint raw = (i < hi) ? uint(ids[i]) : E;
    const bool valid = (i < hi) && (raw < E);
    const uint match = ink2_simd_match(raw, valid, nbits);
    if (valid && (match & below) == 0u) sg_cnt[sg * E + raw] += popcount(match);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (t < E) {
    uint offset = 0u;
    for (uint s = 0u; s < 8u; ++s) {
      const uint c = sg_cnt[s * E + t];
      sg_cnt[s * E + t] = offset;
      offset += c;
    }
    counts[t] = min(offset, a.cap);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i0 = lo; i0 < hi; i0 += 32u) {
    const uint i = i0 + lane;
    const uint raw = (i < hi) ? uint(ids[i]) : E;
    const bool valid = (i < hi) && (raw < E);
    const uint match = ink2_simd_match(raw, valid, nbits);
    const uint rank = popcount(match & below);
    const uint dst_base = valid ? sg_cnt[sg * E + raw] : 0u;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (valid && (match & below) == 0u) sg_cnt[sg * E + raw] = dst_base + popcount(match);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
      const uint pos = dst_base + rank;
      if (pos < a.cap) members[raw * a.cap + pos] = i;
    }
  }
}

