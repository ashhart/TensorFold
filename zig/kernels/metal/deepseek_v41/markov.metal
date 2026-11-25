// DeepSeek-V4.1 DSpark GPU Markov chain (full-vocab shared-w2 multi-ask mode + shared helpers).

#define TF_MK_NONE 0xffffffffu
#define TF_MK_SLICES 16u

struct TfMarkovArgs {
  uint vocab;
  uint rows;
  uint step;
  uint parts;
  uint dim;
  uint asks;
  uint topk;
};

// An ask's draws: pos0 stays at byte offset 8 so early_accept can patch it directly.
struct TfMarkovAsk {
  ulong seed;
  ulong pos0;
  float t;
  uint sampled;
  uint top_k;
  float top_p;
  float min_p;
  uint pad;
};

inline float tf_mk_bf16(uint b) {
  return as_type<float>(b << 16);
}

// Order-preserving uint key for IEEE-754 float: NaN -> 0, -inf -> 0x007FFFFF, +/-0 -> 0x80000000.
inline uint tf_mk_fkey(float v) {
  uint u = as_type<uint>(v);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return 0u;
  if (u == 0x80000000u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Matches host.dspark.rankedBefore: higher value wins, ties go to lower id, TF_MK_NONE always loses.
inline bool tf_mk_better(float va, uint ia, float vb, uint ib) {
  if (ia == TF_MK_NONE) return false;
  if (ib == TF_MK_NONE) return true;
  const uint ka = tf_mk_fkey(va);
  const uint kb = tf_mk_fkey(vb);
  return (ka > kb) || (ka == kb && ia < ib);
}

template <uint PER>
inline void tf_mk_row(device const ushort* t, ulong row, uint lane, thread float* out) {
  device const ushort* p = t + row * (32u * PER) + lane * PER;
  if (PER == 8u) {
    const uint4 q = *(device const uint4*)p;
    for (uint i = 0; i < 4u; ++i) {
      out[2u * i] = tf_mk_bf16(q[i] & 0xFFFFu);
      out[2u * i + 1u] = tf_mk_bf16(q[i] >> 16);
    }
  } else {
    for (uint i = 0; i < PER; ++i) out[i] = tf_mk_bf16(p[i]);
  }
}

inline uint tf_mk_prev(constant TfMarkovArgs& a, device const uint* ids, device const uint* drafts, uint j) {
  return a.step == 0u ? ids[j * a.rows] : drafts[j * a.rows + a.step - 1u];
}

template <uint N>
inline void tf_mk_vals(device const ushort* t, ulong row, uint rank, uint col, thread float* out) {
  device const ushort* p = t + row * rank + col;
  if (N % 8u == 0u) {
    for (uint w = 0; w < N / 8u; ++w) {
      const uint4 q = ((device const uint4*)p)[w];
      for (uint i = 0; i < 4u; ++i) {
        out[8u * w + 2u * i] = tf_mk_bf16(q[i] & 0xFFFFu);
        out[8u * w + 2u * i + 1u] = tf_mk_bf16(q[i] >> 16);
      }
    }
  } else {
    for (uint i = 0; i < N; ++i) out[i] = tf_mk_bf16(p[i]);
  }
}

// Full-vocab GPU Markov mode (topk == 0): shares each w2[v] load across up to 4 asks.
template <uint PER>
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_markov_z(
    constant TfMarkovArgs& a [[buffer(0)]], device const float* logits [[buffer(1)]],
    device const ushort* w1 [[buffer(2)]], device const ushort* w2 [[buffer(3)]], device const uint* ids [[buffer(4)]],
    device const uint* drafts [[buffer(5)]], device float* part_s [[buffer(6)]], device float* part_z [[buffer(7)]],
    device uint* part_id [[buffer(8)]], device const TfMarkovAsk* asks [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]) {
  constexpr uint N = 4u * PER;
  constexpr uint RANK = 32u * PER;
  threadgroup float ts[32];
  threadgroup float tz[32];
  threadgroup uint ti[32];
  threadgroup float te[4][RANK];

  const uint j0 = tg.y * 4u;
  const uint nj = min(4u, a.asks - j0);
  const uint sub = lane % 8u, quad = lane / 8u;
  const uint stride = 4u * a.parts * nsg;

  if (nj == 1u) {
    const uint j = j0;
    const TfMarkovAsk k = asks[j];
    const ulong base = tf_key_base(k.seed, k.pos0 + a.step);
    const uint prev = tf_mk_prev(a, ids, drafts, j);
    float e[N];
    tf_mk_vals<N>(w1, prev, RANK, sub * N, e);
    device const float* row = logits + ulong(j * a.rows + a.step) * a.vocab;
    float bs = -INFINITY, bz = -INFINITY;
    uint bi = TF_MK_NONE;
    for (uint v0 = 4u * (tg.x * nsg + sg); v0 < a.vocab; v0 += stride) {
      const uint v = v0 + quad;
      const bool live = v < a.vocab;
      float h[N];
      tf_mk_vals<N>(w2, live ? v : v0, RANK, sub * N, h);
      float s = 0.0f;
      for (uint i = 0; i < N; ++i) s = fma(e[i], h[i], s);
      for (ushort m = 1; m < 8; m <<= 1) s += simd_shuffle_xor(s, m);
      if (sub == 0u && live) {
        const float z = row[v] + s;
        tf_key_beats(bs, bz, bi, k.sampled != 0u ? z / k.t + tf_key_gumbel(tf_key_mix(base ^ ulong(v))) : z, z, v);
      }
    }
    for (ushort off = 16; off > 0; off >>= 1)
      tf_key_beats(bs, bz, bi, simd_shuffle_down(bs, off), simd_shuffle_down(bz, off), simd_shuffle_down(bi, off));
    if (lane == 0u) { ts[sg] = bs; tz[sg] = bz; ti[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0u) return;
    bs = lane < nsg ? ts[lane] : -INFINITY;
    bz = lane < nsg ? tz[lane] : -INFINITY;
    bi = lane < nsg ? ti[lane] : TF_MK_NONE;
    for (ushort off = 16; off > 0; off >>= 1)
      tf_key_beats(bs, bz, bi, simd_shuffle_down(bs, off), simd_shuffle_down(bz, off), simd_shuffle_down(bi, off));
    if (lane == 0u) {
      part_s[j * a.parts + tg.x] = bs;
      part_z[j * a.parts + tg.x] = bz;
      part_id[j * a.parts + tg.x] = bi;
    }
    return;
  }

  TfMarkovAsk ks[4];
  ulong bases[4];
  for (uint q = 0; q < nj; ++q) {
    const uint j = j0 + q;
    ks[q] = asks[j];
    bases[q] = tf_key_base(ks[q].seed, ks[q].pos0 + a.step);
    const uint prev = tf_mk_prev(a, ids, drafts, j);
    if (tid < RANK) te[q][tid] = tf_mk_bf16(w1[ulong(prev) * RANK + tid]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float bs[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  float bz[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  uint bi[4] = {TF_MK_NONE, TF_MK_NONE, TF_MK_NONE, TF_MK_NONE};
  for (uint v0 = 4u * (tg.x * nsg + sg); v0 < a.vocab; v0 += stride) {
    const uint v = v0 + quad;
    const bool live = v < a.vocab;
    float h[N];
    tf_mk_vals<N>(w2, live ? v : v0, RANK, sub * N, h);
    for (uint q = 0; q < nj; ++q) {
      float s = 0.0f;
      for (uint i = 0; i < N; ++i) s = fma(te[q][sub * N + i], h[i], s);
      for (ushort m = 1; m < 8; m <<= 1) s += simd_shuffle_xor(s, m);
      if (sub == 0u && live) {
        const float z = logits[ulong((j0 + q) * a.rows + a.step) * a.vocab + v] + s;
        tf_key_beats(bs[q], bz[q], bi[q],
                     ks[q].sampled != 0u ? z / ks[q].t + tf_key_gumbel(tf_key_mix(bases[q] ^ ulong(v))) : z, z, v);
      }
    }
  }
  for (uint q = 0; q < nj; ++q) {
    float s0 = bs[q], z0 = bz[q];
    uint i0 = bi[q];
    for (ushort off = 16; off > 0; off >>= 1)
      tf_key_beats(s0, z0, i0, simd_shuffle_down(s0, off), simd_shuffle_down(z0, off), simd_shuffle_down(i0, off));
    if (lane == 0u) { ts[sg] = s0; tz[sg] = z0; ti[sg] = i0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      s0 = lane < nsg ? ts[lane] : -INFINITY;
      z0 = lane < nsg ? tz[lane] : -INFINITY;
      i0 = lane < nsg ? ti[lane] : TF_MK_NONE;
      for (ushort off = 16; off > 0; off >>= 1)
        tf_key_beats(s0, z0, i0, simd_shuffle_down(s0, off), simd_shuffle_down(z0, off), simd_shuffle_down(i0, off));
      if (lane == 0u) {
        const uint j = j0 + q;
        part_s[j * a.parts + tg.x] = s0;
        part_z[j * a.parts + tg.x] = z0;
        part_id[j * a.parts + tg.x] = i0;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <uint PER>
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_markov_pick(
    constant TfMarkovArgs& a [[buffer(0)]], device const float* part_s [[buffer(1)]], device const float* part_z [[buffer(2)]],
    device const uint* part_id [[buffer(3)]], device const ushort* w1 [[buffer(4)]], device const float* cw [[buffer(5)]],
    device const float* hid [[buffer(6)]], device const uint* ids [[buffer(7)]], device uint* drafts [[buffer(8)]],
    device float* conf [[buffer(9)]], uint j [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  float bs = -INFINITY, bz = -INFINITY;
  uint bi = TF_MK_NONE;
  for (uint p = lane; p < a.parts; p += 32u) {
    const uint at = j * a.parts + p;
    tf_key_beats(bs, bz, bi, part_s[at], part_z[at], part_id[at]);
  }
  for (ushort off = 16; off > 0; off >>= 1)
    tf_key_beats(bs, bz, bi, simd_shuffle_down(bs, off), simd_shuffle_down(bz, off), simd_shuffle_down(bi, off));
  const uint prev = tf_mk_prev(a, ids, drafts, j);
  float e[PER];
  tf_mk_row<PER>(w1, prev, lane, e);
  float s = 0.0f;
  for (uint i = 0; i < PER; ++i) s = fma(e[i], cw[a.dim + PER * lane + i], s);
  s = simd_sum(s);
  if (lane == 0u) {
    drafts[j * a.rows + a.step] = bi == TF_MK_NONE ? prev : bi;
    conf[j * a.rows + a.step] = 1.0f / (1.0f + precise::exp(-(hid[j * a.rows + a.step] + s)));
  }
}

[[max_total_threads_per_threadgroup(1024)]] kernel void tf_dsv41_markov_hid(
    constant TfMarkovArgs& a [[buffer(0)]], device const ushort* hf [[buffer(1)]], device const float* cw [[buffer(2)]],
    device float* hid [[buffer(3)]], uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]], uint nt [[threads_per_threadgroup]]) {
  threadgroup float ts[32];
  float s = 0.0f;
  for (uint t = tid; t < a.dim; t += nt) s = fma(cw[t], tf_mk_bf16(hf[ulong(r) * a.dim + t]), s);
  s = simd_sum(s);
  if (lane == 0u) ts[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0u) return;
  s = simd_sum(lane < nsg ? ts[lane] : 0.0f);
  if (lane == 0u) hid[r] = s;
}
