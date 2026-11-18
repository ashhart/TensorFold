// DeepSeek-V4.1's DSpark Markov chain on the GPU over the whole vocabulary (drafts only), after keyed.metal.

#define TF_MK_NONE 0xffffffffu

struct TfMarkovArgs {
  uint vocab;  // logits a row and rows of the tables
  uint rows;   // an ask's rows (its drafts)
  uint step;   // the draft this launch picks
  uint parts;  // threadgroups an ask's search is cut into
  uint dim;    // the hidden size (the confidence head's first part)
};

// An ask's draws: greedy, or keyed (seed, the first draft's position, T) as the host races a row.
struct TfMarkovAsk {
  ulong seed;
  ulong pos0;
  float t;
  uint sampled;
};

inline float tf_mk_bf16(uint b) {
  return as_type<float>(b << 16);
}

// A lane's PER values of a table row (rank 32 * PER): eight from one word quad, else one at a time.
template <uint PER>
inline void tf_mk_row(device const ushort* t, ulong row, uint lane, thread float* out) {
  device const ushort* p = t + row * (32u * PER) + lane * PER;
  if (PER == 8u) {
    const uint4 q = *(device const uint4*)p;
    for (uint i = 0; i < 4u; ++i) {
      out[2 * i] = tf_mk_bf16(q[i] & 0xFFFFu);
      out[2 * i + 1] = tf_mk_bf16(q[i] >> 16);
    }
  } else {
    for (uint i = 0; i < PER; ++i) out[i] = tf_mk_bf16(p[i]);
  }
}

// The token a step's row follows: the ask's anchor (its block's first id) at step 0, else the draft before.
inline uint tf_mk_prev(constant TfMarkovArgs& a, device const uint* ids, device const uint* drafts, uint j) {
  return a.step == 0u ? ids[j * a.rows] : drafts[j * a.rows + a.step - 1u];
}

// A lane's N values of a table row from `col` (N bf16 in a row): word quads when N is a whole number of eight.
template <uint N>
inline void tf_mk_vals(device const ushort* t, ulong row, uint rank, uint col, thread float* out) {
  device const ushort* p = t + row * rank + col;
  if (N % 8u == 0u) {
    for (uint w = 0; w < N / 8u; ++w) {
      const uint4 q = ((device const uint4*)p)[w];
      for (uint i = 0; i < 4u; ++i) {
        out[8 * w + 2 * i] = tf_mk_bf16(q[i] & 0xFFFFu);
        out[8 * w + 2 * i + 1] = tf_mk_bf16(q[i] >> 16);
      }
    }
  } else {
    for (uint i = 0; i < N; ++i) out[i] = tf_mk_bf16(p[i]);
  }
}

// Part tg.x of ask tg.y: logits + markov_head[v] . markov_embed[prev], four rows a simdgroup, eight lanes a row.
template <uint PER>
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_markov_z(
    constant TfMarkovArgs& a [[buffer(0)]], device const float* logits [[buffer(1)]],
    device const ushort* w1 [[buffer(2)]], device const ushort* w2 [[buffer(3)]], device const uint* ids [[buffer(4)]],
    device const uint* drafts [[buffer(5)]], device float* part_s [[buffer(6)]], device float* part_z [[buffer(7)]],
    device uint* part_id [[buffer(8)]], device const TfMarkovAsk* asks [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  constexpr uint N = 4u * PER;  // a lane's ranks: the row's 32 * PER over eight lanes
  threadgroup float ts[32];
  threadgroup float tz[32];
  threadgroup uint ti[32];
  const uint j = tg.y;
  const TfMarkovAsk k = asks[j];
  const ulong base = tf_key_base(k.seed, k.pos0 + a.step);
  const uint prev = tf_mk_prev(a, ids, drafts, j);
  const uint sub = lane % 8u, quad = lane / 8u;
  float e[N];
  tf_mk_vals<N>(w1, prev, 32u * PER, sub * N, e);
  device const float* row = logits + ulong(j * a.rows + a.step) * a.vocab;
  float bs = -INFINITY, bz = -INFINITY;
  uint bi = TF_MK_NONE;
  const uint stride = 4u * a.parts * nsg;
  for (uint v0 = 4u * (tg.x * nsg + sg); v0 < a.vocab; v0 += stride) {
    const uint v = v0 + quad;
    const bool live = v < a.vocab;
    float h[N];
    tf_mk_vals<N>(w2, live ? v : v0, 32u * PER, sub * N, h);
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
  if (lane == 0u) {
    ts[sg] = bs;
    tz[sg] = bz;
    ti[sg] = bi;
  }
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
}

// Ask tg's step: the parts' best is the draft; its confidence sigmoid(hid + conf_w[dim ..] . markov_embed[prev]).
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

// Each row's hidden part of its confidence: conf_w[0 .. dim] . h (the last stage's collapsed streams, bf16).
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

#define TF_MK(PER) \
  template [[host_name("tf_dsv41_markov_z_" #PER)]] [[kernel]] decltype(tf_dsv41_markov_z<PER>) tf_dsv41_markov_z<PER>; \
  template [[host_name("tf_dsv41_markov_pick_" #PER)]] [[kernel]] decltype(tf_dsv41_markov_pick<PER>) tf_dsv41_markov_pick<PER>;
TF_MK(1)
TF_MK(2)
TF_MK(4)
TF_MK(8)
