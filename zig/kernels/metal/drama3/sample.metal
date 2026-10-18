// Drama 3 frame picks: argmax, drama3_sample.cuh's top_k <= 64 Philox race with RAS, and the frame's bookkeeping.
#include "d3_common.h"

constant constexpr uint CAP = 256;

// Greedy pick, ties to the lower index: cand_ids != 0 picks a main token (frame[0..2]), else a code (frame[0]).
struct ArgmaxArgs {
  uint count, has_ids;
  int sem_begin, cb;
};

[[max_total_threads_per_threadgroup(1024)]] kernel void d3_argmax(device const float* logits [[buffer(0)]], device const int* cand_ids [[buffer(1)]],
                      device int* frame [[buffer(2)]], constant ArgmaxArgs& a [[buffer(3)]],
                      uint t [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  float bv = -INFINITY;
  int bi = 0x7FFFFFFF;
  for (uint i = t; i < a.count; i += nt) {
    const float v = logits[i];
    if (v > bv || (v == bv && int(i) < bi)) {
      bv = v;
      bi = int(i);
    }
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ov = simd_shuffle_xor(bv, o);
    const int oi = simd_shuffle_xor(bi, o);
    if (ov > bv || (ov == bv && oi < bi)) {
      bv = ov;
      bi = oi;
    }
  }
  threadgroup float sv[32];
  threadgroup int si[32];
  if (lane == 0) {
    sv[sg] = bv;
    si[sg] = bi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    for (uint k = 1; k < nt / 32; ++k)
      if (sv[k] > bv || (sv[k] == bv && si[k] < bi)) {
        bv = sv[k];
        bi = si[k];
      }
    if (a.has_ids) {
      const int main_tok = cand_ids[bi];
      frame[0] = main_tok;
      frame[1] = clamp(main_tok - a.sem_begin, 0, a.cb - 1);
    } else {
      frame[0] = bi;
    }
  }
}

inline uint philox_first(ulong seed, uint c0, uint c1, uint c2, uint c3) {
  uint k0 = uint(seed & 0xFFFFFFFFul), k1 = uint(seed >> 32);
  for (int r = 0; r < 10; ++r) {
    const uint _c0 = c0, _c2 = c2;
    c0 = mulhi(0xCD9E8D57u, _c2) ^ c1 ^ k0;
    c2 = mulhi(0xD2511F53u, _c0) ^ c3 ^ k1;
    c1 = 0xCD9E8D57u * _c2;
    c3 = 0xD2511F53u * _c0;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return c0;
}

inline float exp_noise(ulong seed, ulong step, int row, int stream, int index) {
  const ulong key = seed ^ ((step >> 32) << 32);
  const uint bits = philox_first(key, uint(index), uint(row), uint(stream), uint(step & 0xFFFFFFFFul));
  const float u = (float(bits >> 9) + 0.5f) * (1.0f / 8388608.0f);
  return -precise::log(u);
}

// Order-preserving key of a score with its index (lower index wins ties).
inline uint ord32(float s) {
  const uint b = as_type<uint>(s == 0.0f ? 0.0f : s);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
inline ulong score_key(float s, int idx) { return (ulong(ord32(s)) << 32) | ulong(0xFFFFFFFFu - uint(idx)); }
inline float key_score(ulong k) {
  uint b = uint(k >> 32);
  b = (b & 0x80000000u) ? (b & 0x7FFFFFFFu) : ~b;
  return as_type<float>(b);
}
inline int key_index(ulong k) { return int(0xFFFFFFFFu - uint(k & 0xFFFFFFFFul)); }

// Bitonic sort of CAP keys descending; nt >= CAP threads.
inline void sort_desc(threadgroup ulong* keys, uint t) {
  for (uint k = 2; k <= CAP; k <<= 1) {
    for (uint j = k >> 1; j > 0; j >>= 1) {
      if (t < CAP) {
        const uint ixj = t ^ j;
        if (ixj > t) {
          const ulong x = keys[t], y = keys[ixj];
          const bool desc = (t & k) == 0;
          if (desc ? (x < y) : (x > y)) {
            keys[t] = y;
            keys[ixj] = x;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

// The `need` largest of count logits plus boundary ties (at most CAP) into cand, sorted descending; returns the count.
inline int select_top(device const float* logits, uint count, int need, threadgroup ulong* cand,
                      threadgroup atomic_int* hist, threadgroup uint* st, threadgroup atomic_int* sn, uint t, uint nt) {
  // st: prefix, mask, above, rem, done
  if (t == 0) {
    st[0] = 0u;
    st[1] = 0u;
    st[2] = 0u;
    st[3] = uint(need);
    st[4] = 0u;
    atomic_store_explicit(sn, 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (uint i = t; i < 256; i += nt) atomic_store_explicit(&hist[i], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint prefix = st[0], mask = st[1];
    for (uint i = t; i < count; i += nt) {
      const uint u = ord32(logits[i]);
      if ((u & mask) == prefix) atomic_fetch_add_explicit(&hist[(u >> shift) & 255u], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      int acc = 0, b = 255;
      const int rem = int(st[3]);
      for (; b > 0; --b) {
        const int hb = atomic_load_explicit(&hist[b], memory_order_relaxed);
        if (acc + hb >= rem) break;
        acc += hb;
      }
      const int hb = atomic_load_explicit(&hist[b], memory_order_relaxed);
      if (int(st[2]) + acc + hb <= int(CAP) || shift == 0) {
        st[0] = prefix | (uint(b) << shift);
        st[1] = 0u;
        st[4] = 1u;
      } else {
        st[0] = prefix | (uint(b) << shift);
        st[1] = mask | (255u << shift);
        st[2] = st[2] + uint(acc);
        st[3] = uint(rem - acc);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (st[4]) break;
  }
  const uint thr = st[0];
  for (uint i = t; i < CAP; i += nt) cand[i] = 0ul;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = t; i < count; i += nt) {
    const float v = logits[i];
    if (ord32(v) >= thr) {
      const int at = atomic_fetch_add_explicit(sn, 1, memory_order_relaxed);
      if (at < int(CAP)) cand[at] = score_key(v, int(i));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sort_desc(cand, t);
  return min(atomic_load_explicit(sn, memory_order_relaxed), int(CAP));
}

// Sum over all count logits of exp(l / t - mx), the full softmax's denominator.
inline float softmax_sum(device const float* logits, uint count, float temp, float mx, threadgroup float* scratch,
                         uint t, uint nt, uint sg, uint lane) {
  float s = 0.0f;
  for (uint i = t; i < count; i += nt) s += precise::exp(logits[i] / temp - mx);
  return group_sum(s, scratch, sg, lane, nt / 32);
}

// One draw among sorted candidates: kept = rank 0, or rank < top_k with cumulative p <= top_p; race w / Exp(1).
inline int draw_top(threadgroup const ulong* cand, int n, float temperature, int top_k, float top_p, float sum,
                    ulong seed, ulong step, int stream, threadgroup float* wsum, threadgroup float* bbest,
                    threadgroup int* brank, threadgroup int* chosen, uint i, uint nt, uint w, uint lane) {
  const float temp = max(temperature, 1e-5f);
  const float mx = key_score(cand[0]) / temp;
  const float wv = (int(i) < n) ? precise::exp(key_score(cand[i]) / temp - mx) : 0.0f;
  const float incl = simd_prefix_inclusive_sum(wv);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 31) wsum[w] = incl;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float before = 0.0f;
  for (uint k = 0; k < w && k < CAP / 32; ++k) before += wsum[k];
  const float cum = (before + incl) / sum;
  float best = -1.0f;
  int rank = 0x7FFFFFFF;
  if (int(i) < n && (i == 0 || (int(i) < top_k && cum <= top_p))) {
    best = wv / exp_noise(seed, step, 0, stream, key_index(cand[i]));
    rank = int(i);
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ob = simd_shuffle_xor(best, o);
    const int orr = simd_shuffle_xor(rank, o);
    if (ob > best || (ob == best && orr < rank)) {
      best = ob;
      rank = orr;
    }
  }
  if (lane == 0) {
    bbest[w] = best;
    brank[w] = rank;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (i == 0) {
    float b = bbest[0];
    int r = brank[0];
    for (uint k = 1; k < nt / 32; ++k)
      if (bbest[k] > b || (bbest[k] == b && brank[k] < r)) {
        b = bbest[k];
        r = brank[k];
      }
    *chosen = key_index(cand[r]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return *chosen;
}

struct SampleArgs {
  D3Sampling sp;
  uint count;
  int stream, sem_begin, cb;
};

#define D3_SAMPLER_SCRATCH                    \
  threadgroup ulong cand[CAP];                \
  threadgroup atomic_int hist[256];           \
  threadgroup uint st[8];                     \
  threadgroup atomic_int sn;                  \
  threadgroup float scratch[32], wsum[32], bbest[32]; \
  threadgroup int brank[32], chosen

// Main token and code 0: the request's draw (stream 0), or RAS's (stream 1) when it repeats in the 16-token window.
[[max_total_threads_per_threadgroup(256)]] kernel void d3_sample_main(device const float* logits [[buffer(0)]], device const int* cand_ids [[buffer(1)]],
                           device const int* window [[buffer(2)]], device const ulong* step_ptr [[buffer(3)]],
                           device int* frame [[buffer(4)]], constant SampleArgs& a [[buffer(5)]],
                           uint t [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  D3_SAMPLER_SCRATCH;
  const int n = select_top(logits, a.count, max(a.sp.top_k, 30), cand, hist, st, &sn, t, nt);
  const float temp = max(a.sp.temperature, 1e-5f);
  const float top = key_score(cand[0]);
  const float zt = softmax_sum(logits, a.count, temp, top / temp, scratch, t, nt, sg, lane);
  const float z1 = softmax_sum(logits, a.count, 1.0f, top, scratch, t, nt, sg, lane);
  const ulong step = *step_ptr;
  const int first = draw_top(cand, n, a.sp.temperature, a.sp.top_k, a.sp.top_p, zt, a.sp.seed, step, 0, wsum, bbest,
                             brank, &chosen, t, nt, sg, lane);
  const int ras = draw_top(cand, n, 1.0f, 30, 0.95f, z1, a.sp.seed, step, 1, wsum, bbest, brank, &chosen, t, nt, sg,
                           lane);
  if (t == 0) {
    const int ft = cand_ids[first], rt = cand_ids[ras];
    bool repeat = false;
    for (int i = 0; i < 16; ++i) repeat |= window[i] == ft;
    const bool sem = ft >= a.sem_begin && ft < a.sem_begin + a.cb;
    const int main_tok = (repeat && sem) ? rt : ft;
    frame[0] = main_tok;
    frame[1] = clamp(main_tok - a.sem_begin, 0, a.cb - 1);
  }
}

// A codebook head's code from its fp32 logits, on stream 2 + head.
[[max_total_threads_per_threadgroup(256)]] kernel void d3_sample_code(device const float* logits [[buffer(0)]], device const ulong* step_ptr [[buffer(1)]],
                           device int* out [[buffer(2)]], constant SampleArgs& a [[buffer(3)]],
                           uint t [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  D3_SAMPLER_SCRATCH;
  const int n = select_top(logits, a.count, a.sp.top_k, cand, hist, st, &sn, t, nt);
  const float temp = max(a.sp.temperature, 1e-5f);
  const float z = softmax_sum(logits, a.count, temp, key_score(cand[0]) / temp, scratch, t, nt, sg, lane);
  const int c = draw_top(cand, n, a.sp.temperature, a.sp.top_k, a.sp.top_p, z, a.sp.seed, *step_ptr, a.stream, wsum,
                         bbest, brank, &chosen, t, nt, sg, lane);
  if (t == 0) *out = c;
}

// After a frame: append it to frames_out, make it the next input, shift the RAS window, advance pos, step and count.
[[max_total_threads_per_threadgroup(32)]] kernel void d3_finish_frame(device const int* frame [[buffer(0)]], device int* frames_out [[buffer(1)]],
                            device int* next_input [[buffer(2)]], device int* window [[buffer(3)]],
                            device int* pos [[buffer(4)]], device ulong* step [[buffer(5)]], device int* count [[buffer(6)]],
                            uint t [[thread_index_in_threadgroup]]) {
  if (t != 0) return;
  const int n = *count;
  for (int i = 0; i < 11; ++i) {
    frames_out[n * 11 + i] = frame[i];
    next_input[i] = frame[i];
  }
  for (int i = 0; i < 15; ++i) window[i] = window[i + 1];
  window[15] = frame[0];
  *pos += 1;
  *step += 1;
  *count = n + 1;
}

// The next backbone row's input once main, code 0 and code 1 are drawn: bb_input = frame[0..3], *pos_bb = *pos + 1.
[[max_total_threads_per_threadgroup(32)]] kernel void d3_next_input(device const int* frame [[buffer(0)]], device int* bb_input [[buffer(1)]],
                          device const int* pos [[buffer(2)]], device int* pos_bb [[buffer(3)]],
                          uint t [[thread_index_in_threadgroup]]) {
  if (t < 11) bb_input[t] = t < 3 ? frame[t] : 0;
  if (t == 0) *pos_bb = *pos + 1;
}

// Copy n ints (small device-side moves inside a frame).
[[max_total_threads_per_threadgroup(64)]] kernel void d3_copy_i32(device const int* src [[buffer(0)]], device int* dst [[buffer(1)]],
                        constant uint& n [[buffer(2)]], uint t [[thread_index_in_threadgroup]],
                        uint nt [[threads_per_threadgroup]]) {
  for (uint i = t; i < n; i += nt) dst[i] = src[i];
}
