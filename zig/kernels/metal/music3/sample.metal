// music3 AR sampling: the CFG top-k draw and the frame's bookkeeping, all on the device between frames.
#include "m3_common.h"

inline uint ord(float v) {  // float -> uint with the same order
  const uint u = as_type<uint>(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

#define SAMPLE_THREADS 1024
#define SAMPLE_CAP 1024

struct SampleArgs {
  ulong seed;
  uint V;
  float cfg;
  int top_k, rank_cond, slot, has_code, has_tokens, pad;
};

inline float guided(device const float* lg, uint V, uint i, float cfg) { return lg[V + i] + (lg[i] - lg[V + i]) * cfg; }

inline uint key_of(device const float* lg, uint V, uint i, float cfg, int rank_cond) {
  return ord(rank_cond ? lg[i] : guided(lg, V, i, cfg));
}

// A CFG top-k draw of logits [2, V] (cond, uncond) by inverse CDF in index order on a Philox uniform, as on CUDA.
[[max_total_threads_per_threadgroup(1024)]] kernel void m3_sample(device const float* lg [[buffer(0)]], device const ulong* step [[buffer(1)]],
                      device int* pick [[buffer(2)]], device int* code [[buffer(3)]], device int* tokens [[buffer(4)]],
                      device const int* count [[buffer(5)]], constant SampleArgs& a [[buffer(6)]],
                      uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint sel[2];
  threadgroup uint cnt_sh[SAMPLE_THREADS / 32];
  threadgroup int cidx[SAMPLE_CAP];
  threadgroup float cval[SAMPLE_CAP];
  threadgroup int ncand;
  const uint V = a.V;
  const uint per = (V + SAMPLE_THREADS - 1) / SAMPLE_THREADS;
  const uint lo = t * per, hi = min(V, lo + per);
  // radix select of the top_k-th largest key, 8 bits at a time from the top
  uint prefix = 0, mask = 0;
  int want = max(1, min(a.top_k, int(V)));
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lo; i < hi; ++i) {
      const uint k = key_of(lg, V, i, a.cfg, a.rank_cond);
      if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      int acc = 0, dgt = 0;
      for (int dd = 255; dd >= 0; --dd) {
        const int hv = int(atomic_load_explicit(&hist[dd], memory_order_relaxed));
        if (acc + hv >= want) {
          dgt = dd;
          break;
        }
        acc += hv;
      }
      sel[0] = uint(dgt);
      sel[1] = uint(want - acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= sel[0] << shift;
    mask |= 255u << shift;
    want = int(sel[1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint thr = prefix;  // keys >= thr are the candidates
  uint mine = 0;
  for (uint i = lo; i < hi; ++i) mine += key_of(lg, V, i, a.cfg, a.rank_cond) >= thr;
  const uint excl = simd_prefix_exclusive_sum(mine);
  if (lane == 31) cnt_sh[sg] = excl + mine;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    uint s = 0;
    for (uint w = 0; w < SAMPLE_THREADS / 32; ++w) {
      const uint v = cnt_sh[w];
      cnt_sh[w] = s;
      s += v;
    }
    ncand = int(min(s, uint(SAMPLE_CAP)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint at = cnt_sh[sg] + excl;
  for (uint i = lo; i < hi; ++i)
    if (key_of(lg, V, i, a.cfg, a.rank_cond) >= thr) {
      if (at < SAMPLE_CAP) {
        cidx[at] = int(i);
        cval[at] = guided(lg, V, i, a.cfg);
      }
      ++at;
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t != 0) return;
  int n = ncand;
  if (a.rank_cond && n > a.top_k) {
    // ComfyUI's second sample_topk: keep values with fewer than top_k strictly greater ones (the top_k with ties)
    float kth = INFINITY;
    for (int i = 0; i < n; ++i) {
      int greater = 0;
      for (int j = 0; j < n; ++j) greater += cval[j] > cval[i];
      if (greater < a.top_k) kth = min(kth, cval[i]);
    }
    int w = 0;
    for (int i = 0; i < n; ++i)
      if (cval[i] >= kth) {
        cidx[w] = cidx[i];
        cval[w] = cval[i];
        ++w;
      }
    n = w;
  }
  float m = -INFINITY;
  for (int i = 0; i < n; ++i) m = max(m, cval[i]);
  int chosen = cidx[0];
  if (a.top_k <= 1) {
    for (int i = 0; i < n; ++i)
      if (cval[i] == m) {
        chosen = cidx[i];
        break;
      }
  } else {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += precise::exp(cval[i] - m);
    const float u = philox_uniform(a.seed, *step, ulong(a.slot)) * sum;
    float run = 0.0f;
    chosen = cidx[n - 1];
    for (int i = 0; i < n; ++i) {
      run += precise::exp(cval[i] - m);
      if (u < run) {
        chosen = cidx[i];
        break;
      }
    }
  }
  *pick = chosen;
  if (a.has_code) *code = max(chosen - 1, 0);
  if (a.has_tokens) tokens[*count] = chosen;
}

// After a frame: the next position and step, one frame more.
[[max_total_threads_per_threadgroup(32)]] kernel void m3_frame_done(device int* pos [[buffer(0)]], device ulong* step [[buffer(1)]],
                          device int* count [[buffer(2)]], uint t [[thread_index_in_threadgroup]]) {
  if (t == 0) {
    *pos += 1;
    *step += 1;
    *count += 1;
  }
}

// A frame state's part: states[*count, part] = src (bf16 4096).
[[max_total_threads_per_threadgroup(256)]] kernel void m3_state_part(device const uint4* src [[buffer(0)]], device uint4* states [[buffer(1)]],
                          device const int* count [[buffer(2)]], constant uint& part [[buffer(3)]],
                          uint t [[thread_index_in_threadgroup]]) {
  const ulong base = ((ulong)*count * 8 + part) * 512;
  for (uint i = t; i < 512; i += 256) states[base + i] = src[i];
}

// A forced frame's codes (teacher forcing): codes[0..8) = forced[*count, 0..8), its token into tokens[*count].
[[max_total_threads_per_threadgroup(32)]] kernel void m3_force_codes(device const int* forced [[buffer(0)]], device const int* count [[buffer(1)]],
                           device int* codes [[buffer(2)]], device int* tokens [[buffer(3)]],
                           uint t [[thread_index_in_threadgroup]]) {
  const int c = *count;
  if (t < 8) codes[t] = forced[c * 8 + t];
  if (t == 0) tokens[c] = forced[c * 8] + 1;
}

// dst = src, n 32-bit words (in order with the dispatches around it).
[[max_total_threads_per_threadgroup(256)]] kernel void m3_copy(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],
                    constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) dst[i] = src[i];
}
