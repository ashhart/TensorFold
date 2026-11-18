// DeepSeek-V4.1 drafts behind a verify window: its greedy picks, the drafts they accept, the drafter's plan.
#include <metal_stdlib>
using namespace metal;

#define TF_TG 1024u
#define TF_NONE 0xffffffffu

struct TfEarlyPickArgs {
  uint vocab; // logits a row (the row stride)
};

// The better of two (value, id) pairs: the larger value, ties to the lower id (no NaN reaches here).
inline void tf_early_better(thread float& v, thread uint& i, float ov, uint oi) {
  if (ov > v || (ov == v && oi < i)) {
    v = ov;
    i = oi;
  }
}

// The host's greedy pick of each row: the first maximum of its values that are not NaN, 0 when value 0 is NaN.
[[max_total_threads_per_threadgroup(TF_TG)]] kernel void tf_dsv41_early_pick(
    constant TfEarlyPickArgs& args [[buffer(0)]],
    device const float* logits [[buffer(1)]],
    device uint* picks [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part_v[32];
  threadgroup uint part_i[32];
  device const float* x = logits + ulong(row) * args.vocab;
  float bv = -INFINITY;
  uint bi = TF_NONE;
  for (uint i = tid; i < args.vocab; i += TF_TG) {
    const float v = x[i];
    if (!isnan(v)) tf_early_better(bv, bi, v, i);
  }
  for (ushort off = 16; off > 0; off >>= 1) tf_early_better(bv, bi, simd_shuffle_down(bv, off), simd_shuffle_down(bi, off));
  if (lane == 0u) {
    part_v[simd] = bv;
    part_i[simd] = bi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd != 0u) return;
  bv = lane < TF_TG / 32u ? part_v[lane] : -INFINITY;
  bi = lane < TF_TG / 32u ? part_i[lane] : TF_NONE;
  for (ushort off = 16; off > 0; off >>= 1) tf_early_better(bv, bi, simd_shuffle_down(bv, off), simd_shuffle_down(bi, off));
  if (lane == 0u) picks[row] = (isnan(x[0]) || bi == TF_NONE) ? 0u : bi;
}

struct TfEarlyAcceptArgs {
  uint rows;  // the window's rows: the pending token's, then the drafts'
  uint ints;  // a plan's ints (each accepted count's plan in `plans`)
  uint skew;  // tests: 1 swaps an accepted count of 0 and 1, so the chain parts from the host's commit
};

// The drafts the picks accept (the host's rule), each count's plan copied in, the bonus into the block and the result.
[[max_total_threads_per_threadgroup(TF_TG)]] kernel void tf_dsv41_early_accept(
    constant TfEarlyAcceptArgs& args [[buffer(0)]],
    device const uint* tokens [[buffer(1)]],
    device const uint* picks [[buffer(2)]],
    device const int* plans [[buffer(3)]],
    device int* plan [[buffer(4)]],
    device uint* ids [[buffer(5)]],
    device uint* result [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]]) {
  uint acc = 0u;
  while (acc + 1u < args.rows && tokens[acc + 1u] == picks[acc]) acc++;
  if (args.skew != 0u && args.rows > 1u && acc < 2u) acc = 1u - acc;
  device const int* from = plans + ulong(acc) * args.ints;
  for (uint i = tid; i < args.ints; i += TF_TG) plan[i] = from[i];
  if (tid == 0u) {
    ids[0] = picks[acc];
    result[0] = acc;
    result[1] = picks[acc];
  }
}
