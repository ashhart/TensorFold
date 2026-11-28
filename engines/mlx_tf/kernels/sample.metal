// Clean-room multi-core two-stage chunked batched argmax over 248,320-vocab bf16 logits.
// Supports rows = 1..128 in parallel. Deterministic tie-breaking: largest finite value, lowest index.
#include <metal_stdlib>
using namespace metal;

struct ArgmaxParams {
  uint vocab;
  uint chunks;
  uint rows;
};

inline bool better(float v, uint idx, float cur_v, uint cur_idx) {
  return (v > cur_v) || (v == cur_v && idx < cur_idx);
}

// Stage 1: Grid threadgroups (chunks, rows) of 256 threads; each thread scans 4 logits (1024/chunk).
kernel void argmax_stage1(const device bfloat* logits [[buffer(0)]],
                          device float* chunk_val [[buffer(1)]],
                          device uint* chunk_idx [[buffer(2)]],
                          constant ArgmaxParams& p [[buffer(3)]],
                          uint2 grp [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
  const uint chunk = grp.x;
  const uint row_idx = grp.y;
  if (chunk >= p.chunks || row_idx >= p.rows) return;

  const device bfloat* row = logits + (size_t)row_idx * p.vocab;
  const uint base = chunk * 1024 + tid;

  float best_v = -INFINITY;
  uint best_i = 0xFFFFFFFFu;

  for (uint k = 0; k < 4; k++) {
    const uint idx = base + k * 256;
    if (idx < p.vocab) {
      const float v = float(row[idx]);
      if (isfinite(v) && better(v, idx, best_v, best_i)) {
        best_v = v;
        best_i = idx;
      }
    }
  }

  for (ushort offset = 16; offset > 0; offset >>= 1) {
    const float ov = simd_shuffle_down(best_v, offset);
    const uint oi = simd_shuffle_down(best_i, offset);
    if (better(ov, oi, best_v, best_i)) {
      best_v = ov;
      best_i = oi;
    }
  }

  threadgroup float sg_v[8];
  threadgroup uint sg_i[8];
  if (lane == 0) {
    sg_v[sg] = best_v;
    sg_i[sg] = best_i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0) {
    float v = (lane < 8) ? sg_v[lane] : -INFINITY;
    uint idx = (lane < 8) ? sg_i[lane] : 0xFFFFFFFFu;
    for (ushort offset = 4; offset > 0; offset >>= 1) {
      const float ov = simd_shuffle_down(v, offset);
      const uint oi = simd_shuffle_down(idx, offset);
      if (better(ov, oi, v, idx)) {
        v = ov;
        idx = oi;
      }
    }
    if (lane == 0) {
      const size_t out_at = (size_t)row_idx * p.chunks + chunk;
      chunk_val[out_at] = v;
      chunk_idx[out_at] = idx;
    }
  }
}

// Stage 2: Grid threadgroups (1, rows) of 256 threads reducing across all chunks for each row.
kernel void argmax_stage2(const device float* chunk_val [[buffer(0)]],
                          const device uint* chunk_idx [[buffer(1)]],
                          device uint* out_token [[buffer(2)]],
                          device float* out_val [[buffer(3)]],
                          constant ArgmaxParams& p [[buffer(4)]],
                          uint2 grp [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
  const uint row_idx = grp.y;
  if (row_idx >= p.rows) return;

  const size_t base = (size_t)row_idx * p.chunks;
  float best_v = -INFINITY;
  uint best_i = 0xFFFFFFFFu;

  for (uint c = tid; c < p.chunks; c += 256) {
    const float v = chunk_val[base + c];
    const uint idx = chunk_idx[base + c];
    if (better(v, idx, best_v, best_i)) {
      best_v = v;
      best_i = idx;
    }
  }

  for (ushort offset = 16; offset > 0; offset >>= 1) {
    const float ov = simd_shuffle_down(best_v, offset);
    const uint oi = simd_shuffle_down(best_i, offset);
    if (better(ov, oi, best_v, best_i)) {
      best_v = ov;
      best_i = oi;
    }
  }

  threadgroup float sg_v[8];
  threadgroup uint sg_i[8];
  if (lane == 0) {
    sg_v[sg] = best_v;
    sg_i[sg] = best_i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0) {
    float v = (lane < 8) ? sg_v[lane] : -INFINITY;
    uint idx = (lane < 8) ? sg_i[lane] : 0xFFFFFFFFu;
    for (ushort offset = 4; offset > 0; offset >>= 1) {
      const float ov = simd_shuffle_down(v, offset);
      const uint oi = simd_shuffle_down(idx, offset);
      if (better(ov, oi, v, idx)) {
        v = ov;
        idx = oi;
      }
    }
    if (lane == 0) {
      out_token[row_idx] = idx;
      out_val[row_idx] = v;
    }
  }
}

struct CopyParams {
  uint count; // number of uint4 (16-byte) elements
};

// Fast vectorized 16-byte buffer copy (used for tap capture & KV compaction).
kernel void copy_u4(const device uint4* src [[buffer(0)]],
                    device uint4* dst [[buffer(1)]],
                    constant CopyParams& p [[buffer(2)]],
                    uint tid [[thread_position_in_grid]]) {
  if (tid < p.count) dst[tid] = src[tid];
}
