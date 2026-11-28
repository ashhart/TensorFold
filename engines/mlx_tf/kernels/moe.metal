// Clean-room Sparse MoE Metal 3.1+ kernels for Nemotron-H (128 experts, top-6, ReLU^2).
//
// Reads the exact same packed 1152-byte (NT=32, GS=64) qk tile layout per expert so both
// single-token decode and multi-token prefill execute 100% GPU-resident with zero CPU sync!
#include <metal_stdlib>
using namespace metal;

#define UNROLL _Pragma("clang loop unroll(full)")

constant constexpr int GS = 64;
constant constexpr int NT = 32;
constant constexpr int SK = 4;
constant constexpr int CODES = NT * GS / 2;   // 1024 bytes
constant constexpr int BLOCK = CODES + NT * 4; // 1152 bytes

struct GateParams {
  uint rows;
  uint hidden;
  uint n_experts;
};

// BF16 router logits: logits[row, e] = dot(X[row], W_gate[e]) in fp32.
// Grid: threadgroups ((n_experts + 3) / 4, rows, 1) of 128 threads (4 simdgroups = 4 experts).
kernel void moe_gate_logits(const device bfloat* X [[buffer(0)]],
                            const device bfloat* W [[buffer(1)]],
                            device float* logits [[buffer(2)]],
                            constant GateParams& p [[buffer(3)]],
                            uint2 grp [[threadgroup_position_in_grid]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
  const uint e = grp.x * 4 + sg;
  const uint row = grp.y;
  if (e >= p.n_experts || row >= p.rows) return;

  const uint k4 = p.hidden / 4;
  const device vec<bfloat, 4>* xp = (const device vec<bfloat, 4>*)(X + (size_t)row * p.hidden);
  const device vec<bfloat, 4>* wp = (const device vec<bfloat, 4>*)(W + (size_t)e * p.hidden);

  float4 acc = float4(0.0f);
  for (uint i = lane; i < k4; i += 32) {
    acc = fma(float4(xp[i]), float4(wp[i]), acc);
  }
  const float sum = simd_sum((acc[0] + acc[1]) + (acc[2] + acc[3]));
  if (lane == 0) {
    logits[(size_t)row * p.n_experts + e] = sum;
  }
}

struct TopkParams {
  uint rows;
  uint n_experts;
  uint top_k;
  uint norm_topk_prob;
  float routed_scaling_factor;
  uint use_softmax; // 0 = sigmoid (Nemotron-H / DeepSeek-V3), 1 = softmax (Qwen3 / Qwen3.5 / Mixtral)
  uint has_e_bias;  // 1 = add e_bias for selection (Nemotron-H / DeepSeek-V3)
  uint n_group;     // >1 = grouped top-k (DeepSeek-V3 / GLM)
  uint topk_group;
};

// Generalized top-k expert router (supports up to 512 experts, sigmoid or softmax, optional e_bias, optional DeepSeek grouped top-k).
// Grid: threadgroups (1, rows, 1) of 128 threads.
kernel void moe_topk_select(const device float* logits [[buffer(0)]],
                            const device float* e_bias [[buffer(1)]],
                            device uint* out_inds [[buffer(2)]],
                            device float* out_weights [[buffer(3)]],
                            constant TopkParams& p [[buffer(4)]],
                            uint2 grp [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]]) {
  const uint row = grp.y;
  if (row >= p.rows) return;

  threadgroup float orig_s[512];
  threadgroup float score_s[512];
  threadgroup float sg_v[4];
  threadgroup uint sg_i[4];

  const ushort sg = tid >> 5;
  const ushort lane = tid & 31;
  const device float* lrow = logits + (size_t)row * p.n_experts;

  if (p.use_softmax != 0) {
    float loc_max = -INFINITY;
    for (uint e = tid; e < p.n_experts; e += 128) loc_max = max(loc_max, lrow[e]);
    loc_max = simd_max(loc_max);
    if (lane == 0) sg_v[sg] = loc_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gmax = max(max(sg_v[0], sg_v[1]), max(sg_v[2], sg_v[3]));

    float loc_sum = 0.0f;
    for (uint e = tid; e < p.n_experts; e += 128) {
      const float ex = exp(lrow[e] - gmax);
      orig_s[e] = ex;
      loc_sum += ex;
    }
    loc_sum = simd_sum(loc_sum);
    if (lane == 0) sg_v[sg] = loc_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_sum = 1.0f / (sg_v[0] + sg_v[1] + sg_v[2] + sg_v[3] + 1e-20f);
    for (uint e = tid; e < p.n_experts; e += 128) {
      const float pr = orig_s[e] * inv_sum;
      orig_s[e] = pr;
      score_s[e] = (p.has_e_bias != 0) ? (pr + e_bias[e]) : pr;
    }
  } else {
    for (uint e = tid; e < p.n_experts; e += 128) {
      const float sig = 1.0f / (1.0f + exp(-lrow[e]));
      orig_s[e] = sig;
      score_s[e] = (p.has_e_bias != 0) ? (sig + e_bias[e]) : sig;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Optional DeepSeek-V3 / GLM grouped top-k masking
  if (p.n_group > 1 && p.topk_group > 0 && p.topk_group < p.n_group) {
    if (tid == 0) {
      const uint gsz = p.n_experts / p.n_group;
      float gscore[32];
      uint gkeep[32];
      for (uint g = 0; g < p.n_group; g++) {
        float m1 = -INFINITY, m2 = -INFINITY;
        for (uint j = 0; j < gsz; j++) {
          const float v = score_s[g * gsz + j];
          if (v > m1) { m2 = m1; m1 = v; } else if (v > m2) { m2 = v; }
        }
        gscore[g] = m1 + (gsz > 1 ? m2 : 0.0f);
        gkeep[g] = 0;
      }
      for (uint k = 0; k < p.topk_group; k++) {
        float bv = -INFINITY;
        uint bi = 0;
        for (uint g = 0; g < p.n_group; g++) {
          if (gkeep[g] == 0 && gscore[g] > bv) { bv = gscore[g]; bi = g; }
        }
        gkeep[bi] = 1;
      }
      for (uint g = 0; g < p.n_group; g++) {
        if (gkeep[g] == 0) {
          for (uint j = 0; j < gsz; j++) score_s[g * gsz + j] = -INFINITY;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup uint sel_idx[16];
  threadgroup float sel_w[16];

  // Exact top-k via iterative parallel argmax.
  for (uint k = 0; k < p.top_k; k++) {
    float best_v = -INFINITY;
    uint best_i = 0xFFFFFFFFu;
    for (uint e = tid; e < p.n_experts; e += 128) {
      const float v = score_s[e];
      if (v > best_v || (v == best_v && e < best_i)) {
        best_v = v;
        best_i = e;
      }
    }

    UNROLL for (ushort off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_down(best_v, off);
      const uint oi = simd_shuffle_down(best_i, off);
      if (ov > best_v || (ov == best_v && oi < best_i)) {
        best_v = ov;
        best_i = oi;
      }
    }

    if (lane == 0) {
      sg_v[sg] = best_v;
      sg_i[sg] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
      float fv = sg_v[0];
      uint fi = sg_i[0];
      UNROLL for (uint s = 1; s < 4; s++) {
        if (sg_v[s] > fv || (sg_v[s] == fv && sg_i[s] < fi)) {
          fv = sg_v[s];
          fi = sg_i[s];
        }
      }
      sel_idx[k] = fi;
      sel_w[k] = orig_s[fi];
      score_s[fi] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    float denom = 1.0f;
    if (p.top_k > 1 && p.norm_topk_prob != 0) {
      float sum = 0.0f;
      for (uint k = 0; k < p.top_k; k++) sum += sel_w[k];
      denom = sum + 1e-20f;
    }
    const size_t base = (size_t)row * p.top_k;
    for (uint k = 0; k < p.top_k; k++) {
      out_inds[base + k] = sel_idx[k];
      out_weights[base + k] = (sel_w[k] / denom) * p.routed_scaling_factor;
    }
  }
}

struct Relu2Params {
  uint rows;
  uint width;       // multiple of 64
  uint sums_stride; // 0 = skip sums
};

// ReLU^2 activation + optional group-64 sums (used for Nemotron-H shared expert up_proj).
// Grid: threadgroups (width / 64, rows, 1) of 32 threads.
kernel void relu2_sums(const device bfloat* in [[buffer(0)]],
                       device bfloat* out [[buffer(1)]],
                       device float* sums [[buffer(2)]],
                       constant Relu2Params& p [[buffer(3)]],
                       uint2 grp [[threadgroup_position_in_grid]],
                       ushort lane [[thread_index_in_simdgroup]]) {
  const uint g = grp.x;
  const uint row = grp.y;
  if (g >= p.width / 64 || row >= p.rows) return;

  const size_t base = (size_t)row * p.width + (size_t)g * 64;
  const float x0 = float(in[base + lane]);
  const float x1 = float(in[base + 32 + lane]);
  const bfloat y0 = bfloat(x0 > 0.0f ? x0 * x0 : 0.0f);
  const bfloat y1 = bfloat(x1 > 0.0f ? x1 * x1 : 0.0f);
  out[base + lane] = y0;
  out[base + 32 + lane] = y1;

  if (p.sums_stride > 0) {
    const float s = simd_sum(float(y0) + float(y1));
    if (lane == 0) {
      sums[(size_t)g * p.sums_stride + row] = s;
    }
  }
}

struct SharedRelu2Params {
  uint K;
  uint N;
  uint G;
};

inline float moe_q4_dot8(uint w, float4 sa, float4 sb) {
  const ushort lo = ushort(w);
  const ushort hi = ushort(w >> 16);
  const float d0 = fma(float(lo & 0x000Fu), sa[0],
                   fma(float(lo & 0x00F0u), sa[1],
                   fma(float(lo & 0x0F00u), sa[2],
                       float(lo & 0xF000u) * sa[3])));
  const float d1 = fma(float(hi & 0x000Fu), sb[0],
                   fma(float(hi & 0x00F0u), sb[1],
                   fma(float(hi & 0x0F00u), sb[2],
                       float(hi & 0xF000u) * sb[3])));
  return d0 + d1;
}

// Dual-simdgroup 100%-contiguous 32-col 128-bit uint4 Fused 1-row GEMV + ReLU^2 for shared expert up_proj.
// Grid: threadgroups (N / 32, 1, 1) of 64 threads (SK = 2).
kernel void qmv_1_relu2(const device bfloat* X [[buffer(0)]],
                        const device uchar* W [[buffer(1)]],
                        device bfloat* Y [[buffer(2)]],
                        constant SharedRelu2Params& p [[buffer(3)]],
                        uint tile [[threadgroup_position_in_grid]],
                        uint slice [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const uint base_col = tile * NT;
  if (base_col >= p.N) return;

  const uint col0 = lane >> 1;
  const uint col1 = col0 + 16;
  const uint half_idx = lane & 1u;

  const device uchar* tile_w = W + (size_t)tile * (size_t)p.G * BLOCK;
  const uint g_first = p.G * slice / 2u;
  const uint g_last = p.G * (slice + 1u) / 2u;
  const size_t off0 = (size_t)col0 * 32 + (size_t)half_idx * 16;
  const size_t off1 = (size_t)col1 * 32 + (size_t)half_idx * 16;
  const size_t sb_off0 = CODES + (size_t)col0 * 4;
  const size_t sb_off1 = CODES + (size_t)col1 * 4;
  const float4 inv_p2 = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  for (uint g = g_first; g < g_last; g++) {
    const device uchar* blk = tile_w + (size_t)g * BLOCK;
    const uint4 w4_0 = *(const device uint4*)(blk + off0);
    const uint4 w4_1 = *(const device uint4*)(blk + off1);
    const uint sb0 = *(const device uint*)(blk + sb_off0);
    const uint sb1 = *(const device uint*)(blk + sb_off1);

    const device vec<bfloat, 4>* xp = (const device vec<bfloat, 4>*)(X + g * GS + half_idx * 32);
    float d0 = 0.0f;
    float d1 = 0.0f;
    float4 sum4 = float4(0.0f);

    UNROLL for (int wi = 0; wi < 4; wi++) {
      const float4 xa = float4(xp[wi * 2]);
      const float4 xb = float4(xp[wi * 2 + 1]);
      sum4 += xa + xb;
      const float4 sa = xa * inv_p2;
      const float4 sb_x = xb * inv_p2;
      d0 += moe_q4_dot8(w4_0[wi], sa, sb_x);
      d1 += moe_q4_dot8(w4_1[wi], sa, sb_x);
    }

    const float xsum = (sum4[0] + sum4[1]) + (sum4[2] + sum4[3]);
    acc0 = fma(as_type<float>(sb0 << 16), d0, fma(as_type<float>(sb0 & 0xFFFF0000u), xsum, acc0));
    acc1 = fma(as_type<float>(sb1 << 16), d1, fma(as_type<float>(sb1 & 0xFFFF0000u), xsum, acc1));
  }

  acc0 += simd_shuffle_xor(acc0, 1);
  acc1 += simd_shuffle_xor(acc1, 1);

  threadgroup float partial[2][NT];
  if (half_idx == 0) {
    partial[slice][col0] = acc0;
    partial[slice][col1] = acc1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (slice == 0) {
    const float v = float(bfloat(partial[0][lane] + partial[1][lane]));
    Y[base_col + lane] = bfloat(v > 0.0f ? v * v : 0.0f);
  }
}

struct MoeExpertParams {
  uint rows;
  uint top_k;
  uint K;
  uint N;
  uint G;               // K / 64
  uint apply_relu2;     // 1 for fc1, 0 for fc2
  uint x_per_expert;    // 0 for fc1 (shared X[row]), 1 for fc2 (X[row * top_k + slot])
  ulong expert_stride;  // byte stride between packed qk experts in W
};

// Single-simdgroup zero-barrier 100%-contiguous 32-col 128-bit uint4 expert GEMV (fc1+ReLU^2 / fc2).
// Grid: threadgroups (N / 32, top_k, rows) of 32 threads (348..504 SGs on Nemotron-3.5!).
kernel void moe_expert_qmv(const device bfloat* X [[buffer(0)]],
                           const device uchar* W [[buffer(1)]],
                           const device uint* expert_inds [[buffer(2)]],
                           device bfloat* Y [[buffer(3)]],
                           constant MoeExpertParams& p [[buffer(4)]],
                           uint3 grp [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const uint tile = grp.x;
  const uint k_slot = grp.y;
  const uint row = grp.z;
  const uint base_col = tile * NT;
  if (base_col >= p.N || k_slot >= p.top_k || row >= p.rows) return;

  const uint slot_idx = row * p.top_k + k_slot;
  const uint e = expert_inds[slot_idx];

  const uint col0 = lane >> 1;
  const uint col1 = col0 + 16;
  const uint half_idx = lane & 1u;

  const device uchar* tile_w = W + (size_t)e * (size_t)p.expert_stride + (size_t)tile * (size_t)p.G * BLOCK;
  const device bfloat* x_row = X + (size_t)(p.x_per_expert ? slot_idx : row) * p.K;
  const size_t off0 = (size_t)col0 * 32 + (size_t)half_idx * 16;
  const size_t off1 = (size_t)col1 * 32 + (size_t)half_idx * 16;
  const size_t sb_off0 = CODES + (size_t)col0 * 4;
  const size_t sb_off1 = CODES + (size_t)col1 * 4;
  const float4 inv_p2 = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  for (uint g = 0; g < p.G; g++) {
    const device uchar* blk = tile_w + (size_t)g * BLOCK;
    const uint4 w4_0 = *(const device uint4*)(blk + off0);
    const uint4 w4_1 = *(const device uint4*)(blk + off1);
    const uint sb0 = *(const device uint*)(blk + sb_off0);
    const uint sb1 = *(const device uint*)(blk + sb_off1);

    const device vec<bfloat, 4>* xp = (const device vec<bfloat, 4>*)(x_row + g * GS + half_idx * 32);
    float d0 = 0.0f;
    float d1 = 0.0f;
    float4 sum4 = float4(0.0f);

    UNROLL for (int wi = 0; wi < 4; wi++) {
      const float4 xa = float4(xp[wi * 2]);
      const float4 xb = float4(xp[wi * 2 + 1]);
      sum4 += xa + xb;
      const float4 sa = xa * inv_p2;
      const float4 sb_x = xb * inv_p2;
      d0 += moe_q4_dot8(w4_0[wi], sa, sb_x);
      d1 += moe_q4_dot8(w4_1[wi], sa, sb_x);
    }

    const float xsum = (sum4[0] + sum4[1]) + (sum4[2] + sum4[3]);
    acc0 = fma(as_type<float>(sb0 << 16), d0, fma(as_type<float>(sb0 & 0xFFFF0000u), xsum, acc0));
    acc1 = fma(as_type<float>(sb1 << 16), d1, fma(as_type<float>(sb1 & 0xFFFF0000u), xsum, acc1));
  }

  acc0 += simd_shuffle_xor(acc0, 1);
  acc1 += simd_shuffle_xor(acc1, 1);

  if (half_idx == 0) {
    device bfloat* y_out = Y + (size_t)slot_idx * p.N + base_col;
    if (p.apply_relu2 != 0) {
      const float v0 = float(bfloat(acc0));
      const float v1 = float(bfloat(acc1));
      y_out[col0] = bfloat(v0 > 0.0f ? v0 * v0 : 0.0f);
      y_out[col1] = bfloat(v1 > 0.0f ? v1 * v1 : 0.0f);
    } else {
      y_out[col0] = bfloat(acc0);
      y_out[col1] = bfloat(acc1);
    }
  }
}

struct MoeCombineParams {
  uint rows;
  uint hidden;
  uint top_k;
};

// Weighted sum of top-k routed expert outputs + shared expert output -> delta_buf.
// Grid: threads (hidden, rows, 1) with threadgroup (256, 1, 1).
kernel void moe_combine_shared(const device bfloat* moe_out [[buffer(0)]],
                               const device float* expert_weights [[buffer(1)]],
                               device bfloat* delta_buf [[buffer(2)]],
                               constant MoeCombineParams& p [[buffer(3)]],
                               uint2 tid [[thread_position_in_grid]]) {
  const uint col = tid.x;
  const uint row = tid.y;
  if (col >= p.hidden || row >= p.rows) return;

  const size_t base_slot = (size_t)row * p.top_k;
  float acc = 0.0f;
  for (uint k = 0; k < p.top_k; k++) {
    const float w = expert_weights[base_slot + k];
    const float v = float(moe_out[(base_slot + k) * p.hidden + col]);
    acc = fma(v, w, acc);
  }
  const size_t out_idx = (size_t)row * p.hidden + col;
  const float shared = float(delta_buf[out_idx]);
  delta_buf[out_idx] = bfloat(float(bfloat(acc)) + shared);
}

struct MoeBucketParams {
  uint rows;
  uint top_k;
  uint n_experts;
};

// GPU-resident token bucketing by expert for fast batched MoE prefill (rows > 1).
// Groups all (row, k_slot) assignments by expert id e in [0..n_experts-1] in < 2 us (zero CPU sync!).
// Grid: threadgroups (1, 1, 1) of 128 threads (supports n_experts <= 256).
kernel void moe_bucket_tokens(const device uint* expert_inds [[buffer(0)]],
                              device uint* expert_offsets [[buffer(1)]],
                              device uint* expert_counts [[buffer(2)]],
                              device uint* bucket_slots [[buffer(3)]],
                              constant MoeBucketParams& p [[buffer(4)]],
                              ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup uint cnt[256];
  threadgroup uint off[256];

  for (uint e = tid; e < p.n_experts; e += 128) cnt[e] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint total = p.rows * p.top_k;
  if (tid == 0) {
    for (uint s = 0; s < total; s++) {
      cnt[expert_inds[s]]++;
    }
    uint acc = 0;
    for (uint e = 0; e < p.n_experts; e++) {
      off[e] = acc;
      expert_offsets[e] = acc;
      expert_counts[e] = cnt[e];
      acc += cnt[e];
    }
    for (uint s = 0; s < total; s++) {
      const uint e = expert_inds[s];
      const uint pos = off[e]++;
      bucket_slots[pos] = s;
    }
  }
}

struct MoeBatchedParams {
  uint top_k;
  uint n_experts;
  uint K;
  uint N;
  uint G;              // K / 64
  uint apply_relu2;    // 1 for fc1, 0 for fc2
  uint x_per_expert;   // 0 for fc1 (shared X[slot / top_k]), 1 for fc2 (X[slot])
  ulong expert_stride; // byte stride between packed qk experts in W
};

// Batched multi-token 4-bit expert GEMM for prefill: loads each active expert's 4-bit weights ONCE
// and reuses them across up to 8 tokens assigned to that expert (up to 6x DRAM bandwidth reduction!).
// Grid: threadgroups (N / 32, n_experts, 1) of 128 threads (SK = 4).
kernel void moe_expert_qmm_batched(const device bfloat* X [[buffer(0)]],
                                   const device uchar* W [[buffer(1)]],
                                   const device uint* expert_offsets [[buffer(2)]],
                                   const device uint* expert_counts [[buffer(3)]],
                                   const device uint* bucket_slots [[buffer(4)]],
                                   device bfloat* Y [[buffer(5)]],
                                   constant MoeBatchedParams& p [[buffer(6)]],
                                   uint2 grp [[threadgroup_position_in_grid]],
                                   uint slice [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = 8;
  const uint tile = grp.x;
  const uint e = grp.y;
  const uint base_col = tile * NT;
  if (base_col >= p.N || e >= p.n_experts) return;

  const uint count = expert_counts[e];
  if (count == 0) return;

  const uint start_idx = expert_offsets[e];
  const device uchar* blocks = W + (size_t)e * (size_t)p.expert_stride + (size_t)tile * (size_t)p.G * BLOCK;
  const int g_first = (int)p.G * (int)slice / SK;
  const int g_last = (int)p.G * (int)(slice + 1) / SK;
  const size_t col_off = (size_t)lane * 32;
  const size_t sb_off = CODES + (size_t)lane * 4;

  // Shared staging buffer: 4 * 8 * 64 * 2B = 4 KiB (repurposed for partial reduction after K loop)
  threadgroup ushort tg_mem[SK * ROWS * GS];
  threadgroup bfloat* x_slice = ((threadgroup bfloat*)tg_mem) + (size_t)slice * ROWS * GS;

  for (uint chunk = 0; chunk < count; chunk += ROWS) {
    const uint m_active = min((uint)ROWS, count - chunk);
    const uint last_valid = m_active - 1;

    uint slots[ROWS];
    uint x_rows[ROWS];
    UNROLL for (uint r = 0; r < ROWS; r++) {
      const uint s = bucket_slots[start_idx + chunk + min(r, last_valid)];
      slots[r] = s;
      x_rows[r] = p.x_per_expert ? s : (s / p.top_k);
    }

    float acc[ROWS];
    UNROLL for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;

    for (int g = g_first; g < g_last; g++) {
      const int k_base = g * GS + (int)lane * 2;
      float row_sum[ROWS];
      UNROLL for (int r = 0; r < ROWS; r++) {
        const vec<bfloat, 2> v2 = *(const device vec<bfloat, 2>*)(X + (size_t)x_rows[r] * p.K + k_base);
        *(threadgroup vec<bfloat, 2>*)(x_slice + r * GS + lane * 2) = v2;
        row_sum[r] = simd_sum(float(v2[0]) + float(v2[1]));
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);

      const device uchar* blk = blocks + (size_t)g * BLOCK;
      const uint4 w_lo = *(const device uint4*)(blk + col_off);
      const uint4 w_hi = *(const device uint4*)(blk + col_off + 16);
      const uint sb = *(const device uint*)(blk + sb_off);
      const float scale = as_type<float>(sb << 16);
      const float bias = as_type<float>(sb & 0xFFFF0000u);

      float dot[ROWS];
      UNROLL for (int r = 0; r < ROWS; r++) dot[r] = 0.0f;

      UNROLL for (int wi = 0; wi < 4; wi++) {
        const uint w = w_lo[wi];
        const float4 qa = float4(float(w & 0xFu), float((w >> 4) & 0xFu), float((w >> 8) & 0xFu), float((w >> 12) & 0xFu));
        const float4 qb = float4(float((w >> 16) & 0xFu), float((w >> 20) & 0xFu), float((w >> 24) & 0xFu), float((w >> 28) & 0xFu));
        const int k_off = wi * 8;
        UNROLL for (int r = 0; r < ROWS; r++) {
          const float4 xa = float4(*(const threadgroup vec<bfloat, 4>*)(x_slice + r * GS + k_off));
          const float4 xb = float4(*(const threadgroup vec<bfloat, 4>*)(x_slice + r * GS + k_off + 4));
          const float4 p4 = fma(qa, xa, qb * xb);
          dot[r] += (p4[0] + p4[1]) + (p4[2] + p4[3]);
        }
      }
      UNROLL for (int wi = 0; wi < 4; wi++) {
        const uint w = w_hi[wi];
        const float4 qa = float4(float(w & 0xFu), float((w >> 4) & 0xFu), float((w >> 8) & 0xFu), float((w >> 12) & 0xFu));
        const float4 qb = float4(float((w >> 16) & 0xFu), float((w >> 20) & 0xFu), float((w >> 24) & 0xFu), float((w >> 28) & 0xFu));
        const int k_off = 32 + wi * 8;
        UNROLL for (int r = 0; r < ROWS; r++) {
          const float4 xa = float4(*(const threadgroup vec<bfloat, 4>*)(x_slice + r * GS + k_off));
          const float4 xb = float4(*(const threadgroup vec<bfloat, 4>*)(x_slice + r * GS + k_off + 4));
          const float4 p4 = fma(qa, xa, qb * xb);
          dot[r] += (p4[0] + p4[1]) + (p4[2] + p4[3]);
        }
      }

      UNROLL for (int r = 0; r < ROWS; r++) {
        acc[r] = fma(scale, dot[r], fma(bias, row_sum[r], acc[r]));
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* partial = (threadgroup float*)tg_mem;
    if (slice > 0) {
      UNROLL for (int r = 0; r < ROWS; r++) {
        partial[((slice - 1) * ROWS + r) * NT + lane] = acc[r];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (slice == 0) {
      const uint out_col = base_col + lane;
      UNROLL for (uint r = 0; r < ROWS; r++) {
        if (r < m_active) {
          float total = acc[r];
          UNROLL for (int s = 0; s < SK - 1; s++) {
            total += partial[(s * ROWS + r) * NT + lane];
          }
          if (p.apply_relu2 != 0) {
            const float v = float(bfloat(total));
            total = (v > 0.0f) ? (v * v) : 0.0f;
          }
          Y[(size_t)slots[r] * p.N + out_col] = bfloat(total);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

struct MoeGatherSumsParams {
  uint total_slots;
  uint top_k;
  uint K;
  uint sums_stride;
  uint identity; // 0 = gather X[bucket_slots[row] / top_k] into dst + sums; 1 = sums only over src in-place
};

// Gather bucketed token activations & compute group-64 fp32 sums for M5 tensor-unit moe_expert_qk16_batched.
// Grid: threadgroups (K / 64, total_slots, 1) of 32 threads.
kernel void moe_gather_x_sums(const device bfloat* src [[buffer(0)]],
                              const device uint* bucket_slots [[buffer(1)]],
                              device bfloat* dst [[buffer(2)]],
                              device float* sums [[buffer(3)]],
                              constant MoeGatherSumsParams& p [[buffer(4)]],
                              uint2 grp [[threadgroup_position_in_grid]],
                              ushort lane [[thread_index_in_simdgroup]]) {
  const uint g = grp.x;
  const uint row = grp.y;
  if (g >= p.K / 64 || row >= p.total_slots) return;

  const uint src_row = (p.identity != 0) ? row : (bucket_slots[row] / p.top_k);
  const size_t s_off = (size_t)src_row * p.K + (size_t)g * 64;
  const bfloat v0 = src[s_off + lane];
  const bfloat v1 = src[s_off + 32 + lane];

  if (p.identity == 0) {
    const size_t d_off = (size_t)row * p.K + (size_t)g * 64;
    dst[d_off + lane] = v0;
    dst[d_off + 32 + lane] = v1;
  }

  const float s = simd_sum(float(v0) + float(v1));
  if (lane == 0) {
    sums[(size_t)g * p.sums_stride + row] = s;
  }
}

// ============================================================================
// General SwiGLU Sparse MoE Kernels (Qwen3 MoE, Qwen3.5/3.6 MoE, Mixtral, DeepSeek, GLM)
// Supports packed qk4, general 2/3/4/5/6/8-bit affine, and lossless BF16 routed experts!
// ============================================================================

inline float moe_read_sb(const device uchar* buf, size_t idx, uint sb_dtype) {
  if (sb_dtype == 0u) return float(((const device bfloat*)buf)[idx]);
  if (sb_dtype == 1u) return float(((const device half*)buf)[idx]);
  return ((const device float*)buf)[idx];
}

inline uint moe_extract_code(const device uint* row_w, uint k, uint bits) {
  const uint bit_pos = k * bits;
  const uint word_idx = bit_pos >> 5;
  const uint bit_off = bit_pos & 31u;
  const uint mask = (1u << bits) - 1u;
  uint v = row_w[word_idx] >> bit_off;
  if (bit_off + bits > 32u) v |= row_w[word_idx + 1u] << (32u - bit_off);
  return v & mask;
}

struct MoeSwigluParams {
  uint rows;
  uint top_k;
  uint K;
  uint N;
  uint bits;
  uint gs;
  uint sb_dtype;
  uint x_per_expert;
  ulong code_stride;
  ulong sb_stride;
};

// Fused gate + up -> silu(gate) * up for packed qk4 routed experts.
// Grid: threadgroups (N / 32, top_k, rows) of 32 threads.
kernel void moe_swiglu_gu_qk4(const device bfloat* X [[buffer(0)]],
                              const device uchar* W_gate [[buffer(1)]],
                              const device uchar* W_up [[buffer(2)]],
                              const device uint* expert_inds [[buffer(3)]],
                              device bfloat* Y [[buffer(4)]],
                              constant MoeSwigluParams& p [[buffer(5)]],
                              uint3 grp [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const uint tile = grp.x;
  const uint k_slot = grp.y;
  const uint row = grp.z;
  const uint base_col = tile * NT;
  if (base_col >= p.N || k_slot >= p.top_k || row >= p.rows) return;

  const uint slot_idx = row * p.top_k + k_slot;
  const uint e = expert_inds[slot_idx];
  const uint G = p.K / GS;
  const uint col0 = lane >> 1;
  const uint col1 = col0 + 16;
  const uint half_idx = lane & 1u;

  const device uchar* tg = W_gate + (size_t)e * (size_t)p.code_stride + (size_t)tile * (size_t)G * BLOCK;
  const device uchar* tu = W_up + (size_t)e * (size_t)p.code_stride + (size_t)tile * (size_t)G * BLOCK;
  const device bfloat* x_row = X + (size_t)row * p.K;
  const size_t off0 = (size_t)col0 * 32 + (size_t)half_idx * 16;
  const size_t off1 = (size_t)col1 * 32 + (size_t)half_idx * 16;
  const size_t sb_off0 = CODES + (size_t)col0 * 4;
  const size_t sb_off1 = CODES + (size_t)col1 * 4;
  const float4 inv_p2 = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

  float g0 = 0.0f, g1 = 0.0f, u0 = 0.0f, u1 = 0.0f;
  for (uint g = 0; g < G; g++) {
    const device uchar* bg = tg + (size_t)g * BLOCK;
    const device uchar* bu = tu + (size_t)g * BLOCK;
    const uint4 wg0 = *(const device uint4*)(bg + off0);
    const uint4 wg1 = *(const device uint4*)(bg + off1);
    const uint4 wu0 = *(const device uint4*)(bu + off0);
    const uint4 wu1 = *(const device uint4*)(bu + off1);
    const uint sbg0 = *(const device uint*)(bg + sb_off0);
    const uint sbg1 = *(const device uint*)(bg + sb_off1);
    const uint sbu0 = *(const device uint*)(bu + sb_off0);
    const uint sbu1 = *(const device uint*)(bu + sb_off1);

    const device vec<bfloat, 4>* xp = (const device vec<bfloat, 4>*)(x_row + g * GS + half_idx * 32);
    float dg0 = 0.0f, dg1 = 0.0f, du0 = 0.0f, du1 = 0.0f;
    float4 sum4 = float4(0.0f);
    UNROLL for (int wi = 0; wi < 4; wi++) {
      const float4 xa = float4(xp[wi * 2]);
      const float4 xb = float4(xp[wi * 2 + 1]);
      sum4 += xa + xb;
      const float4 sa = xa * inv_p2;
      const float4 sb_x = xb * inv_p2;
      dg0 += moe_q4_dot8(wg0[wi], sa, sb_x);
      dg1 += moe_q4_dot8(wg1[wi], sa, sb_x);
      du0 += moe_q4_dot8(wu0[wi], sa, sb_x);
      du1 += moe_q4_dot8(wu1[wi], sa, sb_x);
    }
    const float xsum = (sum4[0] + sum4[1]) + (sum4[2] + sum4[3]);
    g0 = fma(as_type<float>(sbg0 << 16), dg0, fma(as_type<float>(sbg0 & 0xFFFF0000u), xsum, g0));
    g1 = fma(as_type<float>(sbg1 << 16), dg1, fma(as_type<float>(sbg1 & 0xFFFF0000u), xsum, g1));
    u0 = fma(as_type<float>(sbu0 << 16), du0, fma(as_type<float>(sbu0 & 0xFFFF0000u), xsum, u0));
    u1 = fma(as_type<float>(sbu1 << 16), du1, fma(as_type<float>(sbu1 & 0xFFFF0000u), xsum, u1));
  }
  g0 += simd_shuffle_xor(g0, 1); g1 += simd_shuffle_xor(g1, 1);
  u0 += simd_shuffle_xor(u0, 1); u1 += simd_shuffle_xor(u1, 1);
  if (half_idx == 0) {
    device bfloat* y_row = Y + (size_t)slot_idx * p.N + base_col;
    const float a0 = (g0 / (1.0f + exp(-g0))) * u0;
    const float a1 = (g1 / (1.0f + exp(-g1))) * u1;
    y_row[col0] = bfloat(a0);
    y_row[col1] = bfloat(a1);
  }
}

// Fused gate + up -> silu(gate) * up for general 2/3/4/5/6/8-bit affine routed experts.
// Grid: threadgroups ((N + 3) / 4, top_k, rows) of 128 threads (4 SGs = 4 output rows).
kernel void moe_swiglu_gu_affine(const device bfloat* X [[buffer(0)]],
                                 const device uchar* Wg_c [[buffer(1)]],
                                 const device uchar* Wg_s [[buffer(2)]],
                                 const device uchar* Wg_b [[buffer(3)]],
                                 const device uchar* Wu_c [[buffer(4)]],
                                 const device uchar* Wu_s [[buffer(5)]],
                                 const device uchar* Wu_b [[buffer(6)]],
                                 const device uint* expert_inds [[buffer(7)]],
                                 device bfloat* Y [[buffer(8)]],
                                 constant MoeSwigluParams& p [[buffer(9)]],
                                 uint3 grp [[threadgroup_position_in_grid]],
                                 ushort sg [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
  const uint n = grp.x * 4 + sg;
  const uint k_slot = grp.y;
  const uint row = grp.z;
  if (n >= p.N || k_slot >= p.top_k || row >= p.rows) return;

  const uint slot_idx = row * p.top_k + k_slot;
  const uint e = expert_inds[slot_idx];
  const uint words_per_row = (p.K * p.bits) / 32u;
  const uint groups = p.K / p.gs;
  const device bfloat* xr = X + (size_t)row * p.K;

  const device uint* wg_row = (const device uint*)(Wg_c + (size_t)e * (size_t)p.code_stride) + (size_t)n * words_per_row;
  const device uint* wu_row = (const device uint*)(Wu_c + (size_t)e * (size_t)p.code_stride) + (size_t)n * words_per_row;
  const device uchar* wgs = Wg_s + (size_t)e * (size_t)p.sb_stride;
  const device uchar* wgb = Wg_b + (size_t)e * (size_t)p.sb_stride;
  const device uchar* wus = Wu_s + (size_t)e * (size_t)p.sb_stride;
  const device uchar* wub = Wu_b + (size_t)e * (size_t)p.sb_stride;

  float ag = 0.0f, au = 0.0f;
  for (uint g = 0; g < groups; g++) {
    const size_t sb_i = (size_t)n * groups + g;
    const float sgv = moe_read_sb(wgs, sb_i, p.sb_dtype);
    const float bgv = moe_read_sb(wgb, sb_i, p.sb_dtype);
    const float suv = moe_read_sb(wus, sb_i, p.sb_dtype);
    const float buv = moe_read_sb(wub, sb_i, p.sb_dtype);
    const uint k0 = g * p.gs;
    for (uint i = lane; i < p.gs; i += 32) {
      const float xv = float(xr[k0 + i]);
      const float cg = float(moe_extract_code(wg_row, k0 + i, p.bits));
      const float cu = float(moe_extract_code(wu_row, k0 + i, p.bits));
      ag = fma(xv, fma(cg, sgv, bgv), ag);
      au = fma(xv, fma(cu, suv, buv), au);
    }
  }
  const float gv = simd_sum(ag);
  const float uv = simd_sum(au);
  if (lane == 0) {
    Y[(size_t)slot_idx * p.N + n] = bfloat((gv / (1.0f + exp(-gv))) * uv);
  }
}

// Down projection for general 2/3/4/5/6/8-bit affine routed experts.
// Grid: threadgroups ((N + 3) / 4, top_k, rows) of 128 threads.
kernel void moe_swiglu_dn_affine(const device bfloat* X [[buffer(0)]],
                                 const device uchar* W_c [[buffer(1)]],
                                 const device uchar* W_s [[buffer(2)]],
                                 const device uchar* W_b [[buffer(3)]],
                                 const device uint* expert_inds [[buffer(4)]],
                                 device bfloat* Y [[buffer(5)]],
                                 constant MoeSwigluParams& p [[buffer(6)]],
                                 uint3 grp [[threadgroup_position_in_grid]],
                                 ushort sg [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
  const uint n = grp.x * 4 + sg;
  const uint k_slot = grp.y;
  const uint row = grp.z;
  if (n >= p.N || k_slot >= p.top_k || row >= p.rows) return;

  const uint slot_idx = row * p.top_k + k_slot;
  const uint e = expert_inds[slot_idx];
  const uint words_per_row = (p.K * p.bits) / 32u;
  const uint groups = p.K / p.gs;
  const device bfloat* xr = X + (size_t)slot_idx * p.K;
  const device uint* w_row = (const device uint*)(W_c + (size_t)e * (size_t)p.code_stride) + (size_t)n * words_per_row;
  const device uchar* ws = W_s + (size_t)e * (size_t)p.sb_stride;
  const device uchar* wb = W_b + (size_t)e * (size_t)p.sb_stride;

  float acc = 0.0f;
  for (uint g = 0; g < groups; g++) {
    const size_t sb_i = (size_t)n * groups + g;
    const float sv = moe_read_sb(ws, sb_i, p.sb_dtype);
    const float bv = moe_read_sb(wb, sb_i, p.sb_dtype);
    const uint k0 = g * p.gs;
    for (uint i = lane; i < p.gs; i += 32) {
      acc = fma(float(xr[k0 + i]), fma(float(moe_extract_code(w_row, k0 + i, p.bits)), sv, bv), acc);
    }
  }
  const float sum = simd_sum(acc);
  if (lane == 0) Y[(size_t)slot_idx * p.N + n] = bfloat(sum);
}

// Fused gate + up -> silu(gate) * up for BF16 routed experts.
// Grid: threadgroups ((N + 3) / 4, top_k, rows) of 128 threads.
kernel void moe_swiglu_gu_bf16(const device bfloat* X [[buffer(0)]],
                               const device uchar* W_gate [[buffer(1)]],
                               const device uchar* W_up [[buffer(2)]],
                               const device uint* expert_inds [[buffer(3)]],
                               device bfloat* Y [[buffer(4)]],
                               constant MoeSwigluParams& p [[buffer(5)]],
                               uint3 grp [[threadgroup_position_in_grid]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
  const uint n = grp.x * 4 + sg;
  const uint k_slot = grp.y;
  const uint row = grp.z;
  if (n >= p.N || k_slot >= p.top_k || row >= p.rows) return;

  const uint slot_idx = row * p.top_k + k_slot;
  const uint e = expert_inds[slot_idx];
  const uint k4 = p.K / 4;
  const device vec<bfloat, 4>* xp = (const device vec<bfloat, 4>*)(X + (size_t)row * p.K);
  const device vec<bfloat, 4>* wgp = (const device vec<bfloat, 4>*)((const device bfloat*)(W_gate + (size_t)e * (size_t)p.code_stride) + (size_t)n * p.K);
  const device vec<bfloat, 4>* wup = (const device vec<bfloat, 4>*)((const device bfloat*)(W_up + (size_t)e * (size_t)p.code_stride) + (size_t)n * p.K);

  float4 ag = float4(0.0f), au = float4(0.0f);
  for (uint i = lane; i < k4; i += 32) {
    const float4 xv = float4(xp[i]);
    ag = fma(xv, float4(wgp[i]), ag);
    au = fma(xv, float4(wup[i]), au);
  }
  const float gv = simd_sum((ag[0] + ag[1]) + (ag[2] + ag[3]));
  const float uv = simd_sum((au[0] + au[1]) + (au[2] + au[3]));
  if (lane == 0) Y[(size_t)slot_idx * p.N + n] = bfloat((gv / (1.0f + exp(-gv))) * uv);
}

// Down projection for BF16 routed experts.
// Grid: threadgroups ((N + 3) / 4, top_k, rows) of 128 threads.
kernel void moe_swiglu_dn_bf16(const device bfloat* X [[buffer(0)]],
                               const device uchar* W_dn [[buffer(1)]],
                               const device uint* expert_inds [[buffer(2)]],
                               device bfloat* Y [[buffer(3)]],
                               constant MoeSwigluParams& p [[buffer(4)]],
                               uint3 grp [[threadgroup_position_in_grid]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
  const uint n = grp.x * 4 + sg;
  const uint k_slot = grp.y;
  const uint row = grp.z;
  if (n >= p.N || k_slot >= p.top_k || row >= p.rows) return;

  const uint slot_idx = row * p.top_k + k_slot;
  const uint e = expert_inds[slot_idx];
  const uint k4 = p.K / 4;
  const device vec<bfloat, 4>* xp = (const device vec<bfloat, 4>*)(X + (size_t)slot_idx * p.K);
  const device vec<bfloat, 4>* wp = (const device vec<bfloat, 4>*)((const device bfloat*)(W_dn + (size_t)e * (size_t)p.code_stride) + (size_t)n * p.K);

  float4 acc = float4(0.0f);
  for (uint i = lane; i < k4; i += 32) acc = fma(float4(xp[i]), float4(wp[i]), acc);
  const float sum = simd_sum((acc[0] + acc[1]) + (acc[2] + acc[3]));
  if (lane == 0) Y[(size_t)slot_idx * p.N + n] = bfloat(sum);
}

// Qwen3.5 MoE scalar sigmoid shared-expert gate: out[row] = sigmoid(dot(X[row], W_gate)).
// Grid: threadgroups (1, rows, 1) of 32 threads.
kernel void moe_shared_gate_scalar(const device bfloat* X [[buffer(0)]],
                                   const device bfloat* W [[buffer(1)]],
                                   device float* out_g [[buffer(2)]],
                                   constant GateParams& p [[buffer(3)]],
                                   uint2 grp [[threadgroup_position_in_grid]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
  const uint row = grp.y;
  if (row >= p.rows) return;
  const uint k4 = p.hidden / 4;
  const device vec<bfloat, 4>* xp = (const device vec<bfloat, 4>*)(X + (size_t)row * p.hidden);
  const device vec<bfloat, 4>* wp = (const device vec<bfloat, 4>*)W;
  float4 acc = float4(0.0f);
  for (uint i = lane; i < k4; i += 32) acc = fma(float4(xp[i]), float4(wp[i]), acc);
  const float s = simd_sum((acc[0] + acc[1]) + (acc[2] + acc[3]));
  if (lane == 0) out_g[row] = 1.0f / (1.0f + exp(-s));
}

struct MoeSwigluCombineParams {
  uint rows;
  uint hidden;
  uint top_k;
  uint has_shared;
  uint shared_gated;
  uint add_res;
};

// Combine top-K routed expert outputs + optional (gated) shared expert output (+ optional residual add).
// Grid: threads (hidden, rows, 1).
kernel void moe_combine_swiglu(const device bfloat* routed_out [[buffer(0)]],
                               const device float* expert_weights [[buffer(1)]],
                               const device bfloat* shared_out [[buffer(2)]],
                               const device float* shared_gate [[buffer(3)]],
                               device bfloat* dst [[buffer(4)]],
                               constant MoeSwigluCombineParams& p [[buffer(5)]],
                               uint2 tid [[thread_position_in_grid]]) {
  const uint h = tid.x;
  const uint row = tid.y;
  if (h >= p.hidden || row >= p.rows) return;

  float acc = 0.0f;
  const size_t w_base = (size_t)row * p.top_k;
  for (uint k = 0; k < p.top_k; k++) {
    const float w = expert_weights[w_base + k];
    acc = fma(w, float(routed_out[(size_t)(w_base + k) * p.hidden + h]), acc);
  }
  if (p.has_shared != 0) {
    const float sg = (p.shared_gated != 0) ? shared_gate[row] : 1.0f;
    acc = fma(sg, float(shared_out[(size_t)row * p.hidden + h]), acc);
  }
  const size_t off = (size_t)row * p.hidden + h;
  if (p.add_res != 0) acc += float(dst[off]);
  dst[off] = bfloat(acc);
}
