// Clean-room fused GQA Attention kernels for Qwen3.5 / Qwen3.8 (head_dim = 256, rotary_dim = 64).
//
// Lane-to-head-dim mapping inside each 32-lane simdgroup:
//   For group g in {0, 1, 2, 3}: lane l holds (g*64 + l, g*64 + 32 + l).
//   - Coalesced 64-byte reads/writes across all 32 lanes.
//   - Group 0 (l, l+32) IS the exact split-half partial RoPE pair (zero shuffles!).
//   - Each group g (0..3) IS an exact 64-element o_proj quant group (zero shuffles for group sums!).
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define UNROLL _Pragma("clang loop unroll(full)")

constant constexpr uint HD = 256;

struct AttnPrepParams {
  uint rows;
  uint num_heads;
  uint num_kv_heads;
  uint qkv_stride; // 2*H*256 + 2*KVH*256
  float eps;
};

// Fused Q/K RMSNorm + split-half partial RoPE (dims 0..64) + KV cache store.
// Grid: threadgroups (num_heads + num_kv_heads, rows) of 32 threads.
kernel void attn_qk_rope_store(const device bfloat* qkv_buf [[buffer(0)]],
                               const device bfloat* q_norm [[buffer(1)]],
                               const device bfloat* k_norm [[buffer(2)]],
                               const device float* inv_freq [[buffer(3)]], // [32] f32 precomputed from f64
                               const device uint* positions [[buffer(4)]],
                               const device uint* kv_slots [[buffer(5)]],
                               device bfloat* q_out [[buffer(6)]],
                               device bfloat* k_cache [[buffer(7)]],
                               device bfloat* v_cache [[buffer(8)]],
                               constant AttnPrepParams& p [[buffer(9)]],
                               uint2 grp [[threadgroup_position_in_grid]],
                               ushort lane [[thread_index_in_simdgroup]]) {
  const uint idx = grp.x;
  const uint row = grp.y;
  if (row >= p.rows) return;

  const device bfloat* row_ptr = qkv_buf + (size_t)row * p.qkv_stride;
  const float pos = float(positions[row]);
  const float angle = pos * inv_freq[lane];
  const float c = cos(angle);
  const float s = sin(angle);

  if (idx < p.num_heads) {
    // Query head h: Q occupies first 256 of each 512-element [Q, gate] head slice.
    const device bfloat* src = row_ptr + (size_t)idx * (2 * HD);
    float v[8];
    float sq = 0.0f;
    UNROLL for (uint g = 0; g < 4; g++) {
      const float a = float(src[g * 64 + lane]);
      const float b = float(src[g * 64 + 32 + lane]);
      v[g * 2] = a;
      v[g * 2 + 1] = b;
      sq = fma(a, a, fma(b, b, sq));
    }
    const float inv = rsqrt(simd_sum(sq) / float(HD) + p.eps);
    UNROLL for (uint g = 0; g < 4; g++) {
      v[g * 2] = v[g * 2] * inv * float(q_norm[g * 64 + lane]);
      v[g * 2 + 1] = v[g * 2 + 1] * inv * float(q_norm[g * 64 + 32 + lane]);
    }
    // Partial RoPE on group 0 (0..63)
    const float r0 = v[0] * c - v[1] * s;
    const float r1 = v[1] * c + v[0] * s;
    v[0] = r0;
    v[1] = r1;

    device bfloat* dst = q_out + (size_t)(row * p.num_heads + idx) * HD;
    UNROLL for (uint g = 0; g < 4; g++) {
      dst[g * 64 + lane] = bfloat(v[g * 2]);
      dst[g * 64 + 32 + lane] = bfloat(v[g * 2 + 1]);
    }
    return;
  }

  // KV head kh: normalize + RoPE K, and copy V directly into KV cache at kv_slots[row].
  const uint kh = idx - p.num_heads;
  const uint slot = kv_slots[row];
  const size_t k_off = (size_t)(2 * p.num_heads) * HD + (size_t)kh * HD;
  const size_t v_off = k_off + (size_t)p.num_kv_heads * HD;
  const device bfloat* k_src = row_ptr + k_off;
  const device bfloat* v_src = row_ptr + v_off;

  float kv[8];
  float sq = 0.0f;
  UNROLL for (uint g = 0; g < 4; g++) {
    const float a = float(k_src[g * 64 + lane]);
    const float b = float(k_src[g * 64 + 32 + lane]);
    kv[g * 2] = a;
    kv[g * 2 + 1] = b;
    sq = fma(a, a, fma(b, b, sq));
  }
  const float inv = rsqrt(simd_sum(sq) / float(HD) + p.eps);
  UNROLL for (uint g = 0; g < 4; g++) {
    kv[g * 2] = kv[g * 2] * inv * float(k_norm[g * 64 + lane]);
    kv[g * 2 + 1] = kv[g * 2 + 1] * inv * float(k_norm[g * 64 + 32 + lane]);
  }
  const float rk0 = kv[0] * c - kv[1] * s;
  const float rk1 = kv[1] * c + kv[0] * s;
  kv[0] = rk0;
  kv[1] = rk1;

  const size_t cache_base = (size_t)(slot * p.num_kv_heads + kh) * HD;
  device bfloat* k_dst = k_cache + cache_base;
  device bfloat* v_dst = v_cache + cache_base;
  UNROLL for (uint g = 0; g < 4; g++) {
    k_dst[g * 64 + lane] = bfloat(kv[g * 2]);
    k_dst[g * 64 + 32 + lane] = bfloat(kv[g * 2 + 1]);
    v_dst[g * 64 + lane] = v_src[g * 64 + lane];
    v_dst[g * 64 + 32 + lane] = v_src[g * 64 + 32 + lane];
  }
}

struct AttnFlashParams {
  uint rows;
  uint num_heads;
  uint num_kv_heads;
  uint qkv_stride;
  uint sums_stride;
  float scale; // 1 / sqrt(head_dim)
  uint splits; // used by split-K path
  uint base_pos; // committed KV prefix length; tree rows start at base_pos
  uint is_tree;  // 1 when tree_mask filters slots >= base_pos
  uint sliding_window; // 0 = full causal; >0 = attend to last sliding_window tokens
  float softcap; // 0 = none; >0 = softcap * tanh(score / softcap)
};

// Single-stage fused Flash GQA + sigmoid(gate) + o_proj group sums.
// Supports both causal chains (is_tree=0) and arbitrary speculative trees up to 128 rows (is_tree=1).
// Grid: threadgroups (num_heads, rows) of 128 threads.
kernel void attn_flash_fused(const device bfloat* q_in [[buffer(0)]],
                             const device bfloat* qkv_buf [[buffer(1)]],
                             const device bfloat* k_cache [[buffer(2)]],
                             const device bfloat* v_cache [[buffer(3)]],
                             const device uint* seq_lens [[buffer(4)]], // inclusive KV scan bound per row
                             const device uint4* tree_masks [[buffer(5)]], // 128-bit ancestor bitmask per row
                             device bfloat* out [[buffer(6)]],
                             device float* sums [[buffer(7)]],
                             constant AttnFlashParams& p [[buffer(8)]],
                             uint2 grp [[threadgroup_position_in_grid]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  const uint head = grp.x;
  const uint row = grp.y;
  if (head >= p.num_heads || row >= p.rows) return;

  const uint kv_head = head / (p.num_heads / p.num_kv_heads);
  const uint kv_len = seq_lens[row];
  const uint4 tmask = (p.is_tree != 0) ? tree_masks[row] : uint4(0xFFFFFFFFu);
  const device bfloat* q_ptr = q_in + (size_t)(row * p.num_heads + head) * HD;

  float q[8];
  UNROLL for (uint g = 0; g < 4; g++) {
    q[g * 2] = float(q_ptr[g * 64 + lane]) * p.scale;
    q[g * 2 + 1] = float(q_ptr[g * 64 + 32 + lane]) * p.scale;
  }

  float m_val = -INFINITY;
  float l_val = 0.0f;
  float acc[8];
  UNROLL for (uint i = 0; i < 8; i++) acc[i] = 0.0f;

  const size_t kv_stride = (size_t)p.num_kv_heads * HD;
  const size_t kh_off = (size_t)kv_head * HD;

  const uint prefix_end = (p.is_tree != 0) ? min(kv_len, p.base_pos) : kv_len;

  // Phase 1: Committed prefix [0 .. prefix_end), assigned identically to sequential decode (t % 4 == sg).
  for (uint t = sg; t < prefix_end; t += 4) {
    const device bfloat* kp = k_cache + (size_t)t * kv_stride + kh_off;
    float dot_part = 0.0f;
    UNROLL for (uint g = 0; g < 4; g++) {
      dot_part = fma(q[g * 2], float(kp[g * 64 + lane]), dot_part);
      dot_part = fma(q[g * 2 + 1], float(kp[g * 64 + 32 + lane]), dot_part);
    }
    const float score = simd_sum(dot_part);
    const float new_m = max(m_val, score);
    const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
    const float prob = exp(score - new_m);
    l_val = fma(l_val, alpha, prob);
    m_val = new_m;

    const device bfloat* vp = v_cache + (size_t)t * kv_stride + kh_off;
    UNROLL for (uint g = 0; g < 4; g++) {
      acc[g * 2] = fma(acc[g * 2], alpha, prob * float(vp[g * 64 + lane]));
      acc[g * 2 + 1] = fma(acc[g * 2 + 1], alpha, prob * float(vp[g * 64 + 32 + lane]));
    }
  }

  // Phase 2 (trees only): Walk topological ancestors in exact depth order, assigning depth d to
  // simdgroup ((base_pos + d) & 3) so fp32 online-softmax reduction is 100% bit-identical to c=1 decode!
  if (p.is_tree != 0 && kv_len > p.base_pos) {
    const uint rel_limit = kv_len - p.base_pos;
    uint depth = 0;
    for (uint rel = 0; rel < rel_limit; rel++) {
      if (((tmask[rel >> 5] >> (rel & 31u)) & 1u) == 0u) continue;
      if (((p.base_pos + depth) & 3u) == sg) {
        const uint t = p.base_pos + rel;
        const device bfloat* kp = k_cache + (size_t)t * kv_stride + kh_off;
        float dot_part = 0.0f;
        UNROLL for (uint g = 0; g < 4; g++) {
          dot_part = fma(q[g * 2], float(kp[g * 64 + lane]), dot_part);
          dot_part = fma(q[g * 2 + 1], float(kp[g * 64 + 32 + lane]), dot_part);
        }
        const float score = simd_sum(dot_part);
        const float new_m = max(m_val, score);
        const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
        const float prob = exp(score - new_m);
        l_val = fma(l_val, alpha, prob);
        m_val = new_m;

        const device bfloat* vp = v_cache + (size_t)t * kv_stride + kh_off;
        UNROLL for (uint g = 0; g < 4; g++) {
          acc[g * 2] = fma(acc[g * 2], alpha, prob * float(vp[g * 64 + lane]));
          acc[g * 2 + 1] = fma(acc[g * 2 + 1], alpha, prob * float(vp[g * 64 + 32 + lane]));
        }
      }
      depth++;
    }
  }

  threadgroup float tg_m[4];
  threadgroup float tg_l[4];
  threadgroup float tg_acc[4][HD];

  if (lane == 0) {
    tg_m[sg] = m_val;
    tg_l[sg] = l_val;
  }
  UNROLL for (uint g = 0; g < 4; g++) {
    tg_acc[sg][g * 64 + lane] = acc[g * 2];
    tg_acc[sg][g * 64 + 32 + lane] = acc[g * 2 + 1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg != 0) return;

  float gm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
  float w[4];
  float gl = 0.0f;
  UNROLL for (uint s = 0; s < 4; s++) {
    w[s] = (tg_l[s] > 0.0f) ? exp(tg_m[s] - gm) : 0.0f;
    gl += tg_l[s] * w[s];
  }
  const float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

  // Apply sigmoid(gate) from qkv_buf[row, head*512 + 256 .. +512] and compute 4 group sums.
  const device bfloat* gate_ptr = qkv_buf + (size_t)row * p.qkv_stride + (size_t)head * (2 * HD) + HD;
  device bfloat* out_ptr = out + (size_t)(row * p.num_heads + head) * HD;

  UNROLL for (uint g = 0; g < 4; g++) {
    const uint i0 = g * 64 + lane;
    const uint i1 = g * 64 + 32 + lane;
    float o0 = 0.0f, o1 = 0.0f;
    UNROLL for (uint s = 0; s < 4; s++) {
      o0 = fma(tg_acc[s][i0], w[s], o0);
      o1 = fma(tg_acc[s][i1], w[s], o1);
    }
    o0 *= inv_l;
    o1 *= inv_l;
    const float g0 = 1.0f / (1.0f + exp(-float(gate_ptr[i0])));
    const float g1 = 1.0f / (1.0f + exp(-float(gate_ptr[i1])));
    const bfloat y0 = bfloat(o0 * g0);
    const bfloat y1 = bfloat(o1 * g1);
    out_ptr[i0] = y0;
    out_ptr[i1] = y1;
    const float gs = simd_sum(float(y0) + float(y1));
    if (lane == 0) {
      sums[(size_t)(head * 4 + g) * p.sums_stride + row] = gs;
    }
  }
}


// Split-K Stage 1 for long contexts: Grid threadgroups (splits, num_heads, rows) of 128 threads.
kernel void attn_flash_split(const device bfloat* q_in [[buffer(0)]],
                             const device bfloat* k_cache [[buffer(1)]],
                             const device bfloat* v_cache [[buffer(2)]],
                             const device uint* seq_lens [[buffer(3)]],
                             device float* partial_acc [[buffer(4)]], // [rows][H][splits][256]
                             device float2* partial_ml [[buffer(5)]], // [rows][H][splits]
                             constant AttnFlashParams& p [[buffer(6)]],
                             uint3 grp [[threadgroup_position_in_grid]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  const uint split = grp.x;
  const uint head = grp.y;
  const uint row = grp.z;
  if (split >= p.splits || head >= p.num_heads || row >= p.rows) return;

  const uint kv_len = seq_lens[row];
  const uint chunk = (kv_len + p.splits - 1) / p.splits;
  const uint t0 = split * chunk;
  const uint t1 = min(kv_len, t0 + chunk);

  const uint kv_head = head / (p.num_heads / p.num_kv_heads);
  const device bfloat* q_ptr = q_in + (size_t)(row * p.num_heads + head) * HD;

  float q[8];
  UNROLL for (uint g = 0; g < 4; g++) {
    q[g * 2] = float(q_ptr[g * 64 + lane]) * p.scale;
    q[g * 2 + 1] = float(q_ptr[g * 64 + 32 + lane]) * p.scale;
  }

  float m_val = -INFINITY;
  float l_val = 0.0f;
  float acc[8];
  UNROLL for (uint i = 0; i < 8; i++) acc[i] = 0.0f;

  const size_t kv_stride = (size_t)p.num_kv_heads * HD;
  const size_t kh_off = (size_t)kv_head * HD;

  for (uint t = t0 + sg; t < t1; t += 4) {
    const device bfloat* kp = k_cache + (size_t)t * kv_stride + kh_off;
    float dot_part = 0.0f;
    UNROLL for (uint g = 0; g < 4; g++) {
      dot_part = fma(q[g * 2], float(kp[g * 64 + lane]), dot_part);
      dot_part = fma(q[g * 2 + 1], float(kp[g * 64 + 32 + lane]), dot_part);
    }
    const float score = simd_sum(dot_part);
    const float new_m = max(m_val, score);
    const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
    const float prob = exp(score - new_m);
    l_val = fma(l_val, alpha, prob);
    m_val = new_m;

    const device bfloat* vp = v_cache + (size_t)t * kv_stride + kh_off;
    UNROLL for (uint g = 0; g < 4; g++) {
      acc[g * 2] = fma(acc[g * 2], alpha, prob * float(vp[g * 64 + lane]));
      acc[g * 2 + 1] = fma(acc[g * 2 + 1], alpha, prob * float(vp[g * 64 + 32 + lane]));
    }
  }

  threadgroup float tg_m[4];
  threadgroup float tg_l[4];
  threadgroup float tg_acc[4][HD];

  if (lane == 0) {
    tg_m[sg] = m_val;
    tg_l[sg] = l_val;
  }
  UNROLL for (uint g = 0; g < 4; g++) {
    tg_acc[sg][g * 64 + lane] = acc[g * 2];
    tg_acc[sg][g * 64 + 32 + lane] = acc[g * 2 + 1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;

  float gm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
  float w[4];
  float gl = 0.0f;
  UNROLL for (uint s = 0; s < 4; s++) {
    w[s] = (tg_l[s] > 0.0f) ? exp(tg_m[s] - gm) : 0.0f;
    gl += tg_l[s] * w[s];
  }

  const size_t slot = ((size_t)row * p.num_heads + head) * p.splits + split;
  if (lane == 0) partial_ml[slot] = float2(gm, gl);
  device float* dst = partial_acc + slot * HD;
  UNROLL for (uint g = 0; g < 4; g++) {
    const uint i0 = g * 64 + lane;
    const uint i1 = g * 64 + 32 + lane;
    float o0 = 0.0f, o1 = 0.0f;
    UNROLL for (uint s = 0; s < 4; s++) {
      o0 = fma(tg_acc[s][i0], w[s], o0);
      o1 = fma(tg_acc[s][i1], w[s], o1);
    }
    dst[i0] = o0;
    dst[i1] = o1;
  }
}

// Split-K Stage 2: merge splits + sigmoid(gate) + o_proj group sums.
// Grid: threadgroups (num_heads, rows) of 32 threads.
kernel void attn_merge_gate_sums(const device float* partial_acc [[buffer(0)]],
                                 const device float2* partial_ml [[buffer(1)]],
                                 const device bfloat* qkv_buf [[buffer(2)]],
                                 device bfloat* out [[buffer(3)]],
                                 device float* sums [[buffer(4)]],
                                 constant AttnFlashParams& p [[buffer(5)]],
                                 uint2 grp [[threadgroup_position_in_grid]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
  const uint head = grp.x;
  const uint row = grp.y;
  if (head >= p.num_heads || row >= p.rows) return;

  const size_t base_slot = ((size_t)row * p.num_heads + head) * p.splits;
  float gm = -INFINITY;
  for (uint s = 0; s < p.splits; s++) {
    const float2 ml = partial_ml[base_slot + s];
    if (ml.y > 0.0f) gm = max(gm, ml.x);
  }
  float gl = 0.0f;
  float acc[8];
  UNROLL for (uint i = 0; i < 8; i++) acc[i] = 0.0f;

  for (uint s = 0; s < p.splits; s++) {
    const float2 ml = partial_ml[base_slot + s];
    if (ml.y <= 0.0f) continue;
    const float w = exp(ml.x - gm);
    gl = fma(ml.y, w, gl);
    const device float* pa = partial_acc + (base_slot + s) * HD;
    UNROLL for (uint g = 0; g < 4; g++) {
      acc[g * 2] = fma(pa[g * 64 + lane], w, acc[g * 2]);
      acc[g * 2 + 1] = fma(pa[g * 64 + 32 + lane], w, acc[g * 2 + 1]);
    }
  }
  const float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

  const device bfloat* gate_ptr = qkv_buf + (size_t)row * p.qkv_stride + (size_t)head * (2 * HD) + HD;
  device bfloat* out_ptr = out + (size_t)(row * p.num_heads + head) * HD;

  UNROLL for (uint g = 0; g < 4; g++) {
    const uint i0 = g * 64 + lane;
    const uint i1 = g * 64 + 32 + lane;
    const float g0 = 1.0f / (1.0f + exp(-float(gate_ptr[i0])));
    const float g1 = 1.0f / (1.0f + exp(-float(gate_ptr[i1])));
    const bfloat y0 = bfloat((acc[g * 2] * inv_l) * g0);
    const bfloat y1 = bfloat((acc[g * 2 + 1] * inv_l) * g1);
    out_ptr[i0] = y0;
    out_ptr[i1] = y1;
    const float gs = simd_sum(float(y0) + float(y1));
    if (lane == 0) {
      sums[(size_t)(head * 4 + g) * p.sums_stride + row] = gs;
    }
  }
}


struct KvCompactParams {
  uint base_pos;
  uint keep_len;
  uint row_u4; // (num_kv_heads * 256 * 2) / 16 = num_kv_heads * 32
};

// Stage 1: Gather accepted non-contiguous tree KV rows into contiguous tmp buffer.
// Grid: threads (row_u4, keep_len).
kernel void kv_compact_gather(const device uint4* k_cache [[buffer(0)]],
                              const device uint4* v_cache [[buffer(1)]],
                              const device uint* keep_rows [[buffer(2)]],
                              device uint4* k_tmp [[buffer(3)]],
                              device uint4* v_tmp [[buffer(4)]],
                              constant KvCompactParams& p [[buffer(5)]],
                              uint2 tid [[thread_position_in_grid]]) {
  if (tid.x >= p.row_u4 || tid.y >= p.keep_len) return;
  const uint src_slot = p.base_pos + keep_rows[tid.y];
  const size_t src_off = (size_t)src_slot * p.row_u4 + tid.x;
  const size_t dst_off = (size_t)tid.y * p.row_u4 + tid.x;
  k_tmp[dst_off] = k_cache[src_off];
  v_tmp[dst_off] = v_cache[src_off];
}

// Stage 2: Write contiguous tmp rows back to k_cache/v_cache at [base_pos .. base_pos + keep_len).
kernel void kv_compact_scatter(const device uint4* k_tmp [[buffer(0)]],
                               const device uint4* v_tmp [[buffer(1)]],
                               device uint4* k_cache [[buffer(2)]],
                               device uint4* v_cache [[buffer(3)]],
                               constant KvCompactParams& p [[buffer(4)]],
                               uint2 tid [[thread_position_in_grid]]) {
  if (tid.x >= p.row_u4 || tid.y >= p.keep_len) return;
  const size_t src_off = (size_t)tid.y * p.row_u4 + tid.x;
  const size_t dst_off = (size_t)(p.base_pos + tid.y) * p.row_u4 + tid.x;
  k_cache[dst_off] = k_tmp[src_off];
  v_cache[dst_off] = v_tmp[src_off];
}

struct ForkPrepParams {
  uint rows;
  uint num_heads;
  uint num_kv_heads;
  uint qkv_stride;
  uint suffix_stream_stride; // stride in bfloat elements between streams' suffix KV buffers
  float eps;
};

// Forked Q/K RMSNorm + partial RoPE + store K/V into per-stream suffix KV buffer.
// Works for both per-stream delta prefill (all rows same stream) and lockstep B-stream decode.
// Grid: threadgroups (num_heads + num_kv_heads, rows) of 32 threads.
kernel void attn_qk_rope_store_forked(const device bfloat* qkv_buf [[buffer(0)]],
                                      const device bfloat* q_norm [[buffer(1)]],
                                      const device bfloat* k_norm [[buffer(2)]],
                                      const device float* inv_freq [[buffer(3)]],
                                      const device uint* positions [[buffer(4)]],
                                      const device uint* suffix_slots [[buffer(5)]],
                                      const device uint* row_streams [[buffer(6)]],
                                      device bfloat* q_out [[buffer(7)]],
                                      device bfloat* k_suffix [[buffer(8)]],
                                      device bfloat* v_suffix [[buffer(9)]],
                                      constant ForkPrepParams& p [[buffer(10)]],
                                      uint2 grp [[threadgroup_position_in_grid]],
                                      ushort lane [[thread_index_in_simdgroup]]) {
  const uint idx = grp.x;
  const uint row = grp.y;
  if (row >= p.rows) return;

  const device bfloat* row_ptr = qkv_buf + (size_t)row * p.qkv_stride;
  const float pos = float(positions[row]);
  const float angle = pos * inv_freq[lane];
  const float c = cos(angle);
  const float s = sin(angle);

  if (idx < p.num_heads) {
    const device bfloat* src = row_ptr + (size_t)idx * (2 * HD);
    float v[8];
    float sq = 0.0f;
    UNROLL for (uint g = 0; g < 4; g++) {
      const float a = float(src[g * 64 + lane]);
      const float b = float(src[g * 64 + 32 + lane]);
      v[g * 2] = a;
      v[g * 2 + 1] = b;
      sq = fma(a, a, fma(b, b, sq));
    }
    const float inv = rsqrt(simd_sum(sq) / float(HD) + p.eps);
    UNROLL for (uint g = 0; g < 4; g++) {
      v[g * 2] = v[g * 2] * inv * float(q_norm[g * 64 + lane]);
      v[g * 2 + 1] = v[g * 2 + 1] * inv * float(q_norm[g * 64 + 32 + lane]);
    }
    const float r0 = v[0] * c - v[1] * s;
    const float r1 = v[1] * c + v[0] * s;
    v[0] = r0;
    v[1] = r1;

    device bfloat* dst = q_out + (size_t)(row * p.num_heads + idx) * HD;
    UNROLL for (uint g = 0; g < 4; g++) {
      dst[g * 64 + lane] = bfloat(v[g * 2]);
      dst[g * 64 + 32 + lane] = bfloat(v[g * 2 + 1]);
    }
    return;
  }

  const uint kh = idx - p.num_heads;
  const uint sid = row_streams[row];
  const uint slot = suffix_slots[row];
  const size_t k_off = (size_t)(2 * p.num_heads) * HD + (size_t)kh * HD;
  const size_t v_off = k_off + (size_t)p.num_kv_heads * HD;
  const device bfloat* k_src = row_ptr + k_off;
  const device bfloat* v_src = row_ptr + v_off;

  float kv[8];
  float sq = 0.0f;
  UNROLL for (uint g = 0; g < 4; g++) {
    const float a = float(k_src[g * 64 + lane]);
    const float b = float(k_src[g * 64 + 32 + lane]);
    kv[g * 2] = a;
    kv[g * 2 + 1] = b;
    sq = fma(a, a, fma(b, b, sq));
  }
  const float inv = rsqrt(simd_sum(sq) / float(HD) + p.eps);
  UNROLL for (uint g = 0; g < 4; g++) {
    kv[g * 2] = kv[g * 2] * inv * float(k_norm[g * 64 + lane]);
    kv[g * 2 + 1] = kv[g * 2 + 1] * inv * float(k_norm[g * 64 + 32 + lane]);
  }
  const float rk0 = kv[0] * c - kv[1] * s;
  const float rk1 = kv[1] * c + kv[0] * s;
  kv[0] = rk0;
  kv[1] = rk1;

  const size_t cache_base = (size_t)sid * p.suffix_stream_stride + (size_t)(slot * p.num_kv_heads + kh) * HD;
  device bfloat* k_dst = k_suffix + cache_base;
  device bfloat* v_dst = v_suffix + cache_base;
  UNROLL for (uint g = 0; g < 4; g++) {
    k_dst[g * 64 + lane] = bfloat(kv[g * 2]);
    k_dst[g * 64 + 32 + lane] = bfloat(kv[g * 2 + 1]);
    v_dst[g * 64 + lane] = v_src[g * 64 + lane];
    v_dst[g * 64 + 32 + lane] = v_src[g * 64 + 32 + lane];
  }
}

struct ForkFlashParams {
  uint rows;
  uint num_heads;
  uint num_kv_heads;
  uint qkv_stride;
  uint sums_stride;
  uint prefix_len;
  uint suffix_stream_stride;
  float scale;
};

// Zero-copy forked Flash GQA: attends to read-only shared prefix KV [0..prefix_len) PLUS
// per-stream suffix KV [0..suffix_lens[row]) with fused sigmoid(gate) + o_proj group sums.
// Grid: threadgroups (num_heads, rows) of 128 threads.
kernel void attn_flash_forked(const device bfloat* q_in [[buffer(0)]],
                              const device bfloat* qkv_buf [[buffer(1)]],
                              const device bfloat* k_prefix [[buffer(2)]],
                              const device bfloat* v_prefix [[buffer(3)]],
                              const device bfloat* k_suffix [[buffer(4)]],
                              const device bfloat* v_suffix [[buffer(5)]],
                              const device uint* suffix_lens [[buffer(6)]],
                              const device uint* row_streams [[buffer(7)]],
                              device bfloat* out [[buffer(8)]],
                              device float* sums [[buffer(9)]],
                              constant ForkFlashParams& p [[buffer(10)]],
                              uint2 grp [[threadgroup_position_in_grid]],
                              ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
  const uint head = grp.x;
  const uint row = grp.y;
  if (head >= p.num_heads || row >= p.rows) return;

  const uint sid = row_streams[row];
  const uint kv_head = head / (p.num_heads / p.num_kv_heads);
  const uint s_len = suffix_lens[row];
  const uint total_len = p.prefix_len + s_len;
  const device bfloat* q_ptr = q_in + (size_t)(row * p.num_heads + head) * HD;

  float q[8];
  UNROLL for (uint g = 0; g < 4; g++) {
    q[g * 2] = float(q_ptr[g * 64 + lane]) * p.scale;
    q[g * 2 + 1] = float(q_ptr[g * 64 + 32 + lane]) * p.scale;
  }

  float m_val = -INFINITY;
  float l_val = 0.0f;
  float acc[8];
  UNROLL for (uint i = 0; i < 8; i++) acc[i] = 0.0f;

  const size_t kv_stride = (size_t)p.num_kv_heads * HD;
  const size_t kh_off = (size_t)kv_head * HD;
  const device bfloat* ks_base = k_suffix + (size_t)sid * p.suffix_stream_stride;
  const device bfloat* vs_base = v_suffix + (size_t)sid * p.suffix_stream_stride;

  for (uint t = sg; t < total_len; t += 4) {
    const device bfloat* kp;
    const device bfloat* vp;
    if (t < p.prefix_len) {
      kp = k_prefix + (size_t)t * kv_stride + kh_off;
      vp = v_prefix + (size_t)t * kv_stride + kh_off;
    } else {
      const uint st = t - p.prefix_len;
      kp = ks_base + (size_t)st * kv_stride + kh_off;
      vp = vs_base + (size_t)st * kv_stride + kh_off;
    }
    float dot_part = 0.0f;
    UNROLL for (uint g = 0; g < 4; g++) {
      dot_part = fma(q[g * 2], float(kp[g * 64 + lane]), dot_part);
      dot_part = fma(q[g * 2 + 1], float(kp[g * 64 + 32 + lane]), dot_part);
    }
    const float score = simd_sum(dot_part);
    const float new_m = max(m_val, score);
    const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
    const float prob = exp(score - new_m);
    l_val = fma(l_val, alpha, prob);
    m_val = new_m;

    UNROLL for (uint g = 0; g < 4; g++) {
      acc[g * 2] = fma(acc[g * 2], alpha, prob * float(vp[g * 64 + lane]));
      acc[g * 2 + 1] = fma(acc[g * 2 + 1], alpha, prob * float(vp[g * 64 + 32 + lane]));
    }
  }

  threadgroup float tg_m[4];
  threadgroup float tg_l[4];
  threadgroup float tg_acc[4][HD];

  if (lane == 0) {
    tg_m[sg] = m_val;
    tg_l[sg] = l_val;
  }
  UNROLL for (uint g = 0; g < 4; g++) {
    tg_acc[sg][g * 64 + lane] = acc[g * 2];
    tg_acc[sg][g * 64 + 32 + lane] = acc[g * 2 + 1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;

  const float gm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
  float w[4];
  float gl = 0.0f;
  UNROLL for (uint s = 0; s < 4; s++) {
    w[s] = (tg_l[s] > 0.0f) ? exp(tg_m[s] - gm) : 0.0f;
    gl += tg_l[s] * w[s];
  }
  const float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

  const device bfloat* gate_ptr = qkv_buf + (size_t)row * p.qkv_stride + (size_t)head * (2 * HD) + HD;
  device bfloat* out_ptr = out + (size_t)(row * p.num_heads + head) * HD;

  UNROLL for (uint g = 0; g < 4; g++) {
    const uint i0 = g * 64 + lane;
    const uint i1 = g * 64 + 32 + lane;
    float o0 = 0.0f, o1 = 0.0f;
    UNROLL for (uint s = 0; s < 4; s++) {
      o0 = fma(tg_acc[s][i0], w[s], o0);
      o1 = fma(tg_acc[s][i1], w[s], o1);
    }
    o0 *= inv_l;
    o1 *= inv_l;
    const float g0 = 1.0f / (1.0f + exp(-float(gate_ptr[i0])));
    const float g1 = 1.0f / (1.0f + exp(-float(gate_ptr[i1])));
    const bfloat y0 = bfloat(o0 * g0);
    const bfloat y1 = bfloat(o1 * g1);
    out_ptr[i0] = y0;
    out_ptr[i1] = y1;
    const float gs = simd_sum(float(y0) + float(y1));
    if (lane == 0) {
      sums[(size_t)(head * 4 + g) * p.sums_stride + row] = gs;
    }
  }
}

// ============================================================================
// Standard Dense Transformer GQA kernels (head_dim = 128, full split-half RoPE,
// optional QKV bias, no Q/K norm, no output gate) — Qwen2 / Qwen2.5 / Llama.
// ============================================================================

struct Std128PrepParams {
  uint rows;
  uint num_heads;
  uint num_kv_heads;
  uint qkv_stride;
  uint has_bias;
  uint use_rope;
  uint has_qk_norm;
  float eps;
};

// Grid: threadgroups (num_heads + num_kv_heads, rows) of 32 threads.
kernel void attn_std128_rope_store(const device bfloat* qkv_buf [[buffer(0)]],
                                   const device bfloat* qkv_bias [[buffer(1)]],
                                   const device float* inv_freq [[buffer(2)]],
                                   const device uint* positions [[buffer(3)]],
                                   const device uint* kv_slots [[buffer(4)]],
                                   device bfloat* q_out [[buffer(5)]],
                                   device bfloat* k_cache [[buffer(6)]],
                                   device bfloat* v_cache [[buffer(7)]],
                                   constant Std128PrepParams& p [[buffer(8)]],
                                   const device bfloat* q_norm [[buffer(9)]],
                                   const device bfloat* k_norm [[buffer(10)]],
                                   uint2 grp [[threadgroup_position_in_grid]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint H128 = 128;
  const uint idx = grp.x;
  const uint row = grp.y;
  if (row >= p.rows) return;

  const device bfloat* row_ptr = qkv_buf + (size_t)row * p.qkv_stride;
  float c0 = 1.0f, s0 = 0.0f, c1 = 1.0f, s1 = 0.0f;
  if (p.use_rope != 0) {
    const float pos = float(positions[row]);
    const float a0 = pos * inv_freq[lane];
    const float a1 = pos * inv_freq[lane + 32];
    c0 = cos(a0);
    s0 = sin(a0);
    c1 = cos(a1);
    s1 = sin(a1);
  }

  if (idx < p.num_heads) {
    const size_t q_off = (size_t)idx * H128;
    const device bfloat* src = row_ptr + q_off;
    float x0 = float(src[lane]);
    float x1 = float(src[lane + 32]);
    float x2 = float(src[lane + 64]);
    float x3 = float(src[lane + 96]);
    if (p.has_bias != 0) {
      const device bfloat* bs = qkv_bias + q_off;
      x0 += float(bs[lane]);
      x1 += float(bs[lane + 32]);
      x2 += float(bs[lane + 64]);
      x3 += float(bs[lane + 96]);
    }
    if (p.has_qk_norm != 0) {
      const float sq = fma(x0, x0, fma(x1, x1, fma(x2, x2, x3 * x3)));
      const float inv = rsqrt(simd_sum(sq) / float(H128) + p.eps);
      x0 = x0 * inv * float(q_norm[lane]);
      x1 = x1 * inv * float(q_norm[lane + 32]);
      x2 = x2 * inv * float(q_norm[lane + 64]);
      x3 = x3 * inv * float(q_norm[lane + 96]);
    }
    device bfloat* dst = q_out + (size_t)(row * p.num_heads + idx) * H128;
    dst[lane] = bfloat(x0 * c0 - x2 * s0);
    dst[lane + 64] = bfloat(x2 * c0 + x0 * s0);
    dst[lane + 32] = bfloat(x1 * c1 - x3 * s1);
    dst[lane + 96] = bfloat(x3 * c1 + x1 * s1);
    return;
  }

  const uint kh = idx - p.num_heads;
  const uint slot = kv_slots[row];
  const size_t k_off = (size_t)p.num_heads * H128 + (size_t)kh * H128;
  const size_t v_off = k_off + (size_t)p.num_kv_heads * H128;
  const device bfloat* k_src = row_ptr + k_off;
  const device bfloat* v_src = row_ptr + v_off;

  float k0 = float(k_src[lane]);
  float k1 = float(k_src[lane + 32]);
  float k2 = float(k_src[lane + 64]);
  float k3 = float(k_src[lane + 96]);
  float v0 = float(v_src[lane]);
  float v1 = float(v_src[lane + 32]);
  float v2 = float(v_src[lane + 64]);
  float v3 = float(v_src[lane + 96]);
  if (p.has_bias != 0) {
    const device bfloat* kb = qkv_bias + k_off;
    const device bfloat* vb = qkv_bias + v_off;
    k0 += float(kb[lane]);
    k1 += float(kb[lane + 32]);
    k2 += float(kb[lane + 64]);
    k3 += float(kb[lane + 96]);
    v0 += float(vb[lane]);
    v1 += float(vb[lane + 32]);
    v2 += float(vb[lane + 64]);
    v3 += float(vb[lane + 96]);
  }
  if (p.has_qk_norm != 0) {
    const float sq = fma(k0, k0, fma(k1, k1, fma(k2, k2, k3 * k3)));
    const float inv = rsqrt(simd_sum(sq) / float(H128) + p.eps);
    k0 = k0 * inv * float(k_norm[lane]);
    k1 = k1 * inv * float(k_norm[lane + 32]);
    k2 = k2 * inv * float(k_norm[lane + 64]);
    k3 = k3 * inv * float(k_norm[lane + 96]);
  }

  const size_t cache_base = (size_t)(slot * p.num_kv_heads + kh) * H128;
  device bfloat* k_dst = k_cache + cache_base;
  device bfloat* v_dst = v_cache + cache_base;
  k_dst[lane] = bfloat(k0 * c0 - k2 * s0);
  k_dst[lane + 64] = bfloat(k2 * c0 + k0 * s0);
  k_dst[lane + 32] = bfloat(k1 * c1 - k3 * s1);
  k_dst[lane + 96] = bfloat(k3 * c1 + k1 * s1);
  v_dst[lane] = bfloat(v0);
  v_dst[lane + 32] = bfloat(v1);
  v_dst[lane + 64] = bfloat(v2);
  v_dst[lane + 96] = bfloat(v3);
}

// Fused HD=128 Flash GQA + o_proj group sums (supports causal chain + bit-exact tree).
// Grid: threadgroups (num_heads, rows) of 128 threads (4 simdgroups).
kernel void attn_std128_flash(const device bfloat* q_in [[buffer(0)]],
                              const device bfloat* k_cache [[buffer(1)]],
                              const device bfloat* v_cache [[buffer(2)]],
                              const device uint* seq_lens [[buffer(3)]],
                              const device uint4* tree_masks [[buffer(4)]],
                              device bfloat* out [[buffer(5)]],
                              device float* sums [[buffer(6)]],
                              constant AttnFlashParams& p [[buffer(7)]],
                              uint2 grp [[threadgroup_position_in_grid]],
                              ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint H128 = 128;
  const uint head = grp.x;
  const uint row = grp.y;
  if (head >= p.num_heads || row >= p.rows) return;

  const uint kv_head = head / (p.num_heads / p.num_kv_heads);
  const device bfloat* q_ptr = q_in + (size_t)(row * p.num_heads + head) * H128;

  const float q0 = float(q_ptr[lane]) * p.scale;
  const float q1 = float(q_ptr[32 + lane]) * p.scale;
  const float q2 = float(q_ptr[64 + lane]) * p.scale;
  const float q3 = float(q_ptr[96 + lane]) * p.scale;

  float m_val = -INFINITY;
  float l_val = 0.0f;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

  const size_t kv_stride = (size_t)p.num_kv_heads * H128;
  const size_t kh_off = (size_t)kv_head * H128;
  const uint total_seq = seq_lens[row];
  const uint prefix_end = (p.is_tree != 0) ? p.base_pos : total_seq;
  const uint sw_start = (p.sliding_window != 0 && total_seq > p.sliding_window) ? (total_seq - p.sliding_window) : 0u;
  uint t_first = sg;
  if (t_first < sw_start) t_first += ((sw_start - t_first + 3u) & ~3u);

  for (uint t = t_first; t < prefix_end; t += 4) {
    const device bfloat* kp = k_cache + (size_t)t * kv_stride + kh_off;
    const float dot_part = fma(q0, float(kp[lane]), fma(q1, float(kp[32 + lane]), fma(q2, float(kp[64 + lane]), q3 * float(kp[96 + lane]))));
    const float score = simd_sum(dot_part);
    const float new_m = max(m_val, score);
    const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
    const float prob = exp(score - new_m);
    l_val = fma(l_val, alpha, prob);
    m_val = new_m;

    const device bfloat* vp = v_cache + (size_t)t * kv_stride + kh_off;
    a0 = fma(a0, alpha, prob * float(vp[lane]));
    a1 = fma(a1, alpha, prob * float(vp[32 + lane]));
    a2 = fma(a2, alpha, prob * float(vp[64 + lane]));
    a3 = fma(a3, alpha, prob * float(vp[96 + lane]));
  }

  if (p.is_tree != 0) {
    const uint4 mask = tree_masks[row];
    uint anc_idx = 0;
    for (uint r = 0; r <= row; r++) {
      if ((mask[r >> 5] & (1u << (r & 31))) == 0u) continue;
      if (((p.base_pos + anc_idx) & 3u) == sg) {
        const uint t = p.base_pos + r;
        const device bfloat* kp = k_cache + (size_t)t * kv_stride + kh_off;
        const float dot_part = fma(q0, float(kp[lane]), fma(q1, float(kp[32 + lane]), fma(q2, float(kp[64 + lane]), q3 * float(kp[96 + lane]))));
        const float score = simd_sum(dot_part);
        const float new_m = max(m_val, score);
        const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
        const float prob = exp(score - new_m);
        l_val = fma(l_val, alpha, prob);
        m_val = new_m;

        const device bfloat* vp = v_cache + (size_t)t * kv_stride + kh_off;
        a0 = fma(a0, alpha, prob * float(vp[lane]));
        a1 = fma(a1, alpha, prob * float(vp[32 + lane]));
        a2 = fma(a2, alpha, prob * float(vp[64 + lane]));
        a3 = fma(a3, alpha, prob * float(vp[96 + lane]));
      }
      anc_idx++;
    }
  }

  threadgroup float tg_m[4];
  threadgroup float tg_l[4];
  threadgroup float tg_acc[4][H128];

  if (lane == 0) {
    tg_m[sg] = m_val;
    tg_l[sg] = l_val;
  }
  tg_acc[sg][lane] = a0;
  tg_acc[sg][32 + lane] = a1;
  tg_acc[sg][64 + lane] = a2;
  tg_acc[sg][96 + lane] = a3;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;

  const float gm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
  float w[4];
  float gl = 0.0f;
  UNROLL for (uint s = 0; s < 4; s++) {
    w[s] = (tg_l[s] > 0.0f) ? exp(tg_m[s] - gm) : 0.0f;
    gl += tg_l[s] * w[s];
  }
  const float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

  device bfloat* out_ptr = out + (size_t)(row * p.num_heads + head) * H128;
  UNROLL for (uint g = 0; g < 2; g++) {
    const uint i0 = g * 64 + lane;
    const uint i1 = g * 64 + 32 + lane;
    float o0 = 0.0f, o1 = 0.0f;
    UNROLL for (uint s = 0; s < 4; s++) {
      o0 = fma(tg_acc[s][i0], w[s], o0);
      o1 = fma(tg_acc[s][i1], w[s], o1);
    }
    const bfloat y0 = bfloat(o0 * inv_l);
    const bfloat y1 = bfloat(o1 * inv_l);
    out_ptr[i0] = y0;
    out_ptr[i1] = y1;
    if (p.sums_stride != 0) {
      const float gs = simd_sum(float(y0) + float(y1));
      if (lane == 0) {
        sums[(size_t)(head * 2 + g) * p.sums_stride + row] = gs;
      }
    }
  }
}

struct Std128Decode1Params {
  uint num_heads;
  uint num_kv_heads;
  uint qkv_stride;
  uint has_bias;
  uint use_rope;
  uint pos;
  float scale;
  uint has_qk_norm;
  uint sliding_window;
  float eps;
};

// Super-fused single-token HD=128 GQA decode: Bias + optional Q/K norm + RoPE + KV-cache store + Flash GQA in ONE dispatch!
// Eliminates attn_std128_rope_store, global barrier, and attn_q_buf round-trip on every layer.
// Grid: threadgroups (num_heads, 1, 1) of 128 threads (4 simdgroups).
kernel void attn_std128_decode_1(const device bfloat* qkv_buf [[buffer(0)]],
                                 const device bfloat* qkv_bias [[buffer(1)]],
                                 const device float* inv_freq [[buffer(2)]],
                                 device bfloat* k_cache [[buffer(3)]],
                                 device bfloat* v_cache [[buffer(4)]],
                                 device bfloat* out [[buffer(5)]],
                                 constant Std128Decode1Params& p [[buffer(6)]],
                                 const device bfloat* q_norm [[buffer(7)]],
                                 const device bfloat* k_norm [[buffer(8)]],
                                 uint head [[threadgroup_position_in_grid]],
                                 ushort sg [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint H128 = 128;
  if (head >= p.num_heads) return;

  const uint gqa = p.num_heads / p.num_kv_heads;
  const uint kv_head = head / gqa;
  const bool write_kv = (head % gqa) == 0;
  const uint pos = p.pos;

  threadgroup float tg_q[H128];
  threadgroup bfloat tg_k_new[H128];
  threadgroup bfloat tg_v_new[H128];

  if (sg < 2) {
    float c0 = 1.0f, s0 = 0.0f, c1 = 1.0f, s1 = 0.0f;
    if (p.use_rope != 0) {
      const float fpos = float(pos);
      const float a0 = fpos * inv_freq[lane];
      const float a1 = fpos * inv_freq[lane + 32];
      c0 = cos(a0); s0 = sin(a0);
      c1 = cos(a1); s1 = sin(a1);
    }
    if (sg == 0) {
      const size_t q_off = (size_t)head * H128;
      const device bfloat* src = qkv_buf + q_off;
      float x0 = float(src[lane]);
      float x1 = float(src[lane + 32]);
      float x2 = float(src[lane + 64]);
      float x3 = float(src[lane + 96]);
      if (p.has_bias != 0) {
        const device bfloat* bs = qkv_bias + q_off;
        x0 += float(bs[lane]);
        x1 += float(bs[lane + 32]);
        x2 += float(bs[lane + 64]);
        x3 += float(bs[lane + 96]);
      }
      if (p.has_qk_norm != 0) {
        const float sq = fma(x0, x0, fma(x1, x1, fma(x2, x2, x3 * x3)));
        const float inv = rsqrt(simd_sum(sq) / float(H128) + p.eps);
        x0 = x0 * inv * float(q_norm[lane]);
        x1 = x1 * inv * float(q_norm[lane + 32]);
        x2 = x2 * inv * float(q_norm[lane + 64]);
        x3 = x3 * inv * float(q_norm[lane + 96]);
      }
      tg_q[lane] = float(bfloat(x0 * c0 - x2 * s0)) * p.scale;
      tg_q[lane + 64] = float(bfloat(x2 * c0 + x0 * s0)) * p.scale;
      tg_q[lane + 32] = float(bfloat(x1 * c1 - x3 * s1)) * p.scale;
      tg_q[lane + 96] = float(bfloat(x3 * c1 + x1 * s1)) * p.scale;
    } else {
      const size_t k_off = (size_t)p.num_heads * H128 + (size_t)kv_head * H128;
      const size_t v_off = k_off + (size_t)p.num_kv_heads * H128;
      const device bfloat* k_src = qkv_buf + k_off;
      const device bfloat* v_src = qkv_buf + v_off;
      float k0 = float(k_src[lane]);
      float k1 = float(k_src[lane + 32]);
      float k2 = float(k_src[lane + 64]);
      float k3 = float(k_src[lane + 96]);
      float v0 = float(v_src[lane]);
      float v1 = float(v_src[lane + 32]);
      float v2 = float(v_src[lane + 64]);
      float v3 = float(v_src[lane + 96]);
      if (p.has_bias != 0) {
        const device bfloat* kb = qkv_bias + k_off;
        const device bfloat* vb = qkv_bias + v_off;
        k0 += float(kb[lane]); k1 += float(kb[lane + 32]);
        k2 += float(kb[lane + 64]); k3 += float(kb[lane + 96]);
        v0 += float(vb[lane]); v1 += float(vb[lane + 32]);
        v2 += float(vb[lane + 64]); v3 += float(vb[lane + 96]);
      }
      if (p.has_qk_norm != 0) {
        const float sq = fma(k0, k0, fma(k1, k1, fma(k2, k2, k3 * k3)));
        const float inv = rsqrt(simd_sum(sq) / float(H128) + p.eps);
        k0 = k0 * inv * float(k_norm[lane]);
        k1 = k1 * inv * float(k_norm[lane + 32]);
        k2 = k2 * inv * float(k_norm[lane + 64]);
        k3 = k3 * inv * float(k_norm[lane + 96]);
      }
      const bfloat bk0 = bfloat(k0 * c0 - k2 * s0);
      const bfloat bk2 = bfloat(k2 * c0 + k0 * s0);
      const bfloat bk1 = bfloat(k1 * c1 - k3 * s1);
      const bfloat bk3 = bfloat(k3 * c1 + k1 * s1);
      const bfloat bv0 = bfloat(v0);
      const bfloat bv1 = bfloat(v1);
      const bfloat bv2 = bfloat(v2);
      const bfloat bv3 = bfloat(v3);

      tg_k_new[lane] = bk0; tg_k_new[lane + 32] = bk1;
      tg_k_new[lane + 64] = bk2; tg_k_new[lane + 96] = bk3;
      tg_v_new[lane] = bv0; tg_v_new[lane + 32] = bv1;
      tg_v_new[lane + 64] = bv2; tg_v_new[lane + 96] = bv3;

      if (write_kv) {
        const size_t cache_base = (size_t)(pos * p.num_kv_heads + kv_head) * H128;
        device bfloat* k_dst = k_cache + cache_base;
        device bfloat* v_dst = v_cache + cache_base;
        k_dst[lane] = bk0; k_dst[lane + 32] = bk1;
        k_dst[lane + 64] = bk2; k_dst[lane + 96] = bk3;
        v_dst[lane] = bv0; v_dst[lane + 32] = bv1;
        v_dst[lane + 64] = bv2; v_dst[lane + 96] = bv3;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float4 qv = *(const threadgroup float4*)(&tg_q[lane * 4]);

  float m_val = -INFINITY;
  float l_val = 0.0f;
  float4 acc4 = float4(0.0f);

  const size_t kv_stride = (size_t)p.num_kv_heads * H128;
  const size_t kh_off = (size_t)kv_head * H128 + (size_t)lane * 4;
  const uint sw_start = (p.sliding_window != 0 && (pos + 1u) > p.sliding_window) ? (pos + 1u - p.sliding_window) : 0u;
  uint t_first = sg;
  if (t_first < sw_start) t_first += ((sw_start - t_first + 3u) & ~3u);

  for (uint t = t_first; t < pos; t += 4) {
    const float4 kv = float4(*(const device vec<bfloat, 4>*)(k_cache + (size_t)t * kv_stride + kh_off));
    const float score = simd_sum(dot(qv, kv));
    const float new_m = max(m_val, score);
    const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
    const float prob = exp(score - new_m);
    l_val = fma(l_val, alpha, prob);
    m_val = new_m;

    const float4 vv = float4(*(const device vec<bfloat, 4>*)(v_cache + (size_t)t * kv_stride + kh_off));
    acc4 = fma(acc4, float4(alpha), vv * prob);
  }

  // Handle newly appended token at t == pos directly from TG memory (bit-exact sg assignment!)
  if ((pos & 3u) == sg) {
    const float4 kv = float4(*(const threadgroup vec<bfloat, 4>*)(&tg_k_new[lane * 4]));
    const float score = simd_sum(dot(qv, kv));
    const float new_m = max(m_val, score);
    const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
    const float prob = exp(score - new_m);
    l_val = fma(l_val, alpha, prob);
    m_val = new_m;

    const float4 vv = float4(*(const threadgroup vec<bfloat, 4>*)(&tg_v_new[lane * 4]));
    acc4 = fma(acc4, float4(alpha), vv * prob);
  }

  threadgroup float tg_m[4];
  threadgroup float tg_l[4];
  threadgroup float tg_acc[4][H128];

  if (lane == 0) {
    tg_m[sg] = m_val;
    tg_l[sg] = l_val;
  }
  *(threadgroup float4*)(&tg_acc[sg][lane * 4]) = acc4;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;

  const float gm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
  float w[4];
  float gl = 0.0f;
  UNROLL for (uint s = 0; s < 4; s++) {
    w[s] = (tg_l[s] > 0.0f) ? exp(tg_m[s] - gm) : 0.0f;
    gl += tg_l[s] * w[s];
  }
  const float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

  float4 o4 = float4(0.0f);
  UNROLL for (uint s = 0; s < 4; s++) {
    o4 = fma(*(const threadgroup float4*)(&tg_acc[s][lane * 4]), float4(w[s]), o4);
  }
  *(device vec<bfloat, 4>*)(out + (size_t)head * H128 + lane * 4) = vec<bfloat, 4>(o4 * inv_l);
}

struct Qwen35Decode1Params {
  uint num_heads;
  uint num_kv_heads;
  uint qkv_stride;
  uint pos;
  float eps;
  float scale;
};

// Super-fused single-token Qwen3.5/3.8 HD=256 GQA decode: Q/K RMSNorm + partial RoPE +
// KV-cache store + Flash GQA + sigmoid(gate) in ONE dispatch!
// Grid: threadgroups (num_heads, 1, 1) of 128 threads (4 simdgroups).
kernel void attn_qwen35_decode_1(const device bfloat* qkv_buf [[buffer(0)]],
                                 const device bfloat* q_norm [[buffer(1)]],
                                 const device bfloat* k_norm [[buffer(2)]],
                                 const device float* inv_freq [[buffer(3)]],
                                 device bfloat* k_cache [[buffer(4)]],
                                 device bfloat* v_cache [[buffer(5)]],
                                 device bfloat* out [[buffer(6)]],
                                 constant Qwen35Decode1Params& p [[buffer(7)]],
                                 uint head [[threadgroup_position_in_grid]],
                                 ushort sg [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
  if (head >= p.num_heads) return;

  const uint gqa = p.num_heads / p.num_kv_heads;
  const uint kv_head = head / gqa;
  const bool write_kv = (head % gqa) == 0;
  const uint pos = p.pos;

  threadgroup float tg_q[HD];
  threadgroup bfloat tg_k_new[HD];
  threadgroup bfloat tg_v_new[HD];

  if (sg < 2) {
    const float angle = float(pos) * inv_freq[lane];
    const float c = cos(angle);
    const float s = sin(angle);

    if (sg == 0) {
      const device bfloat* src = qkv_buf + (size_t)head * (2 * HD);
      float v[8];
      float sq = 0.0f;
      UNROLL for (uint g = 0; g < 4; g++) {
        const float a = float(src[g * 64 + lane]);
        const float b = float(src[g * 64 + 32 + lane]);
        v[g * 2] = a;
        v[g * 2 + 1] = b;
        sq = fma(a, a, fma(b, b, sq));
      }
      const float inv = rsqrt(simd_sum(sq) / float(HD) + p.eps);
      UNROLL for (uint g = 0; g < 4; g++) {
        v[g * 2] = v[g * 2] * inv * float(q_norm[g * 64 + lane]);
        v[g * 2 + 1] = v[g * 2 + 1] * inv * float(q_norm[g * 64 + 32 + lane]);
      }
      const float r0 = v[0] * c - v[1] * s;
      const float r1 = v[1] * c + v[0] * s;
      v[0] = r0;
      v[1] = r1;
      UNROLL for (uint g = 0; g < 4; g++) {
        tg_q[g * 64 + lane] = float(bfloat(v[g * 2])) * p.scale;
        tg_q[g * 64 + 32 + lane] = float(bfloat(v[g * 2 + 1])) * p.scale;
      }
    } else {
      const size_t k_off = (size_t)(2 * p.num_heads) * HD + (size_t)kv_head * HD;
      const size_t v_off = k_off + (size_t)p.num_kv_heads * HD;
      const device bfloat* k_src = qkv_buf + k_off;
      const device bfloat* v_src = qkv_buf + v_off;
      float kv[8];
      float sq = 0.0f;
      UNROLL for (uint g = 0; g < 4; g++) {
        const float a = float(k_src[g * 64 + lane]);
        const float b = float(k_src[g * 64 + 32 + lane]);
        kv[g * 2] = a;
        kv[g * 2 + 1] = b;
        sq = fma(a, a, fma(b, b, sq));
      }
      const float inv = rsqrt(simd_sum(sq) / float(HD) + p.eps);
      UNROLL for (uint g = 0; g < 4; g++) {
        kv[g * 2] = kv[g * 2] * inv * float(k_norm[g * 64 + lane]);
        kv[g * 2 + 1] = kv[g * 2 + 1] * inv * float(k_norm[g * 64 + 32 + lane]);
      }
      const float rk0 = kv[0] * c - kv[1] * s;
      const float rk1 = kv[1] * c + kv[0] * s;
      kv[0] = rk0;
      kv[1] = rk1;

      const size_t cache_base = (size_t)(pos * p.num_kv_heads + kv_head) * HD;
      device bfloat* k_dst = k_cache + cache_base;
      device bfloat* v_dst = v_cache + cache_base;
      UNROLL for (uint g = 0; g < 4; g++) {
        const uint i0 = g * 64 + lane;
        const uint i1 = g * 64 + 32 + lane;
        const bfloat bk0 = bfloat(kv[g * 2]);
        const bfloat bk1 = bfloat(kv[g * 2 + 1]);
        const bfloat bv0 = v_src[i0];
        const bfloat bv1 = v_src[i1];
        tg_k_new[i0] = bk0; tg_k_new[i1] = bk1;
        tg_v_new[i0] = bv0; tg_v_new[i1] = bv1;
        if (write_kv) {
          k_dst[i0] = bk0; k_dst[i1] = bk1;
          v_dst[i0] = bv0; v_dst[i1] = bv1;
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float q[8];
  UNROLL for (uint g = 0; g < 4; g++) {
    q[g * 2] = tg_q[g * 64 + lane];
    q[g * 2 + 1] = tg_q[g * 64 + 32 + lane];
  }

  float m_val = -INFINITY;
  float l_val = 0.0f;
  float acc[8];
  UNROLL for (uint i = 0; i < 8; i++) acc[i] = 0.0f;

  const size_t kv_stride = (size_t)p.num_kv_heads * HD;
  const size_t kh_off = (size_t)kv_head * HD;

  for (uint t = sg; t < pos; t += 4) {
    const device bfloat* kp = k_cache + (size_t)t * kv_stride + kh_off;
    float dot_part = 0.0f;
    UNROLL for (uint g = 0; g < 4; g++) {
      dot_part = fma(q[g * 2], float(kp[g * 64 + lane]), dot_part);
      dot_part = fma(q[g * 2 + 1], float(kp[g * 64 + 32 + lane]), dot_part);
    }
    const float score = simd_sum(dot_part);
    const float new_m = max(m_val, score);
    const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
    const float prob = exp(score - new_m);
    l_val = fma(l_val, alpha, prob);
    m_val = new_m;

    const device bfloat* vp = v_cache + (size_t)t * kv_stride + kh_off;
    UNROLL for (uint g = 0; g < 4; g++) {
      acc[g * 2] = fma(acc[g * 2], alpha, prob * float(vp[g * 64 + lane]));
      acc[g * 2 + 1] = fma(acc[g * 2 + 1], alpha, prob * float(vp[g * 64 + 32 + lane]));
    }
  }

  if ((pos & 3u) == sg) {
    float dot_part = 0.0f;
    UNROLL for (uint g = 0; g < 4; g++) {
      dot_part = fma(q[g * 2], float(tg_k_new[g * 64 + lane]), dot_part);
      dot_part = fma(q[g * 2 + 1], float(tg_k_new[g * 64 + 32 + lane]), dot_part);
    }
    const float score = simd_sum(dot_part);
    const float new_m = max(m_val, score);
    const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
    const float prob = exp(score - new_m);
    l_val = fma(l_val, alpha, prob);
    m_val = new_m;

    UNROLL for (uint g = 0; g < 4; g++) {
      acc[g * 2] = fma(acc[g * 2], alpha, prob * float(tg_v_new[g * 64 + lane]));
      acc[g * 2 + 1] = fma(acc[g * 2 + 1], alpha, prob * float(tg_v_new[g * 64 + 32 + lane]));
    }
  }

  threadgroup float tg_m[4];
  threadgroup float tg_l[4];
  threadgroup float tg_acc[4][HD];

  if (lane == 0) {
    tg_m[sg] = m_val;
    tg_l[sg] = l_val;
  }
  UNROLL for (uint g = 0; g < 4; g++) {
    tg_acc[sg][g * 64 + lane] = acc[g * 2];
    tg_acc[sg][g * 64 + 32 + lane] = acc[g * 2 + 1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;

  const float gm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
  float w[4];
  float gl = 0.0f;
  UNROLL for (uint s = 0; s < 4; s++) {
    w[s] = (tg_l[s] > 0.0f) ? exp(tg_m[s] - gm) : 0.0f;
    gl += tg_l[s] * w[s];
  }
  const float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

  const device bfloat* gate_ptr = qkv_buf + (size_t)head * (2 * HD) + HD;
  device bfloat* out_ptr = out + (size_t)head * HD;

  UNROLL for (uint g = 0; g < 4; g++) {
    const uint i0 = g * 64 + lane;
    const uint i1 = g * 64 + 32 + lane;
    float o0 = 0.0f, o1 = 0.0f;
    UNROLL for (uint s = 0; s < 4; s++) {
      o0 = fma(tg_acc[s][i0], w[s], o0);
      o1 = fma(tg_acc[s][i1], w[s], o1);
    }
    o0 *= inv_l;
    o1 *= inv_l;
    const float g0 = 1.0f / (1.0f + exp(-float(gate_ptr[i0])));
    const float g1 = 1.0f / (1.0f + exp(-float(gate_ptr[i1])));
    out_ptr[i0] = bfloat(o0 * g0);
    out_ptr[i1] = bfloat(o1 * g1);
  }
}

// Tiled FlashAttention-2 Prefill for HD=128 (works on all M1-M5 via simdgroup_float8x8!).
// 16x K/V cache reuse across BQ=16 query rows per hardware 8x8 matmul tile + fused o_proj group sums.
// Grid: threadgroups ((rows + 15) / 16, num_heads, 1) of 64 threads (2 simdgroups).
kernel void attn_std128_prefill(const device bfloat* q_in [[buffer(0)]],
                                const device bfloat* k_cache [[buffer(1)]],
                                const device bfloat* v_cache [[buffer(2)]],
                                const device uint* seq_lens [[buffer(3)]],
                                device bfloat* out [[buffer(4)]],
                                device float* sums [[buffer(5)]],
                                constant AttnFlashParams& p [[buffer(6)]],
                                uint2 grp [[threadgroup_position_in_grid]],
                                ushort sg [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint H128 = 128;
  const uint q_tile = grp.x * 16;
  const uint head = grp.y;
  if (q_tile >= p.rows || head >= p.num_heads) return;

  const uint tid = sg * 32 + lane; // 0..63
  const uint kv_head = head / (p.num_heads / p.num_kv_heads);
  const size_t kv_stride = (size_t)p.num_kv_heads * H128;
  const size_t kh_off = (size_t)kv_head * H128;

  threadgroup float tg_q[16][H128];
  threadgroup float tg_kv[16][H128];
  threadgroup float tg_o[16][H128];
  threadgroup float tg_s[2][8][16];
  threadgroup float tg_pv[2][8][32];
  threadgroup float tg_alpha[2][8];
  threadgroup float tg_inv_l[2][8];

  // Vectorized bfloat4 staging of Q[q_tile .. q_tile+15, 0..127] (pre-scaled by p.scale) and zero O.
  // 64 threads x 4 elements = 256 elements = 2 full rows per iteration (8 iterations for 16 rows).
  const uint v_row = tid >> 5;           // 0 or 1
  const uint v_col = (tid & 31u) * 4;    // 0, 4, 8, ..., 124
  UNROLL for (uint r0 = 0; r0 < 16; r0 += 2) {
    const uint r = r0 + v_row;
    const uint qr = q_tile + r;
    float4 qv = float4(0.0f);
    if (qr < p.rows) {
      const device vec<bfloat, 4>* qp = (const device vec<bfloat, 4>*)(q_in + (size_t)(qr * p.num_heads + head) * H128 + v_col);
      qv = float4(*qp) * p.scale;
    }
    *(threadgroup float4*)(&tg_q[r][v_col]) = qv;
    *(threadgroup float4*)(&tg_o[r][v_col]) = float4(0.0f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint last_q = min(p.rows - 1, q_tile + 15);
  const uint max_kv = seq_lens[last_q];

  const uint r_loc = lane >> 2;         // 0..7 row within this simdgroup
  const uint c_base = (lane & 3u) * 4;  // 0, 4, 8, 12 for softmax
  const uint c8 = (lane & 3u) * 8;      // 0, 8, 16, 24 for 8x32 PV accumulate
  const uint my_qr = q_tile + sg * 8 + r_loc;
  const uint my_seq_len = (my_qr < p.rows) ? seq_lens[my_qr] : 0u;

  float m_val = -INFINITY;
  float l_val = 0.0f;

  for (uint k0 = 0; k0 < max_kv; k0 += 16) {
    // 1. Vectorized bfloat4 load of K tile [k0 .. k0+15, 0..127]
    UNROLL for (uint kr0 = 0; kr0 < 16; kr0 += 2) {
      const uint kr = kr0 + v_row;
      const uint kp_idx = k0 + kr;
      float4 kv = float4(0.0f);
      if (kp_idx < max_kv) {
        const device vec<bfloat, 4>* kp = (const device vec<bfloat, 4>*)(k_cache + (size_t)kp_idx * kv_stride + kh_off + v_col);
        kv = float4(*kp);
      }
      *(threadgroup float4*)(&tg_kv[kr][v_col]) = kv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. S[8, 16] = Q[sg*8..+7, 0..127] @ K[0..15, 0..127]^T
    simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 s1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    UNROLL for (uint d = 0; d < H128; d += 8) {
      simdgroup_float8x8 qm, km0, km1;
      simdgroup_load(qm, &tg_q[sg * 8][d], H128);
      simdgroup_load(km0, &tg_kv[0][d], H128, ulong2(0), true);
      simdgroup_load(km1, &tg_kv[8][d], H128, ulong2(0), true);
      simdgroup_multiply_accumulate(s0, qm, km0, s0);
      simdgroup_multiply_accumulate(s1, qm, km1, s1);
    }
    simdgroup_store(s0, &tg_s[sg][0][0], 16);
    simdgroup_store(s1, &tg_s[sg][0][8], 16);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Causal online softmax over 16 keys
    float sc[4];
    float loc_max = -INFINITY;
    UNROLL for (uint j = 0; j < 4; j++) {
      const uint k_pos = k0 + c_base + j;
      const float v = (k_pos < my_seq_len) ? tg_s[sg][r_loc][c_base + j] : -INFINITY;
      sc[j] = v;
      loc_max = max(loc_max, v);
    }
    loc_max = max(loc_max, simd_shuffle_xor(loc_max, 1));
    loc_max = max(loc_max, simd_shuffle_xor(loc_max, 2));

    float alpha = 1.0f;
    if (loc_max > -INFINITY) {
      const float new_m = max(m_val, loc_max);
      alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
      float loc_sum = 0.0f;
      UNROLL for (uint j = 0; j < 4; j++) {
        const float pr = (sc[j] > -INFINITY) ? exp(sc[j] - new_m) : 0.0f;
        tg_s[sg][r_loc][c_base + j] = pr;
        loc_sum += pr;
      }
      loc_sum += simd_shuffle_xor(loc_sum, 1);
      loc_sum += simd_shuffle_xor(loc_sum, 2);
      l_val = fma(l_val, alpha, loc_sum);
      m_val = new_m;
    } else {
      UNROLL for (uint j = 0; j < 4; j++) tg_s[sg][r_loc][c_base + j] = 0.0f;
    }
    if ((lane & 3u) == 0) tg_alpha[sg][r_loc] = alpha;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Vectorized bfloat4 load of V tile [k0 .. k0+15, 0..127]
    UNROLL for (uint kr0 = 0; kr0 < 16; kr0 += 2) {
      const uint kr = kr0 + v_row;
      const uint vp_idx = k0 + kr;
      float4 vv = float4(0.0f);
      if (vp_idx < max_kv) {
        const device vec<bfloat, 4>* vp = (const device vec<bfloat, 4>*)(v_cache + (size_t)vp_idx * kv_stride + kh_off + v_col);
        vv = float4(*vp);
      }
      *(threadgroup float4*)(&tg_kv[kr][v_col]) = vv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5. O = alpha * O + P[8, 16] @ V[16, 128] (batched 32 cols at a time -> 4x fewer barriers!)
    simdgroup_float8x8 p0, p1;
    simdgroup_load(p0, &tg_s[sg][0][0], 16);
    simdgroup_load(p1, &tg_s[sg][0][8], 16);
    const float4 my_alpha4 = float4(tg_alpha[sg][r_loc]);
    UNROLL for (uint d = 0; d < H128; d += 32) {
      UNROLL for (uint sub = 0; sub < 4; sub++) {
        simdgroup_float8x8 vm0, vm1;
        simdgroup_load(vm0, &tg_kv[0][d + sub * 8], H128);
        simdgroup_load(vm1, &tg_kv[8][d + sub * 8], H128);
        simdgroup_float8x8 pv = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_multiply_accumulate(pv, p0, vm0, pv);
        simdgroup_multiply_accumulate(pv, p1, vm1, pv);
        simdgroup_store(pv, &tg_pv[sg][0][sub * 8], 32);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      threadgroup float4* o_dst = (threadgroup float4*)(&tg_o[sg * 8 + r_loc][d + c8]);
      const threadgroup float4* pv_src = (const threadgroup float4*)(&tg_pv[sg][r_loc][c8]);
      o_dst[0] = fma(o_dst[0], my_alpha4, pv_src[0]);
      o_dst[1] = fma(o_dst[1], my_alpha4, pv_src[1]);
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // 6. Epilogue: normalize by inv_l, store bfloat out, compute o_proj group-64 sums.
  if ((lane & 3u) == 0) {
    tg_inv_l[sg][r_loc] = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  for (uint rr = 0; rr < 8; rr++) {
    const uint qr = q_tile + sg * 8 + rr;
    if (qr >= p.rows) break;
    const float inv_l = tg_inv_l[sg][rr];
    device bfloat* out_ptr = out + (size_t)(qr * p.num_heads + head) * H128;
    UNROLL for (uint g = 0; g < 2; g++) {
      const uint i0 = g * 64 + lane;
      const uint i1 = g * 64 + 32 + lane;
      const bfloat y0 = bfloat(tg_o[sg * 8 + rr][i0] * inv_l);
      const bfloat y1 = bfloat(tg_o[sg * 8 + rr][i1] * inv_l);
      out_ptr[i0] = y0;
      out_ptr[i1] = y1;
      if (p.sums_stride != 0) {
        const float gs = simd_sum(float(y0) + float(y1));
        if (lane == 0) {
          sums[(size_t)(head * 2 + g) * p.sums_stride + qr] = gs;
        }
      }
    }
  }
}

// ============================================================================
// Universal General-HD GQA Attention (head_dim in {64, 96, 128, 192, 256, 512})
// Supports Llama 3.2 (HD=64), Mistral/Mixtral (sliding_window), Qwen3 (QK norm),
// Gemma 2/3/4 (HD=256/512, softcap, sliding window, dual RoPE), and LFM2!
// ============================================================================

struct GenPrepParams {
  uint rows;
  uint num_heads;
  uint num_kv_heads;
  uint head_dim;
  uint rope_dims;
  uint qkv_stride;
  uint has_bias;
  uint has_qk_norm;
  uint use_rope;
  uint has_v_norm;       // Gemma 4 parameter-free V RMSNorm
  uint k_eq_v;           // Gemma 4 global layers: V reads raw K projection
  uint rope_interleaved; // 1 = traditional interleaved pairs (2i, 2i+1) for GLM-4
  float eps;
};

// Grid: threadgroups (num_heads + num_kv_heads, rows) of 32 threads.
kernel void attn_gen_rope_store(const device bfloat* qkv_buf [[buffer(0)]],
                                const device bfloat* qkv_bias [[buffer(1)]],
                                const device bfloat* q_norm [[buffer(2)]],
                                const device bfloat* k_norm [[buffer(3)]],
                                const device float* inv_freq [[buffer(4)]],
                                const device uint* positions [[buffer(5)]],
                                const device uint* kv_slots [[buffer(6)]],
                                device bfloat* q_out [[buffer(7)]],
                                device bfloat* k_cache [[buffer(8)]],
                                device bfloat* v_cache [[buffer(9)]],
                                constant GenPrepParams& p [[buffer(10)]],
                                uint2 grp [[threadgroup_position_in_grid]],
                                ushort lane [[thread_index_in_simdgroup]]) {
  const uint idx = grp.x;
  const uint row = grp.y;
  if (row >= p.rows) return;

  const uint hd = p.head_dim;
  const uint half_r = p.rope_dims >> 1;
  const float pos = float(positions[row]);
  const device bfloat* row_ptr = qkv_buf + (size_t)row * p.qkv_stride;

  threadgroup float tg[512];

  if (idx < p.num_heads) {
    const size_t q_off = (size_t)idx * hd;
    const device bfloat* src = row_ptr + q_off;
    float sq = 0.0f;
    for (uint d = lane; d < hd; d += 32) {
      float v = float(src[d]);
      if (p.has_bias != 0) v += float(qkv_bias[q_off + d]);
      tg[d] = v;
      sq = fma(v, v, sq);
    }
    if (p.has_qk_norm != 0) {
      const float inv = rsqrt(simd_sum(sq) / float(hd) + p.eps);
      for (uint d = lane; d < hd; d += 32) {
        tg[d] = tg[d] * inv * float(q_norm[d]);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (p.use_rope != 0 && half_r > 0) {
      for (uint i = lane; i < half_r; i += 32) {
        const float angle = pos * inv_freq[i];
        const float c = cos(angle), s = sin(angle);
        if (p.rope_interleaved != 0) {
          const float a = tg[2 * i], b = tg[2 * i + 1];
          tg[2 * i] = a * c - b * s;
          tg[2 * i + 1] = b * c + a * s;
        } else {
          const float a = tg[i], b = tg[i + half_r];
          tg[i] = a * c - b * s;
          tg[i + half_r] = b * c + a * s;
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    device bfloat* dst = q_out + (size_t)(row * p.num_heads + idx) * hd;
    for (uint d = lane; d < hd; d += 32) dst[d] = bfloat(tg[d]);
    return;
  }

  const uint kh = idx - p.num_heads;
  const uint slot = kv_slots[row];
  const size_t k_off = (size_t)p.num_heads * hd + (size_t)kh * hd;
  const size_t v_off = (p.k_eq_v != 0) ? k_off : (k_off + (size_t)p.num_kv_heads * hd);
  const device bfloat* k_src = row_ptr + k_off;
  const device bfloat* v_src = row_ptr + v_off;

  float sq = 0.0f;
  for (uint d = lane; d < hd; d += 32) {
    float v = float(k_src[d]);
    if (p.has_bias != 0) v += float(qkv_bias[k_off + d]);
    tg[d] = v;
    sq = fma(v, v, sq);
  }
  if (p.has_qk_norm != 0) {
    const float inv = rsqrt(simd_sum(sq) / float(hd) + p.eps);
    for (uint d = lane; d < hd; d += 32) {
      tg[d] = tg[d] * inv * float(k_norm[d]);
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (p.use_rope != 0 && half_r > 0) {
    for (uint i = lane; i < half_r; i += 32) {
      const float angle = pos * inv_freq[i];
      const float c = cos(angle), s = sin(angle);
      if (p.rope_interleaved != 0) {
        const float a = tg[2 * i], b = tg[2 * i + 1];
        tg[2 * i] = a * c - b * s;
        tg[2 * i + 1] = b * c + a * s;
      } else {
        const float a = tg[i], b = tg[i + half_r];
        tg[i] = a * c - b * s;
        tg[i + half_r] = b * c + a * s;
      }
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  const size_t cache_base = (size_t)(slot * p.num_kv_heads + kh) * hd;
  device bfloat* k_dst = k_cache + cache_base;
  device bfloat* v_dst = v_cache + cache_base;
  for (uint d = lane; d < hd; d += 32) k_dst[d] = bfloat(tg[d]);

  if (p.has_v_norm != 0) {
    float vsq = 0.0f;
    for (uint d = lane; d < hd; d += 32) {
      float vv = float(v_src[d]);
      if (p.has_bias != 0) vv += float(qkv_bias[v_off + d]);
      tg[d] = vv;
      vsq = fma(vv, vv, vsq);
    }
    const float vinv = rsqrt(simd_sum(vsq) / float(hd) + p.eps);
    for (uint d = lane; d < hd; d += 32) v_dst[d] = bfloat(tg[d] * vinv);
  } else {
    for (uint d = lane; d < hd; d += 32) {
      float vv = float(v_src[d]);
      if (p.has_bias != 0) vv += float(qkv_bias[v_off + d]);
      v_dst[d] = bfloat(vv);
    }
  }
}

struct GenFlashParams {
  uint rows;
  uint num_heads;
  uint num_kv_heads;
  uint head_dim;
  uint sums_stride;
  uint base_pos;
  uint is_tree;
  uint sliding_window;
  float scale;
  float softcap;
};

// Universal General-HD Flash GQA (HD <= 512, sliding window, logit softcap, causal & tree).
// Grid: threadgroups (num_heads, rows) of 128 threads (4 simdgroups).
kernel void attn_gen_flash(const device bfloat* q_in [[buffer(0)]],
                           const device bfloat* k_cache [[buffer(1)]],
                           const device bfloat* v_cache [[buffer(2)]],
                           const device uint* seq_lens [[buffer(3)]],
                           const device uint4* tree_masks [[buffer(4)]],
                           device bfloat* out [[buffer(5)]],
                           device float* sums [[buffer(6)]],
                           constant GenFlashParams& p [[buffer(7)]],
                           uint2 grp [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
  const uint head = grp.x;
  const uint row = grp.y;
  if (head >= p.num_heads || row >= p.rows) return;

  const uint hd = p.head_dim;
  const uint blocks = hd >> 5; // hd / 32 (<= 16)
  const uint kv_head = head / (p.num_heads / p.num_kv_heads);
  const device bfloat* q_ptr = q_in + (size_t)(row * p.num_heads + head) * hd;

  float q[16];
  float acc[16];
  for (uint b = 0; b < blocks; b++) {
    q[b] = float(q_ptr[b * 32 + lane]) * p.scale;
    acc[b] = 0.0f;
  }

  float m_val = -INFINITY;
  float l_val = 0.0f;
  const size_t kv_stride = (size_t)p.num_kv_heads * hd;
  const size_t kh_off = (size_t)kv_head * hd;
  const uint total_seq = seq_lens[row];
  const uint prefix_end = (p.is_tree != 0) ? p.base_pos : total_seq;
  const uint sw_start = (p.sliding_window != 0 && total_seq > p.sliding_window) ? (total_seq - p.sliding_window) : 0u;
  uint t_first = sg;
  if (t_first < sw_start) t_first += ((sw_start - t_first + 3u) & ~3u);

  for (uint t = t_first; t < prefix_end; t += 4) {
    const device bfloat* kp = k_cache + (size_t)t * kv_stride + kh_off;
    float dot_part = 0.0f;
    for (uint b = 0; b < blocks; b++) dot_part = fma(q[b], float(kp[b * 32 + lane]), dot_part);
    float score = simd_sum(dot_part);
    if (p.softcap > 0.0f) score = p.softcap * tanh(clamp(score / p.softcap, -15.0f, 15.0f));
    const float new_m = max(m_val, score);
    const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
    const float prob = exp(score - new_m);
    l_val = fma(l_val, alpha, prob);
    m_val = new_m;

    const device bfloat* vp = v_cache + (size_t)t * kv_stride + kh_off;
    for (uint b = 0; b < blocks; b++) acc[b] = fma(acc[b], alpha, prob * float(vp[b * 32 + lane]));
  }

  if (p.is_tree != 0) {
    const uint4 mask = tree_masks[row];
    uint anc_idx = 0;
    for (uint r = 0; r <= row; r++) {
      if ((mask[r >> 5] & (1u << (r & 31))) == 0u) continue;
      if (((p.base_pos + anc_idx) & 3u) == sg) {
        const uint t = p.base_pos + r;
        const device bfloat* kp = k_cache + (size_t)t * kv_stride + kh_off;
        float dot_part = 0.0f;
        for (uint b = 0; b < blocks; b++) dot_part = fma(q[b], float(kp[b * 32 + lane]), dot_part);
        float score = simd_sum(dot_part);
        if (p.softcap > 0.0f) score = p.softcap * tanh(clamp(score / p.softcap, -15.0f, 15.0f));
        const float new_m = max(m_val, score);
        const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
        const float prob = exp(score - new_m);
        l_val = fma(l_val, alpha, prob);
        m_val = new_m;

        const device bfloat* vp = v_cache + (size_t)t * kv_stride + kh_off;
        for (uint b = 0; b < blocks; b++) acc[b] = fma(acc[b], alpha, prob * float(vp[b * 32 + lane]));
      }
      anc_idx++;
    }
  }

  threadgroup float tg_m[4];
  threadgroup float tg_l[4];
  threadgroup float tg_acc[4][512];

  if (lane == 0) {
    tg_m[sg] = m_val;
    tg_l[sg] = l_val;
  }
  for (uint b = 0; b < blocks; b++) tg_acc[sg][b * 32 + lane] = acc[b];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;

  const float gm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
  float w[4];
  float gl = 0.0f;
  UNROLL for (uint s = 0; s < 4; s++) {
    w[s] = (tg_l[s] > 0.0f) ? exp(tg_m[s] - gm) : 0.0f;
    gl += tg_l[s] * w[s];
  }
  const float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

  device bfloat* out_ptr = out + (size_t)(row * p.num_heads + head) * hd;
  const uint n_g64 = hd >> 6;
  for (uint g = 0; g < n_g64; g++) {
    const uint i0 = g * 64 + lane;
    const uint i1 = g * 64 + 32 + lane;
    float o0 = 0.0f, o1 = 0.0f;
    UNROLL for (uint s = 0; s < 4; s++) {
      o0 = fma(tg_acc[s][i0], w[s], o0);
      o1 = fma(tg_acc[s][i1], w[s], o1);
    }
    const bfloat y0 = bfloat(o0 * inv_l);
    const bfloat y1 = bfloat(o1 * inv_l);
    out_ptr[i0] = y0;
    out_ptr[i1] = y1;
    if (p.sums_stride != 0) {
      const float gs = simd_sum(float(y0) + float(y1));
      if (lane == 0) sums[(size_t)(head * n_g64 + g) * p.sums_stride + row] = gs;
    }
  }
  // Handle tail if hd is not a multiple of 64 (e.g. hd=96)
  for (uint d = n_g64 * 64 + lane; d < hd; d += 32) {
    float o = 0.0f;
    UNROLL for (uint s = 0; s < 4; s++) o = fma(tg_acc[s][d], w[s], o);
    out_ptr[d] = bfloat(o * inv_l);
  }
}

// ============================================================================
// Multi-Head Latent Attention (MLA) Kernels for DeepSeek-V2 / V3 / R1 / V3.2
// and GLM-4.7-Flash / GLM-5 (glm4_moe_lite, glm_moe_dsa)
// ============================================================================

struct MlaNormParams {
  uint rows;
  uint dim;
  uint sums_stride;
  uint has_bias;
  float eps;
};

// Arbitrary-width RMSNorm (+ optional bias + optional group-64 sums) for q_a_layernorm.
// Grid: threadgroups (rows, 1, 1) of 256 threads.
kernel void mla_rms_norm_sums(const device bfloat* x [[buffer(0)]],
                              const device bfloat* bias [[buffer(1)]],
                              const device bfloat* weight [[buffer(2)]],
                              device bfloat* out [[buffer(3)]],
                              device float* sums [[buffer(4)]],
                              constant MlaNormParams& p [[buffer(5)]],
                              uint row [[threadgroup_position_in_grid]],
                              ushort tid [[thread_index_in_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
  if (row >= p.rows) return;
  const device bfloat* xr = x + (size_t)row * p.dim;
  device bfloat* yr = out + (size_t)row * p.dim;

  float sq = 0.0f;
  for (uint d = tid; d < p.dim; d += 256) {
    float v = float(xr[d]);
    if (p.has_bias != 0) v += float(bias[d]);
    sq = fma(v, v, sq);
  }
  sq = simd_sum(sq);
  threadgroup float sg_sq[8];
  if (lane == 0) sg_sq[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = (lane < 8) ? sg_sq[lane] : 0.0f;
    float tot = simd_sum(v);
    if (lane == 0) sg_sq[0] = rsqrt(tot / float(p.dim) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sg_sq[0];

  for (uint d = tid; d < p.dim; d += 256) {
    float v = float(xr[d]);
    if (p.has_bias != 0) v += float(bias[d]);
    yr[d] = bfloat(v * inv * float(weight[d]));
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (p.sums_stride != 0) {
    const uint groups = p.dim >> 6;
    for (uint g = sg; g < groups; g += 8) {
      const uint base = g * 64 + lane;
      const float gs = simd_sum(float(yr[base]) + float(yr[base + 32]));
      if (lane == 0) sums[(size_t)g * p.sums_stride + row] = gs;
    }
  }
}

struct MlaQPrepParams {
  uint rows;
  uint num_heads;
  uint qk_nope_dim;
  uint qk_rope_dim;
  uint rope_interleaved;
  float rope_mscale;
};

// Copy q_nope and apply RoPE (interleaved or split-half) to q_pe into attn_q_buf.
// Grid: threadgroups (num_heads, rows, 1) of 32 threads.
kernel void mla_q_rope(const device bfloat* q_in [[buffer(0)]],
                       const device float* inv_freq [[buffer(1)]],
                       const device uint* positions [[buffer(2)]],
                       device bfloat* q_out [[buffer(3)]],
                       constant MlaQPrepParams& p [[buffer(4)]],
                       uint2 grp [[threadgroup_position_in_grid]],
                       ushort lane [[thread_index_in_simdgroup]]) {
  const uint head = grp.x;
  const uint row = grp.y;
  if (head >= p.num_heads || row >= p.rows) return;

  const uint q_hd = p.qk_nope_dim + p.qk_rope_dim;
  const size_t base = (size_t)(row * p.num_heads + head) * q_hd;
  const device bfloat* src = q_in + base;
  device bfloat* dst = q_out + base;

  for (uint d = lane; d < p.qk_nope_dim; d += 32) {
    dst[d] = src[d];
  }

  const uint half_r = p.qk_rope_dim >> 1;
  const float pos = float(positions[row]);
  const device bfloat* ps = src + p.qk_nope_dim;
  device bfloat* pd = dst + p.qk_nope_dim;
  for (uint i = lane; i < half_r; i += 32) {
    const float angle = pos * inv_freq[i];
    const float c = cos(angle) * p.rope_mscale;
    const float s = sin(angle) * p.rope_mscale;
    if (p.rope_interleaved != 0) {
      const float a = float(ps[2 * i]);
      const float b = float(ps[2 * i + 1]);
      pd[2 * i] = bfloat(a * c - b * s);
      pd[2 * i + 1] = bfloat(b * c + a * s);
    } else {
      const float a = float(ps[i]);
      const float b = float(ps[i + half_r]);
      pd[i] = bfloat(a * c - b * s);
      pd[i + half_r] = bfloat(b * c + a * s);
    }
  }
}

struct MlaKvAParams {
  uint rows;
  uint kv_lora_rank;
  uint qk_rope_dim;
  uint sums_stride;
  uint has_bias;
  uint rope_interleaved;
  float eps;
  float rope_mscale;
};

// RMSNorm kv_a[0..kv_lora_rank] (+ optional group-64 sums) and RoPE kv_a[kv_lora_rank..+qk_rope_dim].
// Grid: threadgroups (rows, 1, 1) of 256 threads.
kernel void mla_kv_a_norm_rope(const device bfloat* kv_a [[buffer(0)]],
                               const device bfloat* kv_a_bias [[buffer(1)]],
                               const device bfloat* kv_norm [[buffer(2)]],
                               const device float* inv_freq [[buffer(3)]],
                               const device uint* positions [[buffer(4)]],
                               device bfloat* kv_lat_out [[buffer(5)]],
                               device bfloat* kpe_out [[buffer(6)]],
                               device float* sums [[buffer(7)]],
                               constant MlaKvAParams& p [[buffer(8)]],
                               uint row [[threadgroup_position_in_grid]],
                               ushort tid [[thread_index_in_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
  if (row >= p.rows) return;
  const uint kv_a_stride = p.kv_lora_rank + p.qk_rope_dim;
  const device bfloat* row_in = kv_a + (size_t)row * kv_a_stride;
  device bfloat* lat_dst = kv_lat_out + (size_t)row * p.kv_lora_rank;

  float sq = 0.0f;
  for (uint d = tid; d < p.kv_lora_rank; d += 256) {
    float v = float(row_in[d]);
    if (p.has_bias != 0) v += float(kv_a_bias[d]);
    sq = fma(v, v, sq);
  }
  sq = simd_sum(sq);
  threadgroup float sg_sq[8];
  if (lane == 0) sg_sq[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = (lane < 8) ? sg_sq[lane] : 0.0f;
    float tot = simd_sum(v);
    if (lane == 0) sg_sq[0] = rsqrt(tot / float(p.kv_lora_rank) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sg_sq[0];

  for (uint d = tid; d < p.kv_lora_rank; d += 256) {
    float v = float(row_in[d]);
    if (p.has_bias != 0) v += float(kv_a_bias[d]);
    lat_dst[d] = bfloat(v * inv * float(kv_norm[d]));
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (p.sums_stride != 0) {
    const uint groups = p.kv_lora_rank >> 6;
    for (uint g = sg; g < groups; g += 8) {
      const uint base = g * 64 + lane;
      const float gs = simd_sum(float(lat_dst[base]) + float(lat_dst[base + 32]));
      if (lane == 0) sums[(size_t)g * p.sums_stride + row] = gs;
    }
  }

  // RoPE on shared k_pe (1 simdgroup handles up to arbitrary qk_rope_dim)
  if (sg == 0) {
    const uint half_r = p.qk_rope_dim >> 1;
    const float pos = float(positions[row]);
    const device bfloat* ps = row_in + p.kv_lora_rank;
    device bfloat* pd = kpe_out + (size_t)row * p.qk_rope_dim;
    for (uint i = lane; i < half_r; i += 32) {
      const float angle = pos * inv_freq[i];
      const float c = cos(angle) * p.rope_mscale;
      const float s = sin(angle) * p.rope_mscale;
      if (p.rope_interleaved != 0) {
        float a = float(ps[2 * i]);
        float b = float(ps[2 * i + 1]);
        if (p.has_bias != 0) {
          a += float(kv_a_bias[p.kv_lora_rank + 2 * i]);
          b += float(kv_a_bias[p.kv_lora_rank + 2 * i + 1]);
        }
        pd[2 * i] = bfloat(a * c - b * s);
        pd[2 * i + 1] = bfloat(b * c + a * s);
      } else {
        float a = float(ps[i]);
        float b = float(ps[i + half_r]);
        if (p.has_bias != 0) {
          a += float(kv_a_bias[p.kv_lora_rank + i]);
          b += float(kv_a_bias[p.kv_lora_rank + i + half_r]);
        }
        pd[i] = bfloat(a * c - b * s);
        pd[i + half_r] = bfloat(b * c + a * s);
      }
    }
  }
}

struct MlaScatterParams {
  uint rows;
  uint num_heads;
  uint qk_nope_dim;
  uint qk_rope_dim;
  uint v_dim;
  uint kv_row_stride;
};

// Scatter expanded kv_b [rows, num_heads, qk_nope + v_dim] + shared k_pe [rows, qk_rope] into KV cache.
// Grid: threadgroups (num_heads, rows, 1) of 32 threads.
kernel void mla_kv_b_scatter(const device bfloat* kv_b [[buffer(0)]],
                             const device bfloat* kpe [[buffer(1)]],
                             const device uint* kv_slots [[buffer(2)]],
                             device bfloat* k_cache [[buffer(3)]],
                             device bfloat* v_cache [[buffer(4)]],
                             constant MlaScatterParams& p [[buffer(5)]],
                             uint2 grp [[threadgroup_position_in_grid]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  const uint head = grp.x;
  const uint row = grp.y;
  if (head >= p.num_heads || row >= p.rows) return;

  const uint slot = kv_slots[row];
  const uint q_hd = p.qk_nope_dim + p.qk_rope_dim;
  const uint kvb_hd = p.qk_nope_dim + p.v_dim;
  const device bfloat* src = kv_b + (size_t)(row * p.num_heads + head) * kvb_hd;
  const device bfloat* kp = kpe + (size_t)row * p.qk_rope_dim;
  device bfloat* k_dst = k_cache + (size_t)slot * p.kv_row_stride + (size_t)head * q_hd;
  device bfloat* v_dst = v_cache + (size_t)slot * p.kv_row_stride + (size_t)head * p.v_dim;

  for (uint d = lane; d < p.qk_nope_dim; d += 32) k_dst[d] = src[d];
  for (uint d = lane; d < p.qk_rope_dim; d += 32) k_dst[p.qk_nope_dim + d] = kp[d];
  for (uint d = lane; d < p.v_dim; d += 32) v_dst[d] = src[p.qk_nope_dim + d];
}

struct MlaFlashParams {
  uint rows;
  uint num_heads;
  uint qk_dim;
  uint v_dim;
  uint kv_row_stride;
  uint sums_stride;
  uint base_pos;
  uint is_tree;
  float scale;
};

// MLA FlashAttention supporting distinct qk_dim (<= 512) and v_dim (<= 512), causal & tree.
// Grid: threadgroups (num_heads, rows, 1) of 128 threads (4 simdgroups).
kernel void mla_flash_attn(const device bfloat* q_in [[buffer(0)]],
                           const device bfloat* k_cache [[buffer(1)]],
                           const device bfloat* v_cache [[buffer(2)]],
                           const device uint* seq_lens [[buffer(3)]],
                           const device uint4* tree_masks [[buffer(4)]],
                           device bfloat* out [[buffer(5)]],
                           device float* sums [[buffer(6)]],
                           constant MlaFlashParams& p [[buffer(7)]],
                           uint2 grp [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
  const uint head = grp.x;
  const uint row = grp.y;
  if (head >= p.num_heads || row >= p.rows) return;

  const uint qk_dim = p.qk_dim;
  const uint v_dim = p.v_dim;
  const uint q_blocks = (qk_dim + 31u) >> 5;
  const uint v_blocks = (v_dim + 31u) >> 5;
  const device bfloat* q_ptr = q_in + (size_t)(row * p.num_heads + head) * qk_dim;

  float q[16];
  float acc[16];
  for (uint b = 0; b < q_blocks; b++) {
    const uint idx = b * 32 + lane;
    q[b] = (idx < qk_dim) ? (float(q_ptr[idx]) * p.scale) : 0.0f;
  }
  for (uint b = 0; b < v_blocks; b++) acc[b] = 0.0f;

  float m_val = -INFINITY;
  float l_val = 0.0f;
  const size_t kh_off = (size_t)head * qk_dim;
  const size_t vh_off = (size_t)head * v_dim;
  const uint total_seq = seq_lens[row];
  const uint prefix_end = (p.is_tree != 0) ? p.base_pos : total_seq;

  for (uint t = sg; t < prefix_end; t += 4) {
    const device bfloat* kp = k_cache + (size_t)t * p.kv_row_stride + kh_off;
    float dot_part = 0.0f;
    for (uint b = 0; b < q_blocks; b++) {
      const uint idx = b * 32 + lane;
      if (idx < qk_dim) dot_part = fma(q[b], float(kp[idx]), dot_part);
    }
    const float score = simd_sum(dot_part);
    const float new_m = max(m_val, score);
    const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
    const float prob = exp(score - new_m);
    l_val = fma(l_val, alpha, prob);
    m_val = new_m;

    const device bfloat* vp = v_cache + (size_t)t * p.kv_row_stride + vh_off;
    for (uint b = 0; b < v_blocks; b++) {
      const uint idx = b * 32 + lane;
      if (idx < v_dim) acc[b] = fma(acc[b], alpha, prob * float(vp[idx]));
    }
  }

  if (p.is_tree != 0) {
    const uint4 mask = tree_masks[row];
    uint anc_idx = 0;
    for (uint r = 0; r <= row; r++) {
      if ((mask[r >> 5] & (1u << (r & 31))) == 0u) continue;
      if (((p.base_pos + anc_idx) & 3u) == sg) {
        const uint t = p.base_pos + r;
        const device bfloat* kp = k_cache + (size_t)t * p.kv_row_stride + kh_off;
        float dot_part = 0.0f;
        for (uint b = 0; b < q_blocks; b++) {
          const uint idx = b * 32 + lane;
          if (idx < qk_dim) dot_part = fma(q[b], float(kp[idx]), dot_part);
        }
        const float score = simd_sum(dot_part);
        const float new_m = max(m_val, score);
        const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
        const float prob = exp(score - new_m);
        l_val = fma(l_val, alpha, prob);
        m_val = new_m;

        const device bfloat* vp = v_cache + (size_t)t * p.kv_row_stride + vh_off;
        for (uint b = 0; b < v_blocks; b++) {
          const uint idx = b * 32 + lane;
          if (idx < v_dim) acc[b] = fma(acc[b], alpha, prob * float(vp[idx]));
        }
      }
      anc_idx++;
    }
  }

  threadgroup float tg_m[4];
  threadgroup float tg_l[4];
  threadgroup float tg_acc[4][512];

  if (lane == 0) {
    tg_m[sg] = m_val;
    tg_l[sg] = l_val;
  }
  for (uint b = 0; b < v_blocks; b++) {
    const uint idx = b * 32 + lane;
    if (idx < v_dim) tg_acc[sg][idx] = acc[b];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;

  const float gm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
  float w[4];
  float gl = 0.0f;
  UNROLL for (uint s = 0; s < 4; s++) {
    w[s] = (tg_l[s] > 0.0f) ? exp(tg_m[s] - gm) : 0.0f;
    gl += tg_l[s] * w[s];
  }
  const float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

  device bfloat* out_ptr = out + (size_t)(row * p.num_heads + head) * v_dim;
  const uint n_g64 = v_dim >> 6;
  for (uint g = 0; g < n_g64; g++) {
    const uint i0 = g * 64 + lane;
    const uint i1 = g * 64 + 32 + lane;
    float o0 = 0.0f, o1 = 0.0f;
    UNROLL for (uint s = 0; s < 4; s++) {
      o0 = fma(tg_acc[s][i0], w[s], o0);
      o1 = fma(tg_acc[s][i1], w[s], o1);
    }
    const bfloat y0 = bfloat(o0 * inv_l);
    const bfloat y1 = bfloat(o1 * inv_l);
    out_ptr[i0] = y0;
    out_ptr[i1] = y1;
    if (p.sums_stride != 0) {
      const float gs = simd_sum(float(y0) + float(y1));
      if (lane == 0) sums[(size_t)(head * n_g64 + g) * p.sums_stride + row] = gs;
    }
  }
  for (uint d = n_g64 * 64 + lane; d < v_dim; d += 32) {
    float o = 0.0f;
    UNROLL for (uint s = 0; s < 4; s++) o = fma(tg_acc[s][d], w[s], o);
    out_ptr[d] = bfloat(o * inv_l);
  }
}
