// Clean-room Gated DeltaNet (GDN) kernels:
//   1. gdn_conv: vectorized 4-tap depthwise conv1d + SiLU + Q/K L2 norm + decay/mixing,
//      with optional race-free in-place history shift when rows == 1.
//   2. gdn_commit_history: updates the 3-tap rolling conv history after multi-row prefill or tree keep.
//   3. gdn_scan: 4 value-columns per simdgroup depth-first GDN recurrence / tree scan,
//      with optional in-place state commit for chain passes (prefill & c=1 decode).
#include <metal_stdlib>
using namespace metal;

#define UNROLL _Pragma("clang loop unroll(full)")

constant constexpr uint KEEP = 3;
constant constexpr uint DK = 128;
constant constexpr uint DV = 128;
constant constexpr uint SAVE_SHIFT = 24;
constant constexpr uint RESTORE_SHIFT = 16;

struct GdnConvParams {
  uint rows;
  uint nk;
  uint nv;
  uint qkv_dim;        // (2*nk + nv) * 128
  uint in_proj_stride; // full row width of in_proj_all
  uint b_offset;       // column offset of B inside in_proj_all
  uint a_offset;       // column offset of A inside in_proj_all
  uint commit_single;  // 1 if rows == 1 and history should be shifted in-place
  uint save_qkv;       // 1 during speculative verify: save raw QKV into keep_qkv for later history commit
};

// Grid: threadgroups (1, 2*nk + nv, rows) of 32 threads (1 simdgroup per 128-wide head).
kernel void gdn_conv(const device vec<bfloat, 4>* in_proj_all [[buffer(0)]],
                     device vec<bfloat, 4>* history [[buffer(1)]],
                     const device vec<bfloat, 4>* weights [[buffer(2)]],
                     const device float* a_log [[buffer(3)]],
                     const device bfloat* dt_bias [[buffer(4)]],
                     const device uint4* windows [[buffer(5)]],
                     constant GdnConvParams& p [[buffer(6)]],
                     device vec<bfloat, 4>* query [[buffer(7)]],
                     device vec<bfloat, 4>* key [[buffer(8)]],
                     device vec<bfloat, 4>* value [[buffer(9)]],
                     device float* decay [[buffer(10)]],
                     device bfloat* mixing [[buffer(11)]],
                     device vec<bfloat, 4>* keep_qkv [[buffer(12)]],
                     uint3 grp [[threadgroup_position_in_grid]],
                     ushort lane [[thread_index_in_simdgroup]]) {
  const uint head = grp.y;
  const uint row = grp.z;
  if (row >= p.rows) return;

  const uint at4 = head * (DK / 4) + lane;
  const uint qkv_stride4 = p.qkv_dim >> 2;
  const uint proj_stride4 = p.in_proj_stride >> 2;
  const uint4 sources = windows[row];

  float4 tap_values[4];
  UNROLL for (uint tap = 0; tap < 4; tap++) {
    const uint src = sources[tap];
    const device vec<bfloat, 4>* from = src < KEEP
        ? history + (size_t)src * qkv_stride4
        : in_proj_all + (size_t)(src - KEEP) * proj_stride4;
    tap_values[tap] = float4(from[at4]);
  }

  // Save raw QKV row during speculative verify so commitKeep can update conv history cleanly.
  if (p.save_qkv != 0) {
    keep_qkv[(size_t)row * qkv_stride4 + at4] = vec<bfloat, 4>(tap_values[3]);
  }

  // Race-free in-place history shift when rows == 1.
  if (p.commit_single != 0) {
    history[0 * qkv_stride4 + at4] = vec<bfloat, 4>(tap_values[1]);
    history[1 * qkv_stride4 + at4] = vec<bfloat, 4>(tap_values[2]);
    history[2 * qkv_stride4 + at4] = vec<bfloat, 4>(tap_values[3]);
  }

  float act[4];
  UNROLL for (uint part = 0; part < 4; part++) {
    const float4 w = float4(weights[head * DK + lane * 4 + part]);
    float acc = 0.0f;
    UNROLL for (uint tap = 0; tap < 4; tap++) acc = fma(w[tap], tap_values[tap][part], acc);
    const float r = float(bfloat(acc));
    const float sig = 1.0f / (1.0f + exp(-r));
    act[part] = r * sig;
  }

  if (head < 2 * p.nk) {
    float sq = 0.0f;
    UNROLL for (uint part = 0; part < 4; part++) sq = fma(act[part], act[part], sq);
    sq = simd_sum(sq);
    // L2 normalize K; L2 normalize Q and scale by 1/sqrt(DK).
    const float inv = rsqrt(sq / float(DK) + 1e-6f);
    const float scale = (head < p.nk) ? (1.0f / float(DK)) : rsqrt(float(DK));
    float4 out;
    UNROLL for (uint part = 0; part < 4; part++) out[part] = act[part] * inv * scale;
    const uint pos = (row * p.nk + (head % p.nk)) * (DK / 4) + lane;
    if (head < p.nk) {
      query[pos] = vec<bfloat, 4>(out);
    } else {
      key[pos] = vec<bfloat, 4>(out);
    }
    return;
  }

  const uint vhead = head - 2 * p.nk;
  value[(row * p.nv + vhead) * (DV / 4) + lane] = vec<bfloat, 4>(float4(act[0], act[1], act[2], act[3]));
  if (lane == 0) {
    const device bfloat* proj_row = (const device bfloat*)in_proj_all + (size_t)row * p.in_proj_stride;
    const float bv = float(proj_row[p.b_offset + vhead]);
    const float av = float(proj_row[p.a_offset + vhead]);
    const float shifted = av + float(dt_bias[vhead]);
    const float sp = max(shifted, 0.0f) + log(1.0f + exp(-abs(shifted)));
    const uint cell = row * p.nv + vhead;
    decay[cell] = exp(-exp(a_log[vhead]) * sp);
    mixing[cell] = bfloat(1.0f / (1.0f + exp(-bv)));
  }
}

struct GdnHistCommitParams {
  uint qkv_dim;
  uint in_proj_stride;
  uint src0;
  uint src1;
  uint src2;
};

// Update 3-row rolling conv history from any mix of prior history [0..2] and new rows [3..].
// Grid: threads (qkv_dim / 4, 1).
kernel void gdn_commit_history(const device vec<bfloat, 4>* in_proj_all [[buffer(0)]],
                               device vec<bfloat, 4>* history [[buffer(1)]],
                               constant GdnHistCommitParams& p [[buffer(2)]],
                               uint tid [[thread_position_in_grid]]) {
  const uint qkv_stride4 = p.qkv_dim >> 2;
  if (tid >= qkv_stride4) return;
  const uint proj_stride4 = p.in_proj_stride >> 2;

  const uint srcs[3] = { p.src0, p.src1, p.src2 };
  vec<bfloat, 4> vals[3];
  UNROLL for (uint i = 0; i < 3; i++) {
    const uint s = srcs[i];
    vals[i] = (s < KEEP)
        ? history[(size_t)s * qkv_stride4 + tid]
        : in_proj_all[(size_t)(s - KEEP) * proj_stride4 + tid];
  }
  UNROLL for (uint i = 0; i < 3; i++) {
    history[(size_t)i * qkv_stride4 + tid] = vals[i];
  }
}

struct GdnScanParams {
  uint rows;
  uint nk;
  uint nv;
  uint commit_state; // 1 for chain prefill or c=1 decode (writes final state in-place)
};

// 4 value columns per simdgroup share one load of Q, K, decay, and mixing per step.
// Grid: threadgroups (1, DV / 4, nv) = (1, 32, nv) of 32 threads.
template <uint SLOTS>
kernel void gdn_scan(const device vec<bfloat, 4>* query [[buffer(0)]],
                     const device vec<bfloat, 4>* key [[buffer(1)]],
                     const device bfloat* value [[buffer(2)]],
                     const device float* decay [[buffer(3)]],
                     const device bfloat* mixing [[buffer(4)]],
                     device float4* state [[buffer(5)]],
                     const device uint* steps [[buffer(6)]],
                     constant GdnScanParams& p [[buffer(7)]],
                     device bfloat* output [[buffer(8)]],
                     uint3 grp [[threadgroup_position_in_grid]],
                     ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint K = 4;
  threadgroup float4 kept[SLOTS][K][32];

  const uint head = grp.z;
  const uint first = grp.y * K;
  const uint key_head = head / (p.nv / p.nk);

  float4 s[K];
  UNROLL for (uint k = 0; k < K; k++) {
    s[k] = state[(size_t)(head * DV + first + k) * (DK / 4) + lane];
  }

  for (uint at = 0; at < p.rows; at++) {
    const uint step = steps[at];
    const uint row = step & 0xFFFFu;
    if (SLOTS > 1) {
      const uint restore = (step >> RESTORE_SHIFT) & 0xFFu;
      if (restore != 0) {
        UNROLL for (uint k = 0; k < K; k++) s[k] = kept[restore - 1][k][lane];
      }
    }

    const uint cell = row * p.nv + head;
    const uint key_at = (row * p.nk + key_head) * (DK / 4) + lane;
    const float forget = decay[cell];
    const float mix = float(mixing[cell]);
    const float4 ki = float4(key[key_at]);
    const float4 qi = float4(query[key_at]);
    const float4 incoming = float4(*(const device vec<bfloat, 4>*)(value + (size_t)cell * DV + first));

    float4 results;
    UNROLL for (uint k = 0; k < K; k++) {
      float4 t = s[k] * forget;
      const float remembered = simd_sum(dot(t, ki));
      const float correction = (incoming[k] - remembered) * mix;
      t = fma(ki, float4(correction), t);
      s[k] = t;
      results[k] = simd_sum(dot(t, qi));
    }

    if (lane == 0) {
      *(device vec<bfloat, 4>*)(output + (size_t)cell * DV + first) = vec<bfloat, 4>(results);
    }

    if (SLOTS > 1) {
      const uint save = step >> SAVE_SHIFT;
      if (save != 0) {
        UNROLL for (uint k = 0; k < K; k++) kept[save - 1][k][lane] = s[k];
      }
    }
  }

  if (p.commit_state != 0) {
    UNROLL for (uint k = 0; k < K; k++) {
      state[(size_t)(head * DV + first + k) * (DK / 4) + lane] = s[k];
    }
  }
}

template [[host_name("gdn_scan_1")]] kernel void gdn_scan<1>(
    const device vec<bfloat, 4>*, const device vec<bfloat, 4>*, const device bfloat*,
    const device float*, const device bfloat*, device float4*, const device uint*,
    constant GdnScanParams&, device bfloat*, uint3, ushort);
template [[host_name("gdn_scan_2")]] kernel void gdn_scan<2>(
    const device vec<bfloat, 4>*, const device vec<bfloat, 4>*, const device bfloat*,
    const device float*, const device bfloat*, device float4*, const device uint*,
    constant GdnScanParams&, device bfloat*, uint3, ushort);
template [[host_name("gdn_scan_4")]] kernel void gdn_scan<4>(
    const device vec<bfloat, 4>*, const device vec<bfloat, 4>*, const device bfloat*,
    const device float*, const device bfloat*, device float4*, const device uint*,
    constant GdnScanParams&, device bfloat*, uint3, ushort);
template [[host_name("gdn_scan_8")]] kernel void gdn_scan<8>(
    const device vec<bfloat, 4>*, const device vec<bfloat, 4>*, const device bfloat*,
    const device float*, const device bfloat*, device float4*, const device uint*,
    constant GdnScanParams&, device bfloat*, uint3, ushort);

// Forward GDN scan variant that records the already-computed rank-1 delta correction vector
// c_t = beta_t * (v_t - alpha_t * S_{t-1} k_t) in fp32 at ZERO extra FLOPs, enabling
// singularity-free backward inversion S_{t-1} = (S_t - c_t k_t^T) / alpha_t even at beta_t = 1!
// Grid: threadgroups (1, DV / 4, nv) of 32 threads.
kernel void gdn_scan_record_corr(const device vec<bfloat, 4>* query [[buffer(0)]],
                                 const device vec<bfloat, 4>* key [[buffer(1)]],
                                 const device bfloat* value [[buffer(2)]],
                                 const device float* decay [[buffer(3)]],
                                 const device bfloat* mixing [[buffer(4)]],
                                 device float4* state [[buffer(5)]],
                                 const device uint* steps [[buffer(6)]],
                                 constant GdnScanParams& p [[buffer(7)]],
                                 device bfloat* output [[buffer(8)]],
                                 device float* corr_out [[buffer(9)]],
                                 uint3 grp [[threadgroup_position_in_grid]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint K = 4;
  const uint head = grp.z;
  const uint first = grp.y * K;
  const uint key_head = head / (p.nv / p.nk);

  float4 s[K];
  UNROLL for (uint k = 0; k < K; k++) {
    s[k] = state[(size_t)(head * DV + first + k) * (DK / 4) + lane];
  }

  for (uint at = 0; at < p.rows; at++) {
    const uint row = steps[at] & 0xFFFFu;
    const uint cell = row * p.nv + head;
    const uint key_at = (row * p.nk + key_head) * (DK / 4) + lane;
    const float forget = decay[cell];
    const float mix = float(mixing[cell]);
    const float4 ki = float4(key[key_at]);
    const float4 qi = float4(query[key_at]);
    const float4 incoming = float4(*(const device vec<bfloat, 4>*)(value + (size_t)cell * DV + first));

    float4 results;
    float4 corrs;
    UNROLL for (uint k = 0; k < K; k++) {
      float4 t = s[k] * forget;
      const float remembered = simd_sum(dot(t, ki));
      const float correction = (incoming[k] - remembered) * mix;
      corrs[k] = correction;
      t = fma(ki, float4(correction), t);
      s[k] = t;
      results[k] = simd_sum(dot(t, qi));
    }

    if (lane == 0) {
      *(device vec<bfloat, 4>*)(output + (size_t)cell * DV + first) = vec<bfloat, 4>(results);
      *(device float4*)(corr_out + (size_t)cell * DV + first) = corrs;
    }
  }

  if (p.commit_state != 0) {
    UNROLL for (uint k = 0; k < K; k++) {
      state[(size_t)(head * DV + first + k) * (DK / 4) + lane] = s[k];
    }
  }
}

struct GdnBatchConvParams {
  uint rows;           // active streams B (1..16)
  uint nk;
  uint nv;
  uint qkv_dim;
  uint in_proj_stride;
  uint b_offset;
  uint a_offset;
  uint hist_stream_stride4; // stride in vec<bfloat,4> between stream histories for this layer
};

// Lockstep multi-stream GDN conv: row r belongs to distinct stream_id = batch_streams[r].
// Reads and shifts each stream's rolling 3-tap history in-place (100% race-free).
// Grid: threadgroups (1, 2*nk + nv, rows) of 32 threads.
kernel void gdn_conv_batched(const device vec<bfloat, 4>* in_proj_all [[buffer(0)]],
                             device vec<bfloat, 4>* all_histories [[buffer(1)]],
                             const device vec<bfloat, 4>* weights [[buffer(2)]],
                             const device float* a_log [[buffer(3)]],
                             const device bfloat* dt_bias [[buffer(4)]],
                             const device uint* batch_streams [[buffer(5)]],
                             constant GdnBatchConvParams& p [[buffer(6)]],
                             device vec<bfloat, 4>* query [[buffer(7)]],
                             device vec<bfloat, 4>* key [[buffer(8)]],
                             device vec<bfloat, 4>* value [[buffer(9)]],
                             device float* decay [[buffer(10)]],
                             device bfloat* mixing [[buffer(11)]],
                             uint3 grp [[threadgroup_position_in_grid]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  const uint head = grp.y;
  const uint row = grp.z;
  if (row >= p.rows) return;

  const uint sid = batch_streams[row];
  const uint at4 = head * (DK / 4) + lane;
  const uint qkv_stride4 = p.qkv_dim >> 2;
  const uint proj_stride4 = p.in_proj_stride >> 2;
  device vec<bfloat, 4>* hist = all_histories + (size_t)sid * p.hist_stream_stride4;

  float4 t0 = float4(hist[0 * qkv_stride4 + at4]);
  float4 t1 = float4(hist[1 * qkv_stride4 + at4]);
  float4 t2 = float4(hist[2 * qkv_stride4 + at4]);
  float4 t3 = float4(in_proj_all[(size_t)row * proj_stride4 + at4]);

  hist[0 * qkv_stride4 + at4] = vec<bfloat, 4>(t1);
  hist[1 * qkv_stride4 + at4] = vec<bfloat, 4>(t2);
  hist[2 * qkv_stride4 + at4] = vec<bfloat, 4>(t3);

  float4 tap_values[4] = { t0, t1, t2, t3 };
  float act[4];
  UNROLL for (uint part = 0; part < 4; part++) {
    const float4 w = float4(weights[head * DK + lane * 4 + part]);
    float acc = 0.0f;
    UNROLL for (uint tap = 0; tap < 4; tap++) acc = fma(w[tap], tap_values[tap][part], acc);
    const float r = float(bfloat(acc));
    const float sig = 1.0f / (1.0f + exp(-r));
    act[part] = r * sig;
  }

  if (head < 2 * p.nk) {
    float sq = 0.0f;
    UNROLL for (uint part = 0; part < 4; part++) sq = fma(act[part], act[part], sq);
    sq = simd_sum(sq);
    const float inv = rsqrt(sq / float(DK) + 1e-6f);
    const float scale = (head < p.nk) ? (1.0f / float(DK)) : rsqrt(float(DK));
    float4 out;
    UNROLL for (uint part = 0; part < 4; part++) out[part] = act[part] * inv * scale;
    const uint pos = (row * p.nk + (head % p.nk)) * (DK / 4) + lane;
    if (head < p.nk) query[pos] = vec<bfloat, 4>(out);
    else key[pos] = vec<bfloat, 4>(out);
    return;
  }

  const uint vhead = head - 2 * p.nk;
  value[(row * p.nv + vhead) * (DV / 4) + lane] = vec<bfloat, 4>(float4(act[0], act[1], act[2], act[3]));
  if (lane == 0) {
    const device bfloat* proj_row = (const device bfloat*)in_proj_all + (size_t)row * p.in_proj_stride;
    const float bv = float(proj_row[p.b_offset + vhead]);
    const float av = float(proj_row[p.a_offset + vhead]);
    const float shifted = av + float(dt_bias[vhead]);
    const float sp = max(shifted, 0.0f) + log(1.0f + exp(-abs(shifted)));
    const uint cell = row * p.nv + vhead;
    decay[cell] = exp(-exp(a_log[vhead]) * sp);
    mixing[cell] = bfloat(1.0f / (1.0f + exp(-bv)));
  }
}

struct GdnBatchScanParams {
  uint rows;
  uint nk;
  uint nv;
  uint state_stream_stride4; // stride in float4 between stream states for this layer
};

// Lockstep multi-stream GDN recurrence: all B streams advance 1 step in parallel in-place!
// Grid: threadgroups (rows, DV / 4, nv) of 32 threads.
kernel void gdn_scan_batched(const device vec<bfloat, 4>* query [[buffer(0)]],
                             const device vec<bfloat, 4>* key [[buffer(1)]],
                             const device bfloat* value [[buffer(2)]],
                             const device float* decay [[buffer(3)]],
                             const device bfloat* mixing [[buffer(4)]],
                             device float4* all_states [[buffer(5)]],
                             const device uint* batch_streams [[buffer(6)]],
                             constant GdnBatchScanParams& p [[buffer(7)]],
                             device bfloat* output [[buffer(8)]],
                             uint3 grp [[threadgroup_position_in_grid]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint K = 4;
  const uint row = grp.x;
  const uint first = grp.y * K;
  const uint head = grp.z;
  if (row >= p.rows) return;

  const uint sid = batch_streams[row];
  const uint key_head = head / (p.nv / p.nk);
  device float4* st = all_states + (size_t)sid * p.state_stream_stride4;

  const uint cell = row * p.nv + head;
  const uint key_at = (row * p.nk + key_head) * (DK / 4) + lane;
  const float forget = decay[cell];
  const float mix = float(mixing[cell]);
  const float4 ki = float4(key[key_at]);
  const float4 qi = float4(query[key_at]);
  const float4 incoming = float4(*(const device vec<bfloat, 4>*)(value + (size_t)cell * DV + first));

  float4 results;
  UNROLL for (uint k = 0; k < K; k++) {
    const size_t s_idx = (size_t)(head * DV + first + k) * (DK / 4) + lane;
    float4 t = st[s_idx] * forget;
    const float remembered = simd_sum(dot(t, ki));
    const float correction = (incoming[k] - remembered) * mix;
    t = fma(ki, float4(correction), t);
    st[s_idx] = t;
    results[k] = simd_sum(dot(t, qi));
  }

  if (lane == 0) {
    *(device vec<bfloat, 4>*)(output + (size_t)cell * DV + first) = vec<bfloat, 4>(results);
  }
}

struct GdnFusedStepParams {
  uint nk;
  uint nv;
  uint qkv_dim;
  uint in_proj_stride;
  uint z_offset;
  uint b_offset;
  uint a_offset;
  uint sums_stride;
  uint write_sums;
  float eps;
};

// Super-fused single-token GDN step: collapses gdn_conv + gdn_scan_1 + gdn_norm_gate_sums
// into ONE SRAM-resident dispatch with ZERO intermediate DRAM round-trips!
// Uses lock-free ping-pong history buffers (`hist_in` read-only -> `hist_out` write-only)
// so all `nv` threadgroups (48 on 27B, 16 on 0.8B/2B) execute in parallel with zero races.
// Grid: threadgroups (nv, 1, 1) of 256 threads (8 simdgroups).
kernel void gdn_fused_step_1(const device vec<bfloat, 4>* in_proj_all [[buffer(0)]],
                             const device vec<bfloat, 4>* hist_in [[buffer(1)]],
                             device vec<bfloat, 4>* hist_out [[buffer(2)]],
                             const device vec<bfloat, 4>* conv_w [[buffer(3)]],
                             const device float* a_log [[buffer(4)]],
                             const device bfloat* dt_bias [[buffer(5)]],
                             const device bfloat* norm_w [[buffer(6)]],
                             device float4* state [[buffer(7)]],
                             device bfloat* act_out [[buffer(8)]],
                             device float* sums [[buffer(9)]],
                             constant GdnFusedStepParams& p [[buffer(10)]],
                             uint vhead [[threadgroup_position_in_grid]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  if (vhead >= p.nv) return;

  const uint ratio = p.nv / p.nk;
  const uint khead = vhead / ratio;
  const bool is_k_owner = (vhead % ratio) == 0;
  const uint qkv_stride4 = p.qkv_dim >> 2;

  threadgroup float4 tg_q[32];
  threadgroup float4 tg_k[32];
  threadgroup float4 tg_v[32];
  threadgroup float4 tg_out[32];
  threadgroup float tg_decay;
  threadgroup float tg_mix;

  // Phase 1: Parallel 4-tap depthwise conv1d + SiLU + L2 norm + ping-pong history shift.
  if (sg < 3) {
    const uint head = (sg == 0) ? khead : (sg == 1) ? (p.nk + khead) : (2 * p.nk + vhead);
    const uint at4 = head * (DK / 4) + lane;

    const float4 t0 = float4(hist_in[0 * qkv_stride4 + at4]);
    const float4 t1 = float4(hist_in[1 * qkv_stride4 + at4]);
    const float4 t2 = float4(hist_in[2 * qkv_stride4 + at4]);
    const float4 t3 = float4(in_proj_all[at4]);

    if (sg == 2 || is_k_owner) {
      hist_out[0 * qkv_stride4 + at4] = vec<bfloat, 4>(t1);
      hist_out[1 * qkv_stride4 + at4] = vec<bfloat, 4>(t2);
      hist_out[2 * qkv_stride4 + at4] = vec<bfloat, 4>(t3);
    }

    const float4 taps[4] = { t0, t1, t2, t3 };
    float act[4];
    UNROLL for (uint part = 0; part < 4; part++) {
      const float4 w = float4(conv_w[head * DK + lane * 4 + part]);
      float acc = 0.0f;
      UNROLL for (uint tap = 0; tap < 4; tap++) acc = fma(w[tap], taps[tap][part], acc);
      const float r = float(bfloat(acc));
      const float sig = 1.0f / (1.0f + exp(-r));
      act[part] = r * sig;
    }

    if (sg < 2) {
      float sq = 0.0f;
      UNROLL for (uint part = 0; part < 4; part++) sq = fma(act[part], act[part], sq);
      sq = simd_sum(sq);
      const float inv = rsqrt(sq / float(DK) + 1e-6f);
      const float scale = (sg == 0) ? (1.0f / float(DK)) : rsqrt(float(DK));
      float4 out;
      UNROLL for (uint part = 0; part < 4; part++) out[part] = act[part] * inv * scale;
      // Round through bfloat4 to stay 100% bit-exact with unfused gdn_conv -> gdn_scan
      const float4 rounded = float4(vec<bfloat, 4>(out));
      if (sg == 0) tg_q[lane] = rounded;
      else tg_k[lane] = rounded;
    } else {
      tg_v[lane] = float4(vec<bfloat, 4>(float4(act[0], act[1], act[2], act[3])));
      if (lane == 0) {
        const device bfloat* proj_row = (const device bfloat*)in_proj_all;
        const float bv = float(proj_row[p.b_offset + vhead]);
        const float av = float(proj_row[p.a_offset + vhead]);
        const float shifted = av + float(dt_bias[vhead]);
        const float sp = max(shifted, 0.0f) + log(1.0f + exp(-abs(shifted)));
        tg_decay = exp(-exp(a_log[vhead]) * sp);
        tg_mix = float(bfloat(1.0f / (1.0f + exp(-bv))));
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Phase 2: All 8 simdgroups update 16 columns each (4 batches x 4 cols) reusing Q/K in registers!
  const float4 qi = tg_q[lane];
  const float4 ki = tg_k[lane];
  const float forget = tg_decay;
  const float mix = tg_mix;

  UNROLL for (uint b = 0; b < 4; b++) {
    const uint first = sg * 16 + b * 4;
    const float4 incoming = tg_v[first >> 2];
    float4 results;
    UNROLL for (uint k = 0; k < 4; k++) {
      const size_t s_idx = (size_t)(vhead * DV + first + k) * (DK / 4) + lane;
      float4 t = state[s_idx] * forget;
      const float remembered = simd_sum(dot(t, ki));
      const float correction = (incoming[k] - remembered) * mix;
      t = fma(ki, float4(correction), t);
      state[s_idx] = t;
      results[k] = simd_sum(dot(t, qi));
    }
    if (lane == 0) {
      tg_out[first >> 2] = float4(vec<bfloat, 4>(results));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Phase 3: Per-head RMSNorm + SiLU(Z) gate straight to act_out (+ optional sums).
  if (sg == 0) {
    const threadgroup float* so = (const threadgroup float*)tg_out;
    const float x0 = so[lane];
    const float x1 = so[32 + lane];
    const float x2 = so[64 + lane];
    const float x3 = so[96 + lane];

    const float sq = simd_sum(fma(x0, x0, fma(x1, x1, fma(x2, x2, x3 * x3))));
    const float inv = rsqrt(sq / 128.0f + p.eps);

    const device bfloat* z_ptr = (const device bfloat*)in_proj_all + p.z_offset + (size_t)vhead * 128;
    const float z0 = float(z_ptr[lane]);
    const float z1 = float(z_ptr[32 + lane]);
    const float z2 = float(z_ptr[64 + lane]);
    const float z3 = float(z_ptr[96 + lane]);

    const float s0 = z0 / (1.0f + exp(-z0));
    const float s1 = z1 / (1.0f + exp(-z1));
    const float s2 = z2 / (1.0f + exp(-z2));
    const float s3 = z3 / (1.0f + exp(-z3));

    const bfloat y0 = bfloat((x0 * inv * float(norm_w[lane])) * s0);
    const bfloat y1 = bfloat((x1 * inv * float(norm_w[32 + lane])) * s1);
    const bfloat y2 = bfloat((x2 * inv * float(norm_w[64 + lane])) * s2);
    const bfloat y3 = bfloat((x3 * inv * float(norm_w[96 + lane])) * s3);

    device bfloat* dst = act_out + (size_t)vhead * 128;
    dst[lane] = y0;
    dst[32 + lane] = y1;
    dst[64 + lane] = y2;
    dst[96 + lane] = y3;

    if (p.write_sums != 0) {
      const float gs0 = simd_sum(float(y0) + float(y1));
      const float gs1 = simd_sum(float(y2) + float(y3));
      if (lane == 0) {
        sums[(size_t)(vhead * 2) * p.sums_stride] = gs0;
        sums[(size_t)(vhead * 2 + 1) * p.sums_stride] = gs1;
      }
    }
  }
}

// High-occupancy split single-token GDN conv+scan for small models (nv < 32, e.g. 0.8B & 2B).
// Splits each value head's 128 columns across 4 threadgroups -> 64 TGs (256 simdgroups) lighting up
// all 40 M5 Max GPU cores at full memory bandwidth! Followed by tiny gdn_norm_gate_sums.
// Grid: threadgroups (nv, 4, 1) of 128 threads (4 simdgroups).
kernel void gdn_conv_scan_split_1(const device vec<bfloat, 4>* in_proj_all [[buffer(0)]],
                                  const device vec<bfloat, 4>* hist_in [[buffer(1)]],
                                  device vec<bfloat, 4>* hist_out [[buffer(2)]],
                                  const device vec<bfloat, 4>* conv_w [[buffer(3)]],
                                  const device float* a_log [[buffer(4)]],
                                  const device bfloat* dt_bias [[buffer(5)]],
                                  device float4* state [[buffer(6)]],
                                  device bfloat* scan_out [[buffer(7)]],
                                  constant GdnFusedStepParams& p [[buffer(8)]],
                                  uint2 grp [[threadgroup_position_in_grid]],
                                  ushort sg [[simdgroup_index_in_threadgroup]],
                                  ushort lane [[thread_index_in_simdgroup]]) {
  const uint vhead = grp.x;
  const uint chunk = grp.y; // 0..3 (each chunk owns 32 of 128 value columns)
  if (vhead >= p.nv) return;

  const uint ratio = p.nv / p.nk;
  const uint khead = vhead / ratio;
  const bool is_k_owner = (vhead % ratio) == 0 && (chunk == 0);
  const bool is_v_owner = (chunk == 0);
  const uint qkv_stride4 = p.qkv_dim >> 2;

  threadgroup float4 tg_q[32];
  threadgroup float4 tg_k[32];
  threadgroup float4 tg_v[8]; // 32 value elements for this chunk (8 x float4)
  threadgroup float tg_decay;
  threadgroup float tg_mix;

  if (sg < 2) {
    const uint head = (sg == 0) ? khead : (p.nk + khead);
    const uint at4 = head * (DK / 4) + lane;
    const float4 t0 = float4(hist_in[0 * qkv_stride4 + at4]);
    const float4 t1 = float4(hist_in[1 * qkv_stride4 + at4]);
    const float4 t2 = float4(hist_in[2 * qkv_stride4 + at4]);
    const float4 t3 = float4(in_proj_all[at4]);
    if (is_k_owner) {
      hist_out[0 * qkv_stride4 + at4] = vec<bfloat, 4>(t1);
      hist_out[1 * qkv_stride4 + at4] = vec<bfloat, 4>(t2);
      hist_out[2 * qkv_stride4 + at4] = vec<bfloat, 4>(t3);
    }
    const float4 taps[4] = { t0, t1, t2, t3 };
    float act[4];
    UNROLL for (uint part = 0; part < 4; part++) {
      const float4 w = float4(conv_w[head * DK + lane * 4 + part]);
      float acc = 0.0f;
      UNROLL for (uint tap = 0; tap < 4; tap++) acc = fma(w[tap], taps[tap][part], acc);
      const float r = float(bfloat(acc));
      act[part] = r / (1.0f + exp(-r));
    }
    float sq = 0.0f;
    UNROLL for (uint part = 0; part < 4; part++) sq = fma(act[part], act[part], sq);
    sq = simd_sum(sq);
    const float inv = rsqrt(sq / float(DK) + 1e-6f);
    const float scale = (sg == 0) ? (1.0f / float(DK)) : rsqrt(float(DK));
    float4 out;
    UNROLL for (uint part = 0; part < 4; part++) out[part] = act[part] * inv * scale;
    const float4 rounded = float4(vec<bfloat, 4>(out));
    if (sg == 0) tg_q[lane] = rounded;
    else tg_k[lane] = rounded;
  } else if (sg == 2) {
    const uint head = 2 * p.nk + vhead;
    if (lane < 8) {
      const uint sub4 = chunk * 8 + lane;
      const uint at4 = head * (DV / 4) + sub4;
      const float4 t0 = float4(hist_in[0 * qkv_stride4 + at4]);
      const float4 t1 = float4(hist_in[1 * qkv_stride4 + at4]);
      const float4 t2 = float4(hist_in[2 * qkv_stride4 + at4]);
      const float4 t3 = float4(in_proj_all[at4]);
      // Each chunk owns its exact 8-vec4 slice of V history -> 100% race-free!
      hist_out[0 * qkv_stride4 + at4] = vec<bfloat, 4>(t1);
      hist_out[1 * qkv_stride4 + at4] = vec<bfloat, 4>(t2);
      hist_out[2 * qkv_stride4 + at4] = vec<bfloat, 4>(t3);
      const float4 taps[4] = { t0, t1, t2, t3 };
      float act[4];
      UNROLL for (uint part = 0; part < 4; part++) {
        const float4 w = float4(conv_w[head * DV + sub4 * 4 + part]);
        float acc = 0.0f;
        UNROLL for (uint tap = 0; tap < 4; tap++) acc = fma(w[tap], taps[tap][part], acc);
        const float r = float(bfloat(acc));
        act[part] = r / (1.0f + exp(-r));
      }
      tg_v[lane] = float4(vec<bfloat, 4>(float4(act[0], act[1], act[2], act[3])));
    }
    if (lane == 0) {
      const device bfloat* proj_row = (const device bfloat*)in_proj_all;
      const float bv = float(proj_row[p.b_offset + vhead]);
      const float av = float(proj_row[p.a_offset + vhead]);
      const float shifted = av + float(dt_bias[vhead]);
      const float sp = max(shifted, 0.0f) + log(1.0f + exp(-abs(shifted)));
      tg_decay = exp(-exp(a_log[vhead]) * sp);
      tg_mix = float(bfloat(1.0f / (1.0f + exp(-bv))));
    }
  }
  (void)is_v_owner;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float4 qi = tg_q[lane];
  const float4 ki = tg_k[lane];
  const float forget = tg_decay;
  const float mix = tg_mix;

  // All 4 simdgroups update 8 columns each (2 batches x 4 cols) = 32 columns per TG!
  UNROLL for (uint b = 0; b < 2; b++) {
    const uint local4 = sg * 2 + b;
    const uint first = chunk * 32 + local4 * 4;
    const float4 incoming = tg_v[local4];
    float4 results;
    UNROLL for (uint k = 0; k < 4; k++) {
      const size_t s_idx = (size_t)(vhead * DV + first + k) * (DK / 4) + lane;
      float4 t = state[s_idx] * forget;
      const float remembered = simd_sum(dot(t, ki));
      const float correction = (incoming[k] - remembered) * mix;
      t = fma(ki, float4(correction), t);
      state[s_idx] = t;
      results[k] = simd_sum(dot(t, qi));
    }
    if (lane == 0) {
      *(device vec<bfloat, 4>*)(scan_out + (size_t)vhead * DV + first) = vec<bfloat, 4>(results);
    }
  }
}

// ============================================================================
// TF-Rewind (Forget-Invert Duality) Kernel — Singularity-Free c_t-Inverse
//
// Theorem 1 (Singularity-Free Exact Delta Inverse via Correction Tape c_t):
//   Forward : S_t     = alpha_t * S_{t-1} + c_t k_t^T,  c_t = beta_t (v_t - alpha_t S_{t-1} k_t)
//   Backward: S_{t-1} = (S_t - c_t k_t^T) / alpha_t     (ZERO beta singularity, ZERO dot products!)
//
// Theorems 3-4 (Non-Stationary Minimax Duality & Runtime-Certified Replay):
//   With W = r * Delta and decay-drift ratio kappa = ln(alpha_pre) / ln(alpha_post):
//     min( E_back, E_fwd(W) ) <= u^( (r * kappa) / (r * kappa + 1) )  [u^(r/(r+1)) when kappa=1]
//   Any non-stationary head (kappa << 1) exceeding eps_tol is detected on the host from alpha_t
//   and adaptively extended backward via head_start[h] until prod alpha_t <= eps_tol (or t = 0).
//
// Per-head mode:
//   mode == 0 : Backward c_t-Inverse S_{cur_pos} -> S_{target_pos} (slow-decay heads)
//   mode == 1 : Forward Zero-Start Tail-Replay [head_start[h] .. target_pos) (fast-decay heads)
//   mode == 2 : Backward Sherman-Morrison Inverse without c_t tape (ablation baseline)
//   mode == 3 : Forward Anchor-Replay from checkpoint S_{anchor} (optional fallback)
//
// Grid: threadgroups (1, DV / 4, nv) = (1, 32, nv) of 32 threads.
// ============================================================================
struct GdnRewindParams {
  uint nk;
  uint nv;
  uint cur_pos;
  uint target_pos;
};

kernel void gdn_rewind_dual(const device vec<bfloat, 4>* tape_k [[buffer(0)]],
                            const device bfloat* tape_v [[buffer(1)]],
                            const device float* tape_decay [[buffer(2)]],
                            const device bfloat* tape_mix [[buffer(3)]],
                            const device float* tape_corr [[buffer(4)]],
                            const device uint* head_mode [[buffer(5)]],
                            const device uint* head_start [[buffer(6)]],
                            const device float4* anchor_state [[buffer(7)]],
                            device float4* state [[buffer(8)]],
                            constant GdnRewindParams& p [[buffer(9)]],
                            uint3 grp [[threadgroup_position_in_grid]],
                            ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint K = 4;
  const uint head = grp.z;
  const uint first = grp.y * K;
  const uint key_head = head / (p.nv / p.nk);
  const uint mode = head_mode[head];

  float4 s[K];
  if (mode == 0) {
    // Mode 0: Singularity-free Backward c_t-Inverse (1 FMA + 1 mul, ZERO simd_sum, ZERO beta singularity!)
    UNROLL for (uint k = 0; k < K; k++) {
      s[k] = state[(size_t)(head * DV + first + k) * (DK / 4) + lane];
    }
    for (int t = (int)p.cur_pos - 1; t >= (int)p.target_pos; t--) {
      const uint cell = (uint)t * p.nv + head;
      const uint key_at = ((uint)t * p.nk + key_head) * (DK / 4) + lane;
      const float inv_alpha = 1.0f / tape_decay[cell];
      const float4 ki = float4(tape_k[key_at]);
      const float4 ci = *(const device float4*)(tape_corr + (size_t)cell * DV + first);

      UNROLL for (uint k = 0; k < K; k++) {
        s[k] = fma(ki, float4(-ci[k]), s[k]) * inv_alpha;
      }
    }
  } else if (mode == 2) {
    // Mode 2: Pure Sherman-Morrison Inverse without c_t tape (ablation baseline showing beta singularity)
    UNROLL for (uint k = 0; k < K; k++) {
      s[k] = state[(size_t)(head * DV + first + k) * (DK / 4) + lane];
    }
    for (int t = (int)p.cur_pos - 1; t >= (int)p.target_pos; t--) {
      const uint cell = (uint)t * p.nv + head;
      const uint key_at = ((uint)t * p.nk + key_head) * (DK / 4) + lane;
      const float alpha = tape_decay[cell];
      const float beta = float(tape_mix[cell]);
      const float4 ki = float4(tape_k[key_at]);
      const float k_norm2 = simd_sum(dot(ki, ki));
      const float inv_alpha = 1.0f / alpha;
      const float gamma = beta / max(1.0f - beta * k_norm2, 1e-7f);
      const float4 vi = float4(*(const device vec<bfloat, 4>*)(tape_v + (size_t)cell * DV + first));

      UNROLL for (uint k = 0; k < K; k++) {
        const float sk = simd_sum(dot(s[k], ki));
        s[k] = fma(ki, float4(gamma * (sk - vi[k])), s[k]) * inv_alpha;
      }
    }
  } else {
    // Mode 1 (Zero-Start Tail-Replay) or Mode 3 (Anchor-Checkpoint Replay)
    if (mode == 1) {
      UNROLL for (uint k = 0; k < K; k++) s[k] = float4(0.0f);
    } else {
      UNROLL for (uint k = 0; k < K; k++) {
        s[k] = anchor_state[(size_t)(head * DV + first + k) * (DK / 4) + lane];
      }
    }
    const uint start_t = head_start[head];
    for (uint t = start_t; t < p.target_pos; t++) {
      const uint cell = t * p.nv + head;
      const uint key_at = (t * p.nk + key_head) * (DK / 4) + lane;
      const float forget = tape_decay[cell];
      const float mix = float(tape_mix[cell]);
      const float4 ki = float4(tape_k[key_at]);
      const float4 incoming = float4(*(const device vec<bfloat, 4>*)(tape_v + (size_t)cell * DV + first));

      UNROLL for (uint k = 0; k < K; k++) {
        float4 tmp = s[k] * forget;
        const float remembered = simd_sum(dot(tmp, ki));
        const float correction = (incoming[k] - remembered) * mix;
        s[k] = fma(ki, float4(correction), tmp);
      }
    }
  }

  UNROLL for (uint k = 0; k < K; k++) {
    state[(size_t)(head * DV + first + k) * (DK / 4) + lane] = s[k];
  }
}
