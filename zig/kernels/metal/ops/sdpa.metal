// Decode attention over a bf16 KV cache (head dim 128), bit-identical to MLX 0.32.3's sdpa_vector kernels.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
// tf:header

constant constexpr float TF_FLOOR = -0x1.fffffep+127f;   // -FLT_MAX: a running max before any key

// Online softmax update with one key's score: the running max, the sum of exps and the PER weighted values.
template <int PER>
inline void tf_softmax_step(float score, thread float& top, thread float& mass, thread float* acc,
                            const device bfloat* v) {
  const float peak = max(top, score);
  const float rescale = fast::exp(top - peak);
  const float weight = fast::exp(score - peak);
  top = peak;
  mass = mass * rescale + weight;
  for (int j = 0; j < PER; ++j) acc[j] = acc[j] * rescale + weight * v[j];
}

// sdpa_vector: simdgroup s takes keys s, s + 32, ..., lane l dims 4l..4l+3; 32 partials merge by lane transposition.
template <bool CAUSAL, bool QT>
inline void tf_sdpa_one_pass(const device bfloat* q, const device bfloat* k, const device bfloat* v, device bfloat* o,
                             int gqa, int n, long k_head, long k_seq, long v_head, long v_seq, float scale, uint3 tg,
                             uint3 grid_tg, uint sg, uint lane, threadgroup float* merged, threadgroup float* tops,
                             threadgroup float* masses) {
  const int head = int(tg.x), row = int(tg.y);
  const int rows = int(grid_tg.y);
  const int out_at = head * rows + row;
  const int q_at = QT ? int(grid_tg.x) * row + head : out_at;
  const int kv = head / gqa;
  float qv[4], acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int j = 0; j < 4; ++j) qv[j] = static_cast<float>(scale) * q[q_at * 128 + 4 * lane + j];
  float top = TF_FLOOR, mass = 0.0f;
  const device bfloat* kp = k + kv * k_head + long(sg) * k_seq + 4 * lane;
  const device bfloat* vp = v + kv * v_head + long(sg) * v_seq + 4 * lane;
  for (int i = int(sg); i < n; i += 32) {
    if (!CAUSAL || i <= n - rows + row) {
      float kr[4];
      for (int j = 0; j < 4; ++j) kr[j] = kp[j];
      float score = 0.0f;
      for (int j = 0; j < 4; ++j) score += qv[j] * kr[j];
      tf_softmax_step<4>(simd_sum(score), top, mass, acc, vp);
    }
    kp += 32 * int(k_seq);
    vp += 32 * int(v_seq);
  }
  if (lane == 0) {
    tops[sg] = top;
    masses[sg] = mass;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mine = tops[lane];
  const float rescale = fast::exp(mine - simd_max(mine));
  const float total = simd_sum(masses[lane] * rescale);
  for (int j = 0; j < 4; ++j) {
    merged[lane * 32 + sg] = acc[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc[j] = simd_sum(merged[sg * 32 + lane] * rescale);
    acc[j] = total == 0 ? acc[j] : (acc[j] / total);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0)
    for (int j = 0; j < 4; ++j) o[out_at * 128 + 4 * sg + j] = static_cast<bfloat>(acc[j]);
}

// sdpa_vector_2pass_1: block b takes keys b, b + blocks, ... and writes its unnormalised bf16 partial, sum and max.
template <bool CAUSAL, bool QT>
inline void tf_sdpa_blocks(const device bfloat* q, const device bfloat* k, const device bfloat* v, device bfloat* part,
                           device float* sums, device float* maxs, int n, int blocks, long k_head, long k_seq,
                           long v_head, long v_seq, float scale, uint3 tg, uint3 grid_tg, uint3 t_tg, uint3 tg_size,
                           uint lane) {
  const int kv = int(tg.x), batch = int(tg.y), block = int(tg.z);
  const int gqa = int(tg_size.y), rows = int(tg_size.z), row = int(t_tg.z);
  const int kv_heads = int(grid_tg.x), q_heads = kv_heads * gqa;
  const int head = batch * q_heads + gqa * kv + int(t_tg.y);
  const int out_at = head * rows + row;
  const int q_at = QT ? q_heads * row + head : out_at;
  const long kv_at = long(batch * kv_heads + kv);
  float qv[4], acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int j = 0; j < 4; ++j) qv[j] = static_cast<float>(scale) * q[q_at * 128 + 4 * lane + j];
  float top = TF_FLOOR, mass = 0.0f;
  const device bfloat* kp = k + kv_at * k_head + long(block) * k_seq + 4 * lane;
  const device bfloat* vp = v + kv_at * v_head + long(block) * v_seq + 4 * lane;
  for (int i = block; i < n; i += blocks) {
    if (!CAUSAL || i <= n - rows + row) {
      float score = 0.0f;
      for (int j = 0; j < 4; ++j) score += qv[j] * kp[j];
      tf_softmax_step<4>(simd_sum(score), top, mass, acc, vp);
    }
    kp += blocks * int(k_seq);
    vp += blocks * int(v_seq);
  }
  const int slot = out_at * blocks + block;
  if (lane == 0) {
    sums[slot] = mass;
    maxs[slot] = top;
  }
  for (int j = 0; j < 4; ++j) part[long(slot) * 128 + 4 * lane + j] = static_cast<bfloat>(acc[j]);
}

// sdpa_vector_2pass_1_gqa_16: block b's keys in two halves, 2 heads a simdgroup, halves merged before the bf16.
inline void tf_sdpa_blocks_gqa16(const device bfloat* q, const device bfloat* k, const device bfloat* v,
                                 device bfloat* part, device float* sums, device float* maxs, int n, long k_head,
                                 long k_seq, long v_head, long v_seq, float scale, uint3 tg, uint3 grid_tg, uint3 t_tg,
                                 uint lane, threadgroup float* o_sh, threadgroup float* se_sh,
                                 threadgroup float* mx_sh) {
  const int kv = int(tg.x), batch = int(tg.y), block = int(tg.z), blocks = int(grid_tg.z);
  const int g = int(t_tg.y), half_id = g / 8, h0 = (g % 8) * 2;
  const int kv_heads = int(grid_tg.x);
  const int base = batch * kv_heads * 16 + kv * 16;
  const long kv_at = long(batch * kv_heads + kv);
  const int chunk = (n + blocks - 1) / blocks;
  const int lo = block * chunk, hi = min(n, lo + chunk);
  const int sub = (chunk + 1) / 2;
  const int s0 = lo + half_id * sub, s1 = min(hi, s0 + sub);
  float qv[2][4], acc[2][4], top[2], mass[2];
  for (int h = 0; h < 2; ++h) {
    for (int j = 0; j < 4; ++j) {
      qv[h][j] = static_cast<float>(scale) * q[(base + h0 + h) * 128 + 4 * lane + j];
      acc[h][j] = 0.0f;
    }
    top[h] = TF_FLOOR;
    mass[h] = 0.0f;
  }
  const device bfloat* kp = k + kv_at * k_head + long(s0) * k_seq + 4 * lane;
  const device bfloat* vp = v + kv_at * v_head + long(s0) * v_seq + 4 * lane;
  for (int t = s0; t < s1; ++t) {
    float kr[4], vr[4];
    for (int j = 0; j < 4; ++j) kr[j] = kp[j];
    for (int j = 0; j < 4; ++j) vr[j] = vp[j];
    kp += k_seq;
    vp += v_seq;
    for (int h = 0; h < 2; ++h) {
      float score = 0.0f;
      for (int j = 0; j < 4; ++j) score += qv[h][j] * kr[j];
      score = simd_sum(score);
      const float peak = max(top[h], score);
      const float rescale = fast::exp(top[h] - peak);
      const float weight = fast::exp(score - peak);
      top[h] = peak;
      mass[h] = mass[h] * rescale + weight;
      for (int j = 0; j < 4; ++j) acc[h][j] = acc[h][j] * rescale + weight * vr[j];
    }
  }
  for (int h = 0; h < 2; ++h) {
    const int slot = (h0 + h) * 2 + half_id;
    const float inv = mass[h] > 0 ? 1 / mass[h] : 0;
    for (int j = 0; j < 4; ++j) o_sh[slot * 128 + 4 * lane + j] = acc[h][j] * inv;
    if (lane == 0) {
      se_sh[slot] = mass[h];
      mx_sh[slot] = top[h];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float peak = TF_FLOOR;
  for (int s = 0; s < 2; ++s) peak = max(peak, mx_sh[g * 2 + s]);
  float denom = 0, out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int s = 0; s < 2; ++s) {
    const float w = se_sh[g * 2 + s] * fast::exp(mx_sh[g * 2 + s] - peak);
    denom += w;
    for (int j = 0; j < 4; ++j) out[j] += w * o_sh[(g * 2 + s) * 128 + 4 * lane + j];
  }
  const int slot = (base + g) * blocks + block;
  for (int j = 0; j < 4; ++j) part[long(slot) * 128 + 4 * lane + j] = static_cast<bfloat>(out[j]);
  if (lane == 0) {
    sums[slot] = denom;
    maxs[slot] = peak;
  }
}

// sdpa_vector_2pass_2: simdgroup s folds blocks s, s + 32, ... (blocks a multiple of 32), merged by lane transposition.
inline void tf_sdpa_merge(const device bfloat* part, const device float* sums, const device float* maxs,
                          device bfloat* o, int blocks, uint3 tg, uint3 grid_tg, uint sg, uint lane,
                          threadgroup float* merged) {
  const int at = int(tg.x) * int(grid_tg.y) + int(tg.y);
  const device float* ms = maxs + long(at) * blocks;
  const device float* ss = sums + long(at) * blocks;
  float top = TF_FLOOR;
  for (int b = 0; b < blocks / 32; ++b) top = max(top, ms[lane + 32 * b]);
  top = simd_max(top);
  float total = 0.0f;
  for (int b = 0; b < blocks / 32; ++b) {
    const float f = fast::exp(ms[lane + 32 * b] - top);
    total += f * ss[lane + 32 * b];
  }
  total = simd_sum(total);
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const device bfloat* p = part + (long(at) * blocks + sg) * 128 + 4 * lane;
  const device float* mb = ms + sg;
  for (int b = 0; b < blocks / 32; ++b, p += 32 * 128, mb += 32) {
    const float f = fast::exp(mb[0] - top);
    for (int j = 0; j < 4; ++j) acc[j] += f * static_cast<float>(p[j]);
  }
  for (int j = 0; j < 4; ++j) {
    merged[lane * 32 + sg] = acc[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc[j] = simd_sum(merged[sg * 32 + lane]);
    acc[j] = total == 0 ? acc[j] : (acc[j] / total);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0)
    for (int j = 0; j < 4; ++j) o[at * 128 + 4 * sg + j] = static_cast<bfloat>(acc[j]);
}

// tf:kernel tf_sdpa_vec_d128 inputs=Q,K,V,P,ST,SC outputs=O
[[kernel]] void tf_sdpa_vec_d128(
  const device bfloat16_t* Q [[buffer(0)]],
  const device bfloat16_t* K [[buffer(1)]],
  const device bfloat16_t* V [[buffer(2)]],
  const constant int32_t* P [[buffer(3)]],
  const constant int64_t* ST [[buffer(4)]],
  const constant float* SC [[buffer(5)]],
  device bfloat16_t* O [[buffer(6)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 threadgroups_per_grid [[threadgroups_per_grid]]) {
  threadgroup float merged[32 * 32];
  threadgroup float tops[32];
  threadgroup float masses[32];
  tf_sdpa_one_pass<false, false>(Q, K, V, O, P[0], P[1], ST[0], ST[1], ST[2], ST[3], SC[0],
                                 threadgroup_position_in_grid, threadgroups_per_grid, simdgroup_index_in_threadgroup,
                                 thread_index_in_simdgroup, merged, tops, masses);
}

// tf:kernel tf_sdpa_vec_d128_cqt inputs=Q,K,V,P,ST,SC outputs=O
[[kernel]] void tf_sdpa_vec_d128_cqt(
  const device bfloat16_t* Q [[buffer(0)]],
  const device bfloat16_t* K [[buffer(1)]],
  const device bfloat16_t* V [[buffer(2)]],
  const constant int32_t* P [[buffer(3)]],
  const constant int64_t* ST [[buffer(4)]],
  const constant float* SC [[buffer(5)]],
  device bfloat16_t* O [[buffer(6)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 threadgroups_per_grid [[threadgroups_per_grid]]) {
  threadgroup float merged[32 * 32];
  threadgroup float tops[32];
  threadgroup float masses[32];
  tf_sdpa_one_pass<true, true>(Q, K, V, O, P[0], P[1], ST[0], ST[1], ST[2], ST[3], SC[0],
                               threadgroup_position_in_grid, threadgroups_per_grid, simdgroup_index_in_threadgroup,
                               thread_index_in_simdgroup, merged, tops, masses);
}

// tf:kernel tf_sdpa_2p1_d128 inputs=Q,K,V,P,ST,SC outputs=PART,SUMS,MAXS
[[kernel]] void tf_sdpa_2p1_d128(
  const device bfloat16_t* Q [[buffer(0)]],
  const device bfloat16_t* K [[buffer(1)]],
  const device bfloat16_t* V [[buffer(2)]],
  const constant int32_t* P [[buffer(3)]],
  const constant int64_t* ST [[buffer(4)]],
  const constant float* SC [[buffer(5)]],
  device bfloat16_t* PART [[buffer(6)]],
  device float* SUMS [[buffer(7)]],
  device float* MAXS [[buffer(8)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 threadgroups_per_grid [[threadgroups_per_grid]],
  uint3 threads_per_threadgroup [[threads_per_threadgroup]]) {
  tf_sdpa_blocks<false, false>(Q, K, V, PART, SUMS, MAXS, P[0], P[1], ST[0], ST[1], ST[2], ST[3], SC[0],
                               threadgroup_position_in_grid, threadgroups_per_grid, thread_position_in_threadgroup,
                               threads_per_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_sdpa_2p1_d128_cqt inputs=Q,K,V,P,ST,SC outputs=PART,SUMS,MAXS
[[kernel]] void tf_sdpa_2p1_d128_cqt(
  const device bfloat16_t* Q [[buffer(0)]],
  const device bfloat16_t* K [[buffer(1)]],
  const device bfloat16_t* V [[buffer(2)]],
  const constant int32_t* P [[buffer(3)]],
  const constant int64_t* ST [[buffer(4)]],
  const constant float* SC [[buffer(5)]],
  device bfloat16_t* PART [[buffer(6)]],
  device float* SUMS [[buffer(7)]],
  device float* MAXS [[buffer(8)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 threadgroups_per_grid [[threadgroups_per_grid]],
  uint3 threads_per_threadgroup [[threads_per_threadgroup]]) {
  tf_sdpa_blocks<true, true>(Q, K, V, PART, SUMS, MAXS, P[0], P[1], ST[0], ST[1], ST[2], ST[3], SC[0],
                             threadgroup_position_in_grid, threadgroups_per_grid, thread_position_in_threadgroup,
                             threads_per_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_sdpa_2p1_gqa16_d128 inputs=Q,K,V,P,ST,SC outputs=PART,SUMS,MAXS
[[kernel]] void tf_sdpa_2p1_gqa16_d128(
  const device bfloat16_t* Q [[buffer(0)]],
  const device bfloat16_t* K [[buffer(1)]],
  const device bfloat16_t* V [[buffer(2)]],
  const constant int32_t* P [[buffer(3)]],
  const constant int64_t* ST [[buffer(4)]],
  const constant float* SC [[buffer(5)]],
  device bfloat16_t* PART [[buffer(6)]],
  device float* SUMS [[buffer(7)]],
  device float* MAXS [[buffer(8)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 threadgroups_per_grid [[threadgroups_per_grid]]) {
  threadgroup float o_sh[16 * 2 * 128];
  threadgroup float se_sh[16 * 2];
  threadgroup float mx_sh[16 * 2];
  tf_sdpa_blocks_gqa16(Q, K, V, PART, SUMS, MAXS, P[0], ST[0], ST[1], ST[2], ST[3], SC[0],
                       threadgroup_position_in_grid, threadgroups_per_grid, thread_position_in_threadgroup,
                       thread_index_in_simdgroup, o_sh, se_sh, mx_sh);
}

// tf:kernel tf_sdpa_2p2_d128 inputs=PART,SUMS,MAXS,P outputs=O
[[kernel]] void tf_sdpa_2p2_d128(
  const device bfloat16_t* PART [[buffer(0)]],
  const device float* SUMS [[buffer(1)]],
  const device float* MAXS [[buffer(2)]],
  const constant int32_t* P [[buffer(3)]],
  device bfloat16_t* O [[buffer(4)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 threadgroups_per_grid [[threadgroups_per_grid]]) {
  threadgroup float merged[32 * 32];
  tf_sdpa_merge(PART, SUMS, MAXS, O, P[0], threadgroup_position_in_grid, threadgroups_per_grid,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup, merged);
}
