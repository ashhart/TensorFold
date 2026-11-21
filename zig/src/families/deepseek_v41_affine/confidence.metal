#include <metal_stdlib>
using namespace metal;

constant constexpr uint ConfHidden = 5120;
constant constexpr uint ConfRank = 256;
constant constexpr uint ConfStreams = 4;
constant constexpr uint ConfThreads = 256;

struct ConfidenceArgs {
  uint step;
  uint cap;
  uint prune;
  uint markov_groups;
  uint row_base;
  float cost_ms[6];
  uint margin;
};

struct ConfidenceState {
  float survival;
  float expected;
  float best_rate;
  uint chosen;
  uint stop;
  uint score_step;
  uint full_passes;
  uint examined;
  float pending;
};

static float conf_sum(float value, uint t, uint lane, uint sg, threadgroup float* waves) {
  const float subtotal = simd_sum(value);
  if (lane == 0) waves[sg] = subtotal;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0;
  if (t == 0)
    for (uint i = 0; i < ConfThreads / 32; ++i) total += waves[i];
  return total;
}

// The confidence head consumes the collapsed residual before RMSNorm.
[[kernel]] void tf_confidence_hidden(const device bfloat* residual [[buffer(0)]],
                                     const device float* pre [[buffer(1)]],
                                     const device bfloat* weights [[buffer(2)]],
                                     device float* hidden [[buffer(3)]],
                                     uint row [[threadgroup_position_in_grid]],
                                     uint t [[thread_position_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float waves[ConfThreads / 32];
  const device bfloat* x = residual + size_t(row) * ConfStreams * ConfHidden;
  float value = 0;
  for (uint d = t; d < ConfHidden; d += ConfThreads) {
    float collapsed = 0;
    for (uint h = 0; h < ConfStreams; ++h)
      collapsed = fma(pre[row * ConfStreams + h], float(x[h * ConfHidden + d]), collapsed);
    value += float(bfloat(collapsed)) * float(weights[d]);
  }
  const float total = conf_sum(value, t, lane, sg, waves);
  if (t == 0) hidden[row] = total;
}

constant ushort2 upper_counts[125] = {
  ushort2(19, 3),
  ushort2(10, 1),
  ushort2(13, 1),
  ushort2(18, 3),
  ushort2(11, 5),
  ushort2(74, 17),
  ushort2(55, 4),
  ushort2(94, 22),
  ushort2(126, 21),
  ushort2(119, 59),
  ushort2(29, 7),
  ushort2(33, 10),
  ushort2(52, 18),
  ushort2(106, 38),
  ushort2(184, 108),
  ushort2(9, 5),
  ushort2(11, 6),
  ushort2(22, 16),
  ushort2(35, 21),
  ushort2(200, 145),
  ushort2(3, 2),
  ushort2(2, 2),
  ushort2(2, 1),
  ushort2(5, 4),
  ushort2(301, 277),
  ushort2(20, 3),
  ushort2(19, 1),
  ushort2(30, 3),
  ushort2(25, 2),
  ushort2(18, 6),
  ushort2(40, 7),
  ushort2(33, 8),
  ushort2(66, 15),
  ushort2(89, 24),
  ushort2(97, 48),
  ushort2(15, 6),
  ushort2(16, 6),
  ushort2(24, 14),
  ushort2(36, 16),
  ushort2(110, 73),
  ushort2(2, 2),
  ushort2(2, 1),
  ushort2(2, 0),
  ushort2(10, 9),
  ushort2(69, 56),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(1, 1),
  ushort2(69, 62),
  ushort2(16, 2),
  ushort2(13, 2),
  ushort2(16, 2),
  ushort2(24, 5),
  ushort2(11, 4),
  ushort2(23, 4),
  ushort2(22, 5),
  ushort2(30, 8),
  ushort2(47, 14),
  ushort2(54, 29),
  ushort2(5, 1),
  ushort2(5, 1),
  ushort2(7, 2),
  ushort2(13, 7),
  ushort2(40, 30),
  ushort2(0, 0),
  ushort2(2, 2),
  ushort2(1, 1),
  ushort2(2, 0),
  ushort2(20, 18),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(11, 11),
  ushort2(14, 2),
  ushort2(6, 0),
  ushort2(7, 1),
  ushort2(12, 0),
  ushort2(7, 3),
  ushort2(9, 3),
  ushort2(8, 3),
  ushort2(15, 5),
  ushort2(25, 10),
  ushort2(19, 13),
  ushort2(0, 0),
  ushort2(1, 0),
  ushort2(3, 1),
  ushort2(3, 1),
  ushort2(14, 9),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(2, 2),
  ushort2(2, 2),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(1, 1),
  ushort2(6, 0),
  ushort2(2, 0),
  ushort2(4, 2),
  ushort2(7, 0),
  ushort2(6, 1),
  ushort2(5, 2),
  ushort2(5, 1),
  ushort2(7, 2),
  ushort2(5, 1),
  ushort2(7, 4),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(2, 2),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(0, 0),
  ushort2(0, 0)
};

constant ushort2 leader_upper_spans[25] = {
  ushort2(0, 1),
  ushort2(1, 1),
  ushort2(2, 1),
  ushort2(3, 2),
  ushort2(5, 3),
  ushort2(8, 1),
  ushort2(9, 1),
  ushort2(10, 1),
  ushort2(11, 2),
  ushort2(13, 3),
  ushort2(16, 1),
  ushort2(17, 1),
  ushort2(18, 1),
  ushort2(19, 1),
  ushort2(20, 1),
  ushort2(21, 1),
  ushort2(22, 1),
  ushort2(23, 1),
  ushort2(24, 1),
  ushort2(25, 1),
  ushort2(26, 1),
  ushort2(27, 1),
  ushort2(28, 1),
  ushort2(29, 1),
  ushort2(30, 1)
};
constant float2 leader_upper_lines[31] = {
  float2(0.54122621564f, 0.26638477801f),
  float2(0.02421948912f, 0.51608325449f),
  float2(0.00834637454f, 0.64788732394f),
  float2(0.00586940572f, 0.75000000000f),
  float2(0.31604938272f, 0.55802469136f),
  float2(0.00251231624f, 0.92370802175f),
  float2(0.09609609610f, 0.83183183183f),
  float2(0.75294117647f, 0.24705882353f),
  float2(0.39384615385f, 0.30461538462f),
  float2(0.03421545041f, 0.58580593424f),
  float2(0.02003129890f, 0.71236306729f),
  float2(0.03425207386f, 0.78967085898f),
  float2(0.60952380952f, 0.37142857143f),
  float2(0.03291757747f, 0.86897261155f),
  float2(0.31683168317f, 0.61386138614f),
  float2(0.86195286195f, 0.13804713805f),
  float2(0.66149870801f, 0.19379844961f),
  float2(0.07441860465f, 0.59244186047f),
  float2(0.08080808081f, 0.68939393939f),
  float2(0.17582417582f, 0.74175824176f),
  float2(0.31334149327f, 0.68665850673f),
  float2(0.54700854701f, 0.21794871795f),
  float2(0.29527104960f, 0.59054209919f),
  float2(0.27826086957f, 0.47826086957f),
  float2(0.75294117647f, 0.24705882353f),
  float2(0.86195286195f, 0.13804713805f),
  float2(0.79012345679f, 0.16049382716f),
  float2(0.50493096647f, 0.37080867850f),
  float2(0.75294117647f, 0.24705882353f),
  float2(1.00000000000f, 0.00000000000f),
  float2(1.00000000000f, 0.00000000000f)
};

static bool rank_can_improve(ConfidenceState s, constant ConfidenceArgs& a, float p) {
  if (!isfinite(p)) return true;
  const uint ci = uint(p > 0.2f) + uint(p > 0.4f) + uint(p > 0.6f) + uint(p > 0.8f);
  const uint start = (a.step * 5 + ci) * 5;
  float upper = p; // the confidence fallback for a non-finite margin
  if (a.margin == 2) {
    const ushort2 span = leader_upper_spans[a.step * 5 + ci];
    for (uint i = 0; i < span.y; ++i) {
      const float2 line = leader_upper_lines[span.x + i];
      upper = max(upper, line.x * p + line.y);
    }
  } else {
    for (uint i = 0; i < 5; ++i) {
      const ushort2 counts = upper_counts[start + i];
      upper = max(upper, (32.0f * p + float(counts.y)) / (32.0f + float(counts.x)));
    }
  }
  upper = min(1.0f, upper + 0.0000004f);
  for (uint j = a.step + 1; j <= a.cap; ++j) {
    const float rate = (s.expected + float(j - a.step) * s.survival * upper) / a.cost_ms[j];
    if (rate >= s.best_rate - 0.000001f) return true;
  }
  return false;
}

[[kernel]] void tf_confidence_step(const device float* hidden [[buffer(0)]],
                                    const device bfloat* projection [[buffer(1)]],
                                    const device bfloat* markov_embed [[buffer(2)]],
                                    const device uint* draft_ids [[buffer(3)]],
                                    device ConfidenceState* state [[buffer(4)]],
                                    device uint4* dispatch [[buffer(5)]],
                                    constant ConfidenceArgs& a [[buffer(6)]],
                                    uint t [[thread_position_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float waves[ConfThreads / 32];
  if (a.step != 0 && a.step >= state->stop) {
    if (t == 0) {
      state->score_step = 0;
      dispatch[0] = uint4(1, 1, 1, 0);
    }
    return;
  }
  const uint row = a.row_base + a.step;
  const float value = float(projection[ConfHidden + t]) * float(markov_embed[size_t(draft_ids[row]) * ConfRank + t]);
  const float total = conf_sum(value, t, lane, sg, waves);
  if (t != 0) return;

  ConfidenceState s = a.step == 0
    ? ConfidenceState{1.0f, 1.0f, 1.0f / a.cost_ms[0], 0, a.cap, 0, 0, 0, 0.0f}
    : *state;
  const float raw = clamp(hidden[row] + total, -32.0f, 32.0f);
  const float probability = 1.0f / (1.0f + precise::exp(-raw));
  if (a.margin) {
    if (!rank_can_improve(s, a, probability)) {
      s.stop = a.step;
      s.score_step = 0;
      *state = s;
      dispatch[0] = uint4(1, 1, 1, 0);
      return;
    }
    s.pending = probability;
    s.score_step = 1;
    s.full_passes++;
    *state = s;
    dispatch[0] = uint4(a.markov_groups, 1, 1, 0);
    return;
  }
  s.survival *= probability;
  s.expected += s.survival;
  s.examined++;
  const uint depth = a.step + 1;
  const float rate = s.expected / a.cost_ms[depth];
  const bool improved = rate > s.best_rate;
  if (improved) {
    s.best_rate = rate;
    s.chosen = depth;
  }

  bool deeper = false;
  for (uint j = depth + 1; j <= a.cap; ++j) {
    const float optimistic_rate = (s.expected + float(j - depth) * s.survival) / a.cost_ms[j];
    // Leave near ties to the full path too.
    deeper = deeper || optimistic_rate >= s.best_rate - 0.000001f;
  }
  if (a.prune && !deeper) s.stop = depth;
  const bool score = !a.prune || improved || deeper;
  s.score_step = uint(score);
  s.full_passes += uint(score);
  *state = s;
  dispatch[0] = uint4(score ? a.markov_groups : 1, 1, 1, 0);
}

// The unchanged Markov winner rule, guarded so a skipped step consumes no stale tile scores.
[[kernel]] void tf_confidence_pick(const device float2* parts [[buffer(0)]],
                                   device uint* draft_ids [[buffer(1)]],
                                   const device ConfidenceState* state [[buffer(2)]],
                                   constant uint2& a [[buffer(3)]]) {
  if (!state->score_step) return;
  float best = -INFINITY;
  uint winner = 0xffffffffu;
  for (uint j = 0; j < a.x; ++j) {
    const float score = parts[j].x;
    const uint id = as_type<uint>(parts[j].y);
    if (score > best || (score == best && id < winner)) {
      best = score;
      winner = id;
    }
  }
  draft_ids[a.y + 1] = winner;
}
