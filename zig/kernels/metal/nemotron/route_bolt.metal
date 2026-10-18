// Nemotron 3.5 Bolt (nemotron_h_puzzle) router top-k: runtime expert count NE and picks K, so one kernel serves
// every Puzzle layer's own expert count (240/304/368/432/496, all NE % 32 == 16). The scoring is the same as
// route.metal (sigmoid of the gate logit, plus the per-expert correction bias, weights normalised and scaled),
// but the per-lane slices are bounded by NE instead of a compile-time NE/32 so the last partial slice is not
// dropped. Hand-authored for this PR (the generated route.metal bakes <128, 6>); keep in step with route.metal.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

// NE <= latent_max_experts (1024) -> at most 32 lane slices; K <= latent_max_picks (32).
constant constexpr uint MAX_SLICES = 32;
constant constexpr uint MAX_PICKS = 32;

[[kernel]] void tf_nemotron_route_bolt(
  const device bfloat16_t* G [[buffer(0)]],      // [rows, NE] gate logits
  const device float* bias [[buffer(1)]],        // [NE] e_score_correction_bias
  const constant float* scaling [[buffer(2)]],   // routed_scaling_factor
  device uint32_t* IDX [[buffer(3)]],            // [rows, K] picked expert ids
  device float* WT [[buffer(4)]],                // [rows, K] normalised, scaled weights
  const constant uint& NE [[buffer(5)]],
  const constant uint& K [[buffer(6)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

  const uint lane = thread_index_in_simdgroup;
  const uint r = threadgroup_position_in_grid.x;
  const uint slices = (NE + 31) / 32;            // ceil, so a partial last slice (NE % 32 != 0) is kept
  float sel[MAX_SLICES], prob[MAX_SLICES];
  for (uint j = 0; j < slices; j++) {
    const uint e = lane + 32 * j;
    if (e < NE) {
      const float g = float(G[r * NE + e]);
      prob[j] = 1.0f / (1.0f + metal::exp(-g));
      sel[j] = prob[j] + bias[e];
    } else {
      prob[j] = 0.0f;
      sel[j] = -INFINITY;                        // padded lanes never win
    }
  }
  float total = 0.0f;
  float picked[MAX_PICKS];
  for (uint k = 0; k < K; k++) {
    float best = -INFINITY;
    int best_e = 1 << 20;
    for (uint j = 0; j < slices; j++) {
      const uint e = lane + 32 * j;
      if (e < NE && sel[j] > best) { best = sel[j]; best_e = int(e); }
    }
    const float top = simd_max(best);
    const int winner = simd_min(best == top ? best_e : (1 << 20));
    float p = 0.0f;
    for (uint j = 0; j < slices; j++) {
      if (int(lane + 32 * j) == winner) { p = prob[j]; sel[j] = -INFINITY; }
    }
    p = simd_sum(p);
    picked[k] = p;
    total += p;
    if (lane == 0) IDX[r * K + k] = uint(winner);
  }
  if (lane == 0) {
    const float denominator = total + 1e-20f;
    for (uint k = 0; k < K; k++) WT[r * K + k] = picked[k] / denominator * scaling[0];
  }
}
