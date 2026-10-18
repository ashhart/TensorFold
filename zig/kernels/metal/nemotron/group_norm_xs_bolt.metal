// nemotron_h_puzzle (Bolt): the Mamba gated group norm and its 64-group sums, as group_norm_xs_<MP>.metal but at
// Bolt's mamba inner width XD = 128 heads x 64 = 8192 in 8 groups of GS = 1024. Hand-authored for this PR (the
// generated files bake XD = 4096, GS = 512 and one MP each); one template, one instantiation per padded row count.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

template <int MP>
[[kernel]] void tf_nemotron_group_norm_xs_bolt(
  const device bfloat16_t* X [[buffer(0)]],
  const device bfloat16_t* W [[buffer(1)]],
  const constant float* eps [[buffer(2)]],
  const device int32_t* dims [[buffer(3)]],
  device bfloat16_t* OUT [[buffer(4)]],
  device float* XS [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  constexpr int XD = 8192;
  constexpr int GS = 1024;

  const uint t = thread_position_in_threadgroup.x;
  const uint grp = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;
  constexpr int SUB = GS / 64;
  if (int(r) >= dims[0]) {
    for (int s = int(thread_position_in_threadgroup.x); s < SUB; s += GS / 4) XS[(int(grp) * SUB + s) * MP + int(r)] = 0.0f;
    return;
  }
  threadgroup bfloat outv[GS];
  constexpr int T = GS / 4;
  threadgroup float partial[T / 32];
  const int base = int(r) * XD + int(grp) * GS + int(t) * 4;
  float v[4];
  float ss = 0.0f;
  for (int i = 0; i < 4; i++) { v[i] = float(X[base + i]); ss = fma(v[i], v[i], ss); }
  ss = simd_sum(ss);
  if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int s = 0; s < T / 32; s++) total += partial[s];
  const float scale = metal::rsqrt(total / float(GS) + eps[0]);
  for (int i = 0; i < 4; i++) {
    const int c = int(grp) * GS + int(t) * 4 + i;
    const bfloat o = bfloat(float(W[c]) * float(bfloat(v[i] * scale)));
    OUT[base + i] = o;
    outv[int(t) * 4 + i] = o;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) < SUB) {
    float acc = 0.0f;
    for (int i = 0; i < 64; i++) acc += float(outv[int(t) * 64 + i]);
    XS[(int(grp) * SUB + int(t)) * MP + int(r)] = acc;
  }
}

template [[host_name("tf_nemotron_group_norm_xs_bolt_16")]] [[kernel]] decltype(tf_nemotron_group_norm_xs_bolt<16>) tf_nemotron_group_norm_xs_bolt<16>;
template [[host_name("tf_nemotron_group_norm_xs_bolt_32")]] [[kernel]] decltype(tf_nemotron_group_norm_xs_bolt<32>) tf_nemotron_group_norm_xs_bolt<32>;
template [[host_name("tf_nemotron_group_norm_xs_bolt_48")]] [[kernel]] decltype(tf_nemotron_group_norm_xs_bolt<48>) tf_nemotron_group_norm_xs_bolt<48>;
template [[host_name("tf_nemotron_group_norm_xs_bolt_64")]] [[kernel]] decltype(tf_nemotron_group_norm_xs_bolt<64>) tf_nemotron_group_norm_xs_bolt<64>;
template [[host_name("tf_nemotron_group_norm_xs_bolt_80")]] [[kernel]] decltype(tf_nemotron_group_norm_xs_bolt<80>) tf_nemotron_group_norm_xs_bolt<80>;
template [[host_name("tf_nemotron_group_norm_xs_bolt_96")]] [[kernel]] decltype(tf_nemotron_group_norm_xs_bolt<96>) tf_nemotron_group_norm_xs_bolt<96>;
template [[host_name("tf_nemotron_group_norm_xs_bolt_112")]] [[kernel]] decltype(tf_nemotron_group_norm_xs_bolt<112>) tf_nemotron_group_norm_xs_bolt<112>;
template [[host_name("tf_nemotron_group_norm_xs_bolt_128")]] [[kernel]] decltype(tf_nemotron_group_norm_xs_bolt<128>) tf_nemotron_group_norm_xs_bolt<128>;
