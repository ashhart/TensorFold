using tf_metal_learn::Context;
struct tf_gdn_norm_back_args {
  GdnShape s;
  float eps;
};
[[max_total_threads_per_threadgroup(128)]]
kernel void tf_gdn_norm_back(
  const device bf16* y [[buffer(0)]],
  const device bf16* P [[buffer(1)]],
  const device bf16* norm_w [[buffer(2)]],
  const device float* dout [[buffer(3)]],
  device float* dy [[buffer(4)]],
  device float* dproj [[buffer(5)]],
  constant tf_gdn_norm_back_args& a [[buffer(6)]],
  uint3 threadIdx [[thread_position_in_threadgroup]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 blockDim [[threads_per_threadgroup]]) {
  threadgroup float memory[161];
  Context c = {threadIdx, blockIdx, blockDim, memory};
  c.tf_gdn_norm_back(y, P, norm_w, dout, dy, dproj, a.s, a.eps);
}
struct tf_gdn_scan_back_args {
  GdnShape s;
  int from;
};
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_gdn_scan_back(
  const device bf16* act [[buffer(0)]],
  const device float* q [[buffer(1)]],
  const device float* k [[buffer(2)]],
  const device float* g [[buffer(3)]],
  const device float* beta [[buffer(4)]],
  const device float* ckpt [[buffer(5)]],
  const device float* dy [[buffer(6)]],
  device float* work [[buffer(7)]],
  device float* dqp [[buffer(8)]],
  device float* dkp [[buffer(9)]],
  device float* dact [[buffer(10)]],
  device float* dg [[buffer(11)]],
  device float* dbeta [[buffer(12)]],
  constant tf_gdn_scan_back_args& a [[buffer(13)]],
  uint3 threadIdx [[thread_position_in_threadgroup]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 blockDim [[threads_per_threadgroup]]) {
  threadgroup float memory[2384];
  Context c = {threadIdx, blockIdx, blockDim, memory};
  c.tf_gdn_scan_back(act, q, k, g, beta, ckpt, dy, work, dqp, dkp, dact, dg, dbeta, a.s, a.from);
}
struct tf_gdn_qk_back_args {
  GdnShape s;
};
[[max_total_threads_per_threadgroup(128)]]
kernel void tf_gdn_qk_back(
  const device bf16* act [[buffer(0)]],
  const device float* dqp [[buffer(1)]],
  const device float* dkp [[buffer(2)]],
  device float* dact [[buffer(3)]],
  constant tf_gdn_qk_back_args& a [[buffer(4)]],
  uint3 threadIdx [[thread_position_in_threadgroup]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 blockDim [[threads_per_threadgroup]]) {
  threadgroup float memory[32];
  Context c = {threadIdx, blockIdx, blockDim, memory};
  c.tf_gdn_qk_back(act, dqp, dkp, dact, a.s);
}
struct tf_gdn_gates_back_args {
  GdnShape s;
};
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_gdn_gates_back(
  const device bf16* P [[buffer(0)]],
  const device float* a_log [[buffer(1)]],
  const device float* dt_bias [[buffer(2)]],
  const device float* g [[buffer(3)]],
  const device float* dg [[buffer(4)]],
  const device float* dbeta [[buffer(5)]],
  device float* dproj [[buffer(6)]],
  constant tf_gdn_gates_back_args& a [[buffer(7)]],
  uint3 threadIdx [[thread_position_in_threadgroup]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 blockDim [[threads_per_threadgroup]]) {
  threadgroup float memory[1];
  Context c = {threadIdx, blockIdx, blockDim, memory};
  c.tf_gdn_gates_back(P, a_log, dt_bias, g, dg, dbeta, dproj, a.s);
}
struct tf_gdn_conv_back_args {
  GdnShape s;
};
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_gdn_conv_back(
  const device bf16* P [[buffer(0)]],
  const device bf16* hist [[buffer(1)]],
  const device bf16* cw [[buffer(2)]],
  const device float* dact [[buffer(3)]],
  device float* dproj [[buffer(4)]],
  constant tf_gdn_conv_back_args& a [[buffer(5)]],
  uint3 threadIdx [[thread_position_in_threadgroup]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 blockDim [[threads_per_threadgroup]]) {
  threadgroup float memory[1];
  Context c = {threadIdx, blockIdx, blockDim, memory};
  c.tf_gdn_conv_back(P, hist, cw, dact, dproj, a.s);
}
