using tf_metal_learn::Context;
struct tf_attn_q_args {
  Heads a;
};
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_attn_q(
  const device bf16* qkv [[buffer(0)]],
  const device float* dout [[buffer(1)]],
  device float* probs [[buffer(2)]],
  device float* dscores [[buffer(3)]],
  device float* dq [[buffer(4)]],
  const device bf16* pk [[buffer(5)]],
  const device bf16* pv [[buffer(6)]],
  constant tf_attn_q_args& a [[buffer(7)]],
  uint3 threadIdx [[thread_position_in_threadgroup]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 blockDim [[threads_per_threadgroup]]) {
  threadgroup float memory[4640];
  Context c = {threadIdx, blockIdx, blockDim, memory};
  c.tf_attn_q(qkv, dout, probs, dscores, dq, a.a, pk, pv);
}
struct tf_attn_gate_back_args {
  int heads;
  int dim;
};
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_attn_gate_back(
  const device float* dgated [[buffer(0)]],
  const device bf16* gated [[buffer(1)]],
  const device bf16* pq [[buffer(2)]],
  device float* dout [[buffer(3)]],
  device float* dpq [[buffer(4)]],
  constant tf_attn_gate_back_args& a [[buffer(5)]],
  uint3 threadIdx [[thread_position_in_threadgroup]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 blockDim [[threads_per_threadgroup]]) {
  threadgroup float memory[1];
  Context c = {threadIdx, blockIdx, blockDim, memory};
  c.tf_attn_gate_back(dgated, gated, pq, dout, dpq, a.heads, a.dim);
}
struct tf_attn_prep_back_args {
  int pos0;
  int heads;
  int kv_heads;
  int dim;
  int rope_half;
  float eps;
};
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_attn_prep_back(
  const device bf16* pq [[buffer(0)]],
  const device bf16* pk [[buffer(1)]],
  const device float* qscale [[buffer(2)]],
  const device float* kscale [[buffer(3)]],
  const device float* inv [[buffer(4)]],
  const device float* dq [[buffer(5)]],
  const device float* dk [[buffer(6)]],
  device float* dpq [[buffer(7)]],
  device float* dpk [[buffer(8)]],
  constant tf_attn_prep_back_args& a [[buffer(9)]],
  uint3 threadIdx [[thread_position_in_threadgroup]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 blockDim [[threads_per_threadgroup]]) {
  threadgroup float memory[288];
  Context c = {threadIdx, blockIdx, blockDim, memory};
  c.tf_attn_prep_back(pq, pk, qscale, kscale, inv, a.pos0, dq, dk, dpq, dpk, a.heads, a.kv_heads, a.dim, a.rope_half, a.eps);
}
