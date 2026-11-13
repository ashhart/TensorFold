// The stored BF16 post outputs are summed in tf_lane_xsum's group order.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_delta_gdn_post_sums(
    device const bfloat* recurrence [[buffer(0)]], device const bfloat* gate [[buffer(1)]],
    device const void* norm [[buffer(2)]], constant TfDeltaParams& p [[buffer(3)]],
    device bfloat* output [[buffer(4)]], device float* XS [[buffer(5)]], constant uint& stride [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  tf_delta_gdn_post_body<4>(recurrence, gate, norm, p, output, group, lane);
  threadgroup_barrier(mem_flags::mem_device);
  if ((lane & 15) == 0) {
    const uint g = group.y * 2 + (lane >> 4);
    const device bfloat* row = output + size_t(group.z) * p.nv * 128;
    float acc = 0.0f;
    for (int i = 0; i < 64; i++) acc += float(row[g * 64 + i]);
    XS[g * stride + group.z] = acc;
  }
}
