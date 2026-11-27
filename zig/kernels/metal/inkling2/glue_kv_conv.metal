// Paired K and V short convolutions in one dispatch, after glue.metal's inline bodies.

[[max_total_threads_per_threadgroup(32)]]
kernel void tf_inkling2_kv_conv_static(
    const device bfloat* k_taps [[buffer(0)]], const device bfloat* k_weight [[buffer(1)]],
    device bfloat* k_out [[buffer(2)]], const device bfloat* v_taps [[buffer(3)]],
    const device bfloat* v_weight [[buffer(4)]], device bfloat* v_out [[buffer(5)]],
    constant InkConvArgs& a [[buffer(6)]], uint3 at [[thread_position_in_grid]]) {
  if (at.z >= 2) return;
  const bool is_v = at.z != 0;
  ink_conv_static_at(is_v ? v_taps : k_taps, is_v ? v_weight : k_weight, is_v ? v_out : k_out, a, at.x, at.y);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_kv_conv_dynamic(
    const device bfloat* k_taps [[buffer(0)]], const device bfloat* k_input [[buffer(1)]],
    const device bfloat* k_static_w [[buffer(2)]], const device bfloat* k_projection [[buffer(3)]],
    device bfloat* k_out [[buffer(4)]], const device bfloat* v_taps [[buffer(5)]],
    const device bfloat* v_input [[buffer(6)]], const device bfloat* v_static_w [[buffer(7)]],
    const device bfloat* v_projection [[buffer(8)]], device bfloat* v_out [[buffer(9)]],
    constant InkConvArgs& a [[buffer(10)]], uint t [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  if (tg.y >= 2) return;
  threadgroup float projected[32];
  const bool is_v = tg.y != 0;
  ink_conv_dynamic_row(is_v ? v_taps : k_taps, is_v ? v_input : k_input,
                       is_v ? v_static_w : k_static_w, is_v ? v_projection : k_projection,
                       is_v ? v_out : k_out, a, t, tg.x, projected);
}
