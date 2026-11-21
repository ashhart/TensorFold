[[kernel]] void tf_hc_coefficients_audit(
    const device ushort* norm [[buffer(0)]], const device uint* post [[buffer(1)]],
    const device uint* pre [[buffer(2)]], const device uint* comb [[buffer(3)]],
    const device ushort* base_norm [[buffer(4)]], const device uint* base_post [[buffer(5)]],
    const device uint* base_pre [[buffer(6)]], const device uint* base_comb [[buffer(7)]],
    device atomic_uint* differs [[buffer(8)]], constant uint& rows [[buffer(9)]],
    uint i [[thread_position_in_grid]]) {
  uint mask = norm[i] != base_norm[i] ? 1u : 0u;
  if (i < rows * S) {
    if (post[i] != base_post[i]) mask |= 2u;
    if (pre[i] != base_pre[i]) mask |= 4u;
  }
  if (i < rows * S * S && comb[i] != base_comb[i]) mask |= 8u;
  if (mask != 0)
    atomic_fetch_or_explicit(differs, mask, memory_order_relaxed);
}
