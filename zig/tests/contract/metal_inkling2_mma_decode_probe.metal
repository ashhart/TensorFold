// Contract-only entry calls the production packed MMA decoder and exposes its actual BF16 operand bits.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_mma_decode_exact(
    device ushort* out [[buffer(0)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  if (lane != 0u) return;
  const uint base = group * 16u, first = base & 255u;
  Ink2MmaRaw raw = {base >> 8u, uint4(0u)};
  for (uint w = 0u; w < 4u; ++w)
    for (uint b = 0u; b < 4u; ++b) raw.q[w] |= (first + 4u * w + b) << (8u * b);
  threadgroup uint words[INK2_MMA_WORDS];
  ink2_mma_decode<false>(raw, 1.0f, 0u, words);
  for (uint w = 0u; w < 4u; ++w) {
    const uint lo = words[w], hi = words[INK2_MMA_ROWS * 4u + w];
    out[base + 2u * w] = ushort(lo);
    out[base + 2u * w + 1u] = ushort(lo >> 16u);
    out[base + 8u + 2u * w] = ushort(hi);
    out[base + 8u + 2u * w + 1u] = ushort(hi >> 16u);
  }
}
