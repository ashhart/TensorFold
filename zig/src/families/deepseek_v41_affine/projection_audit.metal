[[kernel]] void tf_attention_projection_audit(const device ushort* actual [[buffer(0)]],
                                              const device ushort* expected [[buffer(1)]],
                                              device atomic_uint* mismatch [[buffer(2)]],
                                              constant uint& len [[buffer(3)]],
                                              uint i [[thread_position_in_grid]]) {
    if (i < len && actual[i] != expected[i])
        atomic_fetch_or_explicit(mismatch, 1u, memory_order_relaxed);
}
