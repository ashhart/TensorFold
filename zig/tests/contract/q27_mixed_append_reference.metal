// Frozen from supplied base ec344c2bab: contiguous value rows before mixed addressing.
kernel void q27_attn_append_before_mixed(device const bfloat* K [[buffer(0)]], device const bfloat* V [[buffer(1)]],
    device const Q27CachePtrs* caches [[buffer(2)]], device const int* meta [[buffer(3)]], device const int* nodes [[buffer(4)]],
    constant Q27PackArgs& a [[buffer(5)]], uint3 at [[thread_position_in_grid]]) {
  if (at.x >= a.dim || at.y >= a.kv_heads || at.z >= a.rows) return;
  const int st = nodes[4 * at.z + 1], base = 8 + st * 12;
  const uint local = at.z - uint(meta[base + 6]), row = uint(meta[base + 4]) + local;
  caches[st].keys[ulong(at.y) * uint(meta[base + 7]) + row * a.dim + at.x] = K[(at.z * a.kv_heads + at.y) * a.dim + at.x];
  caches[st].values[ulong(at.y) * uint(meta[base + 8]) + row * a.dim + at.x] = V[(at.z * a.kv_heads + at.y) * a.dim + at.x];
}
