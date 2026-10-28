// Nemotron's KV writes, tree compaction and prompt KV put over core-pool pages: main's kernels, the address changed.

// tf_kv_write through the stream's table; dims (QKV, KOFF, capacity, start) as tf_kv_write's, capacity unused.
kernel void nemotron_kv_write_paged(const device bfloat* qkv [[buffer(0)]],
                                    device uchar* kc [[buffer(1)]],
                                    device uchar* vc [[buffer(2)]],
                                    constant uint4& dims [[buffer(3)]],
                                    device const uint* table [[buffer(4)]],
                                    constant ulong& page_bytes [[buffer(5)]],
                                    uint3 pos [[thread_position_in_grid]],
                                    uint3 size [[threads_per_grid]]) {
  const uint d = pos.x, h = pos.y, t = pos.z, hd = size.x, kvh = size.y;
  const size_t src = size_t(t) * dims.x + dims.y + h * hd + d;
  const ulong dst = nemo_paged_row(table, dims.w + t, page_bytes, h, hd) + ulong(d) * 2;
  *(device bfloat*)(kc + dst) = qkv[src];
  *(device bfloat*)(vc + dst) = qkv[src + kvh * hd];
}

// tf_kv_compact through the stream's table: a kept path's rows len + path[j] move to len + j, in j's order.
kernel void nemotron_kv_compact_paged(device uchar* K [[buffer(0)]],
                                      device uchar* V [[buffer(1)]],
                                      const device int32_t* path [[buffer(2)]],
                                      constant uint4& dims [[buffer(3)]],
                                      device const uint* table [[buffer(4)]],
                                      constant ulong& page_bytes [[buffer(5)]],
                                      uint2 pos [[thread_position_in_grid]]) {
  const uint d = pos.x, hk = pos.y, len = dims.x, n = dims.y, hd = dims.w;
  for (uint j = 0; j < n; j++) {
    const uint src = len + uint(path[j]), dst = len + j;
    if (src == dst) continue;
    const ulong to = nemo_paged_row(table, dst, page_bytes, hk, hd) + ulong(d) * 2;
    const ulong from = nemo_paged_row(table, src, page_bytes, hk, hd) + ulong(d) * 2;
    *(device bfloat*)(K + to) = *(device bfloat*)(K + from);
    *(device bfloat*)(V + to) = *(device bfloat*)(V + from);
  }
}

// prefill's kv_put through the stream's table: P (L, kv_heads, dim, cap, at, old_cap, keep), kept rows from `old`.
kernel void nemotron_kv_put_paged(const device bfloat* src [[buffer(0)]],
                                  const device bfloat* old [[buffer(1)]],
                                  const device int* P [[buffer(2)]],
                                  device uchar* dst [[buffer(3)]],
                                  device const uint* table [[buffer(4)]],
                                  constant ulong& page_bytes [[buffer(5)]],
                                  uint3 at [[thread_position_in_grid]]) {
  const int d = int(at.x), r = int(at.y), kv = int(at.z), D = P[2], start = P[6] ? 0 : P[4], row = start + r;
  if (d >= D || row >= P[4] + P[0]) {
    return;
  }
  const ulong to = nemo_paged_row(table, uint(row), page_bytes, uint(kv), uint(D)) + ulong(d) * 2;
  *(device bfloat*)(dst + to) = row < P[4] ? old[(long(kv) * P[5] + row) * D + d] : src[(long(row - P[4]) * P[1] + kv) * D + d];
}
