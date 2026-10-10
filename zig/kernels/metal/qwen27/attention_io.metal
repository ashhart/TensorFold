// Packing and cache movement copy BF16 bits only; non-prefix commits gather before scattering into overlapping caches.
#include <metal_stdlib>
using namespace metal;
struct Q27CachePtrs { device bfloat* keys; device bfloat* values; };
struct Q27PackArgs { uint rows, packed_rows, heads, kv_heads, dim, group; };
struct Q27KeepRow { uint stream, source, destination, row; };

kernel void q27_attn_pack_a(device const bfloat* Q [[buffer(0)]], device const int* rows [[buffer(1)]],
    device bfloat* QA [[buffer(2)]], constant Q27PackArgs& a [[buffer(3)]], uint3 at [[thread_position_in_grid]]) {
  if (at.x >= a.dim || at.y >= a.packed_rows || at.z >= a.kv_heads) return;
  const uint mapped = uint(rows[at.y]), node = mapped / a.group, head = at.z * a.group + mapped % a.group;
  QA[(at.z * a.packed_rows + at.y) * a.dim + at.x] = Q[(node * a.heads + head) * a.dim + at.x];
}
kernel void q27_attn_pack_b(device const bfloat* Q [[buffer(0)]], device bfloat* QB [[buffer(1)]],
    constant Q27PackArgs& a [[buffer(2)]], uint3 at [[thread_position_in_grid]]) {
  if (at.x >= a.dim || at.y >= 16 || at.z >= a.kv_heads * a.rows) return;
  const uint hk = at.z / a.rows, node = at.z % a.rows;
  QB[((hk * a.rows + node) * 16 + at.y) * a.dim + at.x] = at.y < a.group ? Q[(node * a.heads + hk * a.group + at.y) * a.dim + at.x] : bfloat(0.0f);
}
kernel void q27_attn_append(device const bfloat* K [[buffer(0)]], device const bfloat* V [[buffer(1)]],
    device const Q27CachePtrs* caches [[buffer(2)]], device const int* meta [[buffer(3)]], device const int* nodes [[buffer(4)]],
    constant Q27PackArgs& a [[buffer(5)]], uint3 at [[thread_position_in_grid]]) {
  if (at.x >= a.dim || at.y >= a.kv_heads || at.z >= a.rows) return;
  const int st = nodes[2 * at.z + 1], base = 8 + st * 12;
  const uint local = at.z - uint(meta[base + 6]), row = uint(meta[base + 4]) + local;
  caches[st].keys[ulong(at.y) * uint(meta[base + 7]) + row * a.dim + at.x] = K[(at.z * a.kv_heads + at.y) * a.dim + at.x];
  caches[st].values[ulong(at.y) * uint(meta[base + 8]) + row * a.dim + at.x] = V[(at.z * a.kv_heads + at.y) * a.dim + at.x];
}
kernel void q27_attn_gather(device const Q27CachePtrs* caches [[buffer(0)]], device const int* meta [[buffer(1)]],
    device const Q27KeepRow* map [[buffer(2)]], device bfloat* K [[buffer(3)]], device bfloat* V [[buffer(4)]],
    constant uint& kept [[buffer(5)]], uint3 at [[thread_position_in_grid]]) {
  if (at.x >= 256 || at.y >= 4 || at.z >= kept) return;
  const Q27KeepRow r = map[at.z]; if (r.stream == 0xffffffffu) return; const uint base = 8 + r.stream * 12, source = uint(meta[base + 4]) + r.source;
  const ulong destination = (ulong(r.row) * 4 + at.y) * 256 + at.x;
  K[destination] = caches[r.stream].keys[ulong(at.y) * uint(meta[base + 7]) + source * 256 + at.x];
  V[destination] = caches[r.stream].values[ulong(at.y) * uint(meta[base + 8]) + source * 256 + at.x];
}
kernel void q27_attn_scatter(device const bfloat* K [[buffer(0)]], device const bfloat* V [[buffer(1)]],
    device const Q27CachePtrs* caches [[buffer(2)]], device const int* meta [[buffer(3)]], device const Q27KeepRow* map [[buffer(4)]],
    constant uint& kept [[buffer(5)]], uint3 at [[thread_position_in_grid]]) {
  if (at.x >= 256 || at.y >= 4 || at.z >= kept) return;
  const Q27KeepRow r = map[at.z]; if (r.stream == 0xffffffffu) return; const uint base = 8 + r.stream * 12, destination = uint(meta[base + 4]) + r.destination;
  const ulong source = (ulong(r.row) * 4 + at.y) * 256 + at.x;
  caches[r.stream].keys[ulong(at.y) * uint(meta[base + 7]) + destination * 256 + at.x] = K[source];
  caches[r.stream].values[ulong(at.y) * uint(meta[base + 8]) + destination * 256 + at.x] = V[source];
}
