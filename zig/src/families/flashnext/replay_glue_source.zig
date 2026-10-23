//! Unchanged embedded shader source used by replay compilation.
pub const text =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\// meta: position of row 0, key capacity, rows
    \\kernel void fz_kv_write(device const bfloat* kout [[buffer(0)]], device const bfloat* p [[buffer(1)]],
    \\    device bfloat* keys [[buffer(2)]], device bfloat* vals [[buffer(3)]], device bfloat* raw [[buffer(4)]],
    \\    constant uint* meta [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    \\  const uint pos = meta[0], cap = meta[1], rows = meta[2];
    \\  const uint r = i >> 9, j = i & 511;
    \\  if (r >= rows) return;
    \\  const uint at = ((j >> 8) * cap + pos + r) * 256 + (j & 255);
    \\  keys[at] = kout[r * 512 + j];
    \\  vals[at] = p[r * 13952 + 12800 + j];
    \\  if (j < 128) raw[(pos + r) * 128 + j] = p[r * 13952 + 13824 + j];
    \\}
    \\kernel void fz_argmax(device const bfloat* logits [[buffer(0)]], device uint* out [[buffer(1)]],
    \\    constant uint& n [[buffer(2)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    \\    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    \\  device const bfloat* x = logits + row * n;
    \\  float best = -INFINITY; uint at = 0xffffffffu;
    \\  for (uint i = t; i < n; i += 1024) { const float v = float(x[i]); if (v > best) { best = v; at = i; } }
    \\  for (ushort o = 16; o > 0; o >>= 1) {
    \\    const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(at, o);
    \\    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
    \\  }
    \\  threadgroup float vb[32]; threadgroup uint va[32];
    \\  if (lane == 0) { vb[sg] = best; va[sg] = at; }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (sg != 0) return;
    \\  best = vb[lane]; at = va[lane];
    \\  for (ushort o = 16; o > 0; o >>= 1) {
    \\    const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(at, o);
    \\    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
    \\  }
    \\  if (lane == 0) out[row] = at;
    \\}
    \\// x [R, 4 * 2560] = e [R, 2560] broadcast over the streams + hs [4R, 2560] (MLX's bf16 add)
    \\kernel void fz_bcast_add(device const bfloat* e [[buffer(0)]], device const bfloat* hs [[buffer(1)]],
    \\    device bfloat* x [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    \\  if (i >= n) return;
    \\  x[i] = bfloat(float(e[(i / 10240) * 2560 + i % 2560]) + float(hs[i]));
    \\}
    \\kernel void fz_argmax_ids(device const bfloat* logits [[buffer(0)]], device uint* out [[buffer(1)]],
    \\    constant uint& n [[buffer(2)]], device const uint* ids [[buffer(3)]], uint t [[thread_index_in_threadgroup]],
    \\    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    \\  float best = -INFINITY; uint at = 0xffffffffu;
    \\  for (uint i = t; i < n; i += 1024) { const float v = float(logits[i]); if (v > best) { best = v; at = i; } }
    \\  for (ushort o = 16; o > 0; o >>= 1) {
    \\    const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(at, o);
    \\    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
    \\  }
    \\  threadgroup float vb[32]; threadgroup uint va[32];
    \\  if (lane == 0) { vb[sg] = best; va[sg] = at; }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (sg != 0) return;
    \\  best = vb[lane]; at = va[lane];
    \\  for (ushort o = 16; o > 0; o >>= 1) {
    \\    const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(at, o);
    \\    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
    \\  }
    \\  if (lane == 0) out[0] = ids[at];
    \\}
    \\// fz_argmax in two launches: threadgroup (p, row) keeps its slice's first largest (index and value bits)
    \\kernel void fz_argmax_part(device const bfloat* logits [[buffer(0)]], device uint* part [[buffer(1)]],
    \\    constant uint& n [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]], uint2 ng [[threadgroups_per_grid]],
    \\    uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint sg [[simdgroup_index_in_threadgroup]]) {
    \\  const uint p = tg.x, row = tg.y, parts = ng.x, chunk = (n + parts - 1) / parts, lo = p * chunk, hi = min(n, lo + chunk);
    \\  device const bfloat* x = logits + row * n;
    \\  float best = -INFINITY; uint at = 0xffffffffu;
    \\  for (uint i = lo + t; i < hi; i += 256) { const float v = float(x[i]); if (v > best) { best = v; at = i; } }
    \\  for (ushort o = 16; o > 0; o >>= 1) {
    \\    const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(at, o);
    \\    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
    \\  }
    \\  threadgroup float vb[8]; threadgroup uint va[8];
    \\  if (lane == 0) { vb[sg] = best; va[sg] = at; }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (t != 0) return;
    \\  for (uint k = 1; k < 8; k++) if (vb[k] > best || (vb[k] == best && va[k] < at)) { best = vb[k]; at = va[k]; }
    \\  part[(row * parts + p) * 2] = as_type<uint>(best);
    \\  part[(row * parts + p) * 2 + 1] = at;
    \\}
    \\// the parts' winners, the largest and the lowest index among equals: fz_argmax's pick (ids[pick] when map)
    \\kernel void fz_argmax_join(device const uint* part [[buffer(0)]], device uint* out [[buffer(1)]],
    \\    constant uint& parts [[buffer(2)]], device const uint* ids [[buffer(3)]], constant uint& map [[buffer(4)]],
    \\    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    \\  float best = -INFINITY; uint at = 0xffffffffu;
    \\  for (uint q = lane; q < parts; q += 32) {
    \\    const float v = as_type<float>(part[(row * parts + q) * 2]);
    \\    const uint a = part[(row * parts + q) * 2 + 1];
    \\    if (v > best || (v == best && a < at)) { best = v; at = a; }
    \\  }
    \\  for (ushort o = 16; o > 0; o >>= 1) {
    \\    const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(at, o);
    \\    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
    \\  }
    \\  if (lane == 0) out[row] = map != 0 ? ids[at] : at;
    \\}
    \\// NGramEmbedding.ids for a window: pm = history (2), eos, unused, multipliers (3), head sizes (16), offsets (16)
    \\kernel void fz_ple_ids(device const uint* tok [[buffer(0)]], device const long* pm [[buffer(1)]],
    \\    device uint* out [[buffer(2)]], constant uint& rows [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    \\  const uint row = i / 16, hh = i % 16;
    \\  if (row >= rows) return;
    \\  long seq[18];
    \\  seq[0] = pm[0]; seq[1] = pm[1];
    \\  for (uint q = 0; q <= row; q++) seq[2 + q] = long(tok[q]);
    \\  const long eos = pm[2];
    \\  const uint at = 2 + row;
    \\  long before = -1;
    \\  for (uint q = 0; q < at; q++) if (seq[q] == eos) before = long(q);
    \\  const long in_seg = long(at) - (before + 1);
    \\  ulong mixed = 0;
    \\  const uint ng = hh < 8 ? 2 : 3;
    \\  for (uint q = 0; q < ng; q++) {
    \\    const long sh = in_seg >= long(q) ? seq[at - q] : eos;
    \\    const ulong term = ulong(sh) * ulong(pm[4 + q]);
    \\    mixed = q == 0 ? term : (mixed ^ term);
    \\  }
    \\  const long size = pm[7 + hh];
    \\  long mod = as_type<long>(mixed) % size;
    \\  if (mod < 0) mod += size;
    \\  out[row * 16 + hh] = uint(mod + pm[23 + hh]);
    \\}
    \\// dst[l][i] = src[l][(keep + base) row][i] for i < p.x words: kept DeltaNet states, the PLE tail, the MTP's row
    \\kernel void fz_copy_kept(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],
    \\    device const int* ar [[buffer(2)]], constant uint4& p [[buffer(3)]], constant int& base [[buffer(4)]],
    \\    uint2 gid [[thread_position_in_grid]]) {
    \\  if (gid.x >= p.x) return;
    \\  const uint row = uint(ar[0] + base);
    \\  dst[gid.y * p.w + gid.x] = src[gid.y * p.z + row * p.y + gid.x];
    \\}
    \\// The MTP head's kept row (keep[0] - 1) of its attention output, streams and inject gates into row 0, in place
    \\kernel void fz_mtp_take(device bfloat* aout [[buffer(0)]], device bfloat* h [[buffer(1)]], device bfloat* inj [[buffer(2)]],
    \\    device const int* keep [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    \\  const int k = keep[0] - 1;
    \\  if (k <= 0) return;
    \\  if (i < 6144) aout[i] = aout[k * 6144 + i];
    \\  h[i] = h[k * 10240 + i];
    \\  if (i < 4) inj[i] = inj[k * 4 + i];
    \\}
    \\// An expert row from [part][group][6 words] to [group][part][6 words] (FZ_XPACK): the lanes of one group read
    \\// one block; the sums keep their order. d = (parts, groups a part).
    \\[[kernel]] void fz_repack_w(device uint* W [[buffer(0)]], constant uint2& d [[buffer(1)]],
    \\    uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
    \\  threadgroup uint tmp[480];
    \\  const uint n = d.x * d.y * 6;
    \\  device uint* w = W + size_t(row) * n;
    \\  for (uint i = t; i < n; i += 128) tmp[i] = w[i];
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  for (uint i = t; i < n; i += 128) {
    \\    const uint j = i / (d.x * 6), part = (i / 6) % d.x, k = i % 6;
    \\    w[i] = tmp[part * d.y * 6 + j * 6 + k];
    \\  }
    \\}
    \\[[kernel]] void fz_repack_s(device ushort* S [[buffer(0)]], constant uint2& d [[buffer(1)]],
    \\    uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
    \\  threadgroup ushort tmp[80];
    \\  const uint n = d.x * d.y;
    \\  device ushort* s = S + size_t(row) * n;
    \\  for (uint i = t; i < n; i += 128) tmp[i] = s[i];
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  for (uint i = t; i < n; i += 128) s[i] = tmp[(i % d.x) * d.y + i / d.x];
    \\}
    \\// FZ_PREFETCH: read a weight buffer into the cache while a latency-bound launch runs beside it.
    \\[[kernel]] void fz_touch(const device uint4* w [[buffer(0)]], device uint* sink [[buffer(1)]], constant uint& n [[buffer(2)]],
    \\    uint i [[thread_position_in_grid]], uint gs [[threads_per_grid]]) {
    \\  uint acc = 0;
    \\  for (uint k = i; k < n; k += gs) { const uint4 v = w[k]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    \\  if (acc == 0x9e3779b9u) sink[0] = acc;
    \\}
;
