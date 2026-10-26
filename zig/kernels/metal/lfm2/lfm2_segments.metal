
// A shared round's streams in one dispatch a kernel (after lfm2.metal): each row runs its own stream's arithmetic.

// A stream of a recorded call at one layer: its cache rows, its rows from `base`, its key count and sub-spans.
struct lfm2_seg {
  device bfloat* keys;
  device bfloat* values;
  device bfloat* conv;
  uint base, rows, len, cap, subs, pad;
};
// Partials a row owns: every stream's at its first row times the most sub-spans, so no two streams' overlap.
constant constexpr size_t SEG_PART = size_t(QH) * CHUNKS * SUBS;

// lfm2_conv for every row of a recorded call: row y of stream rowseg[y], its state and snapshot rows.
[[kernel]] void lfm2_conv_segs(const device bfloat* bcx [[buffer(0)]], const device lfm2_seg* segs [[buffer(1)]],
                               const device bfloat* w [[buffer(2)]], device bfloat* y [[buffer(3)]],
                               device bfloat* snap [[buffer(4)]], const device uint* rowseg [[buffer(5)]],
                               uint2 at [[thread_position_in_grid]]) {
  if (at.x >= D) return;
  const lfm2_seg s = segs[rowseg[at.y]];
  lfm2_conv_row(bcx + size_t(s.base) * 3 * D, s.conv, w, y + size_t(s.base) * D, snap + size_t(s.base) * 2 * D, 0, at.x,
                at.y - s.base);
}

// lfm2_qk for every row of a recorded call, each stream's keys and values into its own cache.
[[kernel]] void lfm2_qk_segs(const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]],
                             const device bfloat* v [[buffer(2)]], const device bfloat* qn [[buffer(3)]],
                             const device bfloat* kn [[buffer(4)]], constant lfm2_qk_args& a [[buffer(5)]],
                             device bfloat* queries [[buffer(6)]], const device lfm2_seg* segs [[buffer(7)]],
                             const device uint* rowseg [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
  const lfm2_seg s = segs[rowseg[tg.y]];
  const lfm2_qk_args p = {s.len, s.cap, a.eps, a.log2_theta};
  const size_t b = s.base;
  lfm2_qk_row(q + b * QH * HD, k + b * KVH * HD, v + b * KVH * HD, qn, kn, p, queries + b * QH * HD, s.keys, s.values,
              tg.x, tg.y - s.base, lane);
}

// lfm2_attn_mma over the streams' 8-row tiles: tile tg.x is tiles[tg.x] = (stream << 16) | the stream's tile.
[[kernel]] void lfm2_attn_segs(const device bfloat* queries [[buffer(0)]], const device lfm2_seg* segs [[buffer(1)]],
                               const device uint* tiles [[buffer(2)]], constant lfm2_attn_args& a [[buffer(3)]],
                               device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
                               device float* po [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint ks[2][HD * (KT / 2)];
  threadgroup bfloat vs[2][KB * KP];
  const uint item = tiles[tg.x];
  const lfm2_seg s = segs[item >> 16];
  const lfm2_attn_args p = {s.len, s.cap, a.scale, s.subs};
  const size_t part = size_t(s.base) * SEG_PART;
  lfm2_attn_rows<1>(queries + size_t(s.base) * QH * HD, s.keys, s.values, p, pm + part, pl + part, po + part * HD,
                     s.rows, uint3(item & 0xFFFFu, tg.y, tg.z), tid, sg, lane, ks, vs);
}

// lfm2_attn_merge for every row of a recorded call, from its stream's partials.
[[kernel]] void lfm2_attn_merge_segs(const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]],
                                     const device float* po [[buffer(2)]], constant lfm2_attn_args& a [[buffer(3)]],
                                     device bfloat* y [[buffer(4)]], const device lfm2_seg* segs [[buffer(5)]],
                                     const device uint* rowseg [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  const lfm2_seg s = segs[rowseg[tg.y]];
  const lfm2_attn_args p = {s.len, s.cap, a.scale, s.subs};
  const size_t part = size_t(s.base) * SEG_PART;
  lfm2_merge_row(pm + part, pl + part, po + part * HD, p, y + size_t(s.base) * QH * HD, tg.x, tg.y - s.base, lane);
}
