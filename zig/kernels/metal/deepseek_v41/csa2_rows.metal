// Batched decode windows: one row a slot; state via addr (live, 0, selection, 0), mains (owner rows), pos tables.
struct Dsv41AttentionRows {
  uint rows, heads, width, q_stride, window, ring_stride, chunk_stride, main_stride, ring_off, owner;
  uint ratio, index_topk, dense_rows, splits;
  float scale;
};
static_assert(sizeof(Dsv41AttentionRows) == 60, "Dsv41AttentionRows must mirror csa2.zig's RowsParams");

// Four dims of a packed key, as dsv41_dense_kv decodes them: kind 0 the window's e4 rows, 1 the compressed e2 rows.
inline bfloat4 dsv41_dense_dec4(const device uchar *k, uint width, uint dd, uint kind) {
  bfloat4 v;
  if (kind == 0) {
    const float s = as_type<float>(uint(k[width + (dd >> 5)]) << 23);
    const uchar4 c = *(const device uchar4 *)(k + dd);
    TF_UNROLL
    for (short i = 0; i < 4; i++) v[i] = bfloat(dsv41_bf(dsv41_round(dsv41_e4(c[i]) * s)));
  } else {
    const float s = dsv41_e4(k[width / 2 + (dd >> 4)]);
    const uchar2 c = *(const device uchar2 *)(k + dd / 2);
    v[0] = bfloat(dsv41_bf(dsv41_round(dsv41_e2(c[0] & 15) * s)));
    v[1] = bfloat(dsv41_bf(dsv41_round(dsv41_e2(c[0] >> 4) * s)));
    v[2] = bfloat(dsv41_bf(dsv41_round(dsv41_e2(c[1] & 15) * s)));
    v[3] = bfloat(dsv41_bf(dsv41_round(dsv41_e2(c[1] >> 4) * s)));
  }
  return v;
}

// Every slot's visible keys decoded once into its dense table (ring, this row, picks).
kernel void dsv41_dense_kv_rows(device const ulong *addr [[buffer(0)]], device const uint *pos [[buffer(1)]],
                                device const uchar *chunk [[buffer(2)]], device ushort *dense [[buffer(3)]],
                                constant Dsv41AttentionRows &p [[buffer(4)]], device const ulong *mains [[buffer(5)]],
                                uint3 gid [[thread_position_in_grid]]) {
  const uint slot = gid.z, key = gid.x, dd = gid.y * 4;
  if (slot >= p.rows || dd >= p.width || key >= p.dense_rows) return;
  const uint position = pos[slot];
  const device uchar *k;
  uint kind = 0;
  if (key < p.window) {
    if (key >= min(position, p.window)) return;
    k = reinterpret_cast<const device uchar *>(addr[4 * slot]) + p.ring_off + ulong(key) * p.ring_stride;
  } else if (key == p.window) {
    k = chunk + ulong(slot) * p.chunk_stride;
  } else {
    if (!p.ratio) return;
    const uint comp = (position + 1) / p.ratio;
    const uint i = key - p.window - 1;
    if (i >= min(p.index_topk, comp)) return;
    const int pick = reinterpret_cast<const device int *>(addr[4 * slot + 2])[i];
    if (pick < 0 || uint(pick) >= comp) return;
    k = reinterpret_cast<const device uchar *>(mains[4 * slot + p.owner]) + ulong(pick) * p.main_stride;
    kind = 1;
  }
  *(device bfloat4 *)(dense + (ulong(slot) * p.dense_rows + key) * p.width + dd) = dsv41_dense_dec4(k, p.width, dd, kind);
}

#ifndef TF_SIMD_FRAGS
// dsv41_sparse_nax for decode windows of one row each: row tg.y is slot tg.y, its keys in its own dense table.
[[kernel]] void dsv41_sparse_nax_rows(device const ushort *q [[buffer(0)]], device const ulong *addr [[buffer(1)]],
                                      device const uint *pos [[buffer(2)]], device const float *sink [[buffer(3)]],
                                      device ushort *out [[buffer(4)]], constant Dsv41AttentionRows &p [[buffer(5)]],
                                      device const ushort *dense_all [[buffer(6)]], device float *parts [[buffer(7)]],
                                      uint s [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                      uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float partial[4][32][16];
  threadgroup int keyrow[BLOCK];
  threadgroup uint keyoff[BLOCK];
  if (tg.x * GROUP >= p.heads || tg.y >= p.rows) return;
  const uint row = tg.y, query = pos[row];
  const uint columns = query == 0 ? 1u : p.window;
  const uint comp = p.ratio ? (query + 1) / p.ratio : 0u; // the slot's compressed rows
  const uint compressed = p.ratio ? min(p.index_topk, comp) : 0u; // min(topk, min(compressed rows, visible))
  const uint count = columns + compressed;
  const device int *indices = reinterpret_cast<const device int *>(addr[4 * row + 2]);
  const device ushort *dense = dense_all + ulong(row) * p.dense_rows * p.width;
  const int h0 = int(tg.x) * GROUP, d0 = int(s) * SLICE;
  const short2 home = frag_home(ushort(lane));
  const device bfloat *qb = reinterpret_cast<device const bfloat *>(q + ulong(row) * p.q_stride + ulong(h0) * RANK + uint(d0));
  frag<bfloat> qf[8];
  TF_UNROLL
  for (short t = 0; t < 8; t++) frag_get(qf[t], qb, RANK, 0, 16 * t, home);
  frag<float> acc[4][2];
  float m[2], l[2];
  TF_UNROLL
  for (short c = 0; c < 4; c++) {
    acc[c][0] = frag<float>(0);
    acc[c][1] = frag<float>(0);
  }
  m[0] = m[1] = -1e30f;
  l[0] = l[1] = 0.0f;
  const uint split = tg.z, per = (count + p.splits - 1) / max(p.splits, 1u);
  const uint j_begin = split * per, j_end = min(count, j_begin + per);
  for (uint j0 = j_begin; j0 < j_end; j0 += BLOCK) {
    if (s == 0 && lane < uint(BLOCK)) {
      const uint j = j0 + lane;
      int enc = -1;
      uint dense_row = 0;
      if (j < j_end) {
        if (j < columns) {
          // dsv41_slot_at's window part: absolute position `wpos`, its dense row the ring slot (or this row's)
          uint wpos;
          if (query == 0) wpos = 0;
          else if (query + 1 < p.window) wpos = (query + 1 + j) % p.window;
          else wpos = (query + 1 > p.window ? query + 1 - p.window : 0) + j;
          if (wpos <= query) {
            enc = int(wpos);
            dense_row = wpos >= query ? p.window : wpos % p.window;
          }
        } else {
          const int pick = indices[j - columns];
          if (pick >= 0 && uint(pick) < min(comp, (query + 1) / p.ratio)) {
            enc = int(0x40000000u | uint(pick));
            dense_row = p.window + 1 + (j - columns);
          }
        }
      }
      keyrow[lane] = enc;
      keyoff[lane] = enc < 0 ? 0u : dense_row * p.width;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    frag<float> lo_s = frag<float>(0), hi_s = frag<float>(0);
    TF_UNROLL
    for (short t = 0; t < 8; t++) {
      const uint d = uint(d0) + 16u * uint(t) + uint(home.x);
      const bfloat4 a = dsv41_dense4_off(keyrow[home.y], keyoff[home.y], dense, d);
      const bfloat4 b = dsv41_dense4_off(keyrow[short(home.y + 8)], keyoff[short(home.y + 8)], dense, d);
      const bfloat4 c = dsv41_dense4_off(keyrow[short(16 + home.y)], keyoff[short(16 + home.y)], dense, d);
      const bfloat4 e = dsv41_dense4_off(keyrow[short(24 + home.y)], keyoff[short(24 + home.y)], dense, d);
      mma_16x32<false, true>(lo_s, hi_s, qf[t], frag<bfloat>(a, b), frag<bfloat>(c, e));
    }
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      partial[s][lane][e] = lo_s[e];
      partial[s][lane][8 + e] = hi_s[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc[16];
    TF_UNROLL
    for (short i = 0; i < 16; i++) {
      const float v = ((partial[0][lane][i] + partial[1][lane][i]) + partial[2][lane][i]) + partial[3][lane][i];
      const short e = i & 7;
      const int key = (i < 8 ? 0 : 16) + p_col(e, home);
      sc[i] = keyrow[key] >= 0 ? v * p.scale : NO_SCORE;
    }
    float factor[2];
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      float bmax = NO_SCORE;
      TF_UNROLL
      for (short i = 0; i < 4; i++) bmax = max(bmax, max(sc[4 * h + i], sc[8 + 4 * h + i]));
      const float new_max = max(m[h], row_max(bmax));
      factor[h] = fast::exp(m[h] - new_max);
      m[h] = new_max;
    }
    frag<bfloat> p1[2];
    float bsum[2] = {0.0f, 0.0f};
    TF_UNROLL
    for (short i = 0; i < 16; i++) {
      const short e = i & 7, h = e >> 2, kb = i < 8 ? 0 : 1;
      const float pp = sc[i] > NO_SCORE ? fast::exp(sc[i] - m[h]) : 0.0f;
      bsum[h] += pp;
      p1[kb][e] = bfloat(dsv41_bf(dsv41_round(pp)));
    }
    TF_UNROLL
    for (short h = 0; h < 2; h++) l[h] = l[h] * factor[h] + row_sum(bsum[h]);
    if (!simd_all(factor[0] == 1.0f && factor[1] == 1.0f))
    TF_UNROLL
    for (short c = 0; c < 4; c++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        acc[c][0][e] *= factor[e >> 2];
        acc[c][1][e] *= factor[e >> 2];
      }
    }
    TF_UNROLL
    for (short kb = 0; kb < 2; kb++) {
      TF_UNROLL
      for (short c = 0; c < 4; c++) {
        const uint d = uint(d0) + 32u * uint(c) + uint(home.x);
        const bfloat4 a = dsv41_dense4_off(keyrow[16 * kb + home.y], keyoff[16 * kb + home.y], dense, d);
        const bfloat4 b = dsv41_dense4_off(keyrow[16 * kb + short(home.y + 8)], keyoff[16 * kb + short(home.y + 8)], dense, d);
        frag<bfloat> v0(a, b);
        const bfloat4 a2 = dsv41_dense4_off(keyrow[16 * kb + home.y], keyoff[16 * kb + home.y], dense, d + 16);
        const bfloat4 b2 = dsv41_dense4_off(keyrow[16 * kb + short(home.y + 8)], keyoff[16 * kb + short(home.y + 8)], dense, d + 16);
        frag<bfloat> v1(a2, b2);
        mma_16x32<false, false>(acc[c][0], acc[c][1], p1[kb], v0, v1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (p.splits > 1) {
    device float *part = parts + ((((ulong(tg.y) * ulong(p.splits) + ulong(split)) * ulong(p.heads / GROUP) + ulong(tg.x)) * 4 + ulong(s)) * 32 + ulong(lane)) * 68;
    part[0] = m[0];
    part[1] = m[1];
    part[2] = l[0];
    part[3] = l[1];
    TF_UNROLL
    for (short c = 0; c < 4; c++)
      TF_UNROLL
      for (short j = 0; j < 2; j++)
        TF_UNROLL
        for (short e = 0; e < 8; e++) part[4 + c * 16 + j * 8 + e] = acc[c][j][e];
    return;
  }
  device ushort *obase = out + ulong(row) * p.q_stride + ulong(h0) * RANK + uint(d0);
  TF_UNROLL
  for (short c = 0; c < 4; c++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      frag<float> o = acc[c][j];
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const short h = e >> 2;
        const float denom = l[h] + fast::exp(sink[ulong(h0) + uint(home.y) + 8u * uint(h)] - m[h]);
        o[e] = denom == 0.0f ? 0.0f : o[e] / denom;
      }
      dsv41_frag_put(o, obase, RANK, 0, 32 * c + 16 * j, home);
    }
  }
}

#endif
