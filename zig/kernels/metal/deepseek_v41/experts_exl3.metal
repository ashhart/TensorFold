// DeepSeek V4.1 routed experts over EXL3 trellises: the slots' dots and the gate/up epilogue (after exl3_projection).

struct TfX3Args {
  uint k, n, pairs, experts;
  float limit;
};

// tf_exl3_matmul_row's map of a column's 16 tile rows in a ring of `halves`: word, bit offset and next word.
inline void tf_x3_maps(thread uint (&maps)[16], uint col, uint halves) {
  const uint ring = 128 * halves;
  for (uint r = 0; r < 16; ++r) {
    const uint p = 8 * (4 * (col % 8) + (r % 8) / 2) + 4 * (col / 8) + 2 * (r / 8) + (r % 2);
    const uint end = ((p + 1) * halves - ((halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
    const uint start = (end + ring - 16) % ring;
    const uint wi = start / 32, next = wi + 1 == 4 * halves ? 0 : wi + 1;
    maps[r] = wi | ((start % 32) << 6) | (next << 11);
  }
}

// RP rows of a slot through NW matrices; each row's chain is tf_exl3_matmul_row's, whatever rows share the pass.
template <uint NW, uint RP>
inline void tf_x3_pass(const device half* xr, device float* sums, constant TfX3Args& a,
                       const device uchar* const thread (&column)[NW], const thread ulong (&ks)[NW],
                       const thread uint (&book)[NW], const thread uint (&maps)[NW][16],
                       const thread uint (&rows)[RP], uint col, uint nb) {
  float high[NW][RP], low[NW][RP];
  for (uint v = 0; v < NW; ++v)
    for (uint r = 0; r < RP; ++r) high[v][r] = 0.0f, low[v][r] = 0.0f;
  for (uint kb = 0; kb < a.k; kb += 16) {
    float xs[NW][RP];
    const device ushort* tile[NW];
    for (uint v = 0; v < NW; ++v) {
      tile[v] = reinterpret_cast<const device ushort*>(column[v] + ulong(kb / 16) * ks[v]);
      for (uint r = 0; r < RP; ++r) xs[v][r] = float(xr[(ulong(v) * a.pairs + rows[r]) * a.k + kb + col]);
    }
    for (uint t = 0; t < 16; ++t)
      for (uint v = 0; v < NW; ++v) {
        const float w = float(tf_exl3_value(tf_exl3_mapped_state(tile[v], maps[v][t]), book[v]));
        for (uint r = 0; r < RP; ++r) {
          const float product = simd_shuffle(xs[v][r], ushort(t)) * w;
          const float next = high[v][r] + product;
          const float part = next - high[v][r];
          low[v][r] += (high[v][r] - (next - part)) + (product - part);
          high[v][r] = next;
        }
      }
  }
  for (uint v = 0; v < NW; ++v)
    for (uint r = 0; r < RP; ++r) sums[(ulong(v) * a.pairs + rows[r]) * a.n + nb + col] = high[v][r] + low[v][r];
}

// A slot's pairs (one expert) through its NW matrices' rotated rows by pair, 32 columns a simdgroup, 4/2/1 rows a pass.
template <uint NW>
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_x3_slots(
    const device half* xr [[buffer(0)]], const device TfExl3Expert* t0 [[buffer(1)]],
    const device TfExl3Expert* t1 [[buffer(2)]], const device uint* slots [[buffer(3)]],
    const device uint* nslots [[buffer(4)]], const device uint* pairs [[buffer(5)]], device float* sums [[buffer(6)]],
    constant TfX3Args& a [[buffer(7)]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if (tg.y >= nslots[0]) return;
  const uint e = slots[3 * tg.y], first = slots[3 * tg.y + 1], count = slots[3 * tg.y + 2];
  if (e >= a.experts) return;
  const uint col = lane % 16, nb = tg.x * 32 + (lane & 16u);
  const device uchar* column[NW];
  ulong ks[NW];
  uint book[NW], maps[NW][16];
  for (uint v = 0; v < NW; ++v) {
    const device TfExl3Expert& d = v == 0 ? t0[e] : t1[e];
    column[v] = d.trellis + ulong(nb / 16) * d.tile_n_stride;
    ks[v] = d.tile_k_stride;
    book[v] = d.codebook;
    tf_x3_maps(maps[v], col, d.halves);
  }
  for (uint r0 = 0; r0 < count;) {
    const uint left = count - r0;
    if (left >= 4) {
      const uint rows[4] = {pairs[first + r0], pairs[first + r0 + 1], pairs[first + r0 + 2], pairs[first + r0 + 3]};
      tf_x3_pass<NW, 4>(xr, sums, a, column, ks, book, maps, rows, col, nb);
      r0 += 4;
    } else if (left >= 2) {
      const uint rows[2] = {pairs[first + r0], pairs[first + r0 + 1]};
      tf_x3_pass<NW, 2>(xr, sums, a, column, ks, book, maps, rows, col, nb);
      r0 += 2;
    } else {
      const uint rows[1] = {pairs[first + r0]};
      tf_x3_pass<NW, 1>(xr, sums, a, column, ks, book, maps, rows, col, nb);
      r0 += 1;
    }
  }
}

template [[host_name("tf_dsv41_x3_up")]] [[kernel]] decltype(tf_x3_slots<2>) tf_x3_slots<2>;
template [[host_name("tf_dsv41_x3_down")]] [[kernel]] decltype(tf_x3_slots<1>) tf_x3_slots<1>;

// experts.metal's ds_clip and SwiGLU quotient: a NaN stays NaN; silu(g) = g / (1 + exp(-g)).
inline float tf_x3_clip(float x, float lo, float hi) { return x != x ? x : clamp(x, lo, hi); }
inline float tf_x3_silu(float g) { return g / (1.0f + precise::exp(-g)); }

// Gate and up of a pair as tf_exl3_rotate_out writes them in bf16, then experts.metal's clamped SwiGLU and weight.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_x3_act(
    const device float* sums [[buffer(0)]], const device TfExl3Expert* gate [[buffer(1)]],
    const device TfExl3Expert* up [[buffer(2)]], const device uint* ids [[buffer(3)]],
    const device float* wts [[buffer(4)]], device bfloat* act [[buffer(5)]], constant TfX3Args& a [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const uint p = tg.y, base = tg.x * 128, e = ids[p];
  if (e >= a.experts) return;
  float y[2][4];
  for (uint v = 0; v < 2; ++v) {
    const device TfExl3Expert& d = v == 0 ? gate[e] : up[e];
    float h[4];
    for (uint j = 0; j < 4; ++j) h[j] = sums[(ulong(v) * a.pairs + p) * a.n + base + lane + 32 * j];
    tf_exl3_hadamard(h, lane);
    for (uint j = 0; j < 4; ++j) {
      const uint n = base + lane + 32 * j;
      float out = (h[j] * 0.08838834764831845f) * tf_exl3_scale(d.sv, n, d.sv_signs);
      if (d.has_bias) out += float(d.bias[n]);
      y[v][j] = float(bfloat(out));
    }
  }
  for (uint j = 0; j < 4; ++j) {
    const float g = tf_x3_clip(y[0][j], -INFINITY, a.limit), u = tf_x3_clip(y[1][j], -a.limit, a.limit);
    act[ulong(p) * a.n + base + lane + 32 * j] = bfloat(tf_x3_silu(g) * u * wts[p]);
  }
}

// Qualification only: silu of every bf16 gate, the one value the serial reference takes from the device.
kernel void tf_dsv41_x3_silu(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  if (i < 65536) out[i] = tf_x3_silu(float(as_type<bfloat>(ushort(i))));
}
