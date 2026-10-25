// Qwen3.5-2B's keys and attention over core-pool pages: the contiguous kernels with only the K/V address changed.

inline ulong qwen35_page_row(device const uint* table, uint pos, ulong page_bytes, uint h) {
  return tf_page_offset(table, pos, page_bytes, 256 * 2) + ulong(h) * TF_PAGE_TOKENS * 256 * 2;
}

kernel void qwen35_keys_paged(const device bfloat* k [[buffer(0)]],
                              const device bfloat* v [[buffer(1)]],
                              device uchar* keys [[buffer(2)]],
                              device uchar* values [[buffer(3)]],
                              constant uint4& dims [[buffer(4)]],
                              device const uint* table [[buffer(5)]],
                              constant ulong& page_bytes [[buffer(6)]],
                              uint3 p [[thread_position_in_grid]]) {
  const uint d = p.x, h = p.y, r = p.z;
  const uint m = dims.z + r;
  const ulong at = qwen35_page_row(table, dims.x + r, page_bytes, h);
  ((device bfloat*)(keys + at))[d] = qwen35_rotate(k + (size_t(m) * 2 + h) * 256, d, dims.x + r);
  ((device bfloat*)(values + at))[d] = v[(size_t(m) * 2 + h) * 256 + d];
}

kernel void qwen35_attn_partial_paged(
  const device bfloat* Q [[buffer(0)]],
  const device uchar* K [[buffer(1)]],
  const device uchar* V [[buffer(2)]],
  const constant float* scale [[buffer(3)]],
  const device int32_t* dims [[buffer(4)]],
  device float* PM [[buffer(5)]],
  device float* PL [[buffer(6)]],
  device float* PO [[buffer(7)]],
  device const uint* table [[buffer(8)]],
  constant ulong& page_bytes [[buffer(9)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
constexpr int D = 256;
constexpr int G = 4;
constexpr int CK = 128;
constexpr int SPLIT = 4;
constexpr int BLK = 4;
static_assert(TF_PAGE_TOKENS % CK == 0, "a chunk of keys must sit inside one page");

  const uint lane = thread_index_in_simdgroup;
  const uint sgi = simdgroup_index_in_threadgroup;
  const int g = int(sgi) / SPLIT, s = int(sgi) % SPLIT;
  const int c = int(threadgroup_position_in_grid.y);
  const int P = dims[0], W = dims[1], NCH = dims[3];
  const int h = int(threadgroup_position_in_grid.z) / W;
  const int w = int(threadgroup_position_in_grid.z) % W;
  constexpr int DPL = D / 32;
  const int qh = h * G + g;
  threadgroup float sm[G * SPLIT], sl[G * SPLIT];
  threadgroup float so[G * SPLIT][D];
  {
    const int last = P + w;
    const int k0 = c * CK;
    const int k1 = min(k0 + CK, last + 1);
    // the chunk sits inside one page: its keys' and values' head rows there, from one table entry
    const ulong at = ulong(table[k0 / TF_PAGE_TOKENS]) * page_bytes + ulong(h) * TF_PAGE_TOKENS * D * 2;
    const device bfloat* kp = (const device bfloat*)(K + at);
    const device bfloat* vp = (const device bfloat*)(V + at);
    float q[DPL], o[DPL];
    for (int i = 0; i < DPL; i++) {
      q[i] = float(Q[((qh * dims[5]) + dims[6] + w) * D + int(lane) * DPL + i]);
      o[i] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    for (int base = k0 + s; base < k1; base += SPLIT * BLK) {
      float sc[BLK];
      int rows[BLK];
      float bm = -INFINITY;
      for (int j = 0; j < BLK; j++) {
        const int pos = base + j * SPLIT;
        rows[j] = pos < k1 ? (pos < P ? pos : pos) : -1;
        float d = 0.0f;
        if (rows[j] >= 0) {
          const device bfloat* kr = kp + (rows[j] % TF_PAGE_TOKENS) * D + int(lane) * DPL;
          for (int i = 0; i < DPL; i++) d = fma(q[i], float(kr[i]), d);
        }
        sc[j] = simd_sum(d) * scale[0];
        if (rows[j] >= 0) bm = metal::max(bm, sc[j]);
      }
      const float mn = metal::max(m, bm);
      const float a = metal::exp(m - mn);
      l *= a;
      for (int i = 0; i < DPL; i++) o[i] *= a;
      for (int j = 0; j < BLK; j++) {
        if (rows[j] < 0) continue;
        const float b = metal::exp(sc[j] - mn);
        l += b;
        const device bfloat* vr = vp + (rows[j] % TF_PAGE_TOKENS) * D + int(lane) * DPL;
        for (int i = 0; i < DPL; i++) o[i] = fma(b, float(vr[i]), o[i]);
      }
      m = mn;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { sm[sgi] = m; sl[sgi] = l; }
    for (int i = 0; i < DPL; i++) so[sgi][int(lane) * DPL + i] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s == 0) {
      float mx_ = -INFINITY;
      for (int t = 0; t < SPLIT; t++) mx_ = metal::max(mx_, sm[g * SPLIT + t]);
      float lsum = 0.0f, acc[DPL];
      for (int i = 0; i < DPL; i++) acc[i] = 0.0f;
      for (int t = 0; t < SPLIT; t++) {
        const float e = sl[g * SPLIT + t] > 0.0f ? metal::exp(sm[g * SPLIT + t] - mx_) : 0.0f;
        lsum = fma(sl[g * SPLIT + t], e, lsum);
        for (int i = 0; i < DPL; i++) acc[i] = fma(so[g * SPLIT + t][int(lane) * DPL + i], e, acc[i]);
      }
      const int slot = (qh * W + w) * NCH + c;
      if (lane == 0) { PM[slot] = mx_; PL[slot] = lsum; }
      for (int i = 0; i < DPL; i++) PO[size_t(slot) * D + int(lane) * DPL + i] = acc[i];
    }
  }

}
