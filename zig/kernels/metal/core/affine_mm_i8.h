// Ported whole from BobClawblaw/TensorFold, branch deepseek-v41-main, commit 72a552ed ("deepseek_v41: prompt
// chunks' routed experts in int8 x int8"), Apache-2.0; see THIRD_PARTY_NOTICES.md. Unmodified below except for this
// header.

// The codes as int8 into the tile row (q - 128 at 8 bits): code i of the little-endian bit stream at byte i.
inline void am_codes(thread const Codes& c, threadgroup int8_t* out) {
  threadgroup uint* o = (threadgroup uint*)out;
#if TF_BITS == 2
  TF_UNROLL
  for (short i = 0; i < 2; i++) { // byte j of a word holds codes 4j..4j+3, two bits each from the low end
    const uint w = c.v[i];
    TF_UNROLL
    for (short j = 0; j < 4; j++) {
      const uint b = (w >> (8 * j)) & 0xffu;
      o[4 * i + j] = (b & 3u) | (((b >> 2) & 3u) << 8) | (((b >> 4) & 3u) << 16) | ((b >> 6) << 24);
    }
  }
#elif TF_BITS == 8
  TF_UNROLL
  for (short i = 0; i < 8; i++) o[i] = c.v[i / 4][i % 4] ^ 0x80808080u; // q - 128 as int8: the top bit flipped
#elif TF_BITS == 4
  TF_UNROLL
  for (short i = 0; i < 4; i++) {
    const uint w = c.v[0][i];
    const uint lo = w & 0x0f0f0f0fu, hi = (w >> 4) & 0x0f0f0f0fu; // byte j: codes 2j (low nibble) and 2j + 1
    o[2 * i] = (lo & 0xffu) | ((hi & 0xffu) << 8) | (((lo >> 8) & 0xffu) << 16) | (((hi >> 8) & 0xffu) << 24);
    o[2 * i + 1] = ((lo >> 16) & 0xffu) | (((hi >> 16) & 0xffu) << 8) | ((lo >> 24) << 16) | ((hi >> 24) << 24);
  }
#else
  TF_UNROLL
  for (short i = 0; i < 8; i++) {
    uint packed = 0;
    TF_UNROLL
    for (short j = 0; j < 4; j++) {
      const short at = TF_BITS * (4 * i + j), word = at / 32, shift = at % 32;
      uint v = c.v[word] >> shift;
      if (shift + TF_BITS > 32) v |= c.v[word + 1] << (32 - shift);
      packed |= (v & ((1u << TF_BITS) - 1u)) << (8 * j);
    }
    o[i] = packed;
  }
#endif
}

// A block of the int8 code tile as the transposed right operand (PAD rows, 16-column blocks).
inline void am_tile_frag(thread frag<int8_t>& f, const threadgroup int8_t* p, int ld, int r, int c, short2 home) {
  const threadgroup int8_t* q = p + (r + home.y) * ld + (c + home.x);
  const char4 v0 = *(const threadgroup char4*)q, v1 = *(const threadgroup char4*)(q + 8 * ld);
  f = frag<int8_t>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
}

// The fragment of x at column c from a lane's two row starts, int8.
inline frag<int8_t> am_frag(const device int8_t* r0, const device int8_t* r1, int c, short2 home) {
  const char4 v0 = *(const device char4*)(r0 + c + home.x), v1 = *(const device char4*)(r1 + c + home.x);
  return frag<int8_t>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
}

// am_step in int8: each TF_QG columns' int32 sums scaled by the weights' and x's scales; at a group's end its bias.
template <int TM, int TL>
inline void am_step(thread frag<float> (&acc)[TM][2], thread const AmRows<TM>& rows, int k,
                    const threadgroup AM_T* tile, const threadgroup float* sb, int tn, short2 home) {
  frag<AM_ACC> part[TL][2];
  TF_UNROLL
  for (short q = 0; q < 4; q++) { // the step's four 16-deep slices, every row fragment's chain interleaved
    frag<AM_T> b0, b1, a[TL];
    am_tile_frag(b0, tile, PAD, tn, 16 * q, home);
    am_tile_frag(b1, tile, PAD, tn + 16, 16 * q, home);
    TF_UNROLL
    for (short m = 0; m < TL; m++) {
      a[m] = am_frag(rows.x[m][0], rows.x[m][1], k + 16 * q, home);
    }
    TF_UNROLL
    for (short m = 0; m < TL; m++) {
      if (q % (TF_QG / 16) == 0) {
        am_first(part[m][0], part[m][1], a[m], b0, b1);
      } else {
        mma_16x32<false, true>(part[m][0], part[m][1], a[m], b0, b1);
      }
    }
    if ((q + 1) % (TF_QG / 16) == 0) { // a scale's columns done: weight scale times x's scale; at a group's end its bias
      const int g = q / (TF_GROUP / 16), h = q / (TF_QG / 16);
      const bool ends = (q + 1) % (TF_GROUP / 16) == 0;
      const threadgroup float* sg = sb + g * 128;
      TF_UNROLL
      for (short m = 0; m < TL; m++) {
        const float xa0 = rows.a[m][0][k / TF_QG + h], xa1 = rows.a[m][1][k / TF_QG + h];
        const float xr0 = ends ? rows.s[m][0][k / TF_GROUP + g] : 0.0f, xr1 = ends ? rows.s[m][1][k / TF_GROUP + g] : 0.0f;
        TF_UNROLL
        for (short j = 0; j < 2; j++) {
          TF_UNROLL
          for (short c = 0; c < 4; c++) {
            const int col = tn + 16 * j + home.x + TF_COL(c);
            const float s = sg[col], b = ends ? sg[64 + col] : 0.0f;
            acc[m][j][c] += s * (xa0 * float(part[m][j][c])) + b * xr0;
            acc[m][j][4 + c] += s * (xa1 * float(part[m][j][4 + c])) + b * xr1;
          }
        }
      }
    }
  }
}

// The row sums, and x as int8 (XQ [rows, k]) with each scale (XA, max |x| / 127): x ~ a * q, q rounded to nearest.
[[kernel]] void tf_affine_row_sums(const device bfloat* X [[buffer(0)]], constant MmArgs& a [[buffer(5)]],
                                   device float* XS [[buffer(6)]], device int8_t* XQ [[buffer(7)]],
                                   device float* XA [[buffer(8)]], uint2 gid [[thread_position_in_grid]]) {
  const int g = int(gid.x), r = int(gid.y), groups = a.k / TF_GROUP;
  if (g >= groups || r >= a.rows) return;
  const device bfloat* x = X + long(r) * (a.n > 0 ? a.n : a.k) + g * TF_GROUP;
  float sum = 0.0f;
  for (int i = 0; i < TF_GROUP; i++) {
    sum += float(x[i]);
  }
  XS[long(r) * groups + g] = sum;
  for (int h = 0; h < TF_GROUP / TF_QG; h++) {
    float top = 0.0f;
    for (int i = TF_QG * h; i < TF_QG * (h + 1); i++) top = max(top, fabs(float(x[i])));
    const float inv = top > 0.0f ? 127.0f / top : 0.0f;
    device int8_t* q = XQ + long(r) * a.k + g * TF_GROUP;
    for (int i = TF_QG * h; i < TF_QG * (h + 1); i++) q[i] = int8_t(clamp(rint(float(x[i]) * inv), -127.0f, 127.0f));
    XA[(long(r) * groups + g) * (TF_GROUP / TF_QG) + h] = top / 127.0f;
  }
}
