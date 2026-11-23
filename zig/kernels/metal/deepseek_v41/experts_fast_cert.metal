// Fast non-dd MXFP4 routed expert kernels for DSpark drafting under TF_DSV41_FP64_BF16=1.
inline float k3_scaled_fma(float p, uchar e, float acc) {
  if (e == 255u) return NAN;
  if (e - 1u < 240u) return fma(p, k3_e8m0(e), acc);
  const int d = int(e) - 127, d1 = d / 2;
  const float s1 = 16384.0f * as_type<float>(uint(d1 + 127) << 23);
  const float s2 = as_type<float>(uint(d - d1 + 127) << 23);
  return fma(p * s1, s2, acc);
}

inline float ds_xtree8_fast(float acc) {
  for (ushort m = 1; m < 8; m <<= 1) acc += simd_shuffle_xor(acc, m);
  return acc;
}

template <int RP>
inline void ds_up_out_fast(thread float (&acc)[2][RP], threadgroup float (&vals)[RP][32], device bfloat* act,
                           device const float* wts, device const uint* pairs, constant DsArgs& a, uint first, uint r0,
                           uint rn, uint nl, uint n, uint c) {
  for (int r = 0; r < RP; ++r) {
    const float g = ds_clip(float(bfloat(ds_xtree8_fast(acc[0][r]))), -INFINITY, a.limit);
    const float u = ds_clip(float(bfloat(ds_xtree8_fast(acc[1][r]))), -a.limit, a.limit);
    const float v = g / (1.0f + precise::exp(-g)) * u * (uint(r) < rn ? wts[pairs[first + r0 + r]] : 0.0f);
    if (c == 0) vals[r][nl] = float(bfloat(v));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int r = 0; r < RP; ++r) {
    if (c != 0 || uint(r) >= rn) continue;
    float amax = 0.0f;
    bool bad = false;
    for (uint j = 0; j < 32; ++j) bad |= !isfinite(vals[r][j]), amax = max(amax, fabs(vals[r][j]));
    const int s = ds_exp(amax);
    const float v = vals[r][nl];
    const float q = ds_val(ds_code(min(fabs(v) * ds_pow2(-s), 448.0f), signbit(v))) * ds_pow2(s);
    act[(first + r0 + r) * a.N + n] = bfloat(bad ? NAN : q);
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_xp_upc_one_fast(
    device const half* XH [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device const uint* nslots [[buffer(3)]],
    device const uint* pairs [[buffer(4)]], device bfloat* act [[buffer(5)]], constant DsArgs& a [[buffer(6)]],
    device const float* wts [[buffer(7)]], device const float* XS [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half4 xh[DS_KMAX / 32 * DS_GS / 4];
  threadgroup float xs[DS_KMAX / 32], vals[1][32];
  if (tg.y >= nslots[0] || slots[3 * tg.y + 2] != 1) return;
  const uint e = slots[3 * tg.y], first = slots[3 * tg.y + 1];
  const ExpertPtrs w = table[e];
  const uint nl = sg * 4 + (lane >> 3), n = tg.x * 32 + nl, c = lane & 7, rowb = a.K / 2, groups = a.K / 32;
  device const uchar* const wp[2] = {w.w1p + n * rowb, w.w3p + n * rowb};
  device const uchar* const ws[2] = {w.w1s + n * groups, w.w3s + n * groups};
  const uint row = pairs[first] / a.topk;
  for (uint i = t; i < a.K / 4; i += 256) xh[(i / 8) * (DS_GS / 4) + i % 8] = ((device const half4*)(XH + row * a.K))[i];
  for (uint i = t; i < groups; i += 256) xs[i] = XS[row * groups + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc[2][1] = {{0.0f}, {0.0f}};
  for (uint g = c; g < groups; g += 8) {
    uint2 q[2][2];
    for (int v = 0; v < 2; ++v) q[v][0] = *(device const uint2*)(wp[v] + 16 * g), q[v][1] = *(device const uint2*)(wp[v] + 16 * g + 8);
    threadgroup const half4* x = xh + g * (DS_GS / 4);
    float s[2] = {0.0f, 0.0f};
    for (int h = 0; h < 2; ++h) {
      float wv[2][16];
      for (int v = 0; v < 2; ++v) {
        float wlo[8], whi[8];
        ds_fp4_word(q[v][h].x, wlo);
        ds_fp4_word(q[v][h].y, whi);
        for (int j = 0; j < 8; ++j) wv[v][j] = wlo[j], wv[v][8 + j] = whi[j];
      }
      for (int i = 0; i < 4; ++i) {
        const float4 x4 = float4(x[4 * h + i]);
        for (int j = 0; j < 4; ++j)
          for (int v = 0; v < 2; ++v) s[v] = fma(x4[j], wv[v][4 * i + j], s[v]);
      }
    }
    const float xsc = xs[g];
    for (int v = 0; v < 2; ++v) acc[v][0] = k3_scaled_fma(s[v] * xsc, ws[v][g], acc[v][0]);
  }
  ds_up_out_fast<1>(acc, vals, act, wts, pairs, a, first, 0, 1, nl, n, c);
}

template <uint RP, uint NW, typename T, bool DOWN>
inline void ds_rows_chains_fast(thread float (&acc)[NW][RP], device const T* X, device const float* XS,
                                const thread uint (&src)[RP], uint K, device const uchar* const thread (&wp)[NW],
                                device const uchar* const thread (&ws)[NW], uint c) {
  const uint G = K / 32;
  for (uint v = 0; v < NW; ++v)
    for (uint r = 0; r < RP; ++r) acc[v][r] = 0.0f;
  for (uint g = c; g < G; g += 8) {
    float s[NW][RP];
    for (uint v = 0; v < NW; ++v)
      for (uint r = 0; r < RP; ++r) s[v][r] = 0.0f;
    for (int h = 0; h < 2; ++h) {
      float w[NW][16];
      for (uint v = 0; v < NW; ++v) {
        const uint2 q = *(device const uint2*)(wp[v] + 16 * g + 8 * h);
        for (int b = 0; b < 8; ++b) {
          const float2 f = k3_fp4_pair(((b < 4 ? q.x : q.y) >> (8 * (b & 3))) & 0xFFu);
          w[v][2 * b] = f.x, w[v][2 * b + 1] = f.y;
        }
      }
      for (uint r = 0; r < RP; ++r) {
        device const vec<T, 4>* x = (device const vec<T, 4>*)(X + src[r] * K + 32 * g + 16 * h);
        for (int i = 0; i < 4; ++i) {
          const float4 x4 = float4(x[i]);
          for (int j = 0; j < 4; ++j)
            for (uint v = 0; v < NW; ++v) s[v][r] = fma(x4[j], w[v][4 * i + j], s[v][r]);
        }
      }
    }
    for (uint v = 0; v < NW; ++v)
      for (uint r = 0; r < RP; ++r) {
        const float xsc = DOWN ? 1.0f : XS[src[r] * G + g];
        acc[v][r] = k3_scaled_fma(s[v][r] * xsc, ws[v][g], acc[v][r]);
      }
  }
}

template <uint RP>
inline void ds_upc_pass_fast(threadgroup float* vals, device const half* XH, device const float* XS,
                             device const uint* pairs, device bfloat* act, constant DsArgs& a, device const float* wts,
                             device const uchar* const thread (&wp)[2], device const uchar* const thread (&ws)[2],
                             uint first, uint r0, uint rn, uint nl, uint n, uint c) {
  uint src[RP];
  for (uint r = 0; r < RP; ++r) src[r] = pairs[first + r0 + min(r, rn - 1)] / a.topk;
  float acc[2][RP];
  ds_rows_chains_fast<RP, 2, half, false>(acc, XH, XS, src, a.K, wp, ws, c);
  ds_up_out_fast<RP>(acc, *(threadgroup float(*)[RP][32])vals, act, wts, pairs, a, first, r0, rn, nl, n, c);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_xp_upc_rows_fast(
    device const half* XH [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device const uint* nslots [[buffer(3)]],
    device const uint* pairs [[buffer(4)]], device bfloat* act [[buffer(5)]], constant DsArgs& a [[buffer(6)]],
    device const float* wts [[buffer(7)]], device const float* XS [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float vals[4 * 32];
  if (tg.y >= nslots[0] || slots[3 * tg.y + 2] < 2) return;
  const uint e = slots[3 * tg.y], first = slots[3 * tg.y + 1], count = slots[3 * tg.y + 2];
  const ExpertPtrs w = table[e];
  const uint nl = sg * 4 + (lane >> 3), n = tg.x * 32 + nl, c = lane & 7, rowb = a.K / 2, groups = a.K / 32;
  device const uchar* const wp[2] = {w.w1p + n * rowb, w.w3p + n * rowb};
  device const uchar* const ws[2] = {w.w1s + n * groups, w.w3s + n * groups};
  for (uint r0 = 0, rp; r0 < count; r0 += rp) {
    rp = ds_tier(count - r0, false);
    const uint rn = min(rp, count - r0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rp == 2) ds_upc_pass_fast<2>(vals, XH, XS, pairs, act, a, wts, wp, ws, first, r0, rn, nl, n, c);
    else ds_upc_pass_fast<4>(vals, XH, XS, pairs, act, a, wts, wp, ws, first, r0, rn, nl, n, c);
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_xp_down_one_fast(
    device const bfloat* act [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device const uint* nslots [[buffer(3)]],
    device const uint* pairs [[buffer(4)]], device bfloat* out [[buffer(5)]], constant XpArgs& a [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup ushort4 xb[3072 / 32 * DS_GS / 4];
  if (tg.y >= nslots[0] || slots[3 * tg.y + 2] != 1) return;
  const uint e = slots[3 * tg.y], first = slots[3 * tg.y + 1];
  const ExpertPtrs w = table[e];
  const uint n = tg.x * 32 + sg * 4 + (lane >> 3), c = lane & 7, rowb = a.K / 2, groups = a.K / 32;
  for (uint i = t; i < a.K / 4; i += 256) xb[(i / 8) * (DS_GS / 4) + i % 8] = ((device const ushort4*)(act + first * a.K))[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device const uchar* wp = w.w2p + n * rowb;
  device const uchar* ws = w.w2s + n * groups;
  float acc = 0.0f;
  for (uint g = c; g < groups; g += 8) {
    const uint2 q0 = *(device const uint2*)(wp + 16 * g), q1 = *(device const uint2*)(wp + 16 * g + 8);
    threadgroup const ushort4* x = xb + g * (DS_GS / 4);
    float s = 0.0f;
    for (int h = 0; h < 2; ++h) {
      float wlo[8], whi[8];
      ds_fp4_word(h == 0 ? q0.x : q1.x, wlo);
      ds_fp4_word(h == 0 ? q0.y : q1.y, whi);
      for (int i = 0; i < 4; ++i) {
        const float4 x4 = float4(as_type<bfloat4>(x[4 * h + i]));
        for (int j = 0; j < 4; ++j)
          s = fma(x4[j], (4 * i + j < 8 ? wlo[4 * i + j] : whi[4 * i + j - 8]), s);
      }
    }
    acc = k3_scaled_fma(s, ws[g], acc);
  }
  const bfloat v = bfloat(ds_xtree8_fast(acc));
  if (c == 0) out[first * a.N + n] = v;
}

template <uint RP>
inline void ds_down_pass_fast(device const bfloat* act, device bfloat* out, constant XpArgs& a,
                              device const uchar* const thread (&wp)[1], device const uchar* const thread (&ws)[1],
                              uint first, uint r0, uint rn, uint n, uint c) {
  uint src[RP];
  for (uint r = 0; r < RP; ++r) src[r] = first + r0 + min(r, rn - 1);
  float acc[1][RP];
  ds_rows_chains_fast<RP, 1, bfloat, true>(acc, act, nullptr, src, a.K, wp, ws, c);
  for (uint r = 0; r < RP; ++r) {
    const bfloat v = bfloat(ds_xtree8_fast(acc[0][r]));
    if (c == 0 && r < rn) out[(first + r0 + r) * a.N + n] = v;
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_xp_down_rows_fast(
    device const bfloat* act [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device const uint* nslots [[buffer(3)]],
    device const uint* pairs [[buffer(4)]], device bfloat* out [[buffer(5)]], constant XpArgs& a [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (tg.y >= nslots[0] || slots[3 * tg.y + 2] < 2) return;
  const uint e = slots[3 * tg.y], first = slots[3 * tg.y + 1], count = slots[3 * tg.y + 2];
  const ExpertPtrs w = table[e];
  const uint n = tg.x * 32 + sg * 4 + (lane >> 3), c = lane & 7, rowb = a.K / 2, groups = a.K / 32;
  device const uchar* const wp[1] = {w.w2p + n * rowb};
  device const uchar* const ws[1] = {w.w2s + n * groups};
  for (uint r0 = 0, rp; r0 < count; r0 += rp) {
    rp = ds_tier(count - r0, true);
    const uint rn = min(rp, count - r0);
    if (rp == 2) ds_down_pass_fast<2>(act, out, a, wp, ws, first, r0, rn, n, c);
    else ds_down_pass_fast<4>(act, out, a, wp, ws, first, r0, rn, n, c);
  }
}
