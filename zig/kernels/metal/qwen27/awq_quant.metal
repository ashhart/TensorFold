// GPU BF16 -> AWQ W4A16 g128 quantizer packing directly to .m5_tile or .m3_row layout.
struct SnAwqQuantArgs {
  uint n;
  uint k;
  uint m5_layout;
  uint n_eff;
};

[[max_total_threads_per_threadgroup(256)]]
[[kernel]] void sn_awq_quantize_bf16(
    const device bfloat* src [[buffer(0)]],
    constant SnAwqQuantArgs& a [[buffer(1)]],
    device uint* out_w [[buffer(2)]],
    device float2* out_sb [[buffer(3)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
#pragma clang fp contract(off)
  const uint j = tg.x * 8u + sg;
  const uint g = tg.y;
  const uint kg = a.k >> 7;
  if (j >= a.n_eff || g >= kg) return;

  if (j >= a.n) {
    if (a.m5_layout != 0u && (lane & 1u) == 0u) {
      const uint tile = j >> 5;
      const uint col = j & 31u;
      const uint w16 = lane >> 1;
      out_w[(((size_t)tile * kg + g) * 32u + col) * 16u + w16] = 0u;
    }
    return;
  }

  const device bfloat4* xp = (const device bfloat4*)(src + (size_t)j * a.k + (size_t)g * 128u);
  const float4 v = float4(xp[lane]);
  const float lmin = min(min(v.x, v.y), min(v.z, v.w));
  const float lmax = max(max(v.x, v.y), max(v.z, v.w));
  const float min_v = min(simd_min(lmin), 0.0f);
  const float max_v = max(simd_max(lmax), 0.0f);
  const float scale = (max_v - min_v) / 15.0f;

  float eff_scale = 0.0f;
  float bias = min_v;
  uint4 q = uint4(0u);
  if (scale != 0.0f) {
    const uint zp = uint(clamp(int(rint(-min_v / scale)), 0, 15));
    eff_scale = scale;
    bias = -float(zp) * eff_scale;
    q = uint4(clamp(int4(rint((v - bias) / eff_scale)), int4(0), int4(15)));
  }

  if (lane == 0u) {
    const size_t sb_idx = (a.m5_layout != 0u) ? ((size_t)g * a.n + j) : ((size_t)j * kg + g);
    out_sb[sb_idx] = float2(eff_scale, bias);
  }

  const uint half_word = q.x | (q.y << 4) | (q.z << 8) | (q.w << 12);
  const uint peer = simd_shuffle_xor(half_word, 1);
  if ((lane & 1u) == 0u) {
    const uint packed_k = half_word | (peer << 16);
    const uint w16 = lane >> 1;
    const size_t w_idx = (a.m5_layout != 0u)
        ? ((((size_t)(j >> 5) * kg + g) * 32u + (j & 31u)) * 16u + w16)
        : ((size_t)j * (a.k >> 3) + (size_t)g * 16u + w16);
    out_w[w_idx] = packed_k;
  }
}
