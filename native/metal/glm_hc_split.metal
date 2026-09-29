
  uint tid  = thread_position_in_threadgroup.x;
  uint row  = threadgroup_position_in_grid.x;
  uint lane = tid % 32;
  uint sg   = tid / 32;
  constexpr int MIX      = (2 + HC) * HC;
  constexpr int BASE_OFF = 2 * HC;
  constexpr float EPS = EPS_INT * 1e-9;
  const device float* mix      = (const device float*)mixes + row * MIX;
  device float*       post_out = (device float*)post + row * HC;
  device float*       comb_out = (device float*)comb + row * HC * HC;
  threadgroup float pre_shared[HC];
  if (sg == 0) {
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
    const uint  llane  = metal::min(lane, (uint)(HC - 1));
    float pre_z  = mix[llane]      * pre_scale  + base[llane];
    float post_z = mix[HC + llane] * post_scale + base[HC + llane];
    float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
    float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
    if (lane < (uint)HC) {
      pre_shared[lane] = pre_v;
      post_out[lane]   = post_v;
    }
    float4 v = (*(const device float4*)(mix  + BASE_OFF + llane * HC) * comb_scale
              + *(const device float4*)(base + BASE_OFF + llane * HC)) * active;
    float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
    float4 e = metal::fast::exp(v - row_max) * active;
    float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS)) + EPS * active;
    float4 col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
    r *= col_inv;
    for (int iter = 1; iter < ITERS; ++iter) {
      r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
      col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
      r *= col_inv;
    }
    if (lane < (uint)HC) {
      *(device float4*)(comb_out + lane * HC) = r;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float p0 = pre_shared[0];
  const float p1 = pre_shared[1];
  const float p2 = pre_shared[2];
  const float p3 = pre_shared[3];
  const device T* x_row  = (const device T*)x_in + row * (HC * D);
  device T*       out_row = (device T*)collapsed + row * D;
  using T4 = vec<T, 4>;
  const device T4* x_row0 = (const device T4*)(x_row + 0*D);
  const device T4* x_row1 = (const device T4*)(x_row + 1*D);
  const device T4* x_row2 = (const device T4*)(x_row + 2*D);
  const device T4* x_row3 = (const device T4*)(x_row + 3*D);
  device T4*       out4   = (device T4*)out_row;
  constexpr uint D4 = (uint)D / 4;
  for (uint d4 = tid; d4 < D4; d4 += 256) {
    float4 x0 = float4(x_row0[d4]);
    float4 x1 = float4(x_row1[d4]);
    float4 x2 = float4(x_row2[d4]);
    float4 x3 = float4(x_row3[d4]);
    float4 result = fma(float4(p0), x0, fma(float4(p1), x1, fma(float4(p2), x2, float4(p3) * x3)));
    out4[d4] = T4(result);
  }
