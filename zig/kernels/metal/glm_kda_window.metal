
// A window's KDA step in kda_rows' arithmetic: RB rows' gates and norms at once, the recurrence, outputs at once.
template <int H, int D, int TAPS, int TY, int FB, int GB, int RB>
[[max_total_threads_per_threadgroup(1024)]]
[[kernel]] void glm_kda_window(const device bfloat16_t* P [[buffer(0)]], const constant int* P_shape [[buffer(1)]],
                               const device bfloat16_t* CS [[buffer(2)]], const device float* CW [[buffer(3)]],
                               const device uint32_t* FBW [[buffer(4)]], const device bfloat16_t* FBS [[buffer(5)]],
                               const device bfloat16_t* FBB [[buffer(6)]], const device uint32_t* GBW [[buffer(7)]],
                               const device bfloat16_t* GBS [[buffer(8)]], const device bfloat16_t* GBB [[buffer(9)]],
                               const device float* A [[buffer(10)]], const device float* DTB [[buffer(11)]],
                               const device float* ST [[buffer(12)]], const device float* ONW [[buffer(13)]],
                               const constant float* LB [[buffer(14)]], const constant float* EPS [[buffer(15)]],
                               device bfloat16_t* Y [[buffer(16)]], device float* ST_OUT [[buffer(17)]],
                               device bfloat16_t* CS_OUT [[buffer(18)]], uint sgi [[simdgroup_index_in_threadgroup]],
                               uint tid [[thread_index_in_threadgroup]], uint3 tpos [[thread_position_in_threadgroup]],
                               uint3 tgpos [[threadgroup_position_in_grid]]) {
  const uint h = tgpos.z;
  const uint lane = tpos.x;
  const uint ty = tpos.y;
  constexpr int NT = 32 * TY;
  constexpr int NDK = D / 32;
  constexpr int NDV = D / TY;
  constexpr int RBLK = D / 128;
  constexpr int REXTRA = D - RBLK * 128;
  constexpr uint W = (uint)(H * D);
  constexpr uint C3 = 3u * W;
  constexpr uint FA = C3;
  constexpr uint GA = C3 + (uint)D;
  constexpr uint BO = C3 + 2u * (uint)D;
  const int R = int(P_shape[0]);
  const uint PS = (uint)P_shape[1];

  // kda_rows' sq, sk, sv, sgate and sy are bf16 values held as floats: bf16 here holds them exactly
  threadgroup bfloat wq[RB][D];
  threadgroup bfloat wk[RB][D];
  threadgroup bfloat wv[RB][D];
  threadgroup bfloat wgate[RB][D];
  threadgroup bfloat wy[RB][D];
  threadgroup float wg[RB][D];
  threadgroup float wbeta[RB], wrq[RB], wrk[RB], wrn[RB];

  device const float* si = ST + (size_t)h * D * D;
  float st[NDV][NDK];
  for (int j = 0; j < NDV; ++j) {
    uint dv = ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i) st[j][i] = si[(size_t)dv * D + NDK * lane + i];
  }
  const float a_h = A[h];
  const float lb = LB[0];
  const float eps = EPS[0];

  for (int r0 = 0; r0 < R; r0 += RB) {
    const int n = min(RB, R - r0);
    { // f_b and g_b of every row (a quad an output, kda_rows' sums), each decay computed where its f_b lands
      constexpr int PER = D / 4;
      constexpr int KG = D / 64;
      constexpr int FKB = D * FB / 8;
      constexpr int GKB = D * GB / 8;
      const uint q_id = tid / 4u, qlid = tid % 4u;
      for (uint t = q_id; t < 2u * (uint)D * (uint)n; t += (uint)(NT / 4)) {
        const uint rl = t / (2u * (uint)D);
        const uint tt = t - rl * 2u * (uint)D;
        const uint proj = tt / (uint)D;
        const uint d = tt - proj * (uint)D;
        const uint row = h * (uint)D + d;
        device const bfloat* prow = P + (size_t)(r0 + (int)rl) * PS;
        device const bfloat* x = prow + (proj == 0u ? FA : GA) + qlid * (uint)PER;
        const uint gi = row * (uint)KG + qlid / (uint)(64 / PER);
        const float s = float(proj == 0u ? FBS[gi] : GBS[gi]);
        const float bb = float(proj == 0u ? FBB[gi] : GBB[gi]);
        float result;
        if (proj == 0u) {
          device const uint8_t* wb = (device const uint8_t*)FBW + (size_t)row * FKB + qlid * (PER * FB / 8);
          result = quad_dot<FB, PER>(x, wb, s, bb);
        } else {
          device const uint8_t* wb = (device const uint8_t*)GBW + (size_t)row * GKB + qlid * (PER * GB / 8);
          result = quad_dot<GB, PER>(x, wb, s, bb);
        }
        const float v = quad_sum(result);
        if (qlid == 0u) {
          if (proj == 0u) {
            const float sa = float(bfloat(v));
            const float av = float(bfloat(sa)) + DTB[h * (uint)D + d];
            wg[rl][d] = metal::precise::exp(lb * mlx_sigmoid_precise<float>(a_h * av));
          } else {
            wgate[rl][d] = bfloat(v);
          }
        }
      }
    }
    for (uint idx = tid; idx < 3u * (uint)D * (uint)n; idx += NT) { // the short convolutions, SiLU
      const uint rl = idx / (3u * (uint)D);
      const uint rem = idx - rl * 3u * (uint)D;
      const uint part = rem / (uint)D;
      const uint d = rem - part * (uint)D;
      const uint c = part * W + h * (uint)D + d;
      const int r = r0 + (int)rl;
      float acc = 0.0f;
      for (int j = 0; j < TAPS; ++j) {
        const int e = r + j;
        const bfloat xv = e < TAPS - 1 ? CS[(size_t)e * C3 + c] : P[(size_t)(e - (TAPS - 1)) * PS + c];
        const float term = float(xv) * CW[(size_t)j * C3 + c];
        acc = j == 0 ? term : acc + term;
      }
      const bfloat xb = bfloat(acc);
      const bfloat sl = xb * mlx_sigmoid_precise<bfloat>(xb);
      if (part == 0u) wq[rl][d] = sl;
      else if (part == 1u) wk[rl][d] = sl;
      else wv[rl][d] = sl;
    }
    if (tid < (uint)n) wbeta[tid] = float(mlx_sigmoid_precise<bfloat>(P[(size_t)(r0 + (int)tid) * PS + BO + h]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi < (uint)n) { // row sgi's q and k norms, kda_rows' order
      float pq = 0.0f, pk = 0.0f;
      for (int blk = 0; blk < RBLK; ++blk) {
        const uint base = (uint)(blk * 128) + 4u * lane;
        for (int i = 0; i < 4; ++i) { pq = sq_acc(pq, float(wq[sgi][base + i])); pk = sq_acc(pk, float(wk[sgi][base + i])); }
      }
      for (int i = 0; 4u * lane + (uint)i < (uint)REXTRA && i < 4; ++i) {
        const uint at = (uint)(RBLK * 128) + 4u * lane + (uint)i;
        pq = sq_acc(pq, float(wq[sgi][at])); pk = sq_acc(pk, float(wk[sgi][at]));
      }
      pq = simd_sum(pq);
      pk = simd_sum(pk);
      if (lane == 0u) {
        wrq[sgi] = metal::precise::rsqrt(pq + 1.0e-6f);
        wrk[sgi] = metal::precise::rsqrt(pk + 1.0e-6f);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      const float qscale = metal::precise::rsqrt(float(D));
      for (uint idx = tid; idx < (uint)D * (uint)n; idx += NT) {
        const uint rl = idx / (uint)D, d = idx - rl * (uint)D;
        wq[rl][d] = bfloat((float(wq[rl][d]) * wrq[rl]) * qscale);
        wk[rl][d] = bfloat(float(wk[rl][d]) * wrk[rl]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int rl = 0; rl < n; ++rl) { // the recurrence: each simdgroup its value columns, no other's
      const float beta = wbeta[rl];
      for (int j = 0; j < NDV; ++j) {
        const uint dv = ty + (uint)TY * (uint)j;
        float kv = 0.0f;
        for (int i = 0; i < NDK; ++i) {
          const uint s = NDK * lane + i;
          st[j][i] = st[j][i] * wg[rl][s];
          kv += st[j][i] * float(wk[rl][s]);
        }
        kv = simd_sum(kv);
        const float delta = (float(wv[rl][dv]) - kv) * beta;
        float o = 0.0f;
        for (int i = 0; i < NDK; ++i) {
          const uint s = NDK * lane + i;
          st[j][i] = st[j][i] + float(wk[rl][s]) * delta;
          o += st[j][i] * float(wq[rl][s]);
        }
        o = simd_sum(o);
        if (lane == 0u) wy[rl][dv] = bfloat(o);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi < (uint)n) {
      float po = 0.0f;
      for (int blk = 0; blk < RBLK; ++blk) {
        const uint base = (uint)(blk * 128) + 4u * lane;
        for (int i = 0; i < 4; ++i) po = sq_acc(po, float(wy[sgi][base + i]));
      }
      for (int i = 0; 4u * lane + (uint)i < (uint)REXTRA && i < 4; ++i)
        po = sq_acc(po, float(wy[sgi][(uint)(RBLK * 128) + 4u * lane + (uint)i]));
      po = simd_sum(po);
      if (lane == 0u) wrn[sgi] = metal::precise::rsqrt(po / (float)D + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < (uint)D * (uint)n; idx += NT) {
      const uint rl = idx / (uint)D, d = idx - rl * (uint)D;
      float x = float(wy[rl][d]) * wrn[rl];
      x = ONW[d] * x;
      x = x * mlx_sigmoid_precise<float>(float(bfloat(float(wgate[rl][d]))));
      Y[(size_t)(r0 + (int)rl) * W + h * (uint)D + d] = bfloat(x);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  device float* so = ST_OUT + (size_t)h * D * D;
  for (int j = 0; j < NDV; ++j) {
    const uint dv = ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i) so[(size_t)dv * D + NDK * lane + i] = st[j][i];
  }
  for (uint idx = tid; idx < 3u * (uint)D * (uint)(TAPS - 1); idx += NT) {
    const uint m = idx / (3u * (uint)D);
    const uint rem = idx - m * 3u * (uint)D;
    const uint part = rem / (uint)D;
    const uint d = rem - part * (uint)D;
    const uint c = part * W + h * (uint)D + d;
    const int e = R + int(m);
    CS_OUT[(size_t)m * C3 + c] = e < TAPS - 1 ? CS[(size_t)e * C3 + c] : P[(size_t)(e - (TAPS - 1)) * PS + c];
  }
}
template [[host_name("glm_kda_window")]] [[kernel]] decltype(glm_kda_window<64, 128, 4, 32, 4, 4, 8>) glm_kda_window<64, 128, 4, 32, 4, 4, 8>;
template [[host_name("glm_kda_window_tp")]] [[kernel]] decltype(glm_kda_window<32, 128, 4, 32, 4, 4, 8>) glm_kda_window<32, 128, 4, 32, 4, 4, 8>;
template [[host_name("glm_kda_window_q8")]] [[kernel]] decltype(glm_kda_window<64, 128, 4, 32, 8, 8, 8>) glm_kda_window<64, 128, 4, 32, 8, 8, 8>;
