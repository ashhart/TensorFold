// A window's KDA step, each head on G threadgroups (D / G value columns each), then glm_kda_norm: kda_rows' values.
template <int H, int D, int TAPS, int TY, int FB, int GB, int RB, int G>
[[max_total_threads_per_threadgroup(32 * TY)]]
[[kernel]] void glm_kda_parts(const device bfloat16_t* P [[buffer(0)]], const constant int* P_shape [[buffer(1)]],
                              const device bfloat16_t* CS [[buffer(2)]], const device float* CW [[buffer(3)]],
                              const device uint32_t* FBW [[buffer(4)]], const device bfloat16_t* FBS [[buffer(5)]],
                              const device bfloat16_t* FBB [[buffer(6)]], const device uint32_t* GBW [[buffer(7)]],
                              const device bfloat16_t* GBS [[buffer(8)]], const device bfloat16_t* GBB [[buffer(9)]],
                              const device float* A [[buffer(10)]], const device float* DTB [[buffer(11)]],
                              const device float* ST [[buffer(12)]], device bfloat16_t* GATE [[buffer(13)]],
                              const constant float* LB [[buffer(14)]], device bfloat16_t* Y [[buffer(16)]],
                              device float* ST_OUT [[buffer(17)]], device bfloat16_t* CS_OUT [[buffer(18)]],
                              uint sgi [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]],
                              uint3 tpos [[thread_position_in_threadgroup]], uint3 tgpos [[threadgroup_position_in_grid]]) {
  const uint h = tgpos.z;
  const uint part = tgpos.y; // value columns [dv0, dv0 + DV): their v, gates, recurrence; decays, q and k whole
  const uint lane = tpos.x;
  const uint ty = tpos.y;
  constexpr int NT = 32 * TY;
  constexpr int DV = D / G;
  constexpr int NDK = D / 32;
  constexpr int NDV = DV / TY;
  constexpr int RBLK = D / 128;
  constexpr int REXTRA = D - RBLK * 128;
  constexpr uint W = (uint)(H * D);
  constexpr uint C3 = 3u * W;
  constexpr uint FA = C3;
  constexpr uint GA = C3 + (uint)D;
  constexpr uint BO = C3 + 2u * (uint)D;
  static_assert(DV % TY == 0 && NDV >= 1, "whole value columns a thread");
  const uint dv0 = part * (uint)DV;
  const int R = int(P_shape[0]);
  const uint PS = (uint)P_shape[1];

  threadgroup bfloat wq[RB][D];
  threadgroup bfloat wk[RB][D];
  threadgroup bfloat wv[RB][DV];
  threadgroup float wg[RB][D];
  threadgroup float wbeta[RB], wrq[RB], wrk[RB];

  device const float* si = ST + (size_t)h * D * D;
  float st[NDV][NDK];
  for (int j = 0; j < NDV; ++j) {
    const uint dv = dv0 + ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i) st[j][i] = si[(size_t)dv * D + NDK * lane + i];
  }
  constexpr uint CSN = 3u * (uint)D * (uint)(TAPS - 1); // the conv window out (inputs only): a G-th each
  for (uint idx = part * CSN / G + tid; idx < (part + 1u) * CSN / G; idx += NT) {
    const uint m = idx / (3u * (uint)D);
    const uint rem = idx - m * 3u * (uint)D;
    const uint pp = rem / (uint)D;
    const uint d = rem - pp * (uint)D;
    const uint c = pp * W + h * (uint)D + d;
    const int e = R + int(m);
    CS_OUT[(size_t)m * C3 + c] = e < TAPS - 1 ? CS[(size_t)e * C3 + c] : P[(size_t)(e - (TAPS - 1)) * PS + c];
  }
  const float a_h = A[h];
  const float lb = LB[0];

  for (int r0 = 0; r0 < R; r0 += RB) {
    const int n = min(RB, R - r0);
    if (r0 > 0) threadgroup_barrier(mem_flags::mem_threadgroup);
    { // every row's f_b (all D: the decays) and its columns' g_b, a quad an output (kda_rows' sums)
      constexpr int PER = D / 4;
      constexpr int KG = D / 64;
      constexpr int FKB = D * FB / 8;
      constexpr int GKB = D * GB / 8;
      constexpr uint PO = (uint)(D + DV); // outputs a row
      const uint q_id = tid / 4u, qlid = tid % 4u;
      for (uint t = q_id; t < PO * (uint)n; t += (uint)(NT / 4)) {
        const uint rl = t / PO;
        const uint tt = t - rl * PO;
        const uint proj = tt < (uint)D ? 0u : 1u;
        const uint d = proj == 0u ? tt : dv0 + tt - (uint)D;
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
            GATE[(size_t)(r0 + (int)rl) * W + h * (uint)D + d] = bfloat(v);
          }
        }
      }
    }
    constexpr uint CO = 2u * (uint)D + (uint)DV; // q and k whole, this group's v
    for (uint idx = tid; idx < CO * (uint)n; idx += NT) { // the short convolutions, SiLU
      const uint rl = idx / CO;
      const uint rem = idx - rl * CO;
      const uint pp = rem < 2u * (uint)D ? rem / (uint)D : 2u;
      const uint d = pp < 2u ? rem - pp * (uint)D : dv0 + rem - 2u * (uint)D;
      const uint c = pp * W + h * (uint)D + d;
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
      if (pp == 0u) wq[rl][d] = sl;
      else if (pp == 1u) wk[rl][d] = sl;
      else wv[rl][d - dv0] = sl;
    }
    if (tid < (uint)n) wbeta[tid] = float(mlx_sigmoid_precise<bfloat>(P[(size_t)(r0 + (int)tid) * PS + BO + h]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint rl = sgi; rl < (uint)n; rl += (uint)TY) { // row rl's q and k norms, kda_rows' order
      float pq = 0.0f, pk = 0.0f;
      for (int blk = 0; blk < RBLK; ++blk) {
        const uint base = (uint)(blk * 128) + 4u * lane;
        for (int i = 0; i < 4; ++i) { pq = sq_acc(pq, float(wq[rl][base + i])); pk = sq_acc(pk, float(wk[rl][base + i])); }
      }
      for (int i = 0; 4u * lane + (uint)i < (uint)REXTRA && i < 4; ++i) {
        const uint at = (uint)(RBLK * 128) + 4u * lane + (uint)i;
        pq = sq_acc(pq, float(wq[rl][at])); pk = sq_acc(pk, float(wk[rl][at]));
      }
      pq = simd_sum(pq);
      pk = simd_sum(pk);
      if (lane == 0u) {
        wrq[rl] = metal::precise::rsqrt(pq + 1.0e-6f);
        wrk[rl] = metal::precise::rsqrt(pk + 1.0e-6f);
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
    for (int rl = 0; rl < n; ++rl) { // the recurrence on this group's columns; raw outputs straight out
      const float beta = wbeta[rl];
      for (int j = 0; j < NDV; ++j) {
        const uint dl = ty + (uint)TY * (uint)j;
        float kv = 0.0f;
        for (int i = 0; i < NDK; ++i) {
          const uint s = NDK * lane + i;
          st[j][i] = st[j][i] * wg[rl][s];
          kv += st[j][i] * float(wk[rl][s]);
        }
        kv = simd_sum(kv);
        const float delta = (float(wv[rl][dl]) - kv) * beta;
        float o = 0.0f;
        for (int i = 0; i < NDK; ++i) {
          const uint s = NDK * lane + i;
          st[j][i] = st[j][i] + float(wk[rl][s]) * delta;
          o += st[j][i] * float(wq[rl][s]);
        }
        o = simd_sum(o);
        if (lane == 0u) Y[(size_t)(r0 + rl) * W + h * (uint)D + dv0 + dl] = bfloat(o);
      }
    }
  }
  device float* so = ST_OUT + (size_t)h * D * D;
  for (int j = 0; j < NDV; ++j) {
    const uint dv = dv0 + ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i) so[(size_t)dv * D + NDK * lane + i] = st[j][i];
  }
}

// (H / 8, rows) of 256: a simdgroup a row's head, glm_kda_parts' raw outputs normed and gated in place (kda_rows').
template <int H, int D>
[[kernel]] void glm_kda_norm(device bfloat16_t* Y [[buffer(0)]], const device bfloat16_t* GATE [[buffer(1)]],
                             const device float* ONW [[buffer(2)]], const constant float* EPS [[buffer(3)]],
                             uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                             uint2 tgpos [[threadgroup_position_in_grid]]) {
  static_assert(D == 128, "a lane's four outputs are its norm partial's four");
  constexpr uint W = (uint)(H * D);
  const uint h = tgpos.x * 8u + sgi;
  if (h >= (uint)H) return;
  device bfloat* y = Y + (size_t)tgpos.y * W + h * (uint)D;
  const device bfloat* g = GATE + (size_t)tgpos.y * W + h * (uint)D;
  float sy[4];
  float po = 0.0f;
  for (int i = 0; i < 4; ++i) {
    sy[i] = float(y[4u * lane + (uint)i]);
    po = sq_acc(po, sy[i]);
  }
  po = simd_sum(po);
  const float rn = metal::precise::rsqrt(po / (float)D + EPS[0]);
  for (int i = 0; i < 4; ++i) {
    const uint d = 4u * lane + (uint)i;
    float x = sy[i] * rn;
    x = ONW[d] * x;
    x = x * mlx_sigmoid_precise<float>(float(bfloat(float(g[d]))));
    y[d] = bfloat(x);
  }
}

template [[host_name("glm_kda_parts")]] [[kernel]] decltype(glm_kda_parts<64, 128, 4, 16, 4, 4, 8, 2>) glm_kda_parts<64, 128, 4, 16, 4, 4, 8, 2>;
template [[host_name("glm_kda_norm")]] [[kernel]] decltype(glm_kda_norm<64, 128>) glm_kda_norm<64, 128>;
template [[host_name("glm_kda_parts_q8")]] [[kernel]] decltype(glm_kda_parts<64, 128, 4, 16, 8, 8, 8, 2>) glm_kda_parts<64, 128, 4, 16, 8, 8, 8, 2>;
