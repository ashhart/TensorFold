
  // One threadgroup per head h: 32 lanes x TY rows of threads. Rows r = 0 .. R-1 of the window in order.
  const uint h    = threadgroup_position_in_grid.z;
  const uint lane = thread_position_in_threadgroup.x;
  const uint ty   = thread_position_in_threadgroup.y;
  const uint tid  = thread_index_in_threadgroup;
  constexpr int NT   = 32 * TY;
  constexpr int NDK  = D / 32;          // key elements per lane
  constexpr int NDV  = D / TY;          // value rows per thread
  constexpr int RBLK = D / 128;        // MLX's row reduce: 32 lanes x 4 reads a block, then the rest
  constexpr int REXTRA = D - RBLK * 128;
  constexpr uint W   = (uint)(H * D);   // q / k / v width
  constexpr uint C3  = 3u * W;          // conv channels
  constexpr uint FA  = C3;              // offsets in the stacked projection row
  constexpr uint GA  = C3 + (uint)D;
  constexpr uint BO  = C3 + 2u * (uint)D;
  const int R = int(P_shape[0]);
  const uint PS = (uint)P_shape[1];

  threadgroup float sq[D];
  threadgroup float sk[D];
  threadgroup float sv[D];
  threadgroup float sa[D];
  threadgroup float sg[D];
  threadgroup float sgate[D];
  threadgroup float sy[D];
  threadgroup float shr[3];

  device const float* si = ST + (size_t)h * D * D;
  float st[NDV][NDK];
  for (int j = 0; j < NDV; ++j) {
    uint dv = ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i) st[j][i] = si[(size_t)dv * D + NDK * lane + i];
  }
  const float a_h = A[h];
  const float lb = LB[0];
  const float eps = EPS[0];

  for (int r = 0; r < R; ++r) {
    device const bfloat* prow = P + (size_t)r * PS;
    // ---- f_b / g_b (128 -> H*D, FB- / GB-bit groups of 64): MLX's one-row qmv_quad, as kernels.qmv_quad_rows
    {
      constexpr int PER = D / 4;
      constexpr int KG = D / 64;
      constexpr int FKB = D * FB / 8;                 // bytes a weight row
      constexpr int GKB = D * GB / 8;
      const uint q_id = tid / 4u, qlid = tid % 4u;
      for (uint t = q_id; t < 2u * (uint)D; t += (uint)(NT / 4)) {
        const uint proj = t / (uint)D;
        const uint d = t - proj * (uint)D;
        const uint row = h * (uint)D + d;
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
          if (proj == 0u) sa[d] = float(bfloat(v));
          else            sgate[d] = float(bfloat(v));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // sa / sgate come from other threads' quads
    // ---- causal conv over [window ; rows], fp32 taps in order, then silu (bf16, precise sigmoid)
    for (uint idx = tid; idx < 3u * (uint)D; idx += NT) {
      const uint part = idx / (uint)D;
      const uint d = idx - part * (uint)D;
      const uint c = part * W + h * (uint)D + d;
      float acc = 0.0f;
      for (int j = 0; j < TAPS; ++j) {
        const int e = r + j;                       // position in [window (TAPS-1 rows) ; rows]
        const bfloat xv = e < TAPS - 1 ? CS[(size_t)e * C3 + c] : P[(size_t)(e - (TAPS - 1)) * PS + c];
        const float term = float(xv) * CW[(size_t)j * C3 + c];
        acc = j == 0 ? term : acc + term;
      }
      const bfloat xb = bfloat(acc);
      const bfloat sl = xb * mlx_sigmoid_precise<bfloat>(xb);
      if (part == 0u) sq[d] = float(sl);
      else if (part == 1u) sk[d] = float(sl);
      else sv[d] = float(sl);
    }
    // ---- decays and beta
    for (uint d = tid; d < (uint)D; d += NT) {
      const float av = float(bfloat(sa[d])) + DTB[h * (uint)D + d];
      sg[d] = metal::precise::exp(lb * mlx_sigmoid_precise<float>(a_h * av));
    }
    if (tid == 0u) shr[2] = float(mlx_sigmoid_precise<bfloat>(prow[BO + h]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- l2 norms of q and k (MLX's row-reduce order), q also * D^-1/2, back to bf16
    if (simdgroup_index_in_threadgroup == 0u) {
      float pq = 0.0f, pk = 0.0f;
      for (int blk = 0; blk < RBLK; ++blk) {
        const uint base = (uint)(blk * 128) + 4u * lane;
        for (int i = 0; i < 4; ++i) { pq = sq_acc(pq, sq[base + i]); pk = sq_acc(pk, sk[base + i]); }
      }
      for (int i = 0; 4u * lane + (uint)i < (uint)REXTRA && i < 4; ++i) {
        const uint at = (uint)(RBLK * 128) + 4u * lane + (uint)i;
        pq = sq_acc(pq, sq[at]); pk = sq_acc(pk, sk[at]);
      }
      pq = simd_sum(pq);
      pk = simd_sum(pk);
      if (lane == 0u) {
        shr[0] = metal::precise::rsqrt(pq + 1.0e-6f);
        shr[1] = metal::precise::rsqrt(pk + 1.0e-6f);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      const float rq = shr[0], rk = shr[1];
      const float qscale = metal::precise::rsqrt(float(D));
      for (uint d = tid; d < (uint)D; d += NT) {
        sq[d] = float(bfloat((sq[d] * rq) * qscale));
        sk[d] = float(bfloat(sk[d] * rk));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- gated delta rule, one step (mlx-lm's kernel arithmetic: lane owns NDK key elements, simd_sum)
    {
      const float beta = shr[2];
      for (int j = 0; j < NDV; ++j) {
        const uint dv = ty + (uint)TY * (uint)j;
        float kv = 0.0f;
        for (int i = 0; i < NDK; ++i) {
          const uint s = NDK * lane + i;
          st[j][i] = st[j][i] * sg[s];
          kv += st[j][i] * sk[s];
        }
        kv = simd_sum(kv);
        const float delta = (sv[dv] - kv) * beta;
        float o = 0.0f;
        for (int i = 0; i < NDK; ++i) {
          const uint s = NDK * lane + i;
          st[j][i] = st[j][i] + sk[s] * delta;
          o += st[j][i] * sq[s];
        }
        o = simd_sum(o);
        if (lane == 0u) sy[dv] = float(bfloat(o));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- gated RMSNorm over the value axis (fp32), * sigmoid(gate), to bf16
    if (simdgroup_index_in_threadgroup == 0u) {
      float po = 0.0f;
      for (int blk = 0; blk < RBLK; ++blk) {
        const uint base = (uint)(blk * 128) + 4u * lane;
        for (int i = 0; i < 4; ++i) po = sq_acc(po, sy[base + i]);
      }
      for (int i = 0; 4u * lane + (uint)i < (uint)REXTRA && i < 4; ++i)
        po = sq_acc(po, sy[(uint)(RBLK * 128) + 4u * lane + (uint)i]);
      po = simd_sum(po);
      if (lane == 0u) shr[0] = metal::precise::rsqrt(po / (float)D + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      const float rn = shr[0];
      for (uint d = tid; d < (uint)D; d += NT) {
        float x = sy[d] * rn;
        x = ONW[d] * x;
        x = x * mlx_sigmoid_precise<float>(float(bfloat(sgate[d])));
        Y[(size_t)r * W + h * (uint)D + d] = bfloat(x);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // ---- the state after the last row, and the conv window: the last TAPS-1 rows of [window ; rows]
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
