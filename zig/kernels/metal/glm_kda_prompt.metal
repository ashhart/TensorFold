
// A prompt chunk's KDA layer in three passes (appended to kda_rows' source): the fused step's bits, only the recurrence in sequence.

template <int H, int D, int TAPS, int TY, int FB, int GB>
[[kernel]] void glm_kda_pre(const device bfloat16_t* P [[buffer(0)]], const constant int* P_shape [[buffer(1)]],
                            const device bfloat16_t* CS [[buffer(2)]], const device float* CW [[buffer(3)]],
                            const device uint32_t* FBW [[buffer(4)]], const device bfloat16_t* FBS [[buffer(5)]],
                            const device bfloat16_t* FBB [[buffer(6)]], const device uint32_t* GBW [[buffer(7)]],
                            const device bfloat16_t* GBS [[buffer(8)]], const device bfloat16_t* GBB [[buffer(9)]],
                            const device float* A [[buffer(10)]], const device float* DTB [[buffer(11)]],
                            const constant float* LB [[buffer(12)]], device bfloat* QO [[buffer(13)]],
                            device bfloat* KO [[buffer(14)]], device bfloat* VO [[buffer(15)]],
                            device float* GO [[buffer(16)]], device bfloat* GATE [[buffer(17)]],
                            device float* BETA [[buffer(18)]], uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                            uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
                            uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                            uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  const int r = int(threadgroup_position_in_grid.x);
  const uint h = threadgroup_position_in_grid.y;
  const uint lane = thread_position_in_threadgroup.x;
  const uint tid = thread_index_in_threadgroup;
  constexpr int NT = 32 * TY;
  constexpr int RBLK = D / 128;
  constexpr int REXTRA = D - RBLK * 128;
  constexpr uint W = (uint)(H * D);
  constexpr uint C3 = 3u * W;
  constexpr uint FA = C3;
  constexpr uint GA = C3 + (uint)D;
  constexpr uint BO = C3 + 2u * (uint)D;
  const uint PS = (uint)P_shape[1];
  threadgroup float sq[D];
  threadgroup float sk[D];
  threadgroup float sa[D];
  threadgroup float shr[3];
  const float a_h = A[h];
  const float lb = LB[0];
  const size_t at = (size_t)r * W + h * (uint)D;
  device const bfloat* prow = P + (size_t)r * PS;
  {
    constexpr int PER = D / 4;
    constexpr int KG = D / 64;
    constexpr int FKB = D * FB / 8;
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
        else            GATE[at + d] = bfloat(v);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint idx = tid; idx < 3u * (uint)D; idx += NT) {
    const uint part = idx / (uint)D;
    const uint d = idx - part * (uint)D;
    const uint c = part * W + h * (uint)D + d;
    float acc = 0.0f;
    for (int j = 0; j < TAPS; ++j) {
      const int e = r + j;
      const bfloat xv = e < TAPS - 1 ? CS[(size_t)e * C3 + c] : P[(size_t)(e - (TAPS - 1)) * PS + c];
      const float term = float(xv) * CW[(size_t)j * C3 + c];
      acc = j == 0 ? term : acc + term;
    }
    const bfloat xb = bfloat(acc);
    const bfloat sl = xb * mlx_sigmoid_precise<bfloat>(xb);
    if (part == 0u) sq[d] = float(sl);
    else if (part == 1u) sk[d] = float(sl);
    else VO[at + d] = sl;
  }
  for (uint d = tid; d < (uint)D; d += NT) {
    const float av = float(bfloat(sa[d])) + DTB[h * (uint)D + d];
    GO[at + d] = metal::precise::exp(lb * mlx_sigmoid_precise<float>(a_h * av));
  }
  if (tid == 0u) BETA[(size_t)r * H + h] = float(mlx_sigmoid_precise<bfloat>(prow[BO + h]));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_index_in_threadgroup == 0u) {
    float pq = 0.0f, pk = 0.0f;
    for (int blk = 0; blk < RBLK; ++blk) {
      const uint base = (uint)(blk * 128) + 4u * lane;
      for (int i = 0; i < 4; ++i) { pq = sq_acc(pq, sq[base + i]); pk = sq_acc(pk, sk[base + i]); }
    }
    for (int i = 0; 4u * lane + (uint)i < (uint)REXTRA && i < 4; ++i) {
      const uint at2 = (uint)(RBLK * 128) + 4u * lane + (uint)i;
      pq = sq_acc(pq, sq[at2]); pk = sq_acc(pk, sk[at2]);
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
      QO[at + d] = bfloat((sq[d] * rq) * qscale);
      KO[at + d] = bfloat(sk[d] * rk);
    }
  }
}

template <int H, int D, int TAPS, int TY, int FB, int GB>
[[kernel]] void glm_kda_scan(const device bfloat* QO [[buffer(0)]], const device bfloat* KO [[buffer(1)]],
                             const device bfloat* VO [[buffer(2)]], const device float* GO [[buffer(3)]],
                             const device float* BETA [[buffer(4)]], const device float* ST [[buffer(5)]],
                             device float* ST_OUT [[buffer(6)]], device bfloat* SY [[buffer(7)]],
                             constant int& R [[buffer(8)]], uint lane [[thread_index_in_simdgroup]],
                             uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  const uint ty = threadgroup_position_in_grid.x;
  const uint h = threadgroup_position_in_grid.y;
  constexpr int NDK = D / 32;
  constexpr int NDV = D / TY;
  constexpr uint W = (uint)(H * D);
  device const float* si = ST + (size_t)h * D * D;
  float st[NDV][NDK];
  for (int j = 0; j < NDV; ++j) {
    uint dv = ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i) st[j][i] = si[(size_t)dv * D + NDK * lane + i];
  }
  for (int r = 0; r < R; ++r) {
    const size_t at = (size_t)r * W + h * (uint)D;
    float sg[NDK], sk[NDK], sq[NDK];
    for (int i = 0; i < NDK; ++i) {
      const uint s = NDK * lane + i;
      sg[i] = GO[at + s];
      sk[i] = float(KO[at + s]);
      sq[i] = float(QO[at + s]);
    }
    const float beta = BETA[(size_t)r * H + h];
    for (int j = 0; j < NDV; ++j) {
      const uint dv = ty + (uint)TY * (uint)j;
      float kv = 0.0f;
      for (int i = 0; i < NDK; ++i) {
        st[j][i] = st[j][i] * sg[i];
        kv += st[j][i] * sk[i];
      }
      kv = simd_sum(kv);
      const float delta = (float(VO[at + dv]) - kv) * beta;
      float o = 0.0f;
      for (int i = 0; i < NDK; ++i) {
        st[j][i] = st[j][i] + sk[i] * delta;
        o += st[j][i] * sq[i];
      }
      o = simd_sum(o);
      if (lane == 0u) SY[at + dv] = bfloat(o);
    }
  }
  device float* so = ST_OUT + (size_t)h * D * D;
  for (int j = 0; j < NDV; ++j) {
    const uint dv = ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i) so[(size_t)dv * D + NDK * lane + i] = st[j][i];
  }
}

template <int H, int D, int TAPS, int TY, int FB, int GB>
[[kernel]] void glm_kda_post(const device bfloat* SY [[buffer(0)]], const device bfloat* GATE [[buffer(1)]],
                             const device float* ONW [[buffer(2)]], const constant float* EPS [[buffer(3)]],
                             device bfloat16_t* Y [[buffer(4)]], const device bfloat16_t* P [[buffer(5)]],
                             const constant int* P_shape [[buffer(6)]], const device bfloat16_t* CS [[buffer(7)]],
                             device bfloat16_t* CS_OUT [[buffer(8)]], uint lane [[thread_index_in_simdgroup]],
                             uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  const int r = int(threadgroup_position_in_grid.x);
  const uint h = threadgroup_position_in_grid.y;
  constexpr int RBLK = D / 128;
  constexpr int REXTRA = D - RBLK * 128;
  constexpr uint W = (uint)(H * D);
  constexpr uint C3 = 3u * W;
  const int R = int(P_shape[0]);
  const uint PS = (uint)P_shape[1];
  const float eps = EPS[0];
  const size_t at = (size_t)r * W + h * (uint)D;
  float po = 0.0f;
  for (int blk = 0; blk < RBLK; ++blk) {
    const uint base = (uint)(blk * 128) + 4u * lane;
    for (int i = 0; i < 4; ++i) po = sq_acc(po, float(SY[at + base + i]));
  }
  for (int i = 0; 4u * lane + (uint)i < (uint)REXTRA && i < 4; ++i)
    po = sq_acc(po, float(SY[at + (uint)(RBLK * 128) + 4u * lane + (uint)i]));
  po = simd_sum(po);
  const float rn = metal::precise::rsqrt(po / (float)D + eps);
  for (uint d = lane; d < (uint)D; d += 32u) {
    float x = float(SY[at + d]) * rn;
    x = ONW[d] * x;
    x = x * mlx_sigmoid_precise<float>(float(GATE[at + d]));
    Y[at + d] = bfloat(x);
  }
  if (r != 0) return;
  for (uint idx = lane; idx < 3u * (uint)D * (uint)(TAPS - 1); idx += 32u) { // the next chunk's window: the last rows
    const uint m = idx / (3u * (uint)D);
    const uint rem = idx - m * 3u * (uint)D;
    const uint part = rem / (uint)D;
    const uint d = rem - part * (uint)D;
    const uint c = part * W + h * (uint)D + d;
    const int e = R + int(m);
    CS_OUT[(size_t)m * C3 + c] = e < TAPS - 1 ? CS[(size_t)e * C3 + c] : P[(size_t)(e - (TAPS - 1)) * PS + c];
  }
}

template [[host_name("glm_kda_pre")]] [[kernel]] decltype(glm_kda_pre<64, 128, 4, 32, 4, 4>) glm_kda_pre<64, 128, 4, 32, 4, 4>;
template [[host_name("glm_kda_scan")]] [[kernel]] decltype(glm_kda_scan<64, 128, 4, 32, 4, 4>) glm_kda_scan<64, 128, 4, 32, 4, 4>;
template [[host_name("glm_kda_post")]] [[kernel]] decltype(glm_kda_post<64, 128, 4, 32, 4, 4>) glm_kda_post<64, 128, 4, 32, 4, 4>;
