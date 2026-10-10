
// A prompt chunk's KDA layer (appended to kda_rows' source): f_b and g_b on the tensor units, then prep, the recurrence and the output norm.

// A prompt block's KDA inputs, a head a simdgroup (conv, SiLU, q/k norms, f_b's decay, beta); P_shape: rows, pitch, block start (FB, outputs by block row).
template <int H, int D, int TAPS, int HB>
[[kernel]] void glm_kda_prep(const device bfloat16_t* P [[buffer(0)]], const constant int* P_shape [[buffer(1)]],
                             const device bfloat16_t* CS [[buffer(2)]], const device float* CW [[buffer(3)]],
                             const device bfloat* FB [[buffer(4)]], const device float* A [[buffer(5)]],
                             const device float* DTB [[buffer(6)]], const constant float* LB [[buffer(7)]],
                             device bfloat* QO [[buffer(8)]], device bfloat* KO [[buffer(9)]], device bfloat* VO [[buffer(10)]],
                             device float* GO [[buffer(11)]], device float* BETA [[buffer(12)]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                             uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NDK = D / 32;
  constexpr uint W = (uint)(H * D);
  constexpr uint C3 = 3u * W;
  constexpr uint BO = C3 + 2u * (uint)D;
  const int rl = int(tg.x), r = rl + P_shape[2];
  const uint h = tg.y * (uint)HB + sg;
  const uint PS = (uint)P_shape[1];
  const size_t at = (size_t)rl * W + h * (uint)D;
  float sq[NDK], sk[NDK];
  for (uint part = 0; part < 3u; ++part) {
    for (int i = 0; i < NDK; ++i) {
      const uint d = (uint)NDK * lane + (uint)i;
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
      if (part == 0u) sq[i] = float(sl);
      else if (part == 1u) sk[i] = float(sl);
      else VO[at + d] = sl;
    }
  }
  const float a_h = A[h];
  const float lb = LB[0];
  for (int i = 0; i < NDK; ++i) {
    const uint d = (uint)NDK * lane + (uint)i;
    const float av = float(FB[at + d]) + DTB[h * (uint)D + d];
    GO[at + d] = metal::precise::exp(lb * mlx_sigmoid_precise<float>(a_h * av));
  }
  if (lane == 0u) BETA[(size_t)rl * H + h] = float(mlx_sigmoid_precise<bfloat>(P[(size_t)r * PS + BO + h]));
  float pq = 0.0f, pk = 0.0f;
  for (int i = 0; i < NDK; ++i) {
    pq = sq_acc(pq, sq[i]);
    pk = sq_acc(pk, sk[i]);
  }
  pq = simd_sum(pq);
  pk = simd_sum(pk);
  const float rq = metal::precise::rsqrt(pq + 1.0e-6f), rk = metal::precise::rsqrt(pk + 1.0e-6f);
  const float qscale = metal::precise::rsqrt(float(D));
  for (int i = 0; i < NDK; ++i) {
    const uint d = (uint)NDK * lane + (uint)i;
    QO[at + d] = bfloat((sq[i] * rq) * qscale);
    KO[at + d] = bfloat(sk[i] * rk);
  }
}

// The recurrence over R rows: 8 lanes a value column (16 key dims each, sums over the 8 by xor shuffles), rows staged B at a time.
template <int H, int D, int SG, int B>
[[kernel]] void glm_kda_scan(const device bfloat* QO [[buffer(0)]], const device bfloat* KO [[buffer(1)]],
                             const device bfloat* VO [[buffer(2)]], const device float* GO [[buffer(3)]],
                             const device float* BETA [[buffer(4)]], const device float* ST [[buffer(5)]],
                             device float* ST_OUT [[buffer(6)]], device bfloat* SY [[buffer(7)]],
                             constant int& R [[buffer(8)]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]],
                             uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NK = D / 8;    // a lane's key dims
  constexpr int VC = SG * 4;   // a threadgroup's value columns, four a simdgroup
  constexpr int NT = 32 * SG;
  constexpr uint W = (uint)(H * D);
  threadgroup bfloat sqs[B * D], sks[B * D], svs[B * VC];
  threadgroup float sgs[B * D], sbs[B];
  const uint h = tg.y;
  const int c = int(sg) * 4 + int(lane >> 3); // this lane's value column in the threadgroup
  const uint dv = tg.x * (uint)VC + (uint)c;
  const int k0 = NK * int(lane & 7u);
  device const float* si = ST + ((size_t)h * D + dv) * D + k0;
  float st[NK];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < NK; ++i) st[i] = si[i];
  for (int r0 = 0; r0 < R; r0 += B) {
    const int n = min(B, R - r0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = int(tid); idx < n * D; idx += NT) { // the block's q, k and decay rows, and this threadgroup's v columns
      const int b = idx / D, d = idx - b * D;
      const size_t at = (size_t)(r0 + b) * W + h * (uint)D + (uint)d;
      sqs[idx] = QO[at];
      sks[idx] = KO[at];
      sgs[idx] = GO[at];
    }
    for (int idx = int(tid); idx < n * VC; idx += NT) {
      const int b = idx / VC, cc = idx - b * VC;
      svs[idx] = VO[(size_t)(r0 + b) * W + h * (uint)D + tg.x * (uint)VC + (uint)cc];
    }
    if (int(tid) < n) sbs[tid] = BETA[(size_t)(r0 + int(tid)) * H + h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int b = 0; b < n; ++b) {
      const threadgroup bfloat* qr = sqs + b * D + k0;
      const threadgroup bfloat* kr = sks + b * D + k0;
      const threadgroup float* gr = sgs + b * D + k0;
      float kv = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < NK; ++i) {
        st[i] = st[i] * gr[i];
        kv += st[i] * float(kr[i]);
      }
      kv += simd_shuffle_xor(kv, ushort(1));
      kv += simd_shuffle_xor(kv, ushort(2));
      kv += simd_shuffle_xor(kv, ushort(4));
      const float delta = (float(svs[b * VC + c]) - kv) * sbs[b];
      float o = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < NK; ++i) {
        st[i] = st[i] + float(kr[i]) * delta;
        o += st[i] * float(qr[i]);
      }
      o += simd_shuffle_xor(o, ushort(1));
      o += simd_shuffle_xor(o, ushort(2));
      o += simd_shuffle_xor(o, ushort(4));
      if ((lane & 7u) == 0u) SY[(size_t)(r0 + b) * W + h * (uint)D + dv] = bfloat(o);
    }
  }
  device float* so = ST_OUT + ((size_t)h * D + dv) * D + k0;
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < NK; ++i) so[i] = st[i];
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

template [[host_name("glm_kda_prep")]] [[kernel]] decltype(glm_kda_prep<64, 128, 4, 4>) glm_kda_prep<64, 128, 4, 4>;
template [[host_name("glm_kda_scan")]] [[kernel]] decltype(glm_kda_scan<64, 128, 16, 16>) glm_kda_scan<64, 128, 16, 16>;
template [[host_name("glm_kda_post")]] [[kernel]] decltype(glm_kda_post<64, 128, 4, 32, 4, 4>) glm_kda_post<64, 128, 4, 32, 4, 4>;
// TP2: one Mac's 32 heads
template [[host_name("glm_kda_prep_tp")]] [[kernel]] decltype(glm_kda_prep<32, 128, 4, 4>) glm_kda_prep<32, 128, 4, 4>;
template [[host_name("glm_kda_scan_tp")]] [[kernel]] decltype(glm_kda_scan<32, 128, 16, 16>) glm_kda_scan<32, 128, 16, 16>;
template [[host_name("glm_kda_post_tp")]] [[kernel]] decltype(glm_kda_post<32, 128, 4, 32, 4, 4>) glm_kda_post<32, 128, 4, 32, 4, 4>;
