namespace tfq6 {

// x (TM 16-row fragments) times a bf16 [64 rows, K] block, 64 deep a step: thread t copies row t / 2's half t % 2.
template <typename T, int TM>
inline void k_loop_bf16(thread frag<float> (&acc)[TM][2], const device T* x, int K, int live, bool inside,
                        const device T* wr, threadgroup T* tile, int tn, uint t, short2 home) {
  constexpr int PAD = 64 + 16 / sizeof(T);
  threadgroup T* mine = tile + (t / 2) * PAD + 32 * (t % 2);
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    acc[i][0] = frag<float>(0);
    acc[i][1] = frag<float>(0);
  }
  for (int k = 0; k < K; k += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TF_UNROLL
    for (int i = 0; i < 32; i++) mine[i] = wr[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(disable)
    for (int kk = 0; kk < 64; kk += 32) {
      if (live > 0) {
        frag<T> a[TM][2], b[2][2];
        TF_UNROLL
        for (short i = 0; i < 2; i++) {
          TF_UNROLL
          for (short j = 0; j < 2; j++) {
            frag_get_t(b[j][i], (const threadgroup T*)tile, PAD, tn + 16 * i, kk + 16 * j, home);
          }
        }
        TF_UNROLL
        for (short i = 0; i < TM; i++) {
          TF_UNROLL
          for (short j = 0; j < 2; j++) {
            if (inside) {
              frag_get(a[i][j], x, K, 16 * i, kk + 16 * j, home);
            } else {
              frag_get_in(a[i][j], x, K, 16 * i, kk + 16 * j, home, live, kk + 32);
            }
          }
        }
        TF_UNROLL
        for (short m = 0; m < TM; m++) {
          TF_UNROLL
          for (short j = 0; j < 2; j++) {
            mma_16x32<false, true>(acc[m][0], acc[m][1], a[m][j], b[j][0], b[j][1]);
          }
        }
      }
    }
    x += 64;
    wr += 64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

}  // namespace tfq6

// y (fp32) = x W^T for bf16 W [N, K] (the router): 64x64 tiles, columns below N. P: K N M.
[[kernel]] void tf_mm_bf16_f32_t_nax(const device bfloat16_t* W [[buffer(0)]], const device bfloat16_t* X [[buffer(1)]],
    const device int* P [[buffer(2)]], device float* Y [[buffer(3)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile[64 * (64 + 16 / sizeof(bfloat16_t))];
  const int K = P[0], N = P[1], M = P[2];
  const int row = int(tg.y) * 64, col = int(tg.x) * 64, t = int(sg) * 32 + int(lane);
  const int tm = 32 * int(sg / 2), tn = 32 * int(sg % 2), live = min(32, M - (row + tm));
  const long wrow = long(min(col + t / 2, N - 1));
  frag<float> acc[2][2];
  tfq6::k_loop_bf16<bfloat16_t, 2>(acc, X + long(row + tm) * K, K, live, live == 32, W + wrow * K + 32 * (t % 2), tile,
                                    tn, uint(t), frag_home(ushort(lane)));
  if (col + tn < N && live > 0)
    tfq6::store<float, 2>(acc, Y + long(row + tm) * N + col + tn, N, live, N - (col + tn), frag_home(ushort(lane)));
}

// One K part of x W^T (6-bit g32): fp32 partials PART[z][M][N] for part z of P[3] parts. P: K N M parts.
[[kernel]] void tf_qmm6_splitk_nax(const device uint* W [[buffer(0)]], const device bfloat16_t* S [[buffer(1)]],
    const device bfloat16_t* B [[buffer(2)]], const device bfloat16_t* X [[buffer(3)]], const device int* P [[buffer(4)]],
    device float* PART [[buffer(5)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile[64 * (64 + 16 / sizeof(bfloat16_t))];
  const int K = P[0], N = P[1], M = P[2], parts = P[3], KP = K / parts, z = int(tg.z);
  const int row = int(tg.y) * 64, col = int(tg.x) * 64, t = int(sg) * 32 + int(lane);
  const int tm = 32 * int(sg / 2), tn = 32 * int(sg % 2), live = min(32, M - (row + tm));
  const long wrow = long(min(col + t / 2, N - 1));
  const int WPR = K * 6 / 32, KG = K / 32, k0 = z * KP;
  frag<float> acc[2][2];
  tfq6::k_loop6<bfloat16_t, 2>(acc, X + long(row + tm) * K + k0, KP, K, live, live == 32,
                                W + wrow * WPR + (k0 / 32) * 6 + 6 * (t % 2), S + wrow * KG + k0 / 32 + (t % 2),
                                B + wrow * KG + k0 / 32 + (t % 2), tile, tn, uint(t), frag_home(ushort(lane)));
  if (col + tn < N && live > 0)
    tfq6::store<float, 2>(acc, PART + (long(z) * M + row + tm) * N + col + tn, N, live, N - (col + tn), frag_home(ushort(lane)));
}

// y = bf16 of the parts summed in order. P: parts, M*N.
[[kernel]] void tf_parts_sum(const device float* PART [[buffer(0)]], const device int* P [[buffer(1)]],
    device bfloat16_t* Y [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (int(i) >= P[1]) return;
  float total = 0.0f;
  for (int z = 0; z < P[0]; z++) total += PART[long(z) * P[1] + i];
  Y[i] = bfloat16_t(total);
}

// Flash Next's causal prompt-chunk attention on the tensor units: 64 query rows a threadgroup, 32-key cache blocks, head size 256, 12 query heads a key head, fp32 online softmax; gate bf16(bf16(o) * bsig(gate)) on the way out. P: rows, keys (position + rows), position, cache rows.
constant constexpr float a_masked = -3.402823466e+38f;
inline float a_bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
template <bool MAX>
inline void a_fold(thread const frag<float>& f, thread float (&r)[2]) {
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const short b = 4 * h;
    float t = MAX ? max(max(f[b], f[b + 1]), max(f[b + 2], f[b + 3])) : (f[b] + f[b + 1]) + (f[b + 2] + f[b + 3]);
    const float u = simd_shuffle_xor(t, ushort(1));
    t = MAX ? max(t, u) : t + u;
    const float w = simd_shuffle_xor(t, ushort(8));
    t = MAX ? max(t, w) : t + w;
    r[h] = MAX ? max(r[h], t) : r[h] + t;
  }
}
[[kernel]] void tf_attn256_nax(const device bfloat16_t* Q [[buffer(0)]], const device bfloat16_t* K [[buffer(1)]],
    const device bfloat16_t* V [[buffer(2)]], const device bfloat16_t* GP [[buffer(3)]], const device int* P [[buffer(4)]],
    const device float* F [[buffer(5)]], device bfloat16_t* O [[buffer(6)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int BQ = 64, BK = 32, D = 256, H = 24, GQA = 12, PW = 13952;
  const int qL = P[0], kL = P[1], qoff = P[2], cap = P[3];
  const int h = int(tg.y), q0 = int(tg.x) * BQ;
  const short tm = 16 * short(sg);
  const device bfloat16_t* Qp = Q + long(q0 + tm) * (H * D) + h * D;
  const device bfloat16_t* Kp = K + long(h / GQA) * cap * D;
  const device bfloat16_t* Vp = V + long(h / GQA) * cap * D;
  const float scale2 = F[0] * 1.44269504089f;
  const short2 home = frag_home(ushort(lane));
  frag<float> acc[D / 16];
  TF_UNROLL
  for (short i = 0; i < D / 16; i++) acc[i] = frag<float>(0);
  float top[2] = {a_masked, a_masked}, total[2] = {0.0f, 0.0f};
  const int last_row = qoff + min(q0 + BQ, qL) - 1;
  const int blocks = (min(kL, last_row + 1) + BK - 1) / BK;
  for (int kb = 0; kb < blocks; kb++) {
    const int k0 = kb * BK;
    frag<float> s[2] = {frag<float>(0), frag<float>(0)};
#pragma clang loop unroll_count(4)
    for (short d = 0; d < D / 16; d++) {
      frag<bfloat16_t> q, ka, kb2;
      frag_get(q, Qp, H * D, 0, 16 * d, home);
      frag_get_t(ka, Kp + long(k0) * D, D, 0, 16 * d, home);
      frag_get_t(kb2, Kp + long(k0) * D, D, 16, 16 * d, home);
      mma_16x32<false, true>(s[0], s[1], q, ka, kb2);
    }
    const int r0 = qoff + q0 + tm + home.y;
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int col = k0 + 16 * f + home.x + TF_COL(e), row = r0 + (e >> 2) * 8;
        s[f][e] = (col >= kL || col > row) ? a_masked : s[f][e] * scale2;
      }
    }
    float top_new[2] = {top[0], top[1]};
    a_fold<true>(s[0], top_new);
    a_fold<true>(s[1], top_new);
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) s[f][e] = fast::exp2(s[f][e] - top_new[e >> 2]);
    }
    float scale_old[2];
    TF_UNROLL
    for (short hh = 0; hh < 2; hh++) {
      scale_old[hh] = fast::exp2(top[hh] - top_new[hh]);
      top[hh] = top_new[hh];
      total[hh] = total[hh] * scale_old[hh];
    }
    a_fold<false>(s[0], total);
    a_fold<false>(s[1], total);
    TF_UNROLL
    for (short i = 0; i < D / 16; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * scale_old[e >> 2];
    }
    TF_UNROLL
    for (short d = 0; d < D / 16; d += 2) {
      TF_UNROLL
      for (short k = 0; k < 2; k++) {
        frag<bfloat16_t> v0, v1;
        frag_get(v0, Vp + long(k0) * D, D, 16 * k, 16 * d, home);
        frag_get(v1, Vp + long(k0) * D, D, 16 * k, 16 * d + 16, home);
        mma_16x32<false, false>(acc[d], acc[d + 1], s[k], v0, v1);
      }
    }
  }
  float inv[2] = {1.0f / total[0], 1.0f / total[1]};
  TF_UNROLL
  for (short i = 0; i < D / 16; i++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const int row = q0 + tm + home.y + (e >> 2) * 8, col = 16 * i + home.x + TF_COL(e);
      if (row < qL) {
        const float o = float(bfloat(acc[i][e] * inv[e >> 2]));
        const float g = float(GP[long(row) * PW + h * 2 * D + D + col]);
        O[(long(row) * H + h) * D + col] = bfloat(o * a_bsig(g));
      }
    }
  }
}
