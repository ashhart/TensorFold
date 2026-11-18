// Flash Next prompt chunks: hyper-connection pieces in prefill_hc's arithmetic, router rows, top-k, the expert sort
// (count, offsets, slots), row gathers and scatters, and the activation. Compiled after the gate/up header (simd_topk_all).
inline float pf_bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
inline float pf_bsilu(float x) { return float(bfloat(x / (1.0f + metal::exp(-x)))); }
inline float pf_rinv(const device float* ssp, int r, int s, int nt, int streams, int dims, float eps) {
  float total = 0.0f;
  for (int j = 0; j < nt; j++) total += ssp[(r * nt + j) * streams + s];
  return metal::rsqrt(total / float(dims) + eps);
}

// Thread (e, r): bf16((h * rinv(stream)) * scale), the block's normed input.
[[kernel]] void pf_hc_normed(const device bfloat* HN [[buffer(0)]], const device float* SSP [[buffer(1)]],
    const device float* NW [[buffer(2)]], const device float* eps [[buffer(3)]], device bfloat* NORMED [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]) {
  constexpr int S = 4, D = 2560, W = S * D;
  const int e = int(pos.x), r = int(pos.y);
  const float rv = pf_rinv(SSP, r, e / D, D / 256, S, D, eps[0]);
  NORMED[size_t(r) * W + e] = bfloat((float(HN[size_t(r) * W + e]) * rv) * NW[e]);
}

// Thread (c, r): output c of the down + inject rows / S, then SiLU (c < LOW) or the gate 2 sigmoid.
[[kernel]] void pf_hc_act(const device bfloat* DN [[buffer(0)]], const constant int& nd [[buffer(1)]],
    device bfloat* ACT [[buffer(2)]], device bfloat* INJ [[buffer(3)]], uint2 pos [[thread_position_in_grid]]) {
  constexpr int S = 4, LOW = 320;
  const int c = int(pos.x), r = int(pos.y);
  if (c >= nd) return;
  const float v4 = float(bfloat(float(DN[size_t(r) * nd + c]) / float(S)));
  if (c < LOW) ACT[size_t(r) * LOW + c] = bfloat(pf_bsilu(v4));
  else INJ[size_t(r) * S + (c - LOW)] = bfloat(2.0f * pf_bsig(v4));
}

// Thread (d, r): the mean over streams of bf16(sigmoid(up) * normed).
[[kernel]] void pf_hc_mix(const device bfloat* UP [[buffer(0)]], const device bfloat* NORMED [[buffer(1)]],
    device bfloat* MIXED [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
  constexpr int S = 4, D = 2560, W = S * D;
  const int d = int(pos.x), r = int(pos.y);
  float total = 0.0f;
  for (int s = 0; s < S; s++) {
    const size_t e = size_t(r) * W + s * D + d;
    total += float(bfloat(pf_bsig(float(UP[e])) * float(NORMED[e])));
  }
  MIXED[size_t(r) * D + d] = bfloat(total / float(S));
}

// Router logits in fp32: simdgroup s of threadgroup (j-block, r) sums output 8 j-block + s over K, lanes strided.
[[kernel]] void pf_router(const device bfloat* X [[buffer(0)]], const device bfloat* RW [[buffer(1)]],
    device float* LG [[buffer(2)]], uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int K = 2560, NL = 513;
  const int j = int(tg.x) * 8 + int(sgi), r = int(tg.y);
  if (j >= NL) return;
  float acc = 0.0f;
  for (int k = int(lane); k < K; k += 32) acc += float(X[size_t(r) * K + k]) * float(RW[size_t(j) * K + k]);
  acc = simd_sum(acc);
  if (lane == 0) LG[size_t(r) * NL + j] = acc;
}

// Row r's ten experts (the decode's top-k rounds) and their softmax weights; counts each expert's pairs.
[[kernel]] void pf_route(const device float* LG [[buffer(0)]], device uint* PICK [[buffer(1)]],
    device float* WTS [[buffer(2)]], device atomic_uint* CNT [[buffer(3)]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int TOPK = 10, NE = 512, NL = 513;
  const int r = int(tg.y);
  int ids[TOPK];
  float picked[TOPK];
  simd_topk_all<NE, TOPK>(LG + size_t(r) * NL, lane, ids, picked);
  if (lane == 0) {
    float total = 0.0f;
    float ex[TOPK];
    for (int kk = 0; kk < TOPK; kk++) { ex[kk] = metal::exp(picked[kk] - picked[0]); total += ex[kk]; }
    for (int kk = 0; kk < TOPK; kk++) {
      WTS[r * TOPK + kk] = float(bfloat(ex[kk] / total));
      PICK[r * TOPK + kk] = uint(ids[kk]);
      atomic_fetch_add_explicit(CNT + ids[kk], 1u, memory_order_relaxed);
    }
  }
}

// Each expert's first sorted slot (exclusive scan of the counts); the counts go back to zero for the next layer.
[[kernel]] void pf_offsets(device atomic_uint* CNT [[buffer(0)]], device int* OFF [[buffer(1)]],
    device atomic_uint* CUR [[buffer(2)]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup int counts[512];
  counts[t] = int(atomic_load_explicit(CNT + t, memory_order_relaxed));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int before = 0;
  for (uint i = 0; i < t; i++) before += counts[i];
  OFF[t] = before;
  atomic_store_explicit(CUR + t, uint(before), memory_order_relaxed);
  atomic_store_explicit(CNT + t, 0u, memory_order_relaxed);
}

// Pair p (row * 10 + k) into a slot of its expert's range (order within an expert does not change any row's sums).
[[kernel]] void pf_sort(const device uint* PICK [[buffer(0)]], device atomic_uint* CUR [[buffer(1)]],
    device int* ROW_OF [[buffer(2)]], const constant int& pairs [[buffer(3)]], uint p [[thread_position_in_grid]]) {
  if (int(p) >= pairs) return;
  const uint slot = atomic_fetch_add_explicit(CUR + PICK[p], 1u, memory_order_relaxed);
  ROW_OF[slot] = int(p);
}

// Slot s's input row (the pair's token row), 8 values a thread.
[[kernel]] void pf_gather_rows(const device bfloat* X [[buffer(0)]], const device int* ROW_OF [[buffer(1)]],
    device bfloat* XS [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
  constexpr int K = 2560, TOPK = 10;
  const int c = int(pos.x) * 8, slot = int(pos.y);
  const int row = ROW_OF[slot] / TOPK;
  for (int i = 0; i < 8; i++) XS[size_t(slot) * K + c + i] = X[size_t(row) * K + c + i];
}

// The expert activation as the decode rounds it: bf16(bsilu(gate) * up).
[[kernel]] void pf_act(const device bfloat* G [[buffer(0)]], const device bfloat* U [[buffer(1)]],
    device bfloat* A [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  A[i] = bfloat(pf_bsilu(float(G[i])) * float(U[i]));
}

// Slot s's down output into its pair's place in the combine's layout [rows, 11, D], 8 values a thread.
[[kernel]] void pf_scatter_y(const device bfloat* DS [[buffer(0)]], const device int* ROW_OF [[buffer(1)]],
    device bfloat* YD [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
  constexpr int D = 2560, TOPK = 10, SLOTS = TOPK + 1;
  const int c = int(pos.x) * 8, slot = int(pos.y);
  const int p = ROW_OF[slot], row = p / TOPK, k = p % TOPK;
  for (int i = 0; i < 8; i++) YD[(size_t(row) * SLOTS + k) * D + c + i] = DS[size_t(slot) * D + c + i];
}

[[kernel]] void pf_copy(const device uint* S [[buffer(0)]], device uint* T [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  T[i] = S[i];
}

// DeltaNet over a prompt chunk in three launches with q4_gdn_step's arithmetic: every row's conv, norms and gates at
// once; the recurrence, each simdgroup alone over its four state rows; every row's gated RMS norm at once.
inline float pf_fsig(float x) { return 1.0f / (1.0f + metal::exp(-x)); }
inline float pf_log1p(float x) {
  const float u = 1.0f + x;
  return u == 1.0f ? x : x * (metal::log(u) / (u - 1.0f));
}
inline float pf_softplus(float x) { return x > 20.0f ? x : pf_log1p(metal::exp(x)); }

// Threadgroup (group, r): 128 channels of row r, groups 0..15 q heads, 16..31 k heads, 32..79 v heads. Conv + SiLU,
// the q/k L2 norms, the v head's gates; the conv window after the last row into CSO.
[[kernel]] void pf_gdn_pre(const device bfloat* P [[buffer(0)]], const device bfloat* CS [[buffer(1)]],
    const device bfloat* CW [[buffer(2)]], const device bfloat* ALOG [[buffer(3)]], const device bfloat* DT [[buffer(4)]],
    const constant int& R [[buffer(5)]], device float* QN [[buffer(6)]], device float* KN [[buffer(7)]],
    device float* V [[buffer(8)]], device float* G [[buffer(9)]], device float* BETA [[buffer(10)]],
    device bfloat* CSO [[buffer(11)]], const constant int4& MK [[buffer(12)]], device bfloat* CST [[buffer(13)]],
    uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NK = 16, NV = 48, DK = 128, DV = 128, TAPS = 4;
  constexpr int C = 2 * NK * DK + NV * DV;
  constexpr int PW = C + NV * DV + 2 * NV;
  threadgroup float x[128];
  threadgroup float inv_s;
  const int group = int(tg.x), r = int(tg.y);
  const int c = group < 32 ? group * DK + int(t) : 2 * NK * DK + (group - 32) * DV + int(t);
  float conv = 0.0f;
  for (int tap = 0; tap < TAPS; tap++) {
    const int at = r + tap;
    const float xv = at < TAPS - 1 ? float(CS[at * C + c]) : float(P[size_t(at - (TAPS - 1)) * PW + c]);
    conv = fma(float(CW[c * TAPS + tap]), xv, conv);
  }
  const float act = pf_bsilu(conv);
  if (r == R - 1) {
    for (int j = 0; j < TAPS - 1; j++) {
      const int at = r + 1 + j;
      CSO[j * C + c] = at < TAPS - 1 ? CS[at * C + c] : P[size_t(at - (TAPS - 1)) * PW + c];
    }
  }
  for (int m = 0; m < 4; m++) // a mark after row r: the conv window there, as a chunk ending at it would leave it
    if (MK[m] == r + 1)
      for (int j = 0; j < TAPS - 1; j++) {
        const int at = r + 1 + j;
        CST[(m * (TAPS - 1) + j) * C + c] = at < TAPS - 1 ? CS[at * C + c] : P[size_t(at - (TAPS - 1)) * PW + c];
      }
  if (group >= 32) {
    const int hv = group - 32;
    V[(size_t(r) * NV + hv) * DV + t] = act;
    if (t == 0) {
      const float b = float(P[size_t(r) * PW + C + NV * DV + hv]);
      const float a = float(P[size_t(r) * PW + C + NV * DV + NV + hv]);
      G[r * NV + hv] = metal::exp(-metal::exp(float(ALOG[hv])) * pf_softplus(a + float(DT[hv])));
      BETA[r * NV + hv] = pf_bsig(b);
    }
    return;
  }
  x[t] = act;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool isq = group < 16;
  if (sg == 0) {
    float ss = 0.0f;
    for (int i = 0; i < DK / 32; i++) {
      const float v = x[lane * (DK / 32) + i];
      ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    if (lane == 0) inv_s = metal::rsqrt(ss + 1e-6f) * (isq ? metal::rsqrt(float(DK)) : 1.0f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int hk = isq ? group : group - 16;
  (isq ? QN : KN)[(size_t(r) * NK + hk) * DK + t] = x[t] * inv_s;
}

// Four threadgroups a value head: simdgroup s of threadgroup (hv, quarter) owns state row dv = 32 quarter + s (lane l
// its columns 4 l .. 4 l + 3) through every row with no barrier, the next row's inputs loading while this one computes;
// YS gets bf16-rounded outputs, SO the state after the last row.
[[kernel]] void pf_gdn_scan(const device float* QN [[buffer(0)]], const device float* KN [[buffer(1)]],
    const device float* V [[buffer(2)]], const device float* G [[buffer(3)]], const device float* BETA [[buffer(4)]],
    const device float* SIN [[buffer(5)]], const constant int& R [[buffer(6)]], device float* YS [[buffer(7)]],
    device float* SO [[buffer(8)]], const constant int4& MK [[buffer(9)]], device float* ST [[buffer(10)]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NK = 16, NV = 48, DK = 128, DV = 128;
  const int hv = int(tg.x) / 4, hk = hv / (NV / NK), dv = (int(tg.x) % 4) * 32 + int(sg);
  float state[4];
  for (int i = 0; i < 4; i++) state[i] = SIN[(size_t(hv) * DV + dv) * DK + lane * 4 + i];
  float g_n = G[hv], beta_n = BETA[hv], v_n = V[size_t(hv) * DV + dv], kk_n[4], qq_n[4];
  for (int i = 0; i < 4; i++) {
    kk_n[i] = KN[size_t(hk) * DK + lane * 4 + i];
    qq_n[i] = QN[size_t(hk) * DK + lane * 4 + i];
  }
  for (int r = 0; r < R; r++) {
    const float g = g_n, beta = beta_n, vv = v_n;
    float kk[4], qq[4];
    for (int i = 0; i < 4; i++) { kk[i] = kk_n[i]; qq[i] = qq_n[i]; }
    if (r + 1 < R) {
      const int rn = r + 1;
      g_n = G[rn * NV + hv];
      beta_n = BETA[rn * NV + hv];
      v_n = V[(size_t(rn) * NV + hv) * DV + dv];
      for (int i = 0; i < 4; i++) {
        kk_n[i] = KN[(size_t(rn) * NK + hk) * DK + lane * 4 + i];
        qq_n[i] = QN[(size_t(rn) * NK + hk) * DK + lane * 4 + i];
      }
    }
    float kv = 0.0f;
    for (int i = 0; i < 4; i++) {
      state[i] = state[i] * g;
      kv += state[i] * kk[i];
    }
    kv = simd_sum(kv);
    const float delta = (vv - kv) * beta;
    float out = 0.0f;
    for (int i = 0; i < 4; i++) {
      state[i] = state[i] + kk[i] * delta;
      out += state[i] * qq[i];
    }
    out = simd_sum(out);
    if (lane == 0) YS[(size_t(r) * NV + hv) * DV + dv] = float(bfloat(out));
    for (int m = 0; m < 4; m++)
      if (MK[m] == r + 1)
        for (int i = 0; i < 4; i++) ST[size_t(m) * NV * DV * DK + (size_t(hv) * DV + dv) * DK + lane * 4 + i] = state[i];
  }
  for (int i = 0; i < 4; i++) SO[(size_t(hv) * DV + dv) * DK + lane * 4 + i] = state[i];
}

// The recurrence with four state rows a simdgroup (dv = 32 part + 4 s + i, lane l columns 4 l .. 4 l + 3) and both
// reductions of a row at once: Sk = S k and Sq = S q from the state before the row, then delta = (v - g Sk) beta,
// out = g Sq + delta (k . q), S = g S + delta k^T (the same fp32 terms as pf_gdn_scan, in another order). Threadgroup
// (hv, part) of 8 simdgroups covers 32 state rows; the next row's inputs load while this one computes.
[[kernel]] void pf_gdn_scan4(const device float* QN [[buffer(0)]], const device float* KN [[buffer(1)]],
    const device float* V [[buffer(2)]], const device float* G [[buffer(3)]], const device float* BETA [[buffer(4)]],
    const device float* SIN [[buffer(5)]], const constant int& R [[buffer(6)]], device float* YS [[buffer(7)]],
    device float* SO [[buffer(8)]], const constant int4& MK [[buffer(9)]], device float* ST [[buffer(10)]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NK = 16, NV = 48, DK = 128, DV = 128, NR = 4;
  const int hv = int(tg.x) / 4, hk = hv / (NV / NK), dv0 = (int(tg.x) % 4) * 32 + int(sg) * NR;
  float S[NR][4];
  for (int i = 0; i < NR; i++)
    for (int c = 0; c < 4; c++) S[i][c] = SIN[(size_t(hv) * DV + dv0 + i) * DK + lane * 4 + c];
  float g_n = G[hv], beta_n = BETA[hv], v_n[NR], k_n[4], q_n[4];
  for (int i = 0; i < NR; i++) v_n[i] = V[size_t(hv) * DV + dv0 + i];
  for (int c = 0; c < 4; c++) {
    k_n[c] = KN[size_t(hk) * DK + lane * 4 + c];
    q_n[c] = QN[size_t(hk) * DK + lane * 4 + c];
  }
  for (int r = 0; r < R; r++) {
    const float g = g_n, beta = beta_n;
    float v[NR], k[4], q[4];
    for (int i = 0; i < NR; i++) v[i] = v_n[i];
    for (int c = 0; c < 4; c++) { k[c] = k_n[c]; q[c] = q_n[c]; }
    if (r + 1 < R) {
      const int rn = r + 1;
      g_n = G[rn * NV + hv];
      beta_n = BETA[rn * NV + hv];
      for (int i = 0; i < NR; i++) v_n[i] = V[(size_t(rn) * NV + hv) * DV + dv0 + i];
      for (int c = 0; c < 4; c++) {
        k_n[c] = KN[(size_t(rn) * NK + hk) * DK + lane * 4 + c];
        q_n[c] = QN[(size_t(rn) * NK + hk) * DK + lane * 4 + c];
      }
    }
    float kq = 0.0f, sk[NR], sq[NR];
    for (int c = 0; c < 4; c++) kq = fma(k[c], q[c], kq);
    for (int i = 0; i < NR; i++) {
      sk[i] = 0.0f;
      sq[i] = 0.0f;
      for (int c = 0; c < 4; c++) {
        sk[i] = fma(S[i][c], k[c], sk[i]);
        sq[i] = fma(S[i][c], q[c], sq[i]);
      }
    }
    kq = simd_sum(kq);
    for (int i = 0; i < NR; i++) {
      sk[i] = simd_sum(sk[i]);
      sq[i] = simd_sum(sq[i]);
    }
    for (int i = 0; i < NR; i++) {
      const float delta = (v[i] - g * sk[i]) * beta;
      for (int c = 0; c < 4; c++) S[i][c] = fma(delta, k[c], g * S[i][c]);
      if (lane == 0) YS[(size_t(r) * NV + hv) * DV + dv0 + i] = float(bfloat(fma(delta, kq, g * sq[i])));
    }
    for (int m = 0; m < 4; m++)
      if (MK[m] == r + 1)
        for (int i = 0; i < NR; i++)
          for (int c = 0; c < 4; c++) ST[size_t(m) * NV * DV * DK + (size_t(hv) * DV + dv0 + i) * DK + lane * 4 + c] = S[i][c];
  }
  for (int i = 0; i < NR; i++)
    for (int c = 0; c < 4; c++) SO[(size_t(hv) * DV + dv0 + i) * DK + lane * 4 + c] = S[i][c];
}

// pf_gdn_scan4's arithmetic with a state row over 16 lanes (8 columns each, so each reduction is one shuffle step
// shorter) and 8 rows a simdgroup: lanes 16 hl .. 16 hl + 15 hold rows 2 i + hl. Threadgroup (hv, part) of 8 simdgroups
// covers 64 state rows, two a head.
[[kernel]] void pf_gdn_scan8(const device float* QN [[buffer(0)]], const device float* KN [[buffer(1)]],
    const device float* V [[buffer(2)]], const device float* G [[buffer(3)]], const device float* BETA [[buffer(4)]],
    const device float* SIN [[buffer(5)]], const constant int& R [[buffer(6)]], device float* YS [[buffer(7)]],
    device float* SO [[buffer(8)]], const constant int4& MK [[buffer(9)]], device float* ST [[buffer(10)]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NK = 16, NV = 48, DK = 128, DV = 128, NR = 4;
  const int hv = int(tg.x) / 2, hk = hv / (NV / NK), hl = int(lane) / 16, l16 = int(lane) % 16;
  const int dv0 = (int(tg.x) % 2) * 64 + int(sg) * 8 + hl;
  float S[NR][8];
  for (int i = 0; i < NR; i++)
    for (int c = 0; c < 8; c++) S[i][c] = SIN[(size_t(hv) * DV + dv0 + 2 * i) * DK + l16 * 8 + c];
  float g_n = G[hv], beta_n = BETA[hv], v_n[NR];
  float4 k_n0, k_n1, q_n0, q_n1;
  for (int i = 0; i < NR; i++) v_n[i] = V[size_t(hv) * DV + dv0 + 2 * i];
  {
    const device float4* kp = (const device float4*)(KN + size_t(hk) * DK + l16 * 8);
    const device float4* qp = (const device float4*)(QN + size_t(hk) * DK + l16 * 8);
    k_n0 = kp[0]; k_n1 = kp[1]; q_n0 = qp[0]; q_n1 = qp[1];
  }
  for (int r = 0; r < R; r++) {
    const float g = g_n, beta = beta_n;
    float v[NR];
    for (int i = 0; i < NR; i++) v[i] = v_n[i];
    const float k[8] = {k_n0.x, k_n0.y, k_n0.z, k_n0.w, k_n1.x, k_n1.y, k_n1.z, k_n1.w};
    const float q[8] = {q_n0.x, q_n0.y, q_n0.z, q_n0.w, q_n1.x, q_n1.y, q_n1.z, q_n1.w};
    if (r + 1 < R) {
      const int rn = r + 1;
      g_n = G[rn * NV + hv];
      beta_n = BETA[rn * NV + hv];
      for (int i = 0; i < NR; i++) v_n[i] = V[(size_t(rn) * NV + hv) * DV + dv0 + 2 * i];
      const device float4* kp = (const device float4*)(KN + (size_t(rn) * NK + hk) * DK + l16 * 8);
      const device float4* qp = (const device float4*)(QN + (size_t(rn) * NK + hk) * DK + l16 * 8);
      k_n0 = kp[0]; k_n1 = kp[1]; q_n0 = qp[0]; q_n1 = qp[1];
    }
    float kq = 0.0f, sk[NR], sq[NR];
    for (int c = 0; c < 8; c++) kq = fma(k[c], q[c], kq);
    for (int i = 0; i < NR; i++) {
      sk[i] = 0.0f;
      sq[i] = 0.0f;
      for (int c = 0; c < 8; c++) {
        sk[i] = fma(S[i][c], k[c], sk[i]);
        sq[i] = fma(S[i][c], q[c], sq[i]);
      }
    }
    for (ushort o = 8; o > 0; o >>= 1) { // sums over the 16 lanes of each half
      kq += simd_shuffle_xor(kq, o);
      for (int i = 0; i < NR; i++) {
        sk[i] += simd_shuffle_xor(sk[i], o);
        sq[i] += simd_shuffle_xor(sq[i], o);
      }
    }
    for (int i = 0; i < NR; i++) {
      const float delta = (v[i] - g * sk[i]) * beta;
      for (int c = 0; c < 8; c++) S[i][c] = fma(delta, k[c], g * S[i][c]);
      if (l16 == 0) YS[(size_t(r) * NV + hv) * DV + dv0 + 2 * i] = float(bfloat(fma(delta, kq, g * sq[i])));
    }
    for (int m = 0; m < 4; m++)
      if (MK[m] == r + 1)
        for (int i = 0; i < NR; i++)
          for (int c = 0; c < 8; c++) ST[size_t(m) * NV * DV * DK + (size_t(hv) * DV + dv0 + 2 * i) * DK + l16 * 8 + c] = S[i][c];
  }
  for (int i = 0; i < NR; i++)
    for (int c = 0; c < 8; c++) SO[(size_t(hv) * DV + dv0 + 2 * i) * DK + l16 * 8 + c] = S[i][c];
}

// Threadgroup (hv, r): the sigmoid-gated RMS norm of row r's head hv, bf16 out.
[[kernel]] void pf_gdn_post(const device float* YS [[buffer(0)]], const device bfloat* P [[buffer(1)]],
    const device bfloat* NW [[buffer(2)]], const device float* eps [[buffer(3)]], device bfloat* OUT [[buffer(4)]],
    uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NK = 16, NV = 48, DK = 128, DV = 128;
  constexpr int C = 2 * NK * DK + NV * DV;
  constexpr int PW = C + NV * DV + 2 * NV;
  threadgroup float ys[DV];
  threadgroup float red;
  const int hv = int(tg.x), r = int(tg.y);
  ys[t] = YS[(size_t(r) * NV + hv) * DV + t];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float ss = 0.0f;
    for (int i = 0; i < DV / 32; i++) { const float v = ys[lane * (DV / 32) + i]; ss = fma(v, v, ss); }
    ss = simd_sum(ss);
    if (lane == 0) red = metal::rsqrt(ss / float(DV) + eps[0]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float y = float(bfloat(float(NW[t]) * float(bfloat(ys[t] * red))));
  const float z = float(P[size_t(r) * PW + C + hv * DV + int(t)]);
  OUT[size_t(r) * NV * DV + hv * DV + int(t)] = bfloat(y * pf_fsig(z));
}
