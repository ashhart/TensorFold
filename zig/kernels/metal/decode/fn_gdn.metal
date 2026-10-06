// Flash Next's DeltaNet step for a window of up to 16 rows, value head FZ_H0 + threadgroup (the recorded q4_gdn's
// arithmetic, so each row's outputs and states are its bits at any width): every row's conv, q/k norms and gates at
// once, then the state recurrence row after row with no barrier between rows, then every row's gated RMS norm. The
// includer puts the recorded kernels' helpers (bsilu, bsig, fsig, fsoftplus) before this file and sets FZ_GDN_REPLAY:
// 0 stores every row's state (SO), 1 keeps one state a layer in place (S) and replays the previous window's kept rows
// (AR[0] of them, from that window's k, v, k norms and gates in PREV) before this window's rows, whose own go to CUR.

constant constexpr int GDN_NK = 16, GDN_NV = 48, GDN_DK = 128, GDN_DV = 128, GDN_TAPS = 4, GDN_MAXR = 16;
constant constexpr int GDN_C = 2 * GDN_NK * GDN_DK + GDN_NV * GDN_DV, GDN_PW = GDN_C + GDN_NV * GDN_DV + 2 * GDN_NV;
// a window's replay record a layer: k and v [rows][heads][128] (bf16), then [rows][heads] k norm, gate, beta, pad
constant constexpr int GDN_RV = GDN_MAXR * GDN_NV * GDN_DK, GDN_RS = 2 * GDN_RV;

[[max_total_threads_per_threadgroup(1024)]]
[[kernel]] void fz_gdn(const device bfloat* P [[buffer(0)]], const device bfloat* CS [[buffer(1)]],
#if FZ_GDN_REPLAY
    device float* S [[buffer(2)]],
#else
    const device float* S [[buffer(2)]],
#endif
    const device bfloat* CW [[buffer(3)]], const device bfloat* ALOG [[buffer(4)]],
    const device bfloat* DT [[buffer(5)]], const device bfloat* NW [[buffer(6)]], const constant float* eps [[buffer(7)]],
    const constant int* rows [[buffer(8)]], device bfloat* OUT [[buffer(9)]], device bfloat* CSO [[buffer(10)]],
#if FZ_GDN_REPLAY
    const device bfloat* PREV [[buffer(11)]], constant uint& H0 [[buffer(12)]], device bfloat* CUR [[buffer(13)]],
    const device int* AR [[buffer(14)]],
#else
    device float* SO [[buffer(11)]], constant uint& H0 [[buffer(12)]],
#endif
    uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint tg [[threadgroup_position_in_grid]]) {
  constexpr int NK = GDN_NK, NV = GDN_NV, DK = GDN_DK, DV = GDN_DV, TAPS = GDN_TAPS, C = GDN_C, PW = GDN_PW;
  constexpr int RPS = DV / 32;                                // state rows a simdgroup
  const int hv = int(tg) + int(H0), hk = hv / (NV / NK), R = rows[0];
  threadgroup bfloat xq[GDN_MAXR][DK], xk[GDN_MAXR][DK], xv[GDN_MAXR][DV], ys[GDN_MAXR][DV];
  threadgroup float inv[GDN_MAXR][2], gates[GDN_MAXR][2], red[GDN_MAXR];
  float state[RPS][4];
  const device float4* S4 = (const device float4*)S;
  for (int j = 0; j < RPS; j++) {
    const float4 v = S4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane];
    state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
  }
#if FZ_GDN_REPLAY
  threadgroup bfloat pk[GDN_MAXR][DK], pv[GDN_MAXR][DV];
  threadgroup float pg[GDN_MAXR][3];
  const int KEEP = AR[0];
  for (int i = int(t); i < KEEP * DK; i += 1024) {
    const int r = i / DK, d = i % DK;
    pk[r][d] = PREV[(r * NV + hv) * DK + d];
    pv[r][d] = PREV[GDN_RV + (r * NV + hv) * DV + d];
  }
  if (int(t) < KEEP * 3) {
    const device float* ps = (const device float*)(PREV + GDN_RS);
    pg[int(t) / 3][int(t) % 3] = ps[((int(t) / 3) * NV + hv) * 4 + int(t) % 3];
  }
  device float* cs = (device float*)(CUR + GDN_RS);
#endif
  const bool writes_qk = (hv % (NV / NK)) == 0;
  if (int(t) >= 1024 - GDN_MAXR && int(t) - (1024 - GDN_MAXR) < R) {
    const int r = int(t) - (1024 - GDN_MAXR);
    const float b = float(P[r * PW + C + NV * DV + hv]);
    const float a = float(P[r * PW + C + NV * DV + NV + hv]);
    gates[r][0] = metal::exp(-metal::exp(float(ALOG[hv])) * fsoftplus(a + float(DT[hv])));
    gates[r][1] = bsig(b);
#if FZ_GDN_REPLAY
    cs[(r * NV + hv) * 4 + 1] = gates[r][0];
    cs[(r * NV + hv) * 4 + 2] = gates[r][1];
#endif
  }
  for (int i = int(t); i < R * (2 * DK + DV); i += 1024) { // (row, channel): q (hk), k (hk), v (hv)
    const int r = i / (2 * DK + DV), j = i % (2 * DK + DV);
    const int c = j < DK ? hk * DK + j : j < 2 * DK ? NK * DK + hk * DK + j - DK : 2 * NK * DK + hv * DV + j - 2 * DK;
    float conv = 0.0f;
    for (int tap = 0; tap < TAPS; tap++) {
      const int at = r + tap;
      const float x = at < TAPS - 1 ? float(CS[at * C + c]) : float(P[(at - (TAPS - 1)) * PW + c]);
      conv = fma(float(CW[c * TAPS + tap]), x, conv);
    }
    const bfloat act = bfloat(bsilu(conv));
    if (j < DK) xq[r][j] = act; else if (j < 2 * DK) xk[r][j - DK] = act; else xv[r][j - 2 * DK] = act;
#if FZ_GDN_REPLAY
    if (j >= 2 * DK) CUR[GDN_RV + (r * NV + hv) * DV + j - 2 * DK] = act;
    else if (j >= DK) CUR[(r * NV + hv) * DK + j - DK] = act;
#endif
    if (j < 2 * DK ? writes_qk : true)
      for (int q = 0; q < TAPS - 1; q++) {
        const int at = r + 1 + q;
        CSO[(r * (TAPS - 1) + q) * C + c] = at < TAPS - 1 ? CS[at * C + c] : P[(at - (TAPS - 1)) * PW + c];
      }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
#if FZ_GDN_REPLAY
  for (int r = 0; r < KEEP; r++) { // the previous window's kept rows: their state updates, outputs not needed
    const float g = pg[r][1], beta = pg[r][2], ik = pg[r][0];
    float kk[4], kv[RPS];
    for (int i = 0; i < 4; i++) kk[i] = float(pk[r][lane * 4 + i]) * ik;
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = simd_sum(kv[j]);
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(pv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      for (int i = 0; i < 4; i++) state[j][i] = state[j][i] + kk[i] * delta;
    }
  }
  device float4* SW4 = (device float4*)S;
  for (int j = 0; j < RPS; j++) SW4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane] = float4(state[j][0], state[j][1], state[j][2], state[j][3]);
#endif
  if (int(sg) < 2 * R) { // the q and k norms: x / sqrt(sum(x^2) + 1e-6) in fp32, q also times DK^-0.5
    const int r = int(sg) / 2;
    const bool isq = int(sg) % 2 == 0;
    threadgroup bfloat* x = isq ? xq[r] : xk[r];
    float ss = 0.0f;
    for (int i = 0; i < DK / 32; i++) {
      const float v = float(x[lane * (DK / 32) + i]);
      ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    if (lane == 0) {
      inv[r][isq ? 0 : 1] = metal::rsqrt(ss + 1e-6f) * (isq ? metal::rsqrt(float(DK)) : 1.0f);
#if FZ_GDN_REPLAY
      if (!isq) cs[(r * NV + hv) * 4] = inv[r][1];
#endif
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
#if !FZ_GDN_REPLAY
  device float4* SO4 = (device float4*)SO;
#endif
  for (int r = 0; r < R; r++) { // the four state rows' chains side by side: each element's arithmetic unchanged
    const float g = gates[r][0], beta = gates[r][1], iq = inv[r][0], ik = inv[r][1];
    float kk[4], qq[4], kv[RPS], out[RPS];
    for (int i = 0; i < 4; i++) { kk[i] = float(xk[r][lane * 4 + i]) * ik; qq[i] = float(xq[r][lane * 4 + i]) * iq; }
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = simd_sum(kv[j]);
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(xv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      out[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] + kk[i] * delta;
        out[j] += state[j][i] * qq[i];
      }
    }
    for (int j = 0; j < RPS; j++) out[j] = simd_sum(out[j]);
    if (lane == 0) for (int j = 0; j < RPS; j++) ys[r][int(sg) * RPS + j] = bfloat(out[j]);
#if !FZ_GDN_REPLAY
    for (int j = 0; j < RPS; j++) SO4[((size_t(r) * NV + hv) * DV + sg * RPS + j) * DK / 4 + lane] = float4(state[j][0], state[j][1], state[j][2], state[j][3]);
#endif
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(sg) < R) {
    float ss = 0.0f;
    for (int i = 0; i < DV / 32; i++) { const float v = float(ys[sg][lane * (DV / 32) + i]); ss = fma(v, v, ss); }
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = metal::rsqrt(ss / float(DV) + eps[0]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(t); i < R * DV; i += 1024) { // mx.fast.rms_norm's bf16(w * bf16(y * inv)), times sigmoid(z)
    const int r = i / DV, d = i % DV;
    const float y = float(bfloat(float(NW[d]) * float(bfloat(float(ys[r][d]) * red[r]))));
    const float z = float(P[r * PW + C + hv * DV + d]);
    OUT[r * NV * DV + hv * DV + d] = bfloat(y * fsig(z));
  }
}
