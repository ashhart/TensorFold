"""Metal source templates for row-exact Nemotron-H decode kernels."""

_ADD_NORM = r"""
  // one threadgroup of T threads per row; thread t owns elements t, t + T, t + 2T, ...
  const uint t = thread_position_in_threadgroup.x;
  const uint r = threadgroup_position_in_grid.x;
  constexpr int PER = D / T;
  threadgroup float partial[T / 32];
  float hv[PER];
  float ss = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    const int at = int(r) * D + c;
    float delta;
    MIX
    const bfloat hn = bfloat(float(H[at]) + delta);
    HN[at] = hn;
    hv[i] = float(hn);
    ss = fma(hv[i], hv[i], ss);
  }
  ss = simd_sum(ss);
  if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int s = 0; s < T / 32; s++) total += partial[s];
  const float scale = metal::rsqrt(total / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    OUT[int(r) * D + c] = bfloat(float(W[c]) * (hv[i] * scale));
  }
"""

# plain residual: the block's output
_MIX_PLAIN = "delta = float(X[at]);"

# MoE: sum_e w_e y_e (fp32, experts in order) rounded to bf16, plus the shared expert (bf16 add), as mlx_lm
_MIX_MOE = r"""{
      float routed = 0.0f;
      for (int e = 0; e < E; e++) routed = fma(float(Y[(int(r) * E + e) * D + c]), WE[int(r) * E + e], routed);
      delta = float(bfloat(float(bfloat(routed)) + float(SH[at])));
    }"""

_ROUTE = r"""
  // one simdgroup per row: lane l holds experts l, l + 32, l + 64, l + 96
  const uint lane = thread_index_in_simdgroup;
  const uint r = threadgroup_position_in_grid.x;
  float sel[NE / 32], prob[NE / 32];
  for (int j = 0; j < NE / 32; j++) {
    const int e = int(lane) + 32 * j;
    const float g = float(G[int(r) * NE + e]);
    prob[j] = 1.0f / (1.0f + metal::exp(-g));
    sel[j] = prob[j] + bias[e];
  }
  float total = 0.0f;
  float picked[K];
  for (int k = 0; k < K; k++) {
    float best = -INFINITY;
    int best_e = 1 << 20;
    for (int j = 0; j < NE / 32; j++) {
      const int e = int(lane) + 32 * j;
      if (sel[j] > best) { best = sel[j]; best_e = e; }
    }
    const float top = simd_max(best);
    const int winner = simd_min(best == top ? best_e : (1 << 20));   // ties: the lowest expert id
    float p = 0.0f;
    for (int j = 0; j < NE / 32; j++) {
      if (int(lane) + 32 * j == winner) { p = prob[j]; sel[j] = -INFINITY; }
    }
    p = simd_sum(p);
    picked[k] = p;
    total += p;
    if (lane == 0) IDX[int(r) * K + k] = uint(winner);
  }
  if (lane == 0) {
    const float denominator = total + 1e-20f;
    for (int k = 0; k < K; k++) WT[int(r) * K + k] = picked[k] / denominator * scaling[0];
  }
"""

_MAMBA_CONV = r"""
  // grid (CD, R): channel ch of row rr. Rows come in segments, one per stream (SEG: a row's segment, START: a
  // segment's first row); a segment's taps before its first row come from its conv state, row SLOT[s] of CS_IN.
  // Writes the conv output bf16(silu(bf16(conv))) as mlx_lm rounds it, and the row's conv state (its segment's
  // last KC-1 inputs).
  constexpr int CD = XD + 2 * NG * DS;
  const int ch = int(thread_position_in_grid.x);
  const int rr = int(thread_position_in_grid.y);
  const int s = SEG[rr];
  const int b = START[s];
  const int loc = rr - b;
  const int slot = SLOT[s];
  #define TAP(lp) ((lp) < 0 ? CS_IN[(slot * (KC - 1) + (lp) + KC - 1) * CD + ch] : P[(b + (lp)) * PROJ + XOFF + ch])
  float a = float(CB[ch]);
  for (int k = 0; k < KC; k++) a = fma(CW[k * CD + ch], float(TAP(loc - (KC - 1) + k)), a);
  const float cv = float(bfloat(a));
  XBC[rr * CD + ch] = bfloat(cv / (1.0f + metal::exp(-cv)));
  for (int k = 0; k < KC - 1; k++) CS_OUT[(rr * (KC - 1) + k) * CD + ch] = TAP(loc - (KC - 2) + k);
  #undef TAP
"""

_MAMBA_SCAN = r"""
  // grid (32, DH, H): lane = NS state elements of channel d of head h. Rows in order; segment s starts from its
  // stream's SSM state, row SLOT[s] of S_IN. A row's arithmetic depends only on its own inputs and the state
  // before it.
  const uint lane = thread_position_in_threadgroup.x;
  const uint d = thread_position_in_grid.y;
  const uint h = thread_position_in_grid.z;
  const uint g = h / (H / NG);
  const int R = dims[0];
  constexpr int NS = DS / 32;
  constexpr int CD = XD + 2 * NG * DS;
  const int cx = int(h) * DH + int(d);
  const int cb = XD + int(g) * DS + int(lane) * NS;
  const int cc = XD + NG * DS + int(g) * DS + int(lane) * NS;
  const int sbase = cx * DS + int(lane) * NS;
  const float A = -metal::exp(float(A_LOG[h]));
  const float dskip = float(bfloat(float(DSKIP[h])));
  const float dtb = float(DT_BIAS[h]);
  float st[NS];
  int cur = -1;
  for (int rr = 0; rr < R; rr++) {
    const int s = SEG[rr];
    if (s != cur) {
      cur = s;
      for (int i = 0; i < NS; i++) st[i] = float(S_IN[size_t(SLOT[s]) * SSZ + sbase + i]);
    }
    const float xv = float(XBC[rr * CD + cx]);
    float dt = float(P[rr * PROJ + DTOFF + int(h)]) + dtb;
    dt = metal::max(dt, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(dt)));   // softplus (logaddexp(x, 0))
    dt = metal::clamp(dt, limits[0], limits[1]);
    const float dA = metal::exp(A * dt);
    const float xdt = xv * dt;
    float acc = 0.0f;
    for (int i = 0; i < NS; i++) {
      const float sv = dA * st[i] + xdt * float(XBC[rr * CD + cb + i]);
      st[i] = sv;
      acc += sv * float(XBC[rr * CD + cc + i]);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
      const float y = float(bfloat(acc + xv * dskip));
      const float z = float(P[rr * PROJ + cx]);
      const float sz = float(bfloat(z / (1.0f + metal::exp(-z))));
      Y[rr * XD + cx] = bfloat(sz * y);
    }
    // the SSM state after this row (a verify window keeps the state of its last accepted row)
    for (int i = 0; i < NS; i++) S_OUT[size_t(rr) * SSZ + sbase + i] = st[i];
  }
"""

_GROUP_NORM = r"""
  // one threadgroup of GS / 4 threads per (row, group): thread t owns 4 consecutive elements
  const uint t = thread_position_in_threadgroup.x;
  const uint grp = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;
  constexpr int T = GS / 4;
  threadgroup float partial[T / 32];
  const int base = int(r) * XD + int(grp) * GS + int(t) * 4;
  float v[4];
  float ss = 0.0f;
  for (int i = 0; i < 4; i++) { v[i] = float(X[base + i]); ss = fma(v[i], v[i], ss); }
  ss = simd_sum(ss);
  if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int s = 0; s < T / 32; s++) total += partial[s];
  const float scale = metal::rsqrt(total / float(GS) + eps[0]);
  for (int i = 0; i < 4; i++) {
    const int c = int(grp) * GS + int(t) * 4 + i;
    OUT[base + i] = bfloat(float(W[c]) * float(bfloat(v[i] * scale)));
  }
"""

_ROUTER = r"""
  // bf16 router logits for R rows: one threadgroup of SG simdgroups per (expert, block of MAXR rows). Simdgroup g
  // sums its D / SG inputs (lane l: 4 consecutive inputs at a time, 128 apart), then simd_sum; the simdgroups'
  // sums are added in order. A row's logits have the same bits at any row count (the row count is a runtime value).
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int e = int(threadgroup_position_in_grid.y);
  const int rb = int(threadgroup_position_in_grid.z) * MAXR;
  const int R = min(rows[0] - rb, MAXR);
  constexpr int PART = D / SG;
  threadgroup float part[MAXR][SG];
  float acc[MAXR];
  for (int r = 0; r < MAXR; r++) acc[r] = 0.0f;
  const int begin = int(g) * PART;
  for (int c = begin + 4 * int(lane); c < begin + PART; c += 128) {
    const float w0 = float(GW[size_t(e) * D + c]), w1 = float(GW[size_t(e) * D + c + 1]);
    const float w2 = float(GW[size_t(e) * D + c + 2]), w3 = float(GW[size_t(e) * D + c + 3]);
    for (int r = 0; r < MAXR; r++) {
      if (r >= R) break;
      const device bfloat* xr = X + (rb + r) * D + c;
      acc[r] = fma(float(xr[3]), w3, fma(float(xr[2]), w2, fma(float(xr[1]), w1, fma(float(xr[0]), w0, acc[r]))));
    }
  }
  for (int r = 0; r < MAXR; r++) {
    if (r >= R) break;
    const float total = simd_sum(acc[r]);
    if (lane == 0) part[r][g] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (g == 0 && int(lane) < R) {
    float total = 0.0f;
    for (int k = 0; k < SG; k++) total += part[lane][k];
    OUT[(rb + int(lane)) * NE + e] = bfloat(total);
  }
"""
