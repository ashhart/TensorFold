//! Unchanged embedded shader source used by replay compilation.
pub const text =
    \\inline void fz_codes6w(thread const uint* w, thread float* q) {
    \\  for (int m = 0; m < 4; m++) {
    \\    const int bit = 48 * m, word = bit >> 5, shift = bit & 31;
    \\    const ulong v = ((ulong(w[word + 1]) << 32) | ulong(w[word])) >> shift;
    \\    for (int i = 0; i < 8; i++) q[8 * m + i] = float(uint(v >> (6 * i)) & 63u);
    \\  }
    \\}
    \\[[max_total_threads_per_threadgroup(640)]]
    \\[[kernel]] void fz_hc_up(const device bfloat* HN [[buffer(0)]], const device float* SSP [[buffer(1)]],
    \\    const device float* NW [[buffer(2)]], const device float* PART [[buffer(3)]], const device uint* QW [[buffer(4)]],
    \\    const device bfloat* QS [[buffer(5)]], const device bfloat* QB [[buffer(6)]], const constant float* eps [[buffer(7)]],
    \\    const constant int* rows [[buffer(8)]], device bfloat* MIXED [[buffer(9)]], device bfloat* INJOUT [[buffer(10)]],
    \\    constant uint& NDR [[buffer(11)]], uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    \\    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
    \\  constexpr int S = 4, D = 2560, LOW = 320, BITS = 6, GS = 32, KS = 10, GL = LOW / 32, DW = 16, RT = 4;
    \\  constexpr int W = S * D, NO = S * DW, NT = GL * NO;
    \\  const int ND = int(NDR);
    \\  const int t = int(thread_position_in_threadgroup.x);
    \\  const int R = rows[0];
    \\  const int d0 = int(threadgroup_position_in_grid.x) * DW;
    \\  const int rb = int(threadgroup_position_in_grid.y) * RT;
    \\  const int nr = min(RT, R - rb);
    \\  threadgroup float act[RT][LOW];
    \\  threadgroup float vs[RT][GL];
    \\  threadgroup float ps[GL][NO][RT];
    \\  threadgroup float prod[S][DW][RT];
    \\  const int g = t / NO, n = t % NO;
    \\  const int o = (n / DW) * D + d0 + n % DW;
    \\  uint w6[6];
    \\  for (int i = 0; i < 6; i++) w6[i] = QW[size_t(o) * (LOW * BITS / 32) + g * BITS + i];
    \\  for (int i = t; i < nr * ND; i += NT) {
    \\    const int r = i / ND, cc = i % ND;
    \\    float v = 0.0f;
    \\    for (int k = 0; k < KS; k++) v += PART[(size_t(k) * R + rb + r) * ND + cc];
    \\    const float v4 = float(bfloat(float(bfloat(v)) / float(S)));
    \\    if (cc < LOW) act[r][cc] = bsilu(v4);
    \\    else if (threadgroup_position_in_grid.x == 0) INJOUT[(rb + r) * S + (cc - LOW)] = bfloat(2.0f * bsig(v4));
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (t < nr * GL) vs[t / GL][t % GL] = scalar_sum(&act[t / GL][32 * (t % GL)]);
    \\  {
    \\    float q[32];
    \\    fz_codes6w(w6, q);
    \\    float p[RT];
    \\    for (int r = 0; r < RT; r++) p[r] = 0.0f;
    \\    for (int st = 0; st < 4; st++)
    \\      for (int k = 0; k < 8; k++) {
    \\        const int i = 8 * (k / 2) + 2 * st + k % 2;
    \\        for (int r = 0; r < RT; r++) p[r] = fma(q[i], act[r][32 * g + i], p[r]);
    \\      }
    \\    for (int r = 0; r < RT; r++) ps[g][n][r] = p[r];
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (t < NO * nr) {
    \\    const int n2 = t % NO, r = t / NO, s = n2 / DW, o2 = s * D + d0 + n2 % DW;
    \\    float acc = 0.0f;
    \\    for (int gg = 0; gg < GL; gg++)
    \\      acc = fma(float(QB[size_t(o2) * (LOW / GS) + gg * 32 / GS]), vs[r][gg],
    \\                fma(float(QS[size_t(o2) * (LOW / GS) + gg * 32 / GS]), ps[gg][n2][r], acc));
    \\    const float normed = float(bfloat((float(HN[size_t(rb + r) * W + o2]) * stream_rinv(SSP, rb + r, s, D / 256, S, D, eps[0]))
    \\                                      * NW[o2]));
    \\    prod[s][n2 % DW][r] = float(bfloat(bsig(float(bfloat(acc))) * normed));
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (t < DW * nr) {
    \\    const int d = t % DW, r = t / DW;
    \\    float total = 0.0f;
    \\    for (int k = 0; k < S; k++) total += prod[k][d][r];
    \\    MIXED[size_t(rb + r) * D + d0 + d] = bfloat(total / float(S));
    \\  }
    \\}
    \\
;
