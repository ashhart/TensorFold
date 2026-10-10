// MiniMax H3 / FastH3 on Metal: the row kernels around the attention (norms, head layout, routing, gate mix, heads).

// W (N, K) bf16 to int8 (K, N) with a scale per output channel. Threadgroups [N, 1, 1] of [32, 1, 1].
[[kernel]] void h3_quant_weight_t(
  const device bfloat* W [[buffer(0)]],
  const constant int32_t* P [[buffer(1)]],
  device int8_t* W8 [[buffer(2)]],
  device float* WS [[buffer(3)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int K = P[0], N = P[1], n = int(tg.x);
  const device bfloat* w = W + long(n) * K;
  float top = 0.0f;
  for (int k = int(lane); k < K; k += 32) top = max(top, abs(float(w[k])));
  top = max(simd_max(top), 1e-12f);
  if (lane == 0) WS[n] = top / 127.0f;
  const float inverse = 127.0f / top;
  for (int k = int(lane); k < K; k += 32) W8[long(k) * N + n] = int8_t(clamp(int(rint(float(w[k]) * inverse)), -127, 127));
}

// Y[row0 + r, n] = B[n] + sum over k of X[r, k] W[n, k]: the latent rows into the stream. Threads [N, R, 1].
[[kernel]] void h3_rows_in(
  const device float* X [[buffer(0)]],
  const device float* W [[buffer(1)]],
  const device float* B [[buffer(2)]],
  const constant int32_t* P [[buffer(3)]],
  device bfloat* Y [[buffer(4)]],
  uint3 at [[thread_position_in_grid]]) {
  const int R = P[0], K = P[1], N = P[2], row0 = P[3];
  if (int(at.x) >= N || int(at.y) >= R) return;
  const device float* x = X + long(at.y) * K;
  const device float* w = W + long(at.x) * K;
  float sum = 0.0f;
  for (int k = 0; k < K; k++) sum += x[k] * w[k];
  Y[long(row0 + int(at.y)) * N + at.x] = bfloat(sum + B[at.x]);
}

// The modulated norm one value at a time, rounded to bf16 where the reference's bf16 arithmetic rounds.
inline float h3_mod(float x, float inv, float w, float scale, float shift) {
  return float(bfloat(float(bfloat(float(bfloat(x * inv * w)) * float(bfloat(1.0f + scale)))) + shift));
}

// A block's modulated RMSNorm to int8 with a scale per row, after an optional gated add of Y into X.
[[kernel]] void h3_norm_q8(
  device bfloat* X [[buffer(0)]],
  const device bfloat* Y [[buffer(1)]],
  const device bfloat* W [[buffer(2)]],
  const device bfloat* TG [[buffer(3)]],
  const device bfloat* TAB [[buffer(4)]],
  const device int32_t* LINE [[buffer(5)]],
  const constant int32_t* P [[buffer(6)]],
  const constant float* E [[buffer(7)]],
  device int8_t* Q [[buffer(8)]],
  device float* XS [[buffer(9)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int M = P[0], C = P[1], L = P[2], row = int(tg.x);
  device int8_t* q = Q + long(row) * C;
  if (row >= M) {
    for (int c = int(lane); c < C; c += 32) q[c] = 0;
    if (lane == 0) XS[row] = 0.0f;
    return;
  }
  const int line = LINE[row];
  device bfloat* x = X + long(row) * C;
  float sq = 0.0f;
  if (P[3] >= 0) {
    const device bfloat* g = TG + (long(P[3]) * L + line) * C;
    const device bfloat* y = Y + long(row) * C;
    for (int c = int(lane); c < C; c += 32) {
      const bfloat v = bfloat(float(x[c]) + float(bfloat(float(g[c]) * float(y[c]))));
      x[c] = v;
      sq += float(v) * float(v);
    }
  } else {
    for (int c = int(lane); c < C; c += 32) sq += float(x[c]) * float(x[c]);
  }
  const float inv = rsqrt(simd_sum(sq) / float(C) + E[0]);
  const device bfloat* s = TAB + (long(P[4]) * L + line) * C;
  const device bfloat* h = TAB + (long(P[5]) * L + line) * C;
  float top = 0.0f;
  for (int c = int(lane); c < C; c += 32) top = max(top, abs(h3_mod(float(x[c]), inv, float(W[c]), float(s[c]), float(h[c]))));
  top = max(simd_max(top), 1e-12f);
  if (lane == 0) XS[row] = top / 127.0f;
  const float inverse = 127.0f / top;
  for (int c = int(lane); c < C; c += 32)
    q[c] = int8_t(clamp(int(rint(h3_mod(float(x[c]), inv, float(W[c]), float(s[c]), float(h[c])) * inverse)), -127, 127));
}

// X += TG[part, line] * Y, the last block's MLP branch. P: M, C, L, part. Threadgroups [M, 1, 1] of [32, 1, 1].
[[kernel]] void h3_gate_add(
  device bfloat* X [[buffer(0)]],
  const device bfloat* Y [[buffer(1)]],
  const device bfloat* TG [[buffer(2)]],
  const device int32_t* LINE [[buffer(3)]],
  const constant int32_t* P [[buffer(4)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int C = P[1], L = P[2], row = int(tg.x);
  const device bfloat* g = TG + (long(P[3]) * L + LINE[row]) * C;
  device bfloat* x = X + long(row) * C;
  const device bfloat* y = Y + long(row) * C;
  for (int c = int(lane); c < C; c += 32) x[c] = bfloat(float(x[c]) + float(bfloat(float(g[c]) * float(y[c]))));
}

// q, k, v to int8 in tile order: head RMSNorm and rotary on q and k. P[3]: 0 row scales, 1 scales only, 2 tile scales.
[[kernel]] void h3_heads_q8(
  const device bfloat* QR [[buffer(0)]],
  const device bfloat* KR [[buffer(1)]],
  const device bfloat* VR [[buffer(2)]],
  const device bfloat* NQ [[buffer(3)]],
  const device bfloat* NK [[buffer(4)]],
  const device float* COS [[buffer(5)]],
  const device float* SIN [[buffer(6)]],
  const device int32_t* SLOT [[buffer(7)]],
  const constant int32_t* P [[buffer(8)]],
  const constant float* E [[buffer(9)]],
  device int8_t* Q8 [[buffer(10)]],
  device float* QS [[buffer(11)]],
  device int8_t* K8 [[buffer(12)]],
  device float* KS [[buffer(13)]],
  device int8_t* V8 [[buffer(14)]],
  device float* VS [[buffer(15)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int H = P[1], ROT = P[2], HALF = ROT / 2;
  const int row = int(tg.x), head = int(tg.y);
  const int c0 = 4 * int(lane);
  const long base = (long(row) * H + head) * 128;
  float4 q, k, v;
  for (int j = 0; j < 4; j++) {
    q[j] = float(QR[base + c0 + j]);
    k[j] = float(KR[base + c0 + j]);
    v[j] = float(VR[base + c0 + j]);
  }
  const float qi = rsqrt(simd_sum(dot(q, q)) / 128.0f + E[0]), ki = rsqrt(simd_sum(dot(k, k)) / 128.0f + E[0]);
  const device float* cs = COS + long(row) * ROT;
  const device float* sn = SIN + long(row) * ROT;
  for (int j = 0; j < 4; j++) {
    const int c = c0 + j;
    float a = float(bfloat(q[j] * qi * float(NQ[c]))), b = float(bfloat(k[j] * ki * float(NK[c])));
    if (c < ROT) {
      // rotate-half: channel c pairs with c + HALF below the half and c - HALF above it
      const int p = c < HALF ? c + HALF : c - HALF;
      const float sign = c < HALF ? -1.0f : 1.0f;
      const float qp = float(bfloat(float(QR[base + p]) * qi * float(NQ[p])));
      const float kp = float(bfloat(float(KR[base + p]) * ki * float(NK[p])));
      a = a * cs[c] + sign * qp * sn[c];
      b = b * cs[c] + sign * kp * sn[c];
    }
    q[j] = a;
    k[j] = b;
  }
  const float4 aq = abs(q), ak = abs(k), av = abs(v);
  const float qtop = max(simd_max(max(max(aq.x, aq.y), max(aq.z, aq.w))), 1e-12f);
  const float ktop = max(simd_max(max(max(ak.x, ak.y), max(ak.z, ak.w))), 1e-12f);
  const float vtop = max(simd_max(max(max(av.x, av.y), max(av.z, av.w))), 1e-12f);
  const long at = long(head) * H3_SLOTS + SLOT[row];
  const int mode = P[3];
  if (lane == 0 && mode != 2) {
    KS[at] = ktop / 127.0f;
    VS[at] = vtop / 127.0f;
  }
  if (mode == 1) return;
  if (lane == 0) QS[at] = qtop / 127.0f;
  const float qv = 127.0f / qtop, kv = mode == 2 ? 1.0f / KS[at] : 127.0f / ktop, vv = mode == 2 ? 1.0f / VS[at] : 127.0f / vtop;
  for (int j = 0; j < 4; j++) {
    Q8[at * 128 + c0 + j] = int8_t(clamp(int(rint(q[j] * qv)), -127, 127));
    K8[at * 128 + c0 + j] = int8_t(clamp(int(rint(k[j] * kv)), -127, 127));
    V8[at * 128 + c0 + j] = int8_t(clamp(int(rint(v[j] * vv)), -127, 127));
  }
}

// One k scale and one v scale a (tile, head): the largest of its real rows', written to all 64 slots.
[[kernel]] void h3_tile_scales(
  device float* KS [[buffer(0)]],
  device float* VS [[buffer(1)]],
  const device int32_t* SIZES [[buffer(2)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int size = SIZES[tg.x];
  const long first = long(tg.y) * H3_SLOTS + long(tg.x) * 64;
  float k = 0.0f, v = 0.0f;
  for (int s = int(lane); s < size; s += 32) {
    k = max(k, KS[first + s]);
    v = max(v, VS[first + s]);
  }
  k = max(simd_max(k), 1e-12f / 127.0f);
  v = max(simd_max(v), 1e-12f / 127.0f);
  for (int s = int(lane); s < 64; s += 32) {
    KS[first + s] = k;
    VS[first + s] = v;
  }
}

// Each tile's mean q, k and v over its real rows. QP, KP, VP: (H, tiles, 128) float.
[[kernel]] void h3_pool(
  const device int8_t* Q8 [[buffer(0)]],
  const device float* QS [[buffer(1)]],
  const device int8_t* K8 [[buffer(2)]],
  const device float* KS [[buffer(3)]],
  const device int8_t* V8 [[buffer(4)]],
  const device float* VS [[buffer(5)]],
  const device int32_t* SIZES [[buffer(6)]],
  device float* QP [[buffer(7)]],
  device float* KP [[buffer(8)]],
  device float* VP [[buffer(9)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]],
  uint3 grid [[threadgroups_per_grid]]) {
  const int tile = int(tg.x), head = int(tg.y), size = SIZES[tile], c0 = 4 * int(lane);
  const long first = long(head) * H3_SLOTS + long(tile) * 64;
  float4 q = 0.0f, k = 0.0f, v = 0.0f;
  for (int s = 0; s < size; s++) {
    const long at = first + s;
    const device int8_t* q8 = Q8 + at * 128 + c0;
    const device int8_t* k8 = K8 + at * 128 + c0;
    const device int8_t* v8 = V8 + at * 128 + c0;
    q += float4(q8[0], q8[1], q8[2], q8[3]) * QS[at];
    k += float4(k8[0], k8[1], k8[2], k8[3]) * KS[at];
    v += float4(v8[0], v8[1], v8[2], v8[3]) * VS[at];
  }
  const float inv = 1.0f / float(size);
  const long out = (long(head) * grid.x + tile) * 128 + c0;
  for (int j = 0; j < 4; j++) {
    QP[out + j] = q[j] * inv;
    KP[out + j] = k[j] * inv;
    VP[out + j] = v[j] * inv;
  }
}

// Tile against tile: S[h, tq, tk] = QP[h, tq] . KP[h, tk] / sqrt(128). Threads [tiles, tiles, H].
[[kernel]] void h3_tile_scores(
  const device float* QP [[buffer(0)]],
  const device float* KP [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  device float* S [[buffer(3)]],
  uint3 at [[thread_position_in_grid]]) {
  const int NT = P[0], H = P[1];
  if (int(at.x) >= NT || int(at.y) >= NT || int(at.z) >= H) return;
  const device float* q = QP + (long(at.z) * NT + at.y) * 128;
  const device float* k = KP + (long(at.z) * NT + at.x) * 128;
  float sum = 0.0f;
  for (int c = 0; c < 128; c++) sum += q[c] * k[c];
  S[(long(at.z) * NT + at.y) * NT + at.x] = sum * 0.08838834764831845f;
}

// A video tile's key tiles: the prefix, then its KEEP best video tiles by score, in tile order.
[[kernel]] void h3_topk(
  const device float* S [[buffer(0)]],
  const constant int32_t* P [[buffer(1)]],
  device int32_t* IDX [[buffer(2)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int NT = P[0], P0 = P[1], KEEP = P[2], NV = NT - P0;
  const int tq = int(tg.x), head = int(tg.y);
  const device float* s = S + (long(head) * NT + P0 + tq) * NT + P0;
  device int32_t* out = IDX + (long(head) * NV + tq) * (P0 + KEEP);
  for (int i = int(lane); i < P0; i += 32) out[i] = i;
  // a tile is kept when fewer than KEEP score above it; its place counts the kept tiles before it
  threadgroup uchar kept[H3_TILES];
  for (int j = int(lane); j < NV; j += 32) {
    const float mine = s[j];
    int above = 0;
    for (int i = 0; i < NV; i++) above += (s[i] > mine || (s[i] == mine && i < j)) ? 1 : 0;
    kept[j] = above < KEEP ? 1 : 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int j = int(lane); j < NV; j += 32) {
    if (!kept[j]) continue;
    int before = 0;
    for (int i = 0; i < j; i++) before += kept[i];
    out[P0 + before] = P0 + j;
  }
}

// The pooled branch: C[h, tq] = softmax over tiles of S[h, tq] times VP[h]. C: (H, tiles, 128) float.
[[kernel]] void h3_coarse(
  const device float* S [[buffer(0)]],
  const device float* VP [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  device float* C [[buffer(3)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int NT = P[0], tq = int(tg.x), head = int(tg.y), c0 = 4 * int(lane);
  const device float* s = S + (long(head) * NT + tq) * NT;
  float top = -3.0e38f;
  for (int i = int(lane); i < NT; i += 32) top = max(top, s[i]);
  top = simd_max(top);
  float mass = 0.0f;
  for (int i = int(lane); i < NT; i += 32) mass += exp(s[i] - top);
  const float inv = 1.0f / simd_sum(mass);
  float4 acc = 0.0f;
  for (int i = 0; i < NT; i++) {
    const device float* v = VP + (long(head) * NT + i) * 128 + c0;
    acc += exp(s[i] - top) * float4(v[0], v[1], v[2], v[3]);
  }
  device float* out = C + (long(head) * NT + tq) * 128 + c0;
  for (int j = 0; j < 4; j++) out[j] = acc[j] * inv;
}

// The pooled branch gated into the attention output: Y[row, h] += C[h, tile of row] * G[row, h]. Threads [R, H, 1].
[[kernel]] void h3_gate_mix(
  device bfloat* Y [[buffer(0)]],
  const device bfloat* G [[buffer(1)]],
  const device float* C [[buffer(2)]],
  const device int32_t* SLOT [[buffer(3)]],
  const constant int32_t* P [[buffer(4)]],
  uint3 at [[thread_position_in_grid]]) {
  const int R = P[0], H = P[1], NT = P[2];
  if (int(at.x) >= R || int(at.y) >= H) return;
  const long base = (long(at.x) * H + at.y) * 128;
  const device float* c = C + (long(at.y) * NT + SLOT[at.x] / 64) * 128;
  for (int j = 0; j < 128; j++) Y[base + j] = bfloat(float(Y[base + j]) + float(bfloat(c[j] * float(G[base + j]))));
}

// h3_gate_mix and the rounding of its rows to int8 in one pass; the mixed row is never written.
[[kernel]] void h3_mix_quant(
  const device bfloat* Y [[buffer(0)]],
  const device bfloat* G [[buffer(1)]],
  const device float* C [[buffer(2)]],
  const device int32_t* SLOT [[buffer(3)]],
  const constant int32_t* P [[buffer(4)]],
  device int8_t* Q [[buffer(5)]],
  device float* XS [[buffer(6)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int R = P[0], W = P[1] * 128, NT = P[2], row = int(tg.x);
  const long base = long(row) * W;
  if (row >= R) {
    for (int c = int(lane); c < W; c += 32) Q[base + c] = 0;
    if (lane == 0) XS[row] = 0.0f;
    return;
  }
  const device float* coarse = C + long(SLOT[row] / 64) * 128;
  float top = 0.0f;
  for (int c = int(lane); c < W; c += 32) {
    const float mixed = float(bfloat(float(Y[base + c]) + float(bfloat(coarse[long(c >> 7) * NT * 128 + (c & 127)] * float(G[base + c])))));
    top = max(top, abs(mixed));
  }
  top = max(simd_max(top), 1e-12f);
  if (lane == 0) XS[row] = top / 127.0f;
  const float inverse = 127.0f / top;
  for (int c = int(lane); c < W; c += 32) {
    const float mixed = float(bfloat(float(Y[base + c]) + float(bfloat(coarse[long(c >> 7) * NT * 128 + (c & 127)] * float(G[base + c])))));
    Q[base + c] = int8_t(clamp(int(rint(mixed * inverse)), -127, 127));
  }
}

// h3_i8 products between the stream and the attention width, one activation scale a row.
[[kernel]] void h3_i8_in(
  const device int8_t* X [[buffer(0)]],
  const device float* XS [[buffer(1)]],
  const device int8_t* W [[buffer(2)]],
  const device float* WS [[buffer(3)]],
  device bfloat* Y [[buffer(4)]],
  uint tid [[thread_index_in_threadgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float sc[H3P_TM * H3P_GROUPS];
  h3p_i8_linear_body<H3_INNER, H3P_HIDDEN, H3P_HIDDEN>(X, XS, W, WS, Y, tid, tg, sc);
}

[[kernel]] void h3_i8_out(
  const device int8_t* X [[buffer(0)]],
  const device float* XS [[buffer(1)]],
  const device int8_t* W [[buffer(2)]],
  const device float* WS [[buffer(3)]],
  device bfloat* Y [[buffer(4)]],
  uint tid [[thread_index_in_threadgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float sc[H3P_TM * H3P_GROUPS];
  h3p_i8_linear_body<H3P_HIDDEN, H3_INNER, H3_INNER>(X, XS, W, WS, Y, tid, tg, sc);
}

// The final layer's modulated RMSNorm in float. TAB: (T, 2 C) bf16, shift then scale. Y: (M, C) float.
[[kernel]] void h3_final_norm(
  const device bfloat* X [[buffer(0)]],
  const device bfloat* W [[buffer(1)]],
  const device bfloat* TAB [[buffer(2)]],
  const device int32_t* LINE [[buffer(3)]],
  const constant int32_t* P [[buffer(4)]],
  const constant float* E [[buffer(5)]],
  device float* Y [[buffer(6)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int C = P[1], row = int(tg.x);
  const device bfloat* x = X + long(row) * C;
  const device bfloat* t = TAB + long(LINE[row]) * 2 * C;
  float sq = 0.0f;
  for (int c = int(lane); c < C; c += 32) sq += float(x[c]) * float(x[c]);
  const float inv = rsqrt(simd_sum(sq) / float(C) + E[0]);
  for (int c = int(lane); c < C; c += 32) Y[long(row) * C + c] = h3_mod(float(x[c]), inv, float(W[c]), float(t[C + c]), float(t[c]));
}

// Y[r, n] = B[n] + sum over k of X[row0 + r, k] W[n, k]: an output head over its rows, float throughout.
[[kernel]] void h3_rows_out(
  const device float* X [[buffer(0)]],
  const device float* W [[buffer(1)]],
  const device float* B [[buffer(2)]],
  const constant int32_t* P [[buffer(3)]],
  device float* Y [[buffer(4)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int K = P[1], N = P[2], row0 = P[3];
  const device float* x = X + long(row0 + int(tg.y)) * K;
  const device float* w = W + long(tg.x) * K;
  float sum = 0.0f;
  for (int k = int(lane); k < K; k += 32) sum += x[k] * w[k];
  sum = simd_sum(sum);
  if (lane == 0) Y[long(tg.y) * N + tg.x] = sum + B[tg.x];
}

// h3p_i8_swiglu on tiles of H3_SWT rows: two accumulators of a 128-row tile do not fit the registers.
#ifndef H3_SWT
#define H3_SWT 64
#endif
[[kernel]] void h3_i8_swiglu(
  const device int8_t* X [[buffer(0)]],
  const device float* XS [[buffer(1)]],
  const device int8_t* WG [[buffer(2)]],
  const device float* SG [[buffer(3)]],
  const device int8_t* WV [[buffer(4)]],
  const device float* SV [[buffer(5)]],
  device bfloat* H [[buffer(6)]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  using namespace mpp::tensor_ops;
  constexpr int T = H3_SWT, TN = H3P_TN, TK = H3P_TK;
  constexpr int CAP = T * TN / H3P_THREADS;
  constexpr int M = H3P_ROWS, N = H3P_MLP, K = H3P_HIDDEN;
  constexpr int MP = (M + H3P_T - 1) / H3P_T * H3P_T;
  const int n0 = int(tg.x) * TN, r0 = int(tg.y) * T;
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> x((device int8_t*)X, dextents<int32_t, 2>(K, MP));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> wg((device int8_t*)WG, dextents<int32_t, 2>(N, K));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> wv((device int8_t*)WV, dextents<int32_t, 2>(N, K));
  constexpr auto desc = matmul2d_descriptor(T, TN, TK, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<H3P_SG>> op;
  auto a0 = x.slice<TK, T>(0, r0);
  auto b0 = wg.slice<TN, TK>(n0, 0);
  auto gate = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
  auto value = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
  H3P_UNROLL
  for (ushort i = 0; i < CAP; i++) { gate[i] = 0; value[i] = 0; }
  for (int k0 = 0; k0 < K; k0 += TK) {
    auto a = x.slice<TK, T>(k0, r0);
    auto bg = wg.slice<TN, TK>(n0, k0);
    auto bv = wv.slice<TN, TK>(n0, k0);
    op.run(a, bg, gate);
    op.run(a, bv, value);
  }
  H3P_UNROLL
  for (ushort i = 0; i < CAP; i++) {
    auto ids = gate.get_multidimensional_index(i);
    const int row = r0 + ids[1], n = n0 + ids[0];
    if (row >= M) continue;
    const float xs = XS[row];
    const float g = float(gate[i]) * xs * SG[n];
    const float v = float(value[i]) * xs * SV[n];
    H[long(row) * N + n] = bfloat(g / (1.0f + exp(-g)) * v);
  }
}
