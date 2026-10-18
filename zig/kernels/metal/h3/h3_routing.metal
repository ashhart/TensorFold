// MiniMax H3 / FastH3 on Metal: q, k and v in tile order, and FastH3's routing (tile means, scores, top-k, pooling).

// q, k, v to bf16 in tile order: head RMSNorm, rotate-half on q and k in bf16. P[4]: a row-head's stride (128 or 384).
[[kernel]] void h3_heads(
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
  device bfloat* TQ [[buffer(10)]],
  device bfloat* TK [[buffer(11)]],
  device bfloat* TV [[buffer(12)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int H = P[1], ROT = P[2], HALF = ROT / 2, STRIDE = P[4];
  const int row = int(tg.x), head = int(tg.y);
  const int c0 = 4 * int(lane);
  const long base = (long(row) * H + head) * STRIDE;
  float4 q, k;
  for (int j = 0; j < 4; j++) {
    q[j] = float(QR[base + c0 + j]);
    k[j] = float(KR[base + c0 + j]);
  }
  const float qi = rsqrt(simd_sum(dot(q, q)) / 128.0f + E[0]), ki = rsqrt(simd_sum(dot(k, k)) / 128.0f + E[0]);
  const device float* cs = COS + long(row) * ROT;
  const device float* sn = SIN + long(row) * ROT;
  const long at = (long(head) * H3_SLOTS + SLOT[row]) * 128;
  for (int j = 0; j < 4; j++) {
    const int c = c0 + j;
    float a = float(bfloat(q[j] * qi * float(NQ[c]))), b = float(bfloat(k[j] * ki * float(NK[c])));
    if (c < ROT) {
      // rotate-half: channel c pairs with c + HALF below the half and c - HALF above it; tables rounded to bf16
      const int p = c < HALF ? c + HALF : c - HALF;
      const float sign = c < HALF ? -1.0f : 1.0f;
      const float qp = float(bfloat(float(QR[base + p]) * qi * float(NQ[p])));
      const float kp = float(bfloat(float(KR[base + p]) * ki * float(NK[p])));
      const float co = float(bfloat(cs[c])), si = float(bfloat(sn[c]));
      a = float(bfloat(a * co)) + float(bfloat(sign * qp * si));
      b = float(bfloat(b * co)) + float(bfloat(sign * kp * si));
    }
    TQ[at + c] = bfloat(a);
    TK[at + c] = bfloat(b);
    TV[at + c] = VR[base + c];
  }
}

// Each tile's mean q, k and v over its real rows, in float. QP, KP, VP: (H, tiles, 128).
[[kernel]] void h3_pool(
  const device bfloat* TQ [[buffer(0)]],
  const device bfloat* TK [[buffer(1)]],
  const device bfloat* TV [[buffer(2)]],
  const device int32_t* SIZES [[buffer(3)]],
  device float* QP [[buffer(4)]],
  device float* KP [[buffer(5)]],
  device float* VP [[buffer(6)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]],
  uint3 grid [[threadgroups_per_grid]]) {
  const int tile = int(tg.x), head = int(tg.y), size = SIZES[tile], c0 = 4 * int(lane);
  const long first = long(head) * H3_SLOTS + long(tile) * 64;
  float4 q = 0.0f, k = 0.0f, v = 0.0f;
  for (int s = 0; s < size; s++) {
    const long at = (first + s) * 128 + c0;
    q += float4(float(TQ[at]), float(TQ[at + 1]), float(TQ[at + 2]), float(TQ[at + 3]));
    k += float4(float(TK[at]), float(TK[at + 1]), float(TK[at + 2]), float(TK[at + 3]));
    v += float4(float(TV[at]), float(TV[at + 1]), float(TV[at + 2]), float(TV[at + 3]));
  }
  const float n = float(size);
  const long out = (long(head) * grid.x + tile) * 128 + c0;
  for (int j = 0; j < 4; j++) {
    QP[out + j] = q[j] / n;
    KP[out + j] = k[j] / n;
    VP[out + j] = v[j] / n;
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
  // a tile is kept when fewer than KEEP score above it (a NaN ranks as -inf); its place counts the kept tiles before it
  threadgroup uchar kept[H3_TILES];
  for (int j = int(lane); j < NV; j += 32) {
    const float mine = isnan(s[j]) ? -INFINITY : s[j];
    int above = 0;
    for (int i = 0; i < NV; i++) {
      const float other = isnan(s[i]) ? -INFINITY : s[i];
      above += (other > mine || (other == mine && i < j)) ? 1 : 0;
    }
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
