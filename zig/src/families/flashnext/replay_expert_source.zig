//! Unchanged embedded shader source used by replay compilation.
pub const text =
    \\inline void fz_codes6(const device uint* w, thread float* q) {
    \\  const uint2 a = *(const device uint2*)(w), b = *(const device uint2*)(w + 2), c = *(const device uint2*)(w + 4);
    \\  const ulong lo = ulong(a.x) | (ulong(a.y) << 32), mid = ulong(b.x) | (ulong(b.y) << 32), hi = ulong(c.x) | (ulong(c.y) << 32);
    \\  #pragma unroll
    \\  for (int i = 0; i < 32; i++) {
    \\    const int bit = 6 * i;
    \\    uint v;
    \\    if (bit + 6 <= 64) v = uint(lo >> bit) & 63u;
    \\    else if (bit < 64) v = (uint(lo >> bit) | uint(mid << (64 - bit))) & 63u;
    \\    else if (bit + 6 <= 128) v = uint(mid >> (bit - 64)) & 63u;
    \\    else if (bit < 128) v = (uint(mid >> (bit - 64)) | uint(hi << (128 - bit))) & 63u;
    \\    else v = uint(hi >> (bit - 128)) & 63u;
    \\    q[i] = float(v);
    \\  }
    \\}
    \\inline void fz_group(const device uint* w, const device bfloat* x, thread float& qx, thread float& sx) {
    \\  float q[32];
    \\  fz_codes6(w, q);
    \\  const device bfloat4* x4 = (const device bfloat4*)x;
    \\  qx = 0.0f; sx = 0.0f;
    \\  #pragma unroll
    \\  for (int i = 0; i < 8; i++) {
    \\    const float4 v = float4(x4[i]);
    \\    qx += q[4 * i] * v.x + q[4 * i + 1] * v.y + q[4 * i + 2] * v.z + q[4 * i + 3] * v.w;
    \\    sx += v.x + v.y + v.z + v.w;
    \\  }
    \\}
    \\[[kernel]] void fz_xgu(const device bfloat* X [[buffer(0)]], const device float* LOGITS [[buffer(1)]],
    \\    const device uint* GW [[buffer(2)]], const device bfloat* GS [[buffer(3)]], const device bfloat* GB [[buffer(4)]],
    \\    const device uint* UW [[buffer(5)]], const device bfloat* US [[buffer(6)]], const device bfloat* UB [[buffer(7)]],
    \\    const device uint* SGW [[buffer(8)]], const device bfloat* SGS [[buffer(9)]], const device bfloat* SGB [[buffer(10)]],
    \\    const device uint* SUW [[buffer(11)]], const device bfloat* SUS [[buffer(12)]], const device bfloat* SUB [[buffer(13)]],
    \\    device bfloat* ACT [[buffer(14)]], device uint* PICK [[buffer(15)]], device float* WTS [[buffer(16)]],
    \\    constant uint4& OWN [[buffer(17)]],
    \\    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint3 tg [[threadgroup_position_in_grid]]) {
    \\  constexpr int K = 2560, N = 640, TOPK = 10, NE = 512, NL = 513, SLOTS = TOPK + 1, WPR = K * 6 / 32, KG = K / 32;
    \\  const int p = int(tg.z) + int(OWN.x) * SLOTS; // TP: this rank's rows [OWN.x, OWN.y) only
    \\  const int r = p / SLOTS, slot = p % SLOTS;
    \\  if (uint(r) >= OWN.y) return;
    \\  const bool shared = slot == TOPK;
    \\  float picked[TOPK];
    \\  const size_t e = shared ? 0 : size_t(simd_topk<NE>(LOGITS + r * NL, slot, lane, picked));
    \\  if (!shared && tg.y == 0 && sgi == 0 && lane == 0) {
    \\    PICK[r * TOPK + slot] = uint32_t(e);
    \\    if (slot == TOPK - 1) {
    \\      float total = 0.0f;
    \\      float ex[TOPK];
    \\      for (int kk = 0; kk < TOPK; kk++) { ex[kk] = metal::exp(picked[kk] - picked[0]); total += ex[kk]; }
    \\      for (int kk = 0; kk < TOPK; kk++) WTS[r * TOPK + kk] = float(bfloat(ex[kk] / total));
    \\    }
    \\  }
    \\  const int row = int(tg.y) * 8 + int(sgi) * 2 + int(lane >> 4);
    \\  const int part = int(lane & 15);
    \\  constexpr int GJ = FZ_PACKED ? 96 : 6, SJ = FZ_PACKED ? 16 : 1; // a lane's group stride: words, scales
    \\  const size_t wrow = (shared ? 0 : e * N * WPR) + size_t(row) * WPR + part * (FZ_PACKED ? 6 : 30);
    \\  const size_t grow = (shared ? 0 : e * N * KG) + size_t(row) * KG + part * (FZ_PACKED ? 1 : 5);
    \\  const device uint* gw = (shared ? SGW : GW) + wrow;
    \\  const device uint* uw = (shared ? SUW : UW) + wrow;
    \\  const device bfloat* gs = (shared ? SGS : GS) + grow;
    \\  const device bfloat* gb = (shared ? SGB : GB) + grow;
    \\  const device bfloat* us = (shared ? SUS : US) + grow;
    \\  const device bfloat* ub = (shared ? SUB : UB) + grow;
    \\  const device bfloat* x = X + r * K + part * 160;
    \\  float ag = 0.0f, au = 0.0f;
    \\  for (int j = 0; j < 5; j++) {
    \\    float qg, qu, sx, sx2;
    \\    fz_group(gw + j * GJ, x + j * 32, qg, sx);
    \\    fz_group(uw + j * GJ, x + j * 32, qu, sx2);
    \\    ag += float(gs[j * SJ]) * qg + float(gb[j * SJ]) * sx;
    \\    au += float(us[j * SJ]) * qu + float(ub[j * SJ]) * sx;
    \\  }
    \\  for (ushort o = 8; o > 0; o >>= 1) { ag += simd_shuffle_xor(ag, o); au += simd_shuffle_xor(au, o); }
    \\  if (part == 0) ACT[p * N + row] = bfloat(bsilu(float(bfloat(ag))) * float(bfloat(au)));
    \\}
    \\[[kernel]] void fz_xdown(const device bfloat* ACT [[buffer(0)]], const device uint* PICK [[buffer(1)]],
    \\    const device uint* DW [[buffer(2)]], const device bfloat* DS [[buffer(3)]], const device bfloat* DB [[buffer(4)]],
    \\    const device uint* SDW [[buffer(5)]], const device bfloat* SDS [[buffer(6)]], const device bfloat* SDB [[buffer(7)]],
    \\    const constant int* rows [[buffer(8)]], device bfloat* Y [[buffer(9)]], constant uint4& OWN [[buffer(10)]],
    \\    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint3 tg [[threadgroup_position_in_grid]]) {
    \\  constexpr int NI = 640, D = 2560, TOPK = 10, SLOTS = TOPK + 1, WPR = NI * 6 / 32, KG = NI / 32;
    \\  const int pair = int(tg.z) + int(OWN.x) * SLOTS; // TP: this rank's rows [OWN.x, OWN.y) only
    \\  if (pair >= rows[0] * SLOTS || pair >= int(OWN.y) * SLOTS) return;
    \\  const int r = pair / SLOTS, k = pair % SLOTS;
    \\  const bool shared = k == TOPK;
    \\  const size_t e = shared ? 0 : size_t(PICK[r * TOPK + k]);
    \\  const int d = int(tg.y) * 32 + int(sgi) * 8 + int(lane >> 2);
    \\  const int part = int(lane & 3);
    \\  constexpr int GJ = FZ_PACKED ? 24 : 6, SJ = FZ_PACKED ? 4 : 1;
    \\  const device uint* w = (shared ? SDW : DW + e * D * WPR) + size_t(d) * WPR + part * (FZ_PACKED ? 6 : 30);
    \\  const size_t g0 = (shared ? 0 : e * D * KG) + size_t(d) * KG + part * (FZ_PACKED ? 1 : 5);
    \\  const device bfloat* sc = (shared ? SDS : DS) + g0;
    \\  const device bfloat* bi = (shared ? SDB : DB) + g0;
    \\  const device bfloat* x = ACT + (r * SLOTS + k) * NI + part * 160;
    \\  float acc = 0.0f;
    \\  for (int j = 0; j < 5; j++) {
    \\    float qx, sx;
    \\    fz_group(w + j * GJ, x + j * 32, qx, sx);
    \\    acc += float(sc[j * SJ]) * qx + float(bi[j * SJ]) * sx;
    \\  }
    \\  acc += simd_shuffle_xor(acc, 1);
    \\  acc += simd_shuffle_xor(acc, 2);
    \\  if (part == 0) Y[(r * SLOTS + k) * D + d] = bfloat(acc);
    \\}
    \\inline void fz_dot32(thread const float* q, const device bfloat* x, thread float& qx, thread float& sx) {
    \\  const device bfloat4* x4 = (const device bfloat4*)x;
    \\  qx = 0.0f; sx = 0.0f;
    \\  #pragma unroll
    \\  for (int i = 0; i < 8; i++) {
    \\    const float4 v = float4(x4[i]);
    \\    qx += q[4 * i] * v.x + q[4 * i + 1] * v.y + q[4 * i + 2] * v.z + q[4 * i + 3] * v.w;
    \\    sx += v.x + v.y + v.z + v.w;
    \\  }
    \\}
    \\// A 6-bit dense projection of up to 8 rows from lane_qmm's tiled weights [N/32][K/32][32 cols][6 words] and its
    \\// group-major scale/bias pairs: lane = output column, SK simdgroups split the groups, summed in order.
    \\template <int SK>
    \\inline void fz_dense_body(const device bfloat* X, const device uint* W, const device bfloat* SB, device bfloat* Y,
    \\    constant uint4& dims, uint sgi, uint lane, uint tgi, threadgroup float (*part)[8][32]) {
    \\  const int R = int(dims.x), N = int(dims.y), K = int(dims.z), KG = K / 32;
    \\  const int t = int(tgi) + int(dims.w), n = t * 32 + int(lane); // dims.w: the first output tile
    \\  const int g0 = int(sgi) * (KG / SK), g1 = g0 + KG / SK;
    \\  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    \\  for (int g = g0; g < g1; g++) {
    \\    float q[32];
    \\    fz_codes6(W + (size_t(t * KG + g) * 32 + lane) * 6, q);
    \\    const device bfloat* sb = SB + (size_t(g) * N + n) * 2;
    \\    const float sc = float(sb[0]), bi = float(sb[1]);
    \\    #pragma unroll
    \\    for (int r = 0; r < 8; r++) {
    \\      if (r < R) {
    \\        float qx, sx;
    \\        fz_dot32(q, X + size_t(r) * K + g * 32, qx, sx);
    \\        acc[r] += sc * qx + bi * sx;
    \\      }
    \\    }
    \\  }
    \\  #pragma unroll
    \\  for (int r = 0; r < 8; r++) part[sgi][r][lane] = acc[r];
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (sgi == 0) {
    \\    for (int r = 0; r < R; r++) {
    \\      float v = 0.0f;
    \\      for (int k = 0; k < SK; k++) v += part[k][r][lane];
    \\      Y[size_t(r) * N + n] = bfloat(v);
    \\    }
    \\  }
    \\}
    \\[[kernel]] void fz_dense8(const device bfloat* X [[buffer(0)]], const device uint* W [[buffer(1)]],
    \\    const device bfloat* SB [[buffer(2)]], device bfloat* Y [[buffer(3)]], constant uint4& dims [[buffer(4)]],
    \\    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint tgi [[threadgroup_position_in_grid]]) {
    \\  threadgroup float part[8][8][32];
    \\  fz_dense_body<8>(X, W, SB, Y, dims, sgi, lane, tgi, part);
    \\}
    \\[[kernel]] void fz_dense16(const device bfloat* X [[buffer(0)]], const device uint* W [[buffer(1)]],
    \\    const device bfloat* SB [[buffer(2)]], device bfloat* Y [[buffer(3)]], constant uint4& dims [[buffer(4)]],
    \\    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint tgi [[threadgroup_position_in_grid]]) {
    \\  threadgroup float part[16][8][32];
    \\  fz_dense_body<16>(X, W, SB, Y, dims, sgi, lane, tgi, part);
    \\}
    \\// FZ_XSX=1: fz_xgu and fz_xdown with each input group's sum computed once a threadgroup (fz_group's order) and the
    \\// inputs converted once for gate and up; simdgroup 0 routes while the rest sum. Bits equal fz_xgu / fz_xdown.
    \\inline float fz_qdot(thread const float* q, thread const float4* xv) {
    \\  float qx = 0.0f;
    \\  #pragma unroll
    \\  for (int i = 0; i < 8; i++) qx += q[4 * i] * xv[i].x + q[4 * i + 1] * xv[i].y + q[4 * i + 2] * xv[i].z + q[4 * i + 3] * xv[i].w;
    \\  return qx;
    \\}
    \\inline float fz_xsum32(const device bfloat* x) {
    \\  const device bfloat4* x4 = (const device bfloat4*)x;
    \\  float sx = 0.0f;
    \\  #pragma unroll
    \\  for (int i = 0; i < 8; i++) { const float4 v = float4(x4[i]); sx += v.x + v.y + v.z + v.w; }
    \\  return sx;
    \\}
    \\[[kernel]] void fz_xgu_sx(const device bfloat* X [[buffer(0)]], const device float* LOGITS [[buffer(1)]],
    \\    const device uint* GW [[buffer(2)]], const device bfloat* GS [[buffer(3)]], const device bfloat* GB [[buffer(4)]],
    \\    const device uint* UW [[buffer(5)]], const device bfloat* US [[buffer(6)]], const device bfloat* UB [[buffer(7)]],
    \\    const device uint* SGW [[buffer(8)]], const device bfloat* SGS [[buffer(9)]], const device bfloat* SGB [[buffer(10)]],
    \\    const device uint* SUW [[buffer(11)]], const device bfloat* SUS [[buffer(12)]], const device bfloat* SUB [[buffer(13)]],
    \\    device bfloat* ACT [[buffer(14)]], device uint* PICK [[buffer(15)]], device float* WTS [[buffer(16)]],
    \\    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint3 tg [[threadgroup_position_in_grid]]) {
    \\  constexpr int K = 2560, N = 640, TOPK = 10, NE = 512, NL = 513, SLOTS = TOPK + 1, WPR = K * 6 / 32, KG = K / 32;
    \\  const int p = int(tg.z);
    \\  const int r = p / SLOTS, slot = p % SLOTS;
    \\  const bool shared = slot == TOPK;
    \\  threadgroup float sxs[KG];
    \\  threadgroup uint pick_e;
    \\  if (sgi == 0) {
    \\    float picked[TOPK];
    \\    const uint e0 = shared ? 0u : uint(simd_topk<NE>(LOGITS + r * NL, slot, lane, picked));
    \\    if (lane == 0) {
    \\      pick_e = e0;
    \\      if (!shared && tg.y == 0) {
    \\        PICK[r * TOPK + slot] = e0;
    \\        if (slot == TOPK - 1) {
    \\          float total = 0.0f;
    \\          float ex[TOPK];
    \\          for (int kk = 0; kk < TOPK; kk++) { ex[kk] = metal::exp(picked[kk] - picked[0]); total += ex[kk]; }
    \\          for (int kk = 0; kk < TOPK; kk++) WTS[r * TOPK + kk] = float(bfloat(ex[kk] / total));
    \\        }
    \\      }
    \\    }
    \\  } else {
    \\    const int g = int(sgi - 1) * 32 + int(lane);
    \\    if (g < KG) sxs[g] = fz_xsum32(X + r * K + g * 32);
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  const size_t e = size_t(pick_e);
    \\  const int row = int(tg.y) * 8 + int(sgi) * 2 + int(lane >> 4);
    \\  const int part = int(lane & 15);
    \\  const size_t wrow = (shared ? 0 : e * N * WPR) + size_t(row) * WPR + part * 30;
    \\  const size_t grow = (shared ? 0 : e * N * KG) + size_t(row) * KG + part * 5;
    \\  const device uint* gw = (shared ? SGW : GW) + wrow;
    \\  const device uint* uw = (shared ? SUW : UW) + wrow;
    \\  const device bfloat* gs = (shared ? SGS : GS) + grow;
    \\  const device bfloat* gb = (shared ? SGB : GB) + grow;
    \\  const device bfloat* us = (shared ? SUS : US) + grow;
    \\  const device bfloat* ub = (shared ? SUB : UB) + grow;
    \\  const device bfloat4* x4 = (const device bfloat4*)(X + r * K + part * 160);
    \\  float ag = 0.0f, au = 0.0f;
    \\  for (int j = 0; j < 5; j++) {
    \\    float4 xv[8];
    \\    #pragma unroll
    \\    for (int i = 0; i < 8; i++) xv[i] = float4(x4[j * 8 + i]);
    \\    float q[32];
    \\    fz_codes6(gw + j * 6, q);
    \\    const float qg = fz_qdot(q, xv);
    \\    fz_codes6(uw + j * 6, q);
    \\    const float qu = fz_qdot(q, xv);
    \\    const float sx = sxs[part * 5 + j];
    \\    ag += float(gs[j]) * qg + float(gb[j]) * sx;
    \\    au += float(us[j]) * qu + float(ub[j]) * sx;
    \\  }
    \\  for (ushort o = 8; o > 0; o >>= 1) { ag += simd_shuffle_xor(ag, o); au += simd_shuffle_xor(au, o); }
    \\  if (part == 0) ACT[p * N + row] = bfloat(bsilu(float(bfloat(ag))) * float(bfloat(au)));
    \\}
    \\[[kernel]] void fz_xdown_sx(const device bfloat* ACT [[buffer(0)]], const device uint* PICK [[buffer(1)]],
    \\    const device uint* DW [[buffer(2)]], const device bfloat* DS [[buffer(3)]], const device bfloat* DB [[buffer(4)]],
    \\    const device uint* SDW [[buffer(5)]], const device bfloat* SDS [[buffer(6)]], const device bfloat* SDB [[buffer(7)]],
    \\    const constant int* rows [[buffer(8)]], device bfloat* Y [[buffer(9)]],
    \\    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint3 tg [[threadgroup_position_in_grid]]) {
    \\  constexpr int NI = 640, D = 2560, TOPK = 10, SLOTS = TOPK + 1, WPR = NI * 6 / 32, KG = NI / 32;
    \\  const int pair = int(tg.z);
    \\  if (pair >= rows[0] * SLOTS) return;
    \\  threadgroup float sxs[KG];
    \\  if (sgi == 0 && lane < uint(KG)) sxs[lane] = fz_xsum32(ACT + pair * NI + int(lane) * 32);
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  const int r = pair / SLOTS, k = pair % SLOTS;
    \\  const bool shared = k == TOPK;
    \\  const size_t e = shared ? 0 : size_t(PICK[r * TOPK + k]);
    \\  const int d = int(tg.y) * 32 + int(sgi) * 8 + int(lane >> 2);
    \\  const int part = int(lane & 3);
    \\  const device uint* w = (shared ? SDW : DW + e * D * WPR) + size_t(d) * WPR + part * 30;
    \\  const size_t g0 = (shared ? 0 : e * D * KG) + size_t(d) * KG + part * 5;
    \\  const device bfloat* sc = (shared ? SDS : DS) + g0;
    \\  const device bfloat* bi = (shared ? SDB : DB) + g0;
    \\  const device bfloat4* x4 = (const device bfloat4*)(ACT + pair * NI + part * 160);
    \\  float acc = 0.0f;
    \\  for (int j = 0; j < 5; j++) {
    \\    float4 xv[8];
    \\    #pragma unroll
    \\    for (int i = 0; i < 8; i++) xv[i] = float4(x4[j * 8 + i]);
    \\    float q[32];
    \\    fz_codes6(w + j * 6, q);
    \\    acc += float(sc[j]) * fz_qdot(q, xv) + float(bi[j]) * sxs[part * 5 + j];
    \\  }
    \\  acc += simd_shuffle_xor(acc, 1);
    \\  acc += simd_shuffle_xor(acc, 2);
    \\  if (part == 0) Y[pair * D + d] = bfloat(acc);
    \\}
    \\// Experts in one launch (FZ_XFUSED=1): threadgroup (slice s, pair) runs fz_xgu's sums for intermediate rows
    \\// [160 s, 160 s + 160) and fz_xdown's lane-s sums over those rows for every output; the pair's last slice adds
    \\// the four parts in fz_xdown's shuffle order, so every bit equals fz_xgu then fz_xdown.
    \\inline void fz_dot32t(thread const float* q, threadgroup const float* a, thread float& qx, thread float& sx) {
    \\  qx = 0.0f; sx = 0.0f;
    \\  #pragma unroll
    \\  for (int i = 0; i < 8; i++) {
    \\    const float4 v = float4(a[4 * i], a[4 * i + 1], a[4 * i + 2], a[4 * i + 3]);
    \\    qx += q[4 * i] * v.x + q[4 * i + 1] * v.y + q[4 * i + 2] * v.z + q[4 * i + 3] * v.w;
    \\    sx += v.x + v.y + v.z + v.w;
    \\  }
    \\}
    \\[[kernel]] void fz_xfused(const device bfloat* X [[buffer(0)]], const device float* LOGITS [[buffer(1)]],
    \\    const device uint* GW [[buffer(2)]], const device bfloat* GS [[buffer(3)]], const device bfloat* GB [[buffer(4)]],
    \\    const device uint* UW [[buffer(5)]], const device bfloat* US [[buffer(6)]], const device bfloat* UB [[buffer(7)]],
    \\    const device uint* SGW [[buffer(8)]], const device bfloat* SGS [[buffer(9)]], const device bfloat* SGB [[buffer(10)]],
    \\    const device uint* SUW [[buffer(11)]], const device bfloat* SUS [[buffer(12)]], const device bfloat* SUB [[buffer(13)]],
    \\    const device uint* DW [[buffer(14)]], const device bfloat* DS [[buffer(15)]], const device bfloat* DB [[buffer(16)]],
    \\    const device uint* SDW [[buffer(17)]], const device bfloat* SDS [[buffer(18)]], const device bfloat* SDB [[buffer(19)]],
    \\    device uint* PICK [[buffer(20)]], device float* WTS [[buffer(21)]], device bfloat* Y [[buffer(22)]],
    \\    coherent(device) device float* PART [[buffer(23)]], device atomic_uint* DONE [[buffer(24)]],
    \\    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint tid [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
    \\  constexpr int K = 2560, N = 640, D = 2560, TOPK = 10, NE = 512, NL = 513, SLOTS = TOPK + 1, TGS = 512;
    \\  constexpr int WPR = K * 6 / 32, KG = K / 32, DWPR = N * 6 / 32, DKG = N / 32;
    \\  const int s = int(tg.x), p = int(tg.y);
    \\  const int r = p / SLOTS, slot = p % SLOTS;
    \\  const bool shared = slot == TOPK;
    \\  threadgroup float act[160];
    \\  threadgroup uint last;
    \\  float picked[TOPK];
    \\  const size_t e = shared ? 0 : size_t(simd_topk<NE>(LOGITS + r * NL, slot, lane, picked));
    \\  if (!shared && s == 0 && sgi == 0 && lane == 0) {
    \\    PICK[r * TOPK + slot] = uint32_t(e);
    \\    if (slot == TOPK - 1) {
    \\      float total = 0.0f;
    \\      float ex[TOPK];
    \\      for (int kk = 0; kk < TOPK; kk++) { ex[kk] = metal::exp(picked[kk] - picked[0]); total += ex[kk]; }
    \\      for (int kk = 0; kk < TOPK; kk++) WTS[r * TOPK + kk] = float(bfloat(ex[kk] / total));
    \\    }
    \\  }
    \\  const int part = int(lane & 15);
    \\  const device bfloat* x = X + r * K + part * 160;
    \\  for (int pass = 0; pass < 160 / (TGS / 16); pass++) {
    \\    const int lrow = pass * (TGS / 16) + int(sgi) * 2 + int(lane >> 4);
    \\    const int row = s * 160 + lrow;
    \\    const size_t wrow = (shared ? 0 : e * N * WPR) + size_t(row) * WPR + part * 30;
    \\    const size_t grow = (shared ? 0 : e * N * KG) + size_t(row) * KG + part * 5;
    \\    const device uint* gw = (shared ? SGW : GW) + wrow;
    \\    const device uint* uw = (shared ? SUW : UW) + wrow;
    \\    const device bfloat* gs = (shared ? SGS : GS) + grow;
    \\    const device bfloat* gb = (shared ? SGB : GB) + grow;
    \\    const device bfloat* us = (shared ? SUS : US) + grow;
    \\    const device bfloat* ub = (shared ? SUB : UB) + grow;
    \\    float ag = 0.0f, au = 0.0f;
    \\    for (int j = 0; j < 5; j++) {
    \\      float qg, qu, sx, sx2;
    \\      fz_group(gw + j * 6, x + j * 32, qg, sx);
    \\      fz_group(uw + j * 6, x + j * 32, qu, sx2);
    \\      ag += float(gs[j]) * qg + float(gb[j]) * sx;
    \\      au += float(us[j]) * qu + float(ub[j]) * sx;
    \\    }
    \\    for (ushort o = 8; o > 0; o >>= 1) { ag += simd_shuffle_xor(ag, o); au += simd_shuffle_xor(au, o); }
    \\    if (part == 0) act[lrow] = float(bfloat(bsilu(float(bfloat(ag))) * float(bfloat(au))));
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  for (int i = 0; i < D / TGS; i++) {
    \\    const int d = int(tid) + TGS * i;
    \\    const device uint* w = (shared ? SDW : DW + e * D * DWPR) + size_t(d) * DWPR + s * 30;
    \\    const size_t g0 = (shared ? 0 : e * D * DKG) + size_t(d) * DKG + s * 5;
    \\    const device bfloat* sc = (shared ? SDS : DS) + g0;
    \\    const device bfloat* bi = (shared ? SDB : DB) + g0;
    \\    float acc = 0.0f;
    \\    for (int j = 0; j < 5; j++) {
    \\      float q[32];
    \\      fz_codes6(w + j * 6, q);
    \\      float qx, sx;
    \\      fz_dot32t(q, act + j * 32, qx, sx);
    \\      acc += float(sc[j]) * qx + float(bi[j]) * sx;
    \\    }
    \\    PART[(size_t(p) * 4 + s) * D + d] = acc;
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_device);
    \\  if (tid == 0) {
    \\    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    \\    last = atomic_fetch_add_explicit(DONE + p, 1u, memory_order_relaxed);
    \\    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_device);
    \\  if (last != 3) return;
    \\  for (int i = 0; i < D / TGS; i++) {
    \\    const int d = int(tid) + TGS * i;
    \\    const size_t at = size_t(p) * 4 * D + d;
    \\    Y[p * D + d] = bfloat((PART[at] + PART[at + D]) + (PART[at + 2 * D] + PART[at + 3 * D]));
    \\  }
    \\  if (tid == 0) atomic_store_explicit(DONE + p, 0u, memory_order_relaxed);
    \\}
    \\// Grouped experts (FZ_GROUPED=1): fz_route picks every row's experts (the recorded top-k rounds) and lists each
    \\// distinct expert's (row, slot) pairs; fz_ggu and fz_gdown read each distinct expert once for all its pairs, and
    \\// every pair's sums run in fz_xgu's and fz_xdown's order, so each row's bits are the ungrouped kernels' bits.
    \\constant constexpr int FZ_UMAX = 320, FZ_MMAX = 32;
    \\[[kernel]] void fz_route(const device float* LOGITS [[buffer(0)]], const constant int* rows [[buffer(1)]],
    \\    device uint* PICK [[buffer(2)]], device float* WTS [[buffer(3)]], device int* UL [[buffer(4)]],
    \\    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint tid [[thread_index_in_threadgroup]]) {
    \\  constexpr int TOPK = 10, NE = 512, NL = 513;
    \\  threadgroup int tids[FZ_UMAX];
    \\  threadgroup int first[FZ_UMAX];
    \\  const int R = rows[0], n = R * TOPK, i = int(tid);
    \\  if (int(sgi) < R) {
    \\    int ids[TOPK];
    \\    float picked[TOPK];
    \\    simd_topk_all<NE, TOPK>(LOGITS + int(sgi) * NL, lane, ids, picked);
    \\    if (lane == 0) {
    \\      float total = 0.0f;
    \\      float ex[TOPK];
    \\      for (int kk = 0; kk < TOPK; kk++) { ex[kk] = metal::exp(picked[kk] - picked[0]); total += ex[kk]; }
    \\      for (int kk = 0; kk < TOPK; kk++) {
    \\        WTS[sgi * TOPK + kk] = float(bfloat(ex[kk] / total));
    \\        PICK[sgi * TOPK + kk] = uint(ids[kk]);
    \\        tids[sgi * TOPK + kk] = ids[kk];
    \\      }
    \\    }
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  int f = 0;
    \\  if (i < n) {
    \\    f = 1;
    \\    for (int j = 0; j < i; j++) if (tids[j] == tids[i]) { f = 0; break; }
    \\    first[i] = f;
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (i < n && f == 1) {
    \\    int u = 0;
    \\    for (int j = 0; j < i; j++) u += first[j];
    \\    const int e = tids[i];
    \\    UL[1 + u] = e;
    \\    int c = 0;
    \\    for (int j = i; j < n; j++) if (tids[j] == e) { UL[1 + 2 * FZ_UMAX + u * FZ_MMAX + c] = (j / TOPK) * (TOPK + 1) + j % TOPK; c++; }
    \\    UL[1 + FZ_UMAX + u] = c;
    \\  }
    \\  if (i == 0) { int u = 0; for (int j = 0; j < n; j++) u += first[j]; UL[0] = u; }
    \\}
    \\// A distinct expert's pairs, 8 at a time: pair index (row * 11 + slot) or -1; the shared expert is z = 0.
    \\inline void fz_members(const device int* UL, int R, bool shared, int u, int c0, int cnt, thread int* pr) {
    \\  #pragma unroll
    \\  for (int m = 0; m < 8; m++) pr[m] = c0 + m < cnt ? (shared ? (c0 + m) * 11 + 10 : UL[1 + 2 * FZ_UMAX + u * FZ_MMAX + c0 + m]) : -1;
    \\}
    \\[[kernel]] void fz_ggu(const device bfloat* X [[buffer(0)]], const device int* UL [[buffer(1)]],
    \\    const device uint* GW [[buffer(2)]], const device bfloat* GS [[buffer(3)]], const device bfloat* GB [[buffer(4)]],
    \\    const device uint* UW [[buffer(5)]], const device bfloat* US [[buffer(6)]], const device bfloat* UB [[buffer(7)]],
    \\    const device uint* SGW [[buffer(8)]], const device bfloat* SGS [[buffer(9)]], const device bfloat* SGB [[buffer(10)]],
    \\    const device uint* SUW [[buffer(11)]], const device bfloat* SUS [[buffer(12)]], const device bfloat* SUB [[buffer(13)]],
    \\    device bfloat* ACT [[buffer(14)]], const constant int* rows [[buffer(15)]],
    \\    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint3 tg [[threadgroup_position_in_grid]]) {
    \\  constexpr int K = 2560, N = 640, SLOTS = 11, WPR = K * 6 / 32, KG = K / 32;
    \\  const bool shared = tg.z == 0;
    \\  const int u = int(tg.z) - 1;
    \\  if (!shared && u >= UL[0]) return;
    \\  const int cnt = shared ? rows[0] : UL[1 + FZ_UMAX + u];
    \\  const size_t e = shared ? 0 : size_t(UL[1 + u]);
    \\  const int row = int(tg.y) * 8 + int(sgi) * 2 + int(lane >> 4);
    \\  const int part = int(lane & 15);
    \\  const size_t wrow = e * N * WPR + size_t(row) * WPR + part * 30;
    \\  const size_t grow = e * N * KG + size_t(row) * KG + part * 5;
    \\  const device uint* gw = (shared ? SGW : GW) + wrow;
    \\  const device uint* uw = (shared ? SUW : UW) + wrow;
    \\  const device bfloat* gs = (shared ? SGS : GS) + grow;
    \\  const device bfloat* gb = (shared ? SGB : GB) + grow;
    \\  const device bfloat* us = (shared ? SUS : US) + grow;
    \\  const device bfloat* ub = (shared ? SUB : UB) + grow;
    \\  for (int c0 = 0; c0 < cnt; c0 += 8) {
    \\    int pr[8];
    \\    fz_members(UL, rows[0], shared, u, c0, cnt, pr);
    \\    float ag[8], au[8];
    \\    #pragma unroll
    \\    for (int m = 0; m < 8; m++) { ag[m] = 0.0f; au[m] = 0.0f; }
    \\    #pragma unroll
    \\    for (int j = 0; j < 5; j++) {
    \\      float q[32];
    \\      fz_codes6(gw + j * 6, q);
    \\      #pragma unroll
    \\      for (int m = 0; m < 8; m++) if (pr[m] >= 0) {
    \\        float qx, sx;
    \\        fz_dot32(q, X + (pr[m] / SLOTS) * K + part * 160 + j * 32, qx, sx);
    \\        ag[m] += float(gs[j]) * qx + float(gb[j]) * sx;
    \\      }
    \\      fz_codes6(uw + j * 6, q);
    \\      #pragma unroll
    \\      for (int m = 0; m < 8; m++) if (pr[m] >= 0) {
    \\        float qx, sx;
    \\        fz_dot32(q, X + (pr[m] / SLOTS) * K + part * 160 + j * 32, qx, sx);
    \\        au[m] += float(us[j]) * qx + float(ub[j]) * sx;
    \\      }
    \\    }
    \\    #pragma unroll
    \\    for (int m = 0; m < 8; m++) {
    \\      float a = ag[m], b = au[m];
    \\      for (ushort o = 8; o > 0; o >>= 1) { a += simd_shuffle_xor(a, o); b += simd_shuffle_xor(b, o); }
    \\      if (part == 0 && pr[m] >= 0) ACT[pr[m] * N + row] = bfloat(bsilu(float(bfloat(a))) * float(bfloat(b)));
    \\    }
    \\  }
    \\}
    \\[[kernel]] void fz_gdown(const device bfloat* ACT [[buffer(0)]], const device int* UL [[buffer(1)]],
    \\    const device uint* DW [[buffer(2)]], const device bfloat* DS [[buffer(3)]], const device bfloat* DB [[buffer(4)]],
    \\    const device uint* SDW [[buffer(5)]], const device bfloat* SDS [[buffer(6)]], const device bfloat* SDB [[buffer(7)]],
    \\    const constant int* rows [[buffer(8)]], device bfloat* Y [[buffer(9)]],
    \\    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint3 tg [[threadgroup_position_in_grid]]) {
    \\  constexpr int NI = 640, D = 2560, WPR = NI * 6 / 32, KG = NI / 32;
    \\  const bool shared = tg.z == 0;
    \\  const int u = int(tg.z) - 1;
    \\  if (!shared && u >= UL[0]) return;
    \\  const int cnt = shared ? rows[0] : UL[1 + FZ_UMAX + u];
    \\  const size_t e = shared ? 0 : size_t(UL[1 + u]);
    \\  const int d = int(tg.y) * 32 + int(sgi) * 8 + int(lane >> 2);
    \\  const int part = int(lane & 3);
    \\  const device uint* w = (shared ? SDW : DW + e * D * WPR) + size_t(d) * WPR + part * 30;
    \\  const size_t g0 = (shared ? 0 : e * D * KG) + size_t(d) * KG + part * 5;
    \\  const device bfloat* sc = (shared ? SDS : DS) + g0;
    \\  const device bfloat* bi = (shared ? SDB : DB) + g0;
    \\  for (int c0 = 0; c0 < cnt; c0 += 8) {
    \\    int pr[8];
    \\    fz_members(UL, rows[0], shared, u, c0, cnt, pr);
    \\    float acc[8];
    \\    #pragma unroll
    \\    for (int m = 0; m < 8; m++) acc[m] = 0.0f;
    \\    #pragma unroll
    \\    for (int j = 0; j < 5; j++) {
    \\      float q[32];
    \\      fz_codes6(w + j * 6, q);
    \\      #pragma unroll
    \\      for (int m = 0; m < 8; m++) if (pr[m] >= 0) {
    \\        float qx, sx;
    \\        fz_dot32(q, ACT + pr[m] * NI + part * 160 + j * 32, qx, sx);
    \\        acc[m] += float(sc[j]) * qx + float(bi[j]) * sx;
    \\      }
    \\    }
    \\    #pragma unroll
    \\    for (int m = 0; m < 8; m++) {
    \\      float a = acc[m];
    \\      a += simd_shuffle_xor(a, 1);
    \\      a += simd_shuffle_xor(a, 2);
    \\      if (part == 0 && pr[m] >= 0) Y[pr[m] * D + d] = bfloat(a);
    \\    }
    \\  }
    \\}
    \\// The router's 513 logits a row, one simdgroup a (row, expert): each lane's fma chain (8-input runs 256 apart) and
    \\// the simd sum as the recorded q4_router_float, so the bits are its bits; its rows run in parallel, not in turn.
    \\[[kernel]] void fz_router(const device bfloat* X [[buffer(0)]], const device bfloat* GW [[buffer(1)]],
    \\    const constant int* rows [[buffer(2)]], device float* OUT [[buffer(3)]],
    \\    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint3 tg [[threadgroup_position_in_grid]]) {
    \\  constexpr int D = 2560, NE = 513;
    \\  const int g = int(tg.x) * 8 + int(sgi);
    \\  const int r = g / NE, e = g % NE;
    \\  if (r >= rows[0]) return;
    \\  const device bfloat* w = GW + size_t(e) * D;
    \\  const device bfloat* xr = X + r * D;
    \\  float a = 0.0f;
    \\  _Pragma("clang loop unroll(full)") // the steps' loads go out together; the fma order is unchanged
    \\  for (int c = 8 * int(lane); c < D; c += 256) {
    \\    float wv[8];
    \\    for (int j = 0; j < 8; j++) wv[j] = float(w[c + j]);
    \\    for (int j = 0; j < 8; j++) a = fma(float(xr[c + j]), wv[j], a);
    \\  }
    \\  const float total = simd_sum(a);
    \\  if (lane == 0) OUT[r * NE + e] = total;
    \\}
;
