// Flash Next experts grouped by expert for wide decode windows (4-bit group 32): bit for bit the recorded kernels.
#include <metal_stdlib>
using namespace metal;
#ifndef GX_T
#define GX_T 4
#endif

constant constexpr int K = 2560, N = 640, NE = 512, NL = 513, TOPK = 10, SLOTS = 11, RMAX = 32;
constant constexpr int KG = K / 32, NG = N / 32, NC = N / 16; // groups and 16-input chunks a row
// LIST: counts, threadgroup counts, pairs by expert (row * 11 + slot | expert << 16), twin and tile items (-1: shared)
constant constexpr int L_N = 0, L_NT = 1, L_NM = 2, L_GU = 4, L_DNT = 8, L_DNM = 12;
constant constexpr int L_MEM = 16, L_TWIN = L_MEM + RMAX * TOPK, L_TILE = L_TWIN + 2 * (RMAX * TOPK + RMAX);
constant constexpr int LIST_WORDS = L_TILE + 4 * (RMAX * TOPK + RMAX / 8);

inline float bsilu(float x) { return float(bfloat(x / (1.0f + metal::exp(-x)))); }
// float(v) for v < 2^23 by the exponent trick: the same value as a convert
inline float nib(uint v) { return as_type<float>(0x4B000000u | v) - 8388608.0f; }
inline int tile_fm(int l) { return ((l / 4) & 4) + ((l / 2) % 4); }
inline int tile_fn(int l) { return ((l / 4) & 2) * 2 + (l % 2) * 2; }

// The recorded kernels' 16 codes of a row from input v0 (half nibbles: the same values)
inline void lane_codes(const device uint8_t* row, int v0, thread float* q) {
  const device ushort* h = (const device ushort*)(row + v0 / 2);
  for (int j = 0; j < 4; j += 2) {
    const uint w = uint(h[j]) | (uint(h[j + 1]) << 16);
    const half2 a = as_type<half2>(0x64006400u | (w & 0x000F000Fu)) - half2(1024.0h);
    const half2 b = as_type<half2>(0x64006400u | ((w >> 4) & 0x000F000Fu)) - half2(1024.0h);
    const half2 c = as_type<half2>(0x64006400u | ((w >> 8) & 0x000F000Fu)) - half2(1024.0h);
    const half2 d = as_type<half2>(0x64006400u | ((w >> 12) & 0x000F000Fu)) - half2(1024.0h);
    q[8 * (j / 2) + 0] = float(a.x); q[8 * (j / 2) + 1] = float(b.x); q[8 * (j / 2) + 2] = float(c.x); q[8 * (j / 2) + 3] = float(d.x);
    q[8 * (j / 2) + 4] = float(a.y); q[8 * (j / 2) + 5] = float(b.y); q[8 * (j / 2) + 6] = float(c.y); q[8 * (j / 2) + 7] = float(d.y);
  }
}
// The recorded gateup_rows (4-bit, group 32, K 2560, 4 rows a simdgroup) for one row of input x
inline void gateup_rows(const device uint32_t* GWp, const device uint32_t* UWp, const device bfloat* GSp,
                        const device bfloat* GBp, const device bfloat* USp, const device bfloat* UBp, size_t first,
                        const device bfloat* x, uint lane, thread float* ag, thread float* au) {
  constexpr int RB = K / 2, RPS = 4;
  const device uint8_t* gw = (const device uint8_t*)GWp + first * RB;
  const device uint8_t* uw = (const device uint8_t*)UWp + first * RB;
  for (int v0 = int(lane) * 16; v0 < K; v0 += 512) {
    float xv[16], sum = 0.0f;
    for (int i = 0; i < 16; i++) { xv[i] = float(x[v0 + i]); sum += xv[i]; }
    for (int row = 0; row < RPS; row++) {
      float qg[16], qu[16];
      lane_codes(gw + row * RB, v0, qg);
      lane_codes(uw + row * RB, v0, qu);
      float dg = 0.0f, du = 0.0f;
      for (int i = 0; i < 16; i++) { dg = fma(qg[i], xv[i], dg); du = fma(qu[i], xv[i], du); }
      const size_t at = (first + row) * KG + v0 / 32;
      ag[row] += fma(float(GSp[at]), dg, float(GBp[at]) * sum);
      au[row] += fma(float(USp[at]), du, float(UBp[at]) * sum);
    }
  }
}
// The recorded down_rows_bal (4-bit, group 32, 640 inputs): 8 rows, the second chunk spread over every lane
inline void down_rows_bal(const device uint32_t* W, const device bfloat* S, const device bfloat* B, size_t first,
                          const device bfloat* x, uint lane, thread float* out) {
  constexpr int RB = 320;
  const int m = int(lane & 7);
  float xv0[16], xv1[16], sum0 = 0.0f, sum1 = 0.0f;
  for (int i = 0; i < 16; i++) { xv0[i] = float(x[int(lane) * 16 + i]); sum0 += xv0[i]; }
  for (int i = 0; i < 16; i++) { xv1[i] = float(x[512 + m * 16 + i]); sum1 += xv1[i]; }
  float p1[2];
  for (int t = 0; t < 2; t++) {
    const int row = 2 * int(lane >> 3) + t;
    float q[16];
    lane_codes((const device uint8_t*)W + (first + row) * RB, 512 + m * 16, q);
    float d = 0.0f;
    for (int i = 0; i < 16; i++) d = fma(q[i], xv1[i], d);
    const size_t at = (first + row) * NG + (512 + m * 16) / 32;
    p1[t] = fma(float(S[at]), d, float(B[at]) * sum1);
  }
  for (int row = 0; row < 8; row++) {
    float q[16];
    lane_codes((const device uint8_t*)W + (first + row) * RB, int(lane) * 16, q);
    float d = 0.0f;
    for (int i = 0; i < 16; i++) d = fma(q[i], xv0[i], d);
    const size_t at = (first + row) * NG + int(lane) / 2;
    float acc = 0.0f;
    acc += fma(float(S[at]), d, float(B[at]) * sum0);
    const float q1 = simd_shuffle(p1[row & 1], ushort(8 * (row >> 1) + m));
    if (lane < 8) acc += q1;
    out[row] = acc;
  }
}

// simd_topk's rounds 0 .. TOPK-1 in one pass (the recorded expert kernels' picks): rounds of (largest, lowest id)
inline void topk_all(const device float* logits, uint lane, thread int* ids, thread float* picked) {
  float v[NE / 32];
  for (int j = 0; j < NE / 32; j++) v[j] = logits[j * 32 + int(lane)];
  for (int round = 0; round < TOPK; round++) {
    float best = -INFINITY;
    int bid = NE;
    for (int j = 0; j < NE / 32; j++) {
      const int e = j * 32 + int(lane);
      if (v[j] > best || (v[j] == best && e < bid)) { best = v[j]; bid = e; }
    }
    for (int off = 16; off > 0; off /= 2) {
      const float ob = simd_shuffle_xor(best, off);
      const int oi = simd_shuffle_xor(bid, off);
      if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; }
    }
    picked[round] = best;
    ids[round] = bid;
    if (int(lane) == bid % 32) v[bid / 32] = -INFINITY;
  }
}
// Threadgroup r (one simdgroup): row r's picks and weights, as the recorded gate/up writes them
kernel void gx_pick(const device float* LG [[buffer(0)]], const constant int* rows [[buffer(1)]],
    device uint* PICK [[buffer(2)]], device float* WTS [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int r = int(tg.x);
  if (r >= min(rows[0], RMAX)) return;
  int ids[TOPK];
  float picked[TOPK];
  topk_all(LG + r * NL, lane, ids, picked);
  if (lane != 0) return;
  float total = 0.0f;
  float ex[TOPK];
  for (int k = 0; k < TOPK; k++) { ex[k] = metal::exp(picked[k] - picked[0]); total += ex[k]; }
  for (int k = 0; k < TOPK; k++) {
    WTS[r * TOPK + k] = float(bfloat(ex[k] / total));
    PICK[r * TOPK + k] = uint(ids[k]);
  }
}

// An expert's m rows in down: tiles of 8 from GX_T rows (a last tile when its rest reaches GX_T), the rest twin items
inline int tile_rows(int m) { return m >= GX_T ? 8 * (m / 8) + (m % 8 >= GX_T ? m % 8 : 0) : 0; }
inline int tile_items(int m) { return m >= GX_T ? m / 8 + (m % 8 >= GX_T ? 1 : 0) : 0; }
// e's items: tiles over its pairs first .., then twin items for the rest (pairs from sp, or the shared expert's rows)
inline void put_items(device int* LIST, threadgroup const int* sp, int e, int first, int m, int t0, int m0) {
  const int tr = tile_rows(m);
  for (int q = 0; 8 * q < tr; q++) ((device int4*)(LIST + L_TILE))[m0 + q] = int4(e, first + 8 * q, min(8, tr - 8 * q), 0);
  for (int c = tr; c < m; c++)
    ((device int2*)(LIST + L_TWIN))[t0 + c - tr] = int2(e, e < 0 ? c * SLOTS + TOPK : sp[first + c] & 0xFFFF);
}

// One threadgroup of 512: pairs grouped by expert (their order within an expert varies; no row's bits depend on it)
kernel void gx_list(const device uint* PICK [[buffer(0)]], const constant int* rows [[buffer(1)]],
    device int* LIST [[buffer(2)]], uint3 tid [[thread_position_in_threadgroup]],
    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup atomic_int cnt[NE];
  threadgroup atomic_int cur[NE];
  threadgroup int first[NE];
  threadgroup int tot[16];
  threadgroup int sp[RMAX * TOPK];
  const uint t = tid.x;
  const int R = min(rows[0], RMAX), n = R * TOPK;
  atomic_store_explicit(&cnt[t], 0, memory_order_relaxed);
  atomic_store_explicit(&cur[t], 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int ea = -1, eb = -1; // this thread's picks i = t and t + 512
  if (int(t) < n) { ea = int(PICK[t]); atomic_fetch_add_explicit(&cnt[ea], 1, memory_order_relaxed); }
  if (int(t) + NE < n) { eb = int(PICK[t + NE]); atomic_fetch_add_explicit(&cnt[eb], 1, memory_order_relaxed); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // exclusive prefix over experts in id order of (pairs, twin items, tile items), 10 bits each
  const int m = atomic_load_explicit(&cnt[t], memory_order_relaxed);
  const int packed = m | ((m - tile_rows(m)) << 10) | (tile_items(m) << 20);
  const int before = simd_prefix_exclusive_sum(packed);
  if (lane == 31) tot[sgi] = before + packed;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int p = before;
  for (uint s = 0; s < sgi; s++) p += tot[s];
  first[t] = p & 1023;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int h = 0; h < 2; h++) {
    const int i = int(t) + h * NE, e = h ? eb : ea;
    if (e < 0) continue;
    const int at = first[e] + atomic_fetch_add_explicit(&cur[e], 1, memory_order_relaxed);
    const int v = ((i / TOPK) * SLOTS + i % TOPK) | (e << 16);
    sp[at] = v;
    LIST[L_MEM + at] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int st = R - tile_rows(R), sm = tile_items(R);
  if (m > 0) put_items(LIST, sp, int(t), p & 1023, m, st + ((p >> 10) & 1023), sm + ((p >> 20) & 1023));
  if (t == 0) put_items(LIST, sp, -1, 0, R, 0, 0);
  if (t == uint(NE - 1)) {
    const int all = p + packed, nt = st + ((all >> 10) & 1023), nm = sm + ((all >> 20) & 1023);
    LIST[L_N] = n; LIST[L_NT] = nt; LIST[L_NM] = nm;
    LIST[L_GU] = N / 8; LIST[L_GU + 1] = n + R; LIST[L_GU + 2] = 1;
    LIST[L_DNT] = K / 16; LIST[L_DNT + 1] = nt; LIST[L_DNT + 2] = 1;
    LIST[L_DNM] = K / 16; LIST[L_DNM + 1] = nm; LIST[L_DNM + 2] = 1;
  }
}

// Gate/up: the recorded per-row code a pair a threadgroup, pairs by expert (an expert's rows back to back), then shared
kernel void gx_gateup(const device bfloat* X [[buffer(0)]], const device int* LIST [[buffer(1)]],
    const device uint* GW [[buffer(2)]], const device bfloat* GS [[buffer(3)]], const device bfloat* GB [[buffer(4)]],
    const device uint* UW [[buffer(5)]], const device bfloat* US [[buffer(6)]], const device bfloat* UB [[buffer(7)]],
    const device uint* SGW [[buffer(8)]], const device bfloat* SGS [[buffer(9)]], const device bfloat* SGB [[buffer(10)]],
    const device uint* SUW [[buffer(11)]], const device bfloat* SUS [[buffer(12)]], const device bfloat* SUB [[buffer(13)]],
    device bfloat* ACT [[buffer(14)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const int n = LIST[L_N], y = int(tg.y);
  const bool shared = y >= n;
  const int v = shared ? 0 : LIST[L_MEM + y];
  const int p = shared ? (y - n) * SLOTS + TOPK : v & 0xFFFF;
  const int row0 = int(tg.x) * 8 + int(sgi) * 4;
  float ag[4] = {0.0f, 0.0f, 0.0f, 0.0f}, au[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (shared) gateup_rows(SGW, SUW, SGS, SGB, SUS, SUB, size_t(row0), X + (p / SLOTS) * K, lane, ag, au);
  else gateup_rows(GW, UW, GS, GB, US, UB, size_t(v >> 16) * N + row0, X + (p / SLOTS) * K, lane, ag, au);
  for (int row = 0; row < 4; row++) {
    const float gv = simd_sum(ag[row]), uv = simd_sum(au[row]);
    if (lane == 0) ACT[p * N + row0 + row] = bfloat(bsilu(float(bfloat(gv))) * float(bfloat(uv)));
  }
}

// Down for a pair of an expert with fewer than GX_T rows: threadgroup (16 outputs, item), 64 threads, the recorded code
kernel void gx_down_one(const device bfloat* ACT [[buffer(0)]], const device int* LIST [[buffer(1)]],
    const device uint* DW [[buffer(2)]], const device bfloat* DS [[buffer(3)]], const device bfloat* DB [[buffer(4)]],
    const device uint* SDW [[buffer(5)]], const device bfloat* SDS [[buffer(6)]], const device bfloat* SDB [[buffer(7)]],
    device bfloat* Y [[buffer(8)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const int2 it = ((const device int2*)(LIST + L_TWIN))[tg.y];
  const int d0 = int(tg.x) * 16 + int(sgi) * 8;
  float out[8];
  if (it.x < 0) down_rows_bal(SDW, SDS, SDB, size_t(d0), ACT + it.y * N, lane, out);
  else down_rows_bal(DW, DS, DB, size_t(it.x) * K + d0, ACT + it.y * N, lane, out);
  for (int row = 0; row < 8; row++) {
    const float v = simd_sum(out[row]);
    if (lane == 0) Y[it.y * K + d0 + row] = bfloat(v);
  }
}

// a tile item's member c: the shared expert's rows first .., or the routed list's pairs
inline int gx_pair(const device int* LIST, int4 it, int c) {
  return it.x < 0 ? (it.y + c) * SLOTS + TOPK : LIST[L_MEM + it.y + c] & 0xFFFF;
}

// Down tiles: the recorded sum's 32 lane partials (chunk l, then 32 + l for l < 8) in simd_sum's xor tree 1, 2, 4, 8, 16
kernel void gx_down_tile(const device bfloat* ACT [[buffer(0)]], const device int* LIST [[buffer(1)]],
    const device uint* DW [[buffer(2)]], const device bfloat* DS [[buffer(3)]], const device bfloat* DB [[buffer(4)]],
    const device uint* SDW [[buffer(5)]], const device bfloat* SDS [[buffer(6)]], const device bfloat* SDB [[buffer(7)]],
    device bfloat* Y [[buffer(8)]], uint3 tg [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]],
    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float as[8][NC];
  const uint t = tid.x;
  const int4 it = ((const device int4*)(LIST + L_TILE))[tg.y];
  const bool shared = it.x < 0;
  const size_t e = shared ? 0 : size_t(it.x);
  const int cnt = it.z;
  for (int i = int(t); i < cnt * NC; i += 64) { // each member's chunk sums, 16 inputs in order
    const device bfloat* x = ACT + gx_pair(LIST, it, i / NC) * N + (i % NC) * 16;
    float s = 0.0f;
    for (int v = 0; v < 16; v++) s += float(x[v]);
    as[i / NC][i % NC] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int fm = tile_fm(int(lane)), fn = tile_fn(int(lane));
  const int c0 = min(fn, cnt - 1), c1 = min(fn + 1, cnt - 1);
  const int row = int(tg.x) * 16 + int(sgi) * 8 + fm;
  const device bfloat* x0 = ACT + gx_pair(LIST, it, c0) * N;
  const device bfloat* x1 = ACT + gx_pair(LIST, it, c1) * N;
  const device uint2* w = (const device uint2*)((shared ? SDW : DW + e * K * (N / 8)) + size_t(row) * (N / 8));
  const size_t so = (shared ? 0 : e * K * NG) + size_t(row) * NG;
  const device bfloat* ds = (shared ? SDS : DS) + so;
  const device bfloat* db = (shared ? SDB : DB) + so;
  float sq[4][2];
  for (int q = 0; q < 4; q++) {
    float acc[8][2];
    for (int l = 0; l < 8; l++) { acc[l][0] = 0.0f; acc[l][1] = 0.0f; }
    for (int pass = 0; pass < (q == 0 ? 2 : 1); pass++) {
      for (int l = 0; l < 8; l++) {
        const int c = 8 * q + l + 32 * pass;
        const uint2 cw = w[c];
        simdgroup_matrix<float, 8, 8> P = simdgroup_matrix<float, 8, 8>(0.0f);
        for (int stp = 0; stp < 2; stp++) {
          const int k = 16 * c + 8 * stp + fm;
          simdgroup_matrix<float, 8, 8> am, bm;
          bm.thread_elements()[0] = float(x0[k]);
          bm.thread_elements()[1] = float(x1[k]);
          am.thread_elements()[0] = nib((cw[stp] >> (4 * fn)) & 0xFu);
          am.thread_elements()[1] = nib((cw[stp] >> (4 * fn + 4)) & 0xFu);
          simdgroup_multiply_accumulate(P, am, bm, P); // fp32: the forward FMA chain the recorded dot runs
        }
        const float sc = float(ds[c / 2]), bi = float(db[c / 2]);
        acc[l][0] += fma(sc, P.thread_elements()[0], bi * as[c0][c]);
        acc[l][1] += fma(sc, P.thread_elements()[1], bi * as[c1][c]);
      }
    }
    for (int i = 0; i < 2; i++)
      sq[q][i] = ((acc[0][i] + acc[1][i]) + (acc[2][i] + acc[3][i])) + ((acc[4][i] + acc[5][i]) + (acc[6][i] + acc[7][i]));
  }
  for (int i = 0; i < 2; i++) {
    const int c = fn + i;
    if (c < cnt) Y[gx_pair(LIST, it, c) * K + row] = bfloat((sq[0][i] + sq[1][i]) + (sq[2][i] + sq[3][i]));
  }
}
