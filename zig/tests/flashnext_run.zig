//! Flash Next windows of 1-8 rows on the Python engine's own kernels (tools/zig/flashnext_dump.py): every launch is a
//! recorded variant, every weight the pack's or the checkpoint's. Checks one-row greedy tokens and every row of the
//! Python engine's drafted windows (with rollback), then times each window size.
const std = @import("std");
const mtl = @import("metal");

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

const D = 2560;
const WIDE = 4 * D;
const LAYERS = 48;
const VOCAB = 248320;
const CAP = 8192; // keys an attention layer holds in this runner
const PLE_TAIL = 9;
const GROUPS = 8;
const MAXR = 8;
const CS_ROW = 3 * WIDE * 2; // a DeltaNet conv state row (bytes)
const SO_ROW = 48 * 128 * 128 * 4; // a DeltaNet recurrent state row (bytes)

const Buf = struct { b: mtl.Buffer, off: usize = 0 };
const Entry = struct { fd: std.c.fd_t, at: usize, len: usize };
const Variant = struct { inputs: [][]const u8, outputs: [][]const u8, meta: [][]const u8, pipe: mtl.Pipeline };
const Site = struct { v: *Variant, grid: mtl.Size, tg: mtl.Size };

const glue_source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\// meta: position of row 0, key capacity, rows
    \\kernel void fz_kv_write(device const bfloat* kout [[buffer(0)]], device const bfloat* p [[buffer(1)]],
    \\    device bfloat* keys [[buffer(2)]], device bfloat* vals [[buffer(3)]], device bfloat* raw [[buffer(4)]],
    \\    constant uint* meta [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    \\  const uint pos = meta[0], cap = meta[1], rows = meta[2];
    \\  const uint r = i >> 9, j = i & 511;
    \\  if (r >= rows) return;
    \\  const uint at = ((j >> 8) * cap + pos + r) * 256 + (j & 255);
    \\  keys[at] = kout[r * 512 + j];
    \\  vals[at] = p[r * 13952 + 12800 + j];
    \\  if (j < 128) raw[(pos + r) * 128 + j] = p[r * 13952 + 13824 + j];
    \\}
    \\kernel void fz_argmax(device const bfloat* logits [[buffer(0)]], device uint* out [[buffer(1)]],
    \\    constant uint& n [[buffer(2)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    \\    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    \\  device const bfloat* x = logits + row * n;
    \\  float best = -INFINITY; uint at = 0xffffffffu;
    \\  for (uint i = t; i < n; i += 1024) { const float v = float(x[i]); if (v > best) { best = v; at = i; } }
    \\  for (ushort o = 16; o > 0; o >>= 1) {
    \\    const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(at, o);
    \\    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
    \\  }
    \\  threadgroup float vb[32]; threadgroup uint va[32];
    \\  if (lane == 0) { vb[sg] = best; va[sg] = at; }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (sg != 0) return;
    \\  best = vb[lane]; at = va[lane];
    \\  for (ushort o = 16; o > 0; o >>= 1) {
    \\    const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(at, o);
    \\    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
    \\  }
    \\  if (lane == 0) out[row] = at;
    \\}
    \\// x [R, 4 * 2560] = e [R, 2560] broadcast over the streams + hs [4R, 2560] (MLX's bf16 add)
    \\kernel void fz_bcast_add(device const bfloat* e [[buffer(0)]], device const bfloat* hs [[buffer(1)]],
    \\    device bfloat* x [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    \\  if (i >= n) return;
    \\  x[i] = bfloat(float(e[(i / 10240) * 2560 + i % 2560]) + float(hs[i]));
    \\}
    \\kernel void fz_argmax_ids(device const bfloat* logits [[buffer(0)]], device uint* out [[buffer(1)]],
    \\    constant uint& n [[buffer(2)]], device const uint* ids [[buffer(3)]], uint t [[thread_index_in_threadgroup]],
    \\    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    \\  float best = -INFINITY; uint at = 0xffffffffu;
    \\  for (uint i = t; i < n; i += 1024) { const float v = float(logits[i]); if (v > best) { best = v; at = i; } }
    \\  for (ushort o = 16; o > 0; o >>= 1) {
    \\    const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(at, o);
    \\    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
    \\  }
    \\  threadgroup float vb[32]; threadgroup uint va[32];
    \\  if (lane == 0) { vb[sg] = best; va[sg] = at; }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (sg != 0) return;
    \\  best = vb[lane]; at = va[lane];
    \\  for (ushort o = 16; o > 0; o >>= 1) {
    \\    const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(at, o);
    \\    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
    \\  }
    \\  if (lane == 0) out[0] = ids[at];
    \\}
    \\// NGramEmbedding.ids for a window: pm = history (2), eos, rows, multipliers (3), head sizes (16), offsets (16)
    \\kernel void fz_ple_ids(device const uint* tok [[buffer(0)]], device const long* pm [[buffer(1)]],
    \\    device uint* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    \\  const uint rows = uint(pm[3]), row = i / 16, hh = i % 16;
    \\  if (row >= rows) return;
    \\  long seq[10];
    \\  seq[0] = pm[0]; seq[1] = pm[1];
    \\  for (uint q = 0; q <= row; q++) seq[2 + q] = long(tok[q]);
    \\  const long eos = pm[2];
    \\  const uint at = 2 + row;
    \\  long before = -1;
    \\  for (uint q = 0; q < at; q++) if (seq[q] == eos) before = long(q);
    \\  const long in_seg = long(at) - (before + 1);
    \\  ulong mixed = 0;
    \\  const uint ng = hh < 8 ? 2 : 3;
    \\  for (uint q = 0; q < ng; q++) {
    \\    const long sh = in_seg >= long(q) ? seq[at - q] : eos;
    \\    const ulong term = ulong(sh) * ulong(pm[4 + q]);
    \\    mixed = q == 0 ? term : (mixed ^ term);
    \\  }
    \\  const long size = pm[7 + hh];
    \\  long mod = as_type<long>(mixed) % size;
    \\  if (mod < 0) mod += size;
    \\  out[row * 16 + hh] = uint(mod + pm[23 + hh]);
    \\}
    \\// dst[l][i] = src[l][(keep + base) row][i] for i < p.x words: kept DeltaNet states, the PLE tail, the MTP's row
    \\kernel void fz_copy_kept(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],
    \\    device const int* ar [[buffer(2)]], constant uint4& p [[buffer(3)]], constant int& base [[buffer(4)]],
    \\    uint2 gid [[thread_position_in_grid]]) {
    \\  if (gid.x >= p.x) return;
    \\  const uint row = uint(ar[0] + base);
    \\  dst[gid.y * p.w + gid.x] = src[gid.y * p.z + row * p.y + gid.x];
    \\}
    \\// The round's verdict on the GPU: drafts kept, the emitted tokens to the ring, the next window's pending token,
    \\// positions and n-gram history; ar = keep, target length, round, then the meta blocks the next round reads.
    \\kernel void fz_accept(device uint* wids [[buffer(0)]], device const uint* picks [[buffer(1)]],
    \\    device int* ar [[buffer(2)]], device uint* ring [[buffer(3)]], device long* pm [[buffer(4)]],
    \\    constant uint4& cfg [[buffer(5)]], uint tid [[thread_position_in_grid]]) {
    \\  if (tid != 0) return;
    \\  const int W = int(cfg.x), depth = int(cfg.y), cap = int(cfg.z);
    \\  int keep = 1;
    \\  while (keep < W && wids[keep] == picks[keep - 1]) keep++;
    \\  const int round = ar[2];
    \\  ring[round * 9] = uint(keep);
    \\  for (int i = 0; i < keep; i++) ring[round * 9 + 1 + i] = picks[i];
    \\  for (int i = 0; i < keep; i++) { pm[0] = pm[1]; pm[1] = long(wids[i]); }
    \\  const int t_old = ar[1], t_new = t_old + keep;
    \\  ar[0] = keep; ar[1] = t_new; ar[2] = round + 1;
    \\  for (int i = 0; i < 8; i++) {
    \\    ar[4 + i] = i < W ? t_new + i : 0;
    \\    ar[12 + i] = i < W ? t_new + i + 1 : 0;
    \\    ar[24 + i] = i < W ? t_old + i : 0;
    \\    ar[32 + i] = i < W ? t_old + i + 1 : 0;
    \\  }
    \\  ar[20] = t_new; ar[21] = cap; ar[22] = W;
    \\  ar[40] = t_old; ar[41] = cap; ar[42] = W;
    \\  for (int j = 1; j < depth; j++) {
    \\    const int b = 44 + (j - 1) * 20;
    \\    for (int i = 0; i < 8; i++) { ar[b + i] = 0; ar[b + 8 + i] = 0; }
    \\    ar[b] = t_new + j - 1; ar[b + 8] = t_new + j;
    \\    ar[b + 16] = t_new + j - 1; ar[b + 17] = cap; ar[b + 18] = 1;
    \\  }
    \\  wids[0] = picks[keep - 1];
    \\}
    \\// An expert row from [part][group][6 words] to [group][part][6 words] (FZ_XPACK): the lanes of one group read
    \\// one block; the sums keep their order. d = (parts, groups a part).
    \\[[kernel]] void fz_repack_w(device uint* W [[buffer(0)]], constant uint2& d [[buffer(1)]],
    \\    uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
    \\  threadgroup uint tmp[480];
    \\  const uint n = d.x * d.y * 6;
    \\  device uint* w = W + size_t(row) * n;
    \\  for (uint i = t; i < n; i += 128) tmp[i] = w[i];
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  for (uint i = t; i < n; i += 128) {
    \\    const uint j = i / (d.x * 6), part = (i / 6) % d.x, k = i % 6;
    \\    w[i] = tmp[part * d.y * 6 + j * 6 + k];
    \\  }
    \\}
    \\[[kernel]] void fz_repack_s(device ushort* S [[buffer(0)]], constant uint2& d [[buffer(1)]],
    \\    uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
    \\  threadgroup ushort tmp[80];
    \\  const uint n = d.x * d.y;
    \\  device ushort* s = S + size_t(row) * n;
    \\  for (uint i = t; i < n; i += 128) tmp[i] = s[i];
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  for (uint i = t; i < n; i += 128) s[i] = tmp[(i % d.x) * d.y + i / d.x];
    \\}
    \\// FZ_PREFETCH: read a weight buffer into the cache while a latency-bound launch runs beside it.
    \\[[kernel]] void fz_touch(const device uint4* w [[buffer(0)]], device uint* sink [[buffer(1)]], constant uint& n [[buffer(2)]],
    \\    uint i [[thread_position_in_grid]], uint gs [[threads_per_grid]]) {
    \\  uint acc = 0;
    \\  for (uint k = i; k < n; k += gs) { const uint4 v = w[k]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    \\  if (acc == 0x9e3779b9u) sink[0] = acc;
    \\}
;


/// Routed and shared experts at full width (FZ_XNEW=1): a 6-bit group read as six aligned words, 16 lanes a gate/up
/// row (5 groups each) and 4 lanes a down row, 4 simdgroups a threadgroup; routing (top-k, weights) unchanged.
const xnew_source =
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
    \\    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint3 tg [[threadgroup_position_in_grid]]) {
    \\  constexpr int K = 2560, N = 640, TOPK = 10, NE = 512, NL = 513, SLOTS = TOPK + 1, WPR = K * 6 / 32, KG = K / 32;
    \\  const int p = int(tg.z);
    \\  const int r = p / SLOTS, slot = p % SLOTS;
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
    \\    const constant int* rows [[buffer(8)]], device bfloat* Y [[buffer(9)]],
    \\    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint3 tg [[threadgroup_position_in_grid]]) {
    \\  constexpr int NI = 640, D = 2560, TOPK = 10, SLOTS = TOPK + 1, WPR = NI * 6 / 32, KG = NI / 32;
    \\  const int pair = int(tg.z);
    \\  if (pair >= rows[0] * SLOTS) return;
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
    \\  const int t = int(tgi), n = t * 32 + int(lane);
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
;

fn readAll(fd: std.c.fd_t, dest: []u8, at: usize) !void {
    var done: usize = 0;
    while (done < dest.len) {
        const n = std.c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
}

const Run = struct {
    arena: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    index: std.StringHashMapUnmanaged(Entry) = .empty,
    variants: std.StringHashMapUnmanaged(*Variant) = .empty,
    roles: std.StringHashMapUnmanaged(Site) = .empty,
    shapes: std.StringHashMapUnmanaged(mtl.Buffer) = .empty,
    loaded: usize = 0,
    rows: usize = 1,
    fused_xsum: bool = false,
    serial: bool = false,
    enc: mtl.ComputeEncoder = undefined,
    kv_pipe: mtl.Pipeline = undefined,
    argmax_pipe: mtl.Pipeline = undefined,
    add_pipe: mtl.Pipeline = undefined,
    argids_pipe: mtl.Pipeline = undefined,
    pleids_pipe: mtl.Pipeline = undefined,
    copy_pipe: mtl.Pipeline = undefined,
    accept_pipe: mtl.Pipeline = undefined,
    gpu_round: bool = false,
    ar: Buf = undefined,
    probe: ?Buf = null,
    xnew: bool = false,
    xgu_pipe: mtl.Pipeline = undefined,
    xdown_pipe: mtl.Pipeline = undefined,
    dense: bool = false,
    dense_target: bool = false,
    skip: u32 = 0,
    split: bool = false,
    gdn_step: bool = false,
    hc_mma: bool = false,
    event: mtl.SharedEvent = undefined,
    event_value: u64 = 0,
    dense8_pipe: mtl.Pipeline = undefined,
    grouped: bool = false,
    gskip: u32 = 0,
    xpack: bool = false,
    xfused: bool = false,
    xsx: bool = false,
    copy: bool = false,
    xgu_sx_pipe: mtl.Pipeline = undefined,
    xdown_sx_pipe: mtl.Pipeline = undefined,
    prefetch: bool = false,
    touch_pipe: mtl.Pipeline = undefined,
    sink: Buf = undefined,
    xfused_pipe: mtl.Pipeline = undefined,
    xpart: Buf = undefined,
    xdone: Buf = undefined,
    repack_w: mtl.Pipeline = undefined,
    repack_s: mtl.Pipeline = undefined,
    route_pipe: mtl.Pipeline = undefined,
    ggu_pipe: mtl.Pipeline = undefined,
    gdown_pipe: mtl.Pipeline = undefined,
    ul: Buf = undefined,
    dense16_pipe: mtl.Pipeline = undefined,

    fn buffer(r: *Run, len: usize) !mtl.Buffer {
        const n = @max(len, 64);
        const b = try r.device.buffer(n, opts);
        @memset(b.contents()[0..n], 0);
        return b;
    }

    /// Every tensor of a safetensors file in the index, at its absolute offset.
    fn indexFile(r: *Run, path: [:0]const u8) !void {
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.OpenFailed;
        var head: [8]u8 = undefined;
        try readAll(fd, &head, 0);
        const n = std.mem.readInt(u64, &head, .little);
        const text = try r.arena.alloc(u8, n);
        try readAll(fd, text, 8);
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, r.arena, text, .{});
        var it = parsed.object.iterator();
        while (it.next()) |kv| {
            if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
            const offs = kv.value_ptr.object.get("data_offsets").?.array.items;
            const lo: usize = @intCast(offs[0].integer);
            const hi: usize = @intCast(offs[1].integer);
            try r.index.put(r.arena, kv.key_ptr.*, .{ .fd = fd, .at = 8 + n + lo, .len = hi - lo });
        }
    }

    fn entry(r: *Run, name: []const u8) !Entry {
        return r.index.get(name) orelse {
            std.log.err("no tensor {s}", .{name});
            return error.MissingTensor;
        };
    }

    fn load(r: *Run, name: []const u8) !Buf {
        const e = try r.entry(name);
        const b = try r.device.buffer(@max(e.len, 64), opts);
        try readAll(e.fd, b.contents()[0..e.len], e.at);
        r.loaded += e.len;
        return .{ .b = b };
    }

    fn loadf(r: *Run, comptime fmt: []const u8, args: anytype) !Buf {
        var name: [160]u8 = undefined;
        return r.load(try std.fmt.bufPrint(&name, fmt, args));
    }

    /// Shards `first..first+count` of `suffix` concatenated into one buffer (PleTables' group).
    fn group(r: *Run, first: usize, count: usize, suffix: []const u8) !Buf {
        var total: usize = 0;
        var name: [160]u8 = undefined;
        const fmt = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{d}.{s}";
        for (first..first + count) |s| total += (try r.entry(try std.fmt.bufPrint(&name, fmt, .{ s, suffix }))).len;
        const b = try r.device.buffer(total, opts);
        var at: usize = 0;
        for (first..first + count) |s| {
            const e = try r.entry(try std.fmt.bufPrint(&name, fmt, .{ s, suffix }));
            try readAll(e.fd, b.contents()[at .. at + e.len], e.at);
            at += e.len;
        }
        r.loaded += total;
        return .{ .b = b };
    }

    fn compile(r: *Run, dir: []const u8) !void {
        const path = try std.fmt.allocPrintSentinel(r.arena, "{s}/plan.json", .{dir}, 0);
        const f = try mtl.MappedFile.open(path);
        const plan = try std.json.parseFromSliceLeaky(std.json.Value, r.arena, f.bytes[0..f.size], .{});
        var vit = plan.object.get("variants").?.object.iterator();
        while (vit.next()) |kv| {
            const o = kv.value_ptr.object;
            const src_path = try std.fmt.allocPrintSentinel(r.arena, "{s}/{s}", .{ dir, o.get("file").?.string }, 0);
            const src = try mtl.MappedFile.open(src_path);
            const lib = try mtl.Library.fromSource(r.device, src.bytes[0..src.size], mtl.CompileOptions.mlx());
            const v = try r.arena.create(Variant);
            v.* = .{ .inputs = try strings(r.arena, o.get("inputs").?), .outputs = try strings(r.arena, o.get("outputs").?), .meta = try strings(r.arena, o.get("meta").?), .pipe = try mtl.Pipeline.init(r.device, lib, kv.key_ptr.*, false) };
            try r.variants.put(r.arena, kv.key_ptr.*, v);
        }
        var rit = plan.object.get("roles").?.object.iterator();
        while (rit.next()) |kv| {
            const o = kv.value_ptr.object;
            try r.roles.put(r.arena, kv.key_ptr.*, .{ .v = r.variants.get(o.get("function").?.string).?, .grid = size3(o.get("grid").?), .tg = size3(o.get("threadgroup").?) });
        }
        const glue = try mtl.Library.fromSource(r.device, glue_source, mtl.CompileOptions.mlx());
        r.kv_pipe = try mtl.Pipeline.init(r.device, glue, "fz_kv_write", false);
        r.argmax_pipe = try mtl.Pipeline.init(r.device, glue, "fz_argmax", false);
        r.add_pipe = try mtl.Pipeline.init(r.device, glue, "fz_bcast_add", false);
        r.argids_pipe = try mtl.Pipeline.init(r.device, glue, "fz_argmax_ids", false);
        r.pleids_pipe = try mtl.Pipeline.init(r.device, glue, "fz_ple_ids", false);
        r.copy_pipe = try mtl.Pipeline.init(r.device, glue, "fz_copy_kept", false);
        r.accept_pipe = try mtl.Pipeline.init(r.device, glue, "fz_accept", false);
        r.repack_w = try mtl.Pipeline.init(r.device, glue, "fz_repack_w", false);
        r.repack_s = try mtl.Pipeline.init(r.device, glue, "fz_repack_s", false);
        r.touch_pipe = try mtl.Pipeline.init(r.device, glue, "fz_touch", false);
        r.sink = .{ .b = try r.buffer(64) };
        if (r.xnew) { // the recorded gate/up kernel's header (simd_topk, bsilu) with the full-width expert kernels after it
            var hit: ?[]const u8 = null;
            var it = plan.object.get("variants").?.object.iterator();
            while (it.next()) |kv| if (std.mem.indexOf(u8, kv.key_ptr.*, "qa_expert_gateup") != null) {
                hit = kv.value_ptr.object.get("file").?.string;
            };
            const fp = try std.fmt.allocPrintSentinel(r.arena, "{s}/{s}", .{ dir, hit orelse return error.NoGateup }, 0);
            const ff = try mtl.MappedFile.open(fp);
            const text = ff.bytes[0..ff.size];
            const cut = std.mem.indexOf(u8, text, "[[kernel]]") orelse return error.NoKernel;
            const define: []const u8 = if (r.xpack) "#define FZ_PACKED 1\n" else "#define FZ_PACKED 0\n";
            const full = try std.mem.concat(r.arena, u8, &.{ define, text[0..cut], xnew_source });
            const lib = try mtl.Library.fromSource(r.device, full, mtl.CompileOptions.mlx());
            r.xgu_pipe = try mtl.Pipeline.init(r.device, lib, "fz_xgu", false);
            r.xdown_pipe = try mtl.Pipeline.init(r.device, lib, "fz_xdown", false);
            r.dense8_pipe = try mtl.Pipeline.init(r.device, lib, "fz_dense8", false);
            r.dense16_pipe = try mtl.Pipeline.init(r.device, lib, "fz_dense16", false);
            r.route_pipe = try mtl.Pipeline.init(r.device, lib, "fz_route", false);
            r.xfused_pipe = try mtl.Pipeline.init(r.device, lib, "fz_xfused", false);
            r.xgu_sx_pipe = try mtl.Pipeline.init(r.device, lib, "fz_xgu_sx", false);
            r.xdown_sx_pipe = try mtl.Pipeline.init(r.device, lib, "fz_xdown_sx", false);
            r.xpart = .{ .b = try r.buffer(MAXR * 11 * 4 * D * 4) };
            r.xdone = .{ .b = try r.buffer(MAXR * 11 * 4) };
            r.ggu_pipe = try mtl.Pipeline.init(r.device, lib, "fz_ggu", false);
            r.gdown_pipe = try mtl.Pipeline.init(r.device, lib, "fz_gdown", false);
            r.ul = .{ .b = try r.buffer((1 + 2 * 320 + 320 * 32) * 4) };
        }
    }

    fn strings(a: std.mem.Allocator, v: std.json.Value) ![][]const u8 {
        const out = try a.alloc([]const u8, v.array.items.len);
        for (v.array.items, 0..) |s, i| out[i] = s.string;
        return out;
    }

    fn size3(v: std.json.Value) mtl.Size {
        const a = v.array.items;
        return mtl.Size.of(@intCast(a[0].integer), @intCast(a[1].integer), @intCast(a[2].integer));
    }




    /// FZ_XPACK: an expert set's rows (gate, up, shared gate, shared up, down, shared down) to the packed layout.
    fn repack(r: *Run, ex: []Buf) !void {
        const cb = r.queue.commandBuffer();
        const enc = cb.compute(.concurrent);
        for (0..6) |j| {
            const parts: u32 = if (j < 4) 16 else 4;
            const d = [2]u32{ parts, 5 };
            const words = ex[j * 3].b.length() / 4;
            const rows = words / (parts * 30);
            enc.setPipeline(r.repack_w);
            enc.setBuffer(ex[j * 3].b, ex[j * 3].off, 0);
            enc.setBytes(std.mem.asBytes(&d), 1);
            enc.dispatchThreads(mtl.Size.of(128 * rows, 1, 1), mtl.Size.of(128, 1, 1));
            for (1..3) |k| {
                enc.setPipeline(r.repack_s);
                enc.setBuffer(ex[j * 3 + k].b, ex[j * 3 + k].off, 0);
                enc.setBytes(std.mem.asBytes(&d), 1);
                enc.dispatchThreads(mtl.Size.of(128 * rows, 1, 1), mtl.Size.of(128, 1, 1));
            }
        }
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("repack failed: {s}", .{msg});
            return error.GpuFailed;
        }
    }

    /// FZ_PREFETCH: stream `b` into the cache beside the next launch (no barrier between them).
    fn touch(r: *Run, b: Buf) void {
        if (!r.prefetch) return;
        const n: u32 = @intCast((b.b.length() - b.off) / 16);
        r.enc.setPipeline(r.touch_pipe);
        r.enc.setBuffer(b.b, b.off, 0);
        r.enc.setBuffer(r.sink.b, 0, 1);
        r.enc.setBytes(std.mem.asBytes(&n), 2);
        r.enc.dispatchThreads(mtl.Size.of(256 * 64, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// fz_copy_kept: `layers` copies of `n` words from row (keep + base) of src (row and layer strides in words).
    fn copyKept(r: *Run, src: Buf, dst: Buf, n: usize, row: usize, src_stride: usize, dst_stride: usize, layers: usize, base: i32) void {
        r.enc.setPipeline(r.copy_pipe);
        r.enc.setBuffer(src.b, src.off, 0);
        r.enc.setBuffer(dst.b, dst.off, 1);
        r.enc.setBuffer(r.ar.b, r.ar.off, 2);
        const p = [4]u32{ @intCast(n), @intCast(row), @intCast(src_stride), @intCast(dst_stride) };
        r.enc.setBytes(std.mem.asBytes(&p), 3);
        r.enc.setBytes(std.mem.asBytes(&base), 4);
        r.enc.dispatchThreads(mtl.Size.of(n, layers, 1), mtl.Size.of(256, 1, 1));
        if (!r.serial) r.enc.barrier();
    }
    /// y = x W for `rows` rows of K inputs through fz_dense (blocks of 8 rows; 16 K-slices for narrow outputs).
    fn denseRows(r: *Run, x: Buf, k: usize, l: Lane, rows: usize, y: Buf) void {
        const n = l.wq.b.length() * 4 / (3 * k);
        const narrow = n <= 4096;
        var at: usize = 0;
        while (at < rows) : (at += 8) {
            const rr = @min(8, rows - at);
            const dims = [4]u32{ @intCast(rr), @intCast(n), @intCast(k), 0 };
            r.enc.setPipeline(if (narrow) r.dense16_pipe else r.dense8_pipe);
            r.enc.setBuffer(x.b, x.off + at * k * 2, 0);
            r.enc.setBuffer(l.wq.b, l.wq.off, 1);
            r.enc.setBuffer(l.sbt.b, l.sbt.off, 2);
            r.enc.setBuffer(y.b, y.off + at * n * 2, 3);
            r.enc.setBytes(std.mem.asBytes(&dims), 4);
            const sk: usize = if (narrow) 16 else 8;
            r.enc.dispatchThreads(mtl.Size.of(32 * sk * (n / 32), 1, 1), mtl.Size.of(32 * sk, 1, 1));
            if (!r.serial) r.enc.barrier();
        }
    }
    /// The MoE's routed and shared experts for `rows` rows: gate/up (with routing) then down.
    fn experts(r: *Run, gu_role: []const u8, down_role: []const u8, x: Buf, lg: Buf, e: [18]Buf, act: Buf, pick: Buf, wts: Buf, rows_buf: Buf, y: Buf) !void {
        if (r.skip & (1 << 2) != 0) return;
        if (!r.xnew) {
            try r.call(gu_role, &.{ x, lg, e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11] }, &.{ act, pick, wts });
            try r.call(down_role, &.{ act, pick, e[12], e[13], e[14], e[15], e[16], e[17], rows_buf }, &.{y});
            return;
        }
        if (r.xfused) { // gate/up and down in one launch
            r.enc.setPipeline(r.xfused_pipe);
            const fb = [_]Buf{ x, lg, e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11], e[12], e[13], e[14], e[15], e[16], e[17], pick, wts, y, r.xpart, r.xdone };
            for (fb, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
            r.enc.dispatchThreads(mtl.Size.of(512 * 4, r.rows * 11, 1), mtl.Size.of(512, 1, 1));
            if (!r.serial) r.enc.barrier();
            return;
        }
        if (r.grouped and r.rows > 1) { // each distinct expert read once for every row that picked it
            if (r.gskip & 1 == 0) {
                r.enc.setPipeline(r.route_pipe);
                for ([_]Buf{ lg, rows_buf, pick, wts, r.ul }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
                r.enc.dispatchThreads(mtl.Size.of(32 * r.rows, 1, 1), mtl.Size.of(32 * r.rows, 1, 1));
                if (!r.serial) r.enc.barrier();
            }
            if (r.gskip & 2 != 0) return;
            r.enc.setPipeline(r.ggu_pipe);
            const ggu = [_]Buf{ x, r.ul, e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11], act, rows_buf };
            for (ggu, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
            r.enc.dispatchThreads(mtl.Size.of(128, 80, r.rows * 10 + 1), mtl.Size.of(128, 1, 1));
            if (!r.serial) r.enc.barrier();
            if (r.gskip & 4 != 0) return;
            r.enc.setPipeline(r.gdown_pipe);
            const gdn = [_]Buf{ act, r.ul, e[12], e[13], e[14], e[15], e[16], e[17], rows_buf, y };
            for (gdn, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
            r.enc.dispatchThreads(mtl.Size.of(128, 80, r.rows * 10 + 1), mtl.Size.of(128, 1, 1));
            if (!r.serial) r.enc.barrier();
            return;
        }
        const pairs = r.rows * 11;
        r.enc.setPipeline(if (r.xsx) r.xgu_sx_pipe else r.xgu_pipe);
        const gu = [_]Buf{ x, lg, e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11], act, pick, wts };
        for (gu, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        r.enc.dispatchThreads(mtl.Size.of(128, 80, pairs), mtl.Size.of(128, 1, 1));
        if (!r.serial) r.enc.barrier();
        r.enc.setPipeline(if (r.xsx) r.xdown_sx_pipe else r.xdown_pipe);
        const dn = [_]Buf{ act, pick, e[12], e[13], e[14], e[15], e[16], e[17], rows_buf, y };
        for (dn, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        r.enc.dispatchThreads(mtl.Size.of(128, 80, pairs), mtl.Size.of(128, 1, 1));
        if (!r.serial) r.enc.barrier();
    }

    /// One launch of `role` at the current row count, inputs and outputs in the variant's order.
    /// The launch class of a role, for FZ_PROFILE's knock-outs.
    fn class(role: []const u8) u32 {
        const names = [_][]const u8{ "hc_", "lane_qmm", "expert", "router", "gdn", "attn", "ple", "head" };
        if (std.mem.indexOf(u8, role, "@head") != null) return 1 << 7;
        for (names, 0..) |n, i| if (std.mem.indexOf(u8, role, n) != null) return @as(u32, 1) << @intCast(i);
        return 0;
    }

    fn call(r: *Run, role: []const u8, ins: []const Buf, outs: []const Buf) !void {
        return r.callAs(role, r.rows, ins, outs);
    }

    /// `role` launched as recorded at `as_rows` rows (its kernel reads the row count at run time).
    fn callAs(r: *Run, role: []const u8, as_rows: usize, ins: []const Buf, outs: []const Buf) !void {
        if (r.skip & class(role) != 0) return;
        var key: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&key, "{s}|{d}", .{ role, as_rows });
        const s = r.roles.get(name) orelse {
            std.log.err("no launch site {s}", .{name});
            return error.NoSite;
        };
        const v = s.v;
        if (ins.len != v.inputs.len or outs.len != v.outputs.len) {
            std.log.err("{s}: {d} inputs and {d} outputs given, the kernel takes {d} and {d}", .{ name, ins.len, outs.len, v.inputs.len, v.outputs.len });
            return error.Arity;
        }
        r.enc.setPipeline(v.pipe);
        var at: usize = 0;
        for (v.inputs, ins) |input, b| {
            r.enc.setBuffer(b.b, b.off, at);
            at += 1;
            for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suffix| {
                for (v.meta) |m| {
                    if (m.len == input.len + suffix.len and std.mem.startsWith(u8, m, input) and std.mem.endsWith(u8, m, suffix)) {
                        r.enc.setBuffer(r.shapes.get(m) orelse return error.NoShape, 0, at);
                        at += 1;
                    }
                }
            }
        }
        for (outs) |b| {
            r.enc.setBuffer(b.b, b.off, at);
            at += 1;
        }
        r.enc.dispatchThreads(s.grid, s.tg);
        if (!r.serial) r.enc.barrier();
    }
};

const Hc = struct { scale: Buf, dw: Buf, ds: Buf, db: Buf, uw: Buf, us: Buf, ub: Buf };
const Lane = struct { wq: Buf, sbt: Buf };
const Layer = struct {
    ahc: Hc,
    mhc: Hc,
    linear: bool,
    proj: Lane,
    out: Lane,
    conv: Buf = undefined,
    alog: Buf = undefined,
    dt: Buf = undefined,
    norm: Buf = undefined,
    qn: Buf = undefined,
    kn: Buf = undefined,
    iqn: Buf = undefined,
    router: Buf,
    ex: [18]Buf, // gate, up, shared gate, shared up (weight, scales, biases), then down, shared down
    cs: [2]Buf = undefined,
    so: [2]Buf = undefined,
    keys: Buf = undefined,
    vals: Buf = undefined,
    raw: Buf = undefined,
};

fn hcOf(r: *Run, comptime fmt: []const u8, args: anytype) !Hc {
    var name: [96]u8 = undefined;
    const stem = try std.fmt.bufPrint(&name, fmt, args);
    var full: [128]u8 = undefined;
    const parts = [_][]const u8{ "scale", "down.w", "down.s", "down.b", "up.w", "up.s", "up.b" };
    var out: [7]Buf = undefined;
    for (parts, 0..) |p, i| out[i] = try r.load(try std.fmt.bufPrint(&full, "{s}.{s}", .{ stem, p }));
    return .{ .scale = out[0], .dw = out[1], .ds = out[2], .db = out[3], .uw = out[4], .us = out[5], .ub = out[6] };
}

fn laneOf(r: *Run, comptime fmt: []const u8, args: anytype) !Lane {
    var name: [96]u8 = undefined;
    const stem = try std.fmt.bufPrint(&name, fmt, args);
    var full: [128]u8 = undefined;
    return .{ .wq = try r.load(try std.fmt.bufPrint(&full, "{s}.wq", .{stem})), .sbt = try r.load(try std.fmt.bufPrint(&full, "{s}.sbt", .{stem})) };
}

const Ple = struct {
    kv: Lane,
    ks: Buf,
    qs: Buf,
    cs: Buf,
    conv: Buf,
    starts: Buf,
    tables: [3 * GROUPS]Buf,
    cin: Buf,
    hist: [2]i64,
    eos: i64,
    mult: [3]i64,
    sizes: [16]i64,
    offsets: [16]i64,
};

/// Every intermediate a window of up to MAXR rows writes, and the per-window values the kernels read.
const Tmp = struct {
    h: [2]Buf,
    ssp: Buf,
    part: Buf,
    mixed: Buf,
    inj_a: Buf,
    inj_m: Buf,
    xs: Buf,
    p: Buf,
    gout: Buf,
    branch: Buf,
    lg: Buf,
    act: Buf,
    pick: Buf,
    wts: Buf,
    ydown: Buf,
    q: Buf,
    kout: Buf,
    iq: Buf,
    po: Buf,
    pm: Buf,
    aout: Buf,
    emb: Buf,
    kvp: Buf,
    gated: Buf,
    hout: Buf,
    logits: Buf,
    picks: Buf,
    rows: Buf,
    mdims: Buf,
    eps: Buf,
    ids8: Buf,
    pos8: Buf,
    nk8: Buf,
    zero8: Buf,
    ids81: Buf,
    scale: Buf,
    log2base: Buf,
    ple_ids: Buf,
    ple_meta: Buf,
    kvmeta: Buf,
    vocab: Buf,
};

fn f32Buf(r: *Run, v: f32) !Buf {
    const b = try r.buffer(4);
    b.slice(f32, 1)[0] = v;
    return .{ .b = b };
}

fn i32Buf(r: *Run, vals: []const i32) !Buf {
    const b = try r.buffer(vals.len * 4);
    @memcpy(b.slice(i32, vals.len), vals);
    return .{ .b = b };
}

/// One MTP call's per-row values (every call in a command buffer reads its own).
const Slot = struct { rows: Buf, md: Buf, md4: Buf, ids8: Buf, pos8: Buf, nk8: Buf, kvmeta: Buf, n_add: Buf };

/// The MTP head: its decoder layer and mixer, the input projections, the cut head, its own attention cache.
const Mtp = struct {
    ahc: Hc,
    mhc: Hc,
    mix: Hc,
    proj: Lane,
    out: Lane,
    fce: Lane,
    fch: Lane,
    draft: Lane,
    qn: Buf,
    kn: Buf,
    iqn: Buf,
    enorm: Buf,
    hnorm: Buf,
    router: Buf,
    ids: Buf,
    ids_n: usize,
    ex: [18]Buf,
    keys: Buf,
    vals: Buf,
    raw: Buf,
    pos: usize = 0,
    drafted: usize = 0,
    h: [2]Buf,
    emb: Buf,
    en: Buf,
    e: Buf,
    hn: Buf,
    hs: Buf,
    logits: Buf,
    pick: Buf,
    md1: Buf,
    n_ids: Buf,
    slots: [16]Slot,
    mixsel: Buf = undefined,
    hsel: Buf = undefined,
    last: Buf = undefined, // the last call's output streams, row by row
};

const Model = struct {
    r: *Run,
    layers: [LAYERS]Layer,
    mix: Hc,
    head: Lane,
    embed: [3]Buf,
    ple: Ple,
    t: Tmp,
    pos: usize = 0,
    state: usize = 0, // the DeltaNet buffer pair holding the state
    state_row: usize = 0, // its row
    gpu_seconds: f64 = 0,
    mtp: Mtp = undefined,
    last: Buf = undefined, // the last window's streams before the final mixer

    fn reset(m: *Model) void {
        m.pos = 0;
        m.state = 0;
        m.state_row = 0;
        for (&m.layers) |*L| if (L.linear) {
            @memset(L.cs[0].b.contents()[L.cs[0].off .. L.cs[0].off + CS_ROW], 0);
            @memset(L.so[0].b.contents()[L.so[0].off .. L.so[0].off + SO_ROW], 0);
        };
        m.ple.hist = .{ m.ple.eos, m.ple.eos };
        @memset(m.ple.cin.b.contents()[m.ple.cin.off .. m.ple.cin.off + (PLE_TAIL + MAXR) * WIDE * 2], 0);
    }

    fn lane(m: *Model, x: Buf, k: usize, l: Lane, role: []const u8, y: Buf) !void {
        if (m.r.dense_target) return m.r.denseRows(x, k, l, m.r.rows, y);
        if (!m.r.fused_xsum) try m.r.call(if (k == D) "lane_qmm_xsum#[2560]" else "lane_qmm_xsum#[6144]", &.{ x, m.t.mdims }, &.{m.t.xs});
        try m.r.call(role, &.{ x, m.t.xs, l.wq, l.sbt, m.t.mdims }, &.{y});
    }

    fn hcProject(m: *Model, hn: Buf, hc: Hc, down: []const u8, up: []const u8, inj: Buf) !void {
        const t = &m.t;
        const as_rows = if (m.r.hc_mma and m.r.rows > 1) 8 else m.r.rows;
        try m.r.callAs(down, as_rows, &.{ hn, t.ssp, hc.scale, hc.dw, hc.ds, hc.db, t.eps, t.rows }, &.{t.part});
        try m.r.callAs(up, as_rows, &.{ hn, t.ssp, hc.scale, t.part, hc.uw, hc.us, hc.ub, t.eps, t.rows }, &.{ t.mixed, inj });
    }

    fn grouped(m: *Model, h: Buf, out: Buf) !void {
        const t = &m.t;
        try m.r.call("q4_hc_norm_grouped#[10240]", &.{ h, t.inj_m, t.ydown, t.wts, t.lg }, &.{ out, t.ssp });
    }

    /// The window's n-gram row ids [rows, 16] after the history (NGramEmbedding.ids on [history, tokens]).
    fn pleIds(m: *Model, tokens: []const u32) void {
        const p = &m.ple;
        var seq: [2 + MAXR]i64 = undefined;
        seq[0], seq[1] = .{ p.hist[0], p.hist[1] };
        for (tokens, 0..) |tok, i| seq[2 + i] = tok;
        const out = m.t.ple_ids.b.slice(u32, 16 * MAXR);
        for (0..tokens.len) |row| {
            const at = 2 + row;
            var before: i64 = -1;
            for (0..at) |q| if (seq[q] == p.eos) {
                before = @intCast(q);
            };
            const in_seg = @as(i64, @intCast(at)) - (before + 1);
            var sh: [3]i64 = undefined;
            for (0..3) |s| sh[s] = if (in_seg >= @as(i64, @intCast(s))) seq[at - s] else p.eos;
            for (2..4) |ng| {
                var mixed: i64 = sh[0] *% p.mult[0];
                for (1..ng) |q| mixed ^= sh[q] *% p.mult[q];
                for (0..8) |k| {
                    const hh = (ng - 2) * 8 + k;
                    out[row * 16 + hh] = @intCast(@mod(mixed, p.sizes[hh]) + p.offsets[hh]);
                }
            }
        }
    }

    /// A window's per-row values from the cache position (rows, matmul dims, positions, key counts, kv rows).
    fn windowMeta(m: *Model, rows: usize) void {
        if (m.r.gpu_round) return; // fz_accept wrote them
        const t = &m.t;
        t.rows.b.slice(i32, 1)[0] = @intCast(rows);
        t.mdims.b.slice(i32, 2)[0] = @intCast(rows);
        const pos8 = t.pos8.b.slice(i32, 8);
        const nk8 = t.nk8.b.slice(i32, 8);
        for (0..8) |i| {
            pos8[i] = if (i < rows) @intCast(m.pos + i) else 0;
            nk8[i] = if (i < rows) @intCast(m.pos + i + 1) else 0;
        }
        const kvm = t.kvmeta.b.slice(u32, 3);
        kvm[0], kvm[2] = .{ @intCast(m.pos), @intCast(rows) };
    }

    /// One forward over `tokens` (a window of up to MAXR rows from the cache's position): each row's argmax.
    fn window(m: *Model, tokens: []const u32, picks: []u32) !void {
        const r = m.r;
        const t = &m.t;
        const rows = tokens.len;
        m.windowMeta(rows);
        const ids = t.ids8.b.slice(u32, 8);
        for (0..8) |i| ids[i] = if (i < rows) tokens[i] else 0;
        m.pleIds(tokens);
        const cb = r.queue.commandBuffer();
        r.enc = cb.compute(if (r.serial) .serial else .concurrent);
        try m.windowEncode(rows, t.ids8);
        try m.finish(cb);
        @memcpy(picks[0..rows], t.picks.b.slice(u32, rows));
    }

    fn finish(m: *Model, cb: mtl.CommandBuffer) !void {
        m.r.enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("command buffer failed: {s}", .{msg});
            return error.GpuFailed;
        }
        m.gpu_seconds += cb.gpuSeconds();
    }

    /// The n-gram ids of the window in `ids` hashed on the GPU (the window's drafts never reach the host).
    fn pleIdsGpu(m: *Model, rows: usize, ids: Buf) void {
        const r = m.r;
        const p = &m.ple;
        if (!r.gpu_round) { // in GPU-side rounds fz_accept keeps the history and the rest is set once
            const pm = m.t.ple_meta.b.slice(i64, 39);
            pm[0], pm[1], pm[2], pm[3] = .{ p.hist[0], p.hist[1], p.eos, @intCast(rows) };
            for (0..3) |k| pm[4 + k] = p.mult[k];
            for (0..16) |k| {
                pm[7 + k] = p.sizes[k];
                pm[23 + k] = p.offsets[k];
            }
        }
        r.enc.setPipeline(r.pleids_pipe);
        for ([_]Buf{ ids, m.t.ple_meta, m.t.ple_ids }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        r.enc.dispatchThreads(mtl.Size.of(16 * rows, 1, 1), mtl.Size.of(16 * rows, 1, 1));
        if (!r.serial) r.enc.barrier();
    }

    /// Encode the window's forward (tokens read from `ids`, n-gram ids already in t.ple_ids) and its argmax.
    fn windowEncode(m: *Model, rows: usize, ids: Buf) !void {
        const r = m.r;
        const t = &m.t;
        r.rows = rows;
        var cur: usize = 0;
        try r.call("qa_embed_rows@embed", &.{ ids, m.embed[0], m.embed[1], m.embed[2] }, &.{t.h[0]});
        var pending: enum { none, grouped } = .none;
        for (0..LAYERS) |i| {
            const L = &m.layers[i];
            if (i == 1) { // the PLE layer: write the pending MoE back, then the n-gram gate and conv
                try m.grouped(t.h[cur], t.h[1 - cur]);
                cur = 1 - cur;
                pending = .none;
                const p = &m.ple;
                var tabs: [2 + 3 * GROUPS]Buf = undefined;
                tabs[0] = t.ple_ids;
                tabs[1] = p.starts;
                for (0..3 * GROUPS) |j| tabs[2 + j] = p.tables[j];
                try r.call("qa_ple_lookup@ple", &tabs, &.{t.emb});
                try m.lane(t.emb, D, p.kv, "lane_qmm_bytes_grouped@ple.kv", t.kvp);
                try r.call("q4_ple_gate@ple", &.{ t.kvp, t.h[cur], p.ks, p.qs, p.cs, t.eps }, &.{ t.gated, .{ .b = p.cin.b, .off = PLE_TAIL * WIDE * 2 } });
                try r.call("q4_ple_conv@ple", &.{ p.cin, p.conv, t.gated, t.h[cur] }, &.{t.hout});
                try r.call("q4_hc_norm_none#[10240]", &.{t.hout}, &.{ t.h[1 - cur], t.ssp });
            } else if (pending == .none) {
                try r.call("q4_hc_norm_none#[10240]", &.{t.h[cur]}, &.{ t.h[1 - cur], t.ssp });
            } else {
                try m.grouped(t.h[cur], t.h[1 - cur]);
            }
            cur = 1 - cur;
            try m.hcProject(t.h[cur], L.ahc, "qa_hc_down@ahc", "qa_hc_up@ahc", t.inj_a);
            if (L.linear) {
                try m.lane(t.mixed, D, L.proj, "lane_qmm_bytes_grouped@gdn.in", t.p);
                const a = m.state;
                const cs_in: Buf = .{ .b = L.cs[a].b, .off = L.cs[a].off + m.state_row * CS_ROW };
                const so_in: Buf = .{ .b = L.so[a].b, .off = L.so[a].off + m.state_row * SO_ROW };
                r.touch(L.out.wq);
                r.touch(L.out.sbt);
                try r.callAs("q4_gdn@gdn", if (r.gdn_step and rows > 1) 8 else rows, &.{ t.p, cs_in, so_in, L.conv, L.alog, L.dt, L.norm, t.eps, t.rows }, &.{ t.gout, L.cs[1 - a], L.so[1 - a] });
                try m.lane(t.gout, 6144, L.out, "lane_qmm_bytes_grouped@gdn.out", t.branch);
            } else {
                try m.lane(t.mixed, D, L.proj, "lane_qmm_bytes_grouped@att.proj", t.p);
                r.touch(L.out.wq);
                r.touch(L.out.sbt);
                try r.call("q4_attn_prep@att", &.{ t.p, t.pos8, L.qn, L.kn, L.iqn, t.eps, t.log2base }, &.{ t.q, t.kout, t.iq });
                if (r.skip & (1 << 5) == 0) {
                    r.enc.setPipeline(r.kv_pipe);
                    for ([_]Buf{ t.kout, t.p, L.keys, L.vals, L.raw, t.kvmeta }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
                    r.enc.dispatchThreads(mtl.Size.of(512 * rows, 1, 1), mtl.Size.of(256, 1, 1));
                }
                if (!r.serial) r.enc.barrier();
                try r.call("q4_attn_parts#[24, 256]", &.{ t.q, L.keys, L.vals, t.ids81, t.nk8, t.zero8, t.scale }, &.{ t.po, t.pm });
                try r.call("q4_attn_merge_gate#[24, 16, 256]", &.{ t.po, t.pm, t.p }, &.{t.aout});
                try m.lane(t.aout, 6144, L.out, "lane_qmm_bytes_grouped@att.o", t.branch);
            }
            try r.call("q4_hc_norm_plain#[10240]", &.{ t.h[cur], t.inj_a, t.branch }, &.{ t.h[1 - cur], t.ssp });
            cur = 1 - cur;
            r.touch(L.router);
            try m.hcProject(t.h[cur], L.mhc, "qa_hc_down@mhc", "qa_hc_up@mhc", t.inj_m);
            try r.call("q4_router_float@moe", &.{ t.mixed, L.router, t.rows }, &.{t.lg});
            try r.experts("qa_expert_gateup@moe.gate", "qa_expert_down_y@moe.down", t.mixed, t.lg, L.ex, t.act, t.pick, t.wts, t.rows, t.ydown);
            if (r.probe) |pb| r.copyKept(t.pick, .{ .b = pb.b, .off = pb.off + i * MAXR * 10 * 4 }, rows * 10, 0, 0, 0, 1, -1);
            pending = .grouped;
        }
        try m.grouped(t.h[cur], t.h[1 - cur]);
        cur = 1 - cur;
        m.last = t.h[cur];
        try m.hcProject(t.h[cur], m.mix, "qa_hc_down@mix", "qa_hc_up@mix", t.inj_a);
        try m.lane(t.mixed, D, m.head, "lane_qmm_bytes_grouped@head", t.logits);
        r.enc.setPipeline(r.argmax_pipe);
        r.enc.setBuffer(t.logits.b, 0, 0);
        r.enc.setBuffer(t.picks.b, 0, 1);
        r.enc.setBuffer(t.vocab.b, 0, 2);
        r.enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
        if (!r.serial) r.enc.barrier();
    }

    /// The MTP head on `rows` rows: each row's next token and the streams it follows (target or head output, row
    /// by row from `streams`); returns the draft after the last row (the cut head's argmax through its ids).
    fn mtpRun(m: *Model, nexts: []const u32, streams: Buf) !u32 {
        const r = m.r;
        const h = &m.mtp;
        const rows = nexts.len;
        h.pos -= h.drafted;
        h.drafted = 0;
        const ids = h.slots[0].ids8.b.slice(u32, 8);
        for (0..8) |i| ids[i] = if (i < rows) nexts[i] else 0;
        const cb = r.queue.commandBuffer();
        r.enc = cb.compute(if (r.serial) .serial else .concurrent);
        try m.mtpEncode(0, rows, h.slots[0].ids8, streams, h.pick);
        try m.finish(cb);
        h.pos += rows;
        return h.pick.b.slice(u32, 1)[0];
    }

    /// Encode the MTP head at its position on `rows` rows (tokens from `ids`), its draft written to `out`; meta in
    /// `slot` (each call in one command buffer has its own).
    fn mtpEncode(m: *Model, slot: usize, rows: usize, ids: Buf, streams: Buf, out: Buf) !void {
        const r = m.r;
        const h = &m.mtp;
        const sl = &h.slots[slot];
        if (!r.gpu_round) try m.mtpMeta(sl, rows);
        try m.mtpLayer(sl, rows, ids, streams, out);
    }

    fn mtpMeta(m: *Model, sl: *Slot, rows: usize) !void {
        const h = &m.mtp;
        sl.rows.b.slice(i32, 1)[0] = @intCast(rows);
        sl.md.b.slice(i32, 2)[0] = @intCast(rows);
        sl.md4.b.slice(i32, 2)[0] = @intCast(4 * rows);
        sl.md4.b.slice(i32, 2)[1] = @intCast(16 * ((4 * rows + 15) / 16)); // rows padded to whole 16-row tiles
        const pos8 = sl.pos8.b.slice(i32, 8);
        const nk8 = sl.nk8.b.slice(i32, 8);
        for (0..8) |i| {
            pos8[i] = if (i < rows) @intCast(h.pos + i) else 0;
            nk8[i] = if (i < rows) @intCast(h.pos + i + 1) else 0;
        }
        const kvm = sl.kvmeta.b.slice(u32, 3);
        kvm[0], kvm[1], kvm[2] = .{ @intCast(h.pos), CAP, @intCast(rows) };
        sl.n_add.b.slice(u32, 1)[0] = @intCast(rows * WIDE);
    }

    fn mtpLayer(m: *Model, sl: *Slot, rows: usize, ids: Buf, streams: Buf, out: Buf) !void {
        const r = m.r;
        const t = &m.t;
        const h = &m.mtp;
        r.rows = rows;
        try r.call("mtp:qa_embed_rows@embed", &.{ ids, m.embed[0], m.embed[1], m.embed[2] }, &.{h.emb});
        try r.call("mtp:q4_rms_rows@mtp.enorm", &.{ h.emb, h.enorm, t.eps }, &.{h.en});
        if (!r.fused_xsum) try r.call("mtp:lane_qmm_xsum#[2560]", &.{ h.en, sl.md }, &.{t.xs});
        if (r.dense) r.denseRows(h.en, D, h.fce, rows, h.e) else try r.call("mtp:lane_qmm_bytes_grouped@mtp.fce", &.{ h.en, t.xs, h.fce.wq, h.fce.sbt, sl.md }, &.{h.e});
        try r.call("mtp:q4_rms_rows@mtp.hnorm", &.{ streams, h.hnorm, t.eps }, &.{h.hn});
        if (!r.fused_xsum) try r.call("mtp:lane_qmm_xsum#[4R, 2560]", &.{ h.hn, sl.md4 }, &.{t.xs});
        if (r.dense) r.denseRows(h.hn, D, h.fch, 4 * rows, h.hs) else try r.call("mtp:lane_qmm_bytes_grouped@mtp.fch", &.{ h.hn, t.xs, h.fch.wq, h.fch.sbt, sl.md4 }, &.{h.hs});
        r.enc.setPipeline(r.add_pipe);
        for ([_]Buf{ h.e, h.hs, h.h[0], sl.n_add }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        r.enc.dispatchThreads(mtl.Size.of(rows * WIDE, 1, 1), mtl.Size.of(256, 1, 1));
        if (!r.serial) r.enc.barrier();
        try r.call("mtp:q4_hc_norm_none#[10240]", &.{h.h[0]}, &.{ h.h[1], t.ssp });
        const down = [_][]const u8{ "mtp:qa_hc_down@mtp.ahc", "mtp:qa_hc_down@mtp.mhc", "mtp:qa_hc_down@mtp.mix" };
        const up = [_][]const u8{ "mtp:qa_hc_up@mtp.ahc", "mtp:qa_hc_up@mtp.mhc", "mtp:qa_hc_up@mtp.mix" };
        try m.mtpProject(h.h[1], h.ahc, down[0], up[0], t.inj_a, sl.rows);
        if (!r.fused_xsum) try r.call("mtp:lane_qmm_xsum#[2560]", &.{ t.mixed, sl.md }, &.{t.xs});
        if (r.dense) r.denseRows(t.mixed, D, h.proj, rows, t.p) else try r.call("mtp:lane_qmm_bytes_grouped@mtp.att.proj", &.{ t.mixed, t.xs, h.proj.wq, h.proj.sbt, sl.md }, &.{t.p});
        try r.call("mtp:q4_attn_prep@mtp.att", &.{ t.p, sl.pos8, h.qn, h.kn, h.iqn, t.eps, t.log2base }, &.{ t.q, t.kout, t.iq });
        r.enc.setPipeline(r.kv_pipe);
        for ([_]Buf{ t.kout, t.p, h.keys, h.vals, h.raw, sl.kvmeta }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        r.enc.dispatchThreads(mtl.Size.of(512 * rows, 1, 1), mtl.Size.of(256, 1, 1));
        if (!r.serial) r.enc.barrier();
        try r.call("mtp:q4_attn_parts#[24, 256]", &.{ t.q, h.keys, h.vals, t.ids81, sl.nk8, t.zero8, t.scale }, &.{ t.po, t.pm });
        try r.call("mtp:q4_attn_merge_gate#[24, 16, 256]", &.{ t.po, t.pm, t.p }, &.{t.aout});
        if (!r.fused_xsum) try r.call("mtp:lane_qmm_xsum#[6144]", &.{ t.aout, sl.md }, &.{t.xs});
        if (r.dense) r.denseRows(t.aout, 6144, h.out, rows, t.branch) else try r.call("mtp:lane_qmm_bytes_grouped@mtp.att.o", &.{ t.aout, t.xs, h.out.wq, h.out.sbt, sl.md }, &.{t.branch});
        try r.call("mtp:q4_hc_norm_plain#[10240]", &.{ h.h[1], t.inj_a, t.branch }, &.{ h.h[0], t.ssp });
        try m.mtpProject(h.h[0], h.mhc, down[1], up[1], t.inj_m, sl.rows);
        try r.call("mtp:q4_router_float@mtp.moe", &.{ t.mixed, h.router, sl.rows }, &.{t.lg});
        try r.experts("mtp:qa_expert_gateup@mtp.moe.gate", "mtp:qa_expert_down_y@mtp.moe.down", t.mixed, t.lg, h.ex, t.act, t.pick, t.wts, sl.rows, t.ydown);
        try r.call("mtp:q4_hc_norm_grouped#[10240]", &.{ h.h[0], t.inj_m, t.ydown, t.wts, t.lg }, &.{ h.h[1], t.ssp });
        h.last = .{ .b = h.h[1].b, .off = (rows - 1) * WIDE * 2 };
        try m.mtpProject(h.h[1], h.mix, down[2], up[2], t.inj_a, sl.rows);
        var x: Buf = .{ .b = t.mixed.b, .off = (rows - 1) * D * 2 };
        if (r.gpu_round and rows > 1) { // the kept row's mix and streams, chosen on the GPU
            r.copyKept(t.mixed, h.mixsel, D / 2, D / 2, 0, 0, 1, -1);
            r.copyKept(h.h[1], h.hsel, WIDE / 2, WIDE / 2, 0, 0, 1, -1);
            x = h.mixsel;
            h.last = h.hsel;
        }
        r.rows = 1;
        if (!r.fused_xsum) try r.call("mtp:lane_qmm_xsum#[2560]", &.{ x, h.md1 }, &.{t.xs});
        if (r.dense) r.denseRows(x, D, h.draft, 1, h.logits) else try r.call("mtp:lane_qmm_bytes_grouped@mtp.draft", &.{ x, t.xs, h.draft.wq, h.draft.sbt, h.md1 }, &.{h.logits});
        r.enc.setPipeline(r.argids_pipe);
        for ([_]Buf{ h.logits, out, h.n_ids, h.ids }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        r.enc.dispatchThreads(mtl.Size.of(1024, 1, 1), mtl.Size.of(1024, 1, 1));
        if (!r.serial) r.enc.barrier();
    }

    fn mtpProject(m: *Model, hn: Buf, hc: Hc, down: []const u8, up: []const u8, inj: Buf, rows: Buf) !void {
        const t = &m.t;
        try m.r.call(down, &.{ hn, t.ssp, hc.scale, hc.dw, hc.ds, hc.db, t.eps, rows }, &.{t.part});
        try m.r.call(up, &.{ hn, t.ssp, hc.scale, t.part, hc.uw, hc.us, hc.ub, t.eps, rows }, &.{ t.mixed, inj });
    }

    /// Absorb `nexts.len` rows (chunks of up to MAXR) from `streams`; returns the draft after the last row.
    fn mtpAbsorb(m: *Model, nexts: []const u32, streams: Buf) !u32 {
        var at: usize = 0;
        var d: u32 = 0;
        while (at < nexts.len) {
            const n = @min(MAXR, nexts.len - at);
            d = try m.mtpRun(nexts[at .. at + n], .{ .b = streams.b, .off = streams.off + at * WIDE * 2 });
            at += n;
        }
        return d;
    }

    /// One chained draft after `draft`, from the head's last output row; its cache entry is trimmed next absorb.
    fn mtpChain(m: *Model, draft: u32) !u32 {
        const prev = m.mtp.last;
        const rows_before = m.mtp.drafted;
        m.mtp.drafted = 0;
        const d = try m.mtpRun(&.{draft}, prev);
        m.mtp.drafted = rows_before + 1;
        return d;
    }

    /// Keep the window's first `keep` rows: the DeltaNet state of row keep-1, the n-gram history and conv tail.
    fn keepRows(m: *Model, tokens: []const u32, keep: usize) void {
        m.state = 1 - m.state;
        m.state_row = keep - 1;
        m.pos += keep;
        const cin = m.ple.cin.b.contents();
        std.mem.copyForwards(u8, cin[0 .. PLE_TAIL * WIDE * 2], cin[keep * WIDE * 2 .. (keep + PLE_TAIL) * WIDE * 2]);
        for (tokens[0..keep]) |tok| m.ple.hist = .{ m.ple.hist[1], tok };
    }
};

/// Copy lanes: the longest suffix of `hist` (`min`..8 tokens) seen earlier; the tokens after its latest earlier
/// occurrence go into `out`. Returns how many (0 when nothing matches).
fn copyDrafts(hist: []const u32, min: usize, out: []u32) usize {
    const n = hist.len;
    if (n < min + 1) return 0;
    var len: usize = @min(8, n - 1);
    while (len >= min) : (len -= 1) {
        const suffix = hist[n - len ..];
        var end: usize = n - 1;
        while (end >= len) : (end -= 1) {
            if (std.mem.eql(u32, hist[end - len .. end], suffix)) {
                const k = @min(out.len, n - end);
                @memcpy(out[0..k], hist[end .. end + k]);
                return k;
            }
        }
    }
    return 0;
}

fn jsonInt(v: std.json.Value) i64 {
    return switch (v) {
        .integer => |x| x,
        .float => |x| @intFromFloat(x),
        else => 0,
    };
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: tf-flashnext-run MODEL_DIR DUMP_DIR\n", .{});
        std.process.exit(2);
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const device = try mtl.Device.init();
    var r = Run{ .arena = arena, .device = device, .queue = try device.queue() };
    r.fused_xsum = std.c.getenv("FZ_FUSED_XSUM") != null;
    r.serial = std.c.getenv("FZ_SERIAL") != null;
    r.xnew = std.c.getenv("FZ_XNEW") != null;
    r.split = std.c.getenv("FZ_SPLIT") != null;
    r.gdn_step = std.c.getenv("FZ_GDN_STEP") != null;
    r.hc_mma = std.c.getenv("FZ_HC_MMA") != null;
    r.event = try device.sharedEvent();
    r.dense = r.xnew and std.c.getenv("FZ_DENSE") != null;
    r.dense_target = r.dense and std.c.getenv("FZ_DENSE_TARGET") != null;
    r.xpack = r.xnew and std.c.getenv("FZ_XPACK") != null;
    r.grouped = r.xnew and !r.xpack and std.c.getenv("FZ_GROUPED") != null;
    r.xfused = r.xnew and !r.xpack and !r.grouped and std.c.getenv("FZ_XFUSED") != null;
    r.xsx = r.xnew and std.c.getenv("FZ_XSX") != null;
    r.copy = std.c.getenv("FZ_COPY") != null;
    r.prefetch = !r.serial and std.c.getenv("FZ_PREFETCH") != null;
    const t0 = mtl.clock.seconds();
    try r.compile(args[2]);
    const t1 = mtl.clock.seconds();

    const index_file = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(arena, "{s}/model.safetensors.index.json", .{args[1]}, 0));
    const index = try std.json.parseFromSliceLeaky(std.json.Value, arena, index_file.bytes[0..index_file.size], .{});
    var files: std.StringHashMapUnmanaged(void) = .empty;
    var wit = index.object.get("weight_map").?.object.iterator();
    while (wit.next()) |kv| try files.put(arena, kv.value_ptr.string, {});
    var fit = files.keyIterator();
    while (fit.next()) |name| try r.indexFile(try std.fmt.allocPrintSentinel(arena, "{s}/{s}", .{ args[1], name.* }, 0));
    try r.indexFile(try std.fmt.allocPrintSentinel(arena, "{s}/pack.safetensors", .{args[2]}, 0));

    const own_prompt = std.c.getenv("FZ_REF") != null; // another prompt: passes against Python's records are skipped
    const ref_path = if (std.c.getenv("FZ_REF")) |p| try std.fmt.allocPrintSentinel(arena, "{s}", .{std.mem.span(p)}, 0) else try std.fmt.allocPrintSentinel(arena, "{s}/ref.json", .{args[2]}, 0);
    const ref_file = try mtl.MappedFile.open(ref_path);
    const ref = try std.json.parseFromSliceLeaky(std.json.Value, arena, ref_file.bytes[0..ref_file.size], .{});
    const ple_ref = ref.object.get("ple").?.object;

    const m = try arena.create(Model);
    m.* = .{ .r = &r, .layers = undefined, .mix = undefined, .head = undefined, .embed = undefined, .ple = undefined, .t = undefined };
    for (0..LAYERS) |i| {
        const linear = i % 4 != 3;
        var L: Layer = .{ .ahc = try hcOf(&r, "L{d}.ahc", .{i}), .mhc = try hcOf(&r, "L{d}.mhc", .{i}), .linear = linear, .proj = undefined, .out = undefined, .router = try r.loadf("L{d}.moe.router", .{i}), .ex = undefined };
        if (linear) {
            L.proj = try laneOf(&r, "L{d}.gdn.in", .{i});
            L.out = try laneOf(&r, "L{d}.gdn.out", .{i});
            L.conv = try r.loadf("L{d}.gdn.conv", .{i});
            L.alog = try r.loadf("L{d}.gdn.alog", .{i});
            L.dt = try r.loadf("L{d}.gdn.dt", .{i});
            L.norm = try r.loadf("L{d}.gdn.norm", .{i});
            for (0..2) |j| {
                L.cs[j] = .{ .b = try r.buffer(MAXR * CS_ROW) };
                L.so[j] = .{ .b = try r.buffer(MAXR * SO_ROW) };
            }
        } else {
            L.proj = try laneOf(&r, "L{d}.att.proj", .{i});
            L.out = try laneOf(&r, "L{d}.att.o", .{i});
            L.qn = try r.loadf("L{d}.att.qn", .{i});
            L.kn = try r.loadf("L{d}.att.kn", .{i});
            L.iqn = try r.loadf("L{d}.att.iqn", .{i});
            L.keys = .{ .b = try r.buffer(2 * CAP * 256 * 2) };
            L.vals = .{ .b = try r.buffer(2 * CAP * 256 * 2) };
            L.raw = .{ .b = try r.buffer(CAP * 128 * 2) };
        }
        const projs = [_][]const u8{ "switch_mlp.gate_proj", "switch_mlp.up_proj", "shared_expert.gate_proj", "shared_expert.up_proj", "switch_mlp.down_proj", "shared_expert.down_proj" };
        for (projs, 0..) |proj, j| {
            for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |suffix, k| {
                L.ex[j * 3 + k] = try r.loadf("language_model.model.layers.{d}.mlp.{s}.{s}", .{ i, proj, suffix });
            }
        }
        if (r.xpack) try r.repack(&L.ex);
        m.layers[i] = L;
    }
    m.mix = try hcOf(&r, "mix", .{});
    m.head = try laneOf(&r, "head", .{});
    m.embed = .{ try r.load("language_model.model.embed_tokens.weight"), try r.load("language_model.model.embed_tokens.scales"), try r.load("language_model.model.embed_tokens.biases") };
    m.ple = .{
        .kv = try laneOf(&r, "ple.kv", .{}),
        .ks = try r.load("ple.ks"),
        .qs = try r.load("ple.qs"),
        .cs = try r.load("ple.cs"),
        .conv = try r.load("ple.conv"),
        .starts = try r.load("ple.starts"),
        .tables = undefined,
        .cin = .{ .b = try r.buffer((PLE_TAIL + MAXR) * WIDE * 2) },
        .hist = undefined,
        .eos = jsonInt(ple_ref.get("eos").?),
        .mult = undefined,
        .sizes = undefined,
        .offsets = undefined,
    };
    for (0..3) |k| m.ple.mult[k] = jsonInt(ple_ref.get("multipliers").?.array.items[k]);
    for (0..16) |k| {
        m.ple.sizes[k] = jsonInt(ple_ref.get("sizes").?.array.items[k]);
        m.ple.offsets[k] = jsonInt(ple_ref.get("offsets").?.array.items[k]);
    }
    for (0..GROUPS) |g| {
        m.ple.tables[3 * g + 0] = try r.group(16 * g, 16, "weight");
        m.ple.tables[3 * g + 1] = try r.group(16 * g, 16, "scales");
        m.ple.tables[3 * g + 2] = try r.group(16 * g, 16, "biases");
    }
    const B = struct {
        fn of(rr: *Run, n: usize) !Buf {
            return .{ .b = try rr.buffer(n) };
        }
    };
    m.t = .{
        .h = .{ try B.of(&r, MAXR * WIDE * 2), try B.of(&r, MAXR * WIDE * 2) },
        .ssp = try B.of(&r, MAXR * 10 * 4 * 4),
        .part = try B.of(&r, 10 * MAXR * 324 * 4),
        .mixed = try B.of(&r, MAXR * D * 2),
        .inj_a = try B.of(&r, MAXR * 4 * 2),
        .inj_m = try B.of(&r, MAXR * 4 * 2),
        .xs = try B.of(&r, 192 * 16 * 4),
        .p = try B.of(&r, MAXR * 16480 * 2),
        .gout = try B.of(&r, MAXR * 6144 * 2),
        .branch = try B.of(&r, MAXR * D * 2),
        .lg = try B.of(&r, MAXR * 513 * 4),
        .act = try B.of(&r, MAXR * 11 * 640 * 2),
        .pick = try B.of(&r, MAXR * 10 * 4),
        .wts = try B.of(&r, MAXR * 10 * 4),
        .ydown = try B.of(&r, MAXR * 11 * D * 2),
        .q = try B.of(&r, MAXR * 24 * 256 * 2),
        .kout = try B.of(&r, MAXR * 2 * 256 * 2),
        .iq = try B.of(&r, MAXR * 4 * 128 * 2),
        .po = try B.of(&r, MAXR * 24 * 16 * 256 * 4),
        .pm = try B.of(&r, MAXR * 24 * 16 * 2 * 4),
        .aout = try B.of(&r, MAXR * 6144 * 2),
        .emb = try B.of(&r, MAXR * D * 2),
        .kvp = try B.of(&r, MAXR * (WIDE + D) * 2),
        .gated = try B.of(&r, MAXR * WIDE * 2),
        .hout = try B.of(&r, MAXR * WIDE * 2),
        .logits = try B.of(&r, MAXR * VOCAB * 2),
        .picks = try B.of(&r, MAXR * 4),
        .rows = try i32Buf(&r, &.{1}),
        .mdims = try i32Buf(&r, &.{ 1, 16, 0, 0, 0, 0, 0, 0 }),
        .eps = try r.load("eps"),
        .ids8 = try i32Buf(&r, &(@as([8]i32, @splat(0)))),
        .pos8 = try i32Buf(&r, &(@as([8]i32, @splat(0)))),
        .nk8 = try i32Buf(&r, &(@as([8]i32, @splat(0)))),
        .zero8 = try i32Buf(&r, &(@as([8]i32, @splat(0)))),
        .ids81 = try i32Buf(&r, &(@as([8]i32, @splat(0)))),
        .scale = try f32Buf(&r, @floatCast(ref.object.get("attention_scale").?.float)),
        .log2base = try f32Buf(&r, 23.253496170043945),
        .ple_ids = try i32Buf(&r, &(@as([16 * MAXR]i32, @splat(0)))),
        .ple_meta = try B.of(&r, 39 * 8),
        .kvmeta = try i32Buf(&r, &.{ 0, CAP, 1 }),
        .vocab = try i32Buf(&r, &.{VOCAB}),
    };
    {
        const ids = try r.load("mtp.draft_ids");
        var ex: [18]Buf = undefined;
        const projs = [_][]const u8{ "switch_mlp.gate_proj", "switch_mlp.up_proj", "shared_expert.gate_proj", "shared_expert.up_proj", "switch_mlp.down_proj", "shared_expert.down_proj" };
        for (projs, 0..) |proj, j| {
            for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |suffix, k| {
                ex[j * 3 + k] = try r.loadf("language_model.mtp.layers.0.mlp.{s}.{s}", .{ proj, suffix });
            }
        }
        if (r.xpack) try r.repack(&ex);
        m.mtp = .{
            .ahc = try hcOf(&r, "mtp.ahc", .{}),
            .mhc = try hcOf(&r, "mtp.mhc", .{}),
            .mix = try hcOf(&r, "mtp.mix", .{}),
            .proj = try laneOf(&r, "mtp.att.proj", .{}),
            .out = try laneOf(&r, "mtp.att.o", .{}),
            .fce = try laneOf(&r, "mtp.fce", .{}),
            .fch = try laneOf(&r, "mtp.fch", .{}),
            .draft = try laneOf(&r, "mtp.draft", .{}),
            .qn = try r.load("mtp.att.qn"),
            .kn = try r.load("mtp.att.kn"),
            .iqn = try r.load("mtp.att.iqn"),
            .enorm = try r.load("mtp.enorm.scale"),
            .hnorm = try r.load("mtp.hnorm.scale"),
            .router = try r.load("mtp.moe.router"),
            .ids = ids,
            .ids_n = (try r.entry("mtp.draft_ids")).len / 4,
            .ex = ex,
            .keys = .{ .b = try r.buffer(2 * CAP * 256 * 2) },
            .vals = .{ .b = try r.buffer(2 * CAP * 256 * 2) },
            .raw = .{ .b = try r.buffer(CAP * 128 * 2) },
            .h = .{ try B.of(&r, MAXR * WIDE * 2), try B.of(&r, MAXR * WIDE * 2) },
            .emb = try B.of(&r, MAXR * D * 2),
            .en = try B.of(&r, MAXR * D * 2),
            .e = try B.of(&r, MAXR * D * 2),
            .hn = try B.of(&r, MAXR * WIDE * 2),
            .hs = try B.of(&r, MAXR * WIDE * 2),
            .logits = try B.of(&r, 80000 * 2),
            .pick = try B.of(&r, 16),
            .md1 = try i32Buf(&r, &.{ 1, 16, 0, 0, 0, 0, 0, 0 }),
            .n_ids = undefined,
            .slots = undefined,
        };
        for (&m.mtp.slots) |*sl| sl.* = .{
            .rows = try i32Buf(&r, &.{1}),
            .md = try i32Buf(&r, &.{ 1, 16, 0, 0, 0, 0, 0, 0 }),
            .md4 = try i32Buf(&r, &.{ 4, 16, 0, 0, 0, 0, 0, 0 }),
            .ids8 = try i32Buf(&r, &(@as([8]i32, @splat(0)))),
            .pos8 = try i32Buf(&r, &(@as([8]i32, @splat(0)))),
            .nk8 = try i32Buf(&r, &(@as([8]i32, @splat(0)))),
            .kvmeta = try i32Buf(&r, &.{ 0, CAP, 1 }),
            .n_add = try i32Buf(&r, &.{0}),
        };
        m.mtp.n_ids = try i32Buf(&r, &.{@intCast(m.mtp.ids_n)});
        if (m.mtp.ids_n * 2 > 80000 * 2) return error.DraftVocab;
    }
    try r.shapes.put(arena, "Kc_shape", (try i32Buf(&r, &.{ 1, 2, CAP, 256 })).b);
    try r.shapes.put(arena, "IDS_shape", (try i32Buf(&r, &.{ 8, 1 })).b);
    const t2 = mtl.clock.seconds();
    std.debug.print("compiled in {d:.2} s, loaded {d:.1} GB in {d:.1} s\n", .{ t1 - t0, @as(f64, @floatFromInt(r.loaded)) / 1e9, t2 - t1 });

    if (std.c.getenv("FZ_PROFILE") != null) {
        const names = [_][]const u8{ "none", "hc", "dense", "experts", "router", "gdn", "attn", "ple", "head" };
        var toks: [MAXR]u32 = undefined;
        for (0..MAXR) |i| toks[i] = @intCast(ref.object.get("prompt").?.array.items[i].integer);
        var pk: [MAXR]u32 = undefined;
        m.reset();
        for (0..3) |_| try m.window(toks[0..1], &pk);
        for ([_]usize{ 1, 8 }) |rows| {
            var base: f64 = 0;
            for (names, 0..) |name, c| {
                r.skip = if (c == 0) 0 else @as(u32, 1) << @intCast(c - 1);
                m.gpu_seconds = 0;
                for (0..20) |_| try m.window(toks[0..rows], &pk);
                const ms = m.gpu_seconds * 1e3 / 20;
                if (c == 0) base = ms;
                std.debug.print("rows {d}: without {s:8} {d:6.2} ms GPU  ({d:5.2} ms)\n", .{ rows, name, ms, base - ms });
            }
        }
        r.skip = 0;
        return;
    }
    if (std.c.getenv("FZ_DUAL") != null) { // two lane groups on two queues: one window, two in a row, two at once
        const want0 = ref.object.get("tokens").?.array.items;
        const pr = ref.object.get("prompt").?.array.items;
        var pk: [MAXR]u32 = undefined;
        m.reset();
        for (pr) |x| {
            try m.window(&.{@intCast(x.integer)}, &pk);
            m.keepRows(&.{@intCast(x.integer)}, 1);
        }
        const q2 = try device.queue();
        const ta = m.t;
        var tb = m.t;
        const info = @typeInfo(Tmp).@"struct";
        inline for (info.field_names, info.field_types) |fname, ftype| {
            const f = .{ .name = fname, .type = ftype };
            if (f.type == Buf) {
                const old = @field(ta, f.name);
                const n = old.b.length() - old.off;
                const nb = try r.buffer(n);
                @memcpy(nb.contents()[0..n], old.b.contents()[old.off .. old.off + n]);
                @field(tb, f.name) = .{ .b = nb };
            } else if (f.type == [2]Buf) {
                for (0..2) |k| {
                    const old = @field(ta, f.name)[k];
                    const n = old.b.length() - old.off;
                    @field(tb, f.name)[k] = .{ .b = try r.buffer(n) };
                }
            }
        }
        for ([_]usize{ 1, 4, 8 }) |rows| {
            var toks: [MAXR]u32 = undefined;
            for (0..rows) |i| toks[i] = @intCast(want0[i].integer);
            m.windowMeta(rows);
            const ids = ta.ids8.b.slice(u32, 8);
            for (0..8) |i| ids[i] = if (i < rows) toks[i] else 0;
            m.pleIds(toks[0..rows]);
            var toks_b: [MAXR]u32 = undefined; // the second group: other tokens at the same positions
            for (0..rows) |i| toks_b[i] = @intCast(want0[16 + i].integer);
            const ids_b = tb.ids8.b.slice(u32, 8);
            for (0..8) |i| ids_b[i] = if (i < rows) toks_b[i] else 0;
            m.t = tb;
            m.pleIds(toks_b[0..rows]);
            m.t = ta;
            var wall: [3]f64 = undefined;
            for (0..3) |mode| {
                const n_it: usize = 20;
                const a = mtl.clock.seconds();
                for (0..n_it) |_| {
                    const cb1 = r.queue.commandBuffer();
                    r.enc = cb1.compute(.serial);
                    m.t = ta;
                    try m.windowEncode(rows, ta.ids8);
                    r.enc.end();
                    if (mode == 0) {
                        cb1.commit();
                        cb1.wait();
                        continue;
                    }
                    const cb2 = (if (mode == 1) r.queue else q2).commandBuffer();
                    r.enc = cb2.compute(.serial);
                    m.t = tb;
                    try m.windowEncode(rows, tb.ids8);
                    r.enc.end();
                    cb1.commit();
                    cb2.commit();
                    cb1.wait();
                    cb2.wait();
                }
                wall[mode] = (mtl.clock.seconds() - a) * 1e3 / @as(f64, @floatFromInt(n_it));
            }
            m.t = ta;
            std.debug.print("{d} lanes: one group {d:.2} ms, two groups in a row {d:.2} ms ({d:.2}x), two groups at once {d:.2} ms ({d:.2}x)\n", .{ rows, wall[0], wall[1], wall[1] / wall[0], wall[2], wall[2] / wall[0] });
        }
        return;
    }
    if (std.c.getenv("FZ_GCHECK") != null) { // grouped or fused experts against fz_xgu/fz_xdown: every row's logits, bit for bit
        const want0 = ref.object.get("tokens").?.array.items;
        const pr = ref.object.get("prompt").?.array.items;
        var pk: [MAXR]u32 = undefined;
        m.reset();
        for (pr) |x| {
            try m.window(&.{@intCast(x.integer)}, &pk);
            m.keepRows(&.{@intCast(x.integer)}, 1);
        }
        const keep = try gpa.alloc(u16, MAXR * VOCAB);
        defer gpa.free(keep);
        for (1..MAXR + 1) |rows| {
            var toks: [MAXR]u32 = undefined;
            for (0..rows) |i| toks[i] = @intCast(want0[i].integer);
            const g0, const f0, const x0 = .{ r.grouped, r.xfused, r.xsx };
            r.grouped, r.xfused, r.xsx = .{ false, false, false };
            try m.window(toks[0..rows], &pk);
            @memcpy(keep[0 .. rows * VOCAB], m.t.logits.b.slice(u16, rows * VOCAB));
            r.grouped, r.xfused, r.xsx = .{ g0, f0, x0 };
            try m.window(toks[0..rows], &pk);
            const now = m.t.logits.b.slice(u16, rows * VOCAB);
            var diff: usize = 0;
            for (keep[0 .. rows * VOCAB], now) |a, b| diff += @intFromBool(a != b);
            std.debug.print("rows {d}: {d} of {d} logits differ\n", .{ rows, diff, rows * VOCAB });
        }
        return;
    }
    if (std.c.getenv("FZ_XTIME") != null) { // the experts' share of a window: chain rows against one token repeated
        const want0 = ref.object.get("tokens").?.array.items;
        var pk: [MAXR]u32 = undefined;
        m.reset();
        for (0..3) |_| try m.window(&.{@intCast(want0[0].integer)}, &pk);
        if (std.c.getenv("FZ_GPARTS") != null) for ([_]usize{ 2, 8 }) |rows| {
            var toks: [MAXR]u32 = undefined;
            for (0..rows) |i| toks[i] = @intCast(want0[i].integer);
            for ([_]u32{ 0, 1, 4, 6 }) |gs| {
                r.gskip = gs;
                m.gpu_seconds = 0;
                for (0..30) |_| try m.window(toks[0..rows], &pk);
                std.debug.print("rows {d} gskip {d}: window {d:6.2} ms\n", .{ rows, gs, m.gpu_seconds * 1e3 / 30 });
            }
            r.gskip = 0;
        };
        for ([_]usize{ 1, 2, 4, 8 }) |rows| {
            for ([_]bool{ false, true }) |same| {
                var toks: [MAXR]u32 = undefined;
                for (0..rows) |i| toks[i] = @intCast(want0[if (same) 0 else i].integer);
                var ms: [2]f64 = undefined;
                for ([_]u32{ 0, 1 << 2 }, 0..) |skip, j| {
                    r.skip = skip;
                    m.gpu_seconds = 0;
                    for (0..30) |_| try m.window(toks[0..rows], &pk);
                    ms[j] = m.gpu_seconds * 1e3 / 30;
                }
                r.skip = 0;
                std.debug.print("rows {d} {s}: window {d:6.2} ms, experts {d:5.2} ms\n", .{ rows, if (same) "one token" else "chain    ", ms[0], ms[0] - ms[1] });
            }
        }
        return;
    }
    if (std.c.getenv("FZ_OVERLAP") != null) { // how many experts a window's rows share, layer by layer
        const pr = ref.object.get("prompt").?.array.items;
        const want0 = ref.object.get("tokens").?.array.items;
        r.ar = .{ .b = try r.buffer(64) };
        r.ar.b.slice(i32, 1)[0] = 1;
        const pb: Buf = .{ .b = try r.buffer(LAYERS * MAXR * 10 * 4) };
        var pk: [MAXR]u32 = undefined;
        m.reset();
        for (pr) |x| {
            try m.window(&.{@intCast(x.integer)}, &pk);
            m.keepRows(&.{@intCast(x.integer)}, 1);
        }
        r.probe = pb;
        const picks = pb.b.slice(u32, LAYERS * MAXR * 10);
        const Count = struct {
            fn distinct(sets: []std.AutoHashMap(u32, void)) f64 {
                var n: usize = 0;
                for (sets) |*u| n += u.count();
                return @as(f64, @floatFromInt(n)) / LAYERS;
            }
            fn add(sets: []std.AutoHashMap(u32, void), all: []const u32, rows: usize) !void {
                for (0..LAYERS) |l| for (all[l * MAXR * 10 .. l * MAXR * 10 + rows * 10]) |e| try sets[l].put(e, {});
            }
        };
        var chain_sum: [4]f64 = @splat(0);
        var sib_sum: [4]f64 = @splat(0);
        var far: [LAYERS]std.AutoHashMap(u32, void) = undefined;
        for (&far) |*u| u.* = std.AutoHashMap(u32, void).init(gpa);
        const starts = [_]usize{ 0, 8, 16, 24 };
        for (starts) |s0| {
            for ([_]usize{ 8, 4, 2, 1 }, 0..) |rows, ri| { // chain windows from the pending token, widest first
                var toks: [MAXR]u32 = undefined;
                for (0..rows) |i| toks[i] = @intCast(want0[s0 + i].integer);
                try m.window(toks[0..rows], &pk);
                var sets: [LAYERS]std.AutoHashMap(u32, void) = undefined;
                for (&sets) |*u| u.* = std.AutoHashMap(u32, void).init(gpa);
                try Count.add(&sets, picks, rows);
                chain_sum[3 - ri] += Count.distinct(&sets);
                if (rows == 1) try Count.add(&far, picks, 1);
            }
            // the target's 8 best tokens for the next position, each a row from the same state (tree siblings)
            const lg = m.t.logits.b.slice(u16, VOCAB);
            var best: [8]u32 = undefined;
            var bestv: [8]f32 = @splat(-std.math.inf(f32));
            for (lg, 0..) |raw, id| {
                const v: f32 = @bitCast(@as(u32, raw) << 16);
                if (v <= bestv[7]) continue;
                var k: usize = 7;
                while (k > 0 and v > bestv[k - 1]) : (k -= 1) {
                    bestv[k] = bestv[k - 1];
                    best[k] = best[k - 1];
                }
                bestv[k] = v;
                best[k] = @intCast(id);
            }
            m.keepRows(&.{@intCast(want0[s0].integer)}, 1);
            var sib: [LAYERS]std.AutoHashMap(u32, void) = undefined;
            for (&sib) |*u| u.* = std.AutoHashMap(u32, void).init(gpa);
            for (best, 0..) |cand, ci| {
                try m.window(&.{cand}, &pk);
                try Count.add(&sib, picks, 1);
                const at: ?usize = switch (ci) {
                    0 => 0,
                    1 => 1,
                    3 => 2,
                    7 => 3,
                    else => null,
                };
                if (at) |j| sib_sum[j] += Count.distinct(&sib);
            }
            var toks: [MAXR]u32 = undefined;
            for (0..7) |i| toks[i] = @intCast(want0[s0 + 1 + i].integer);
            try m.window(toks[0..7], &pk);
            m.keepRows(toks[0..7], 7);
        }
        const nstarts: f64 = @floatFromInt(starts.len);
        for ([_]usize{ 1, 2, 4, 8 }, 0..) |rows, j| {
            std.debug.print("{d} rows (of {d} picks a layer): chain {d:.1} distinct, siblings {d:.1}\n", .{ rows, rows * 10, chain_sum[j] / nstarts, sib_sum[j] / nstarts });
        }
        std.debug.print("4 rows 8 tokens apart: {d:.1} distinct\n", .{Count.distinct(&far)});
        return;
    }
    const prompt = ref.object.get("prompt").?.array.items;
    const want = ref.object.get("tokens").?.array.items;
    var pick: [MAXR]u32 = undefined;

    // 1. one-row greedy steps against the Python engine's tokens
    m.reset();
    for (prompt) |tok| {
        try m.window(&.{@intCast(tok.integer)}, &pick);
        m.keepRows(&.{@intCast(tok.integer)}, 1);
    }
    var got: std.ArrayList(u32) = .empty;
    try got.append(gpa, pick[0]);
    const t3 = mtl.clock.seconds();
    m.gpu_seconds = 0;
    while (got.items.len < want.len) {
        const last = got.items[got.items.len - 1];
        try m.window(&.{last}, &pick);
        m.keepRows(&.{last}, 1);
        try got.append(gpa, pick[0]);
    }
    const t4 = mtl.clock.seconds();
    var same: usize = 0;
    while (same < want.len and got.items[same] == @as(u32, @intCast(want[same].integer))) same += 1;
    const steps: f64 = @floatFromInt(want.len - 1);
    std.debug.print("one row: {d}/{d} tokens equal to Python's; {d:.1} tok/s ({d:.2} ms a step, GPU {d:.2} ms)\n", .{ same, want.len, steps / (t4 - t3), (t4 - t3) * 1e3 / steps, m.gpu_seconds * 1e3 / steps });

    // 2. the Python engine's drafted windows, every row's pick, with rollback
    m.reset();
    for (prompt) |tok| {
        try m.window(&.{@intCast(tok.integer)}, &pick);
        m.keepRows(&.{@intCast(tok.integer)}, 1);
    }
    var bad: usize = 0;
    var total_rows: usize = 0;
    const rounds = if (own_prompt) &[_]std.json.Value{} else ref.object.get("rounds").?.array.items;
    for (rounds, 0..) |round, ri| {
        const o = round.object;
        const win = o.get("window").?.array.items;
        var tokens: [MAXR]u32 = undefined;
        for (win, 0..) |x, i| tokens[i] = @intCast(x.integer);
        try m.window(tokens[0..win.len], &pick);
        const exp = o.get("picks").?.array.items;
        for (exp, 0..) |x, i| {
            total_rows += 1;
            if (pick[i] != @as(u32, @intCast(x.integer))) {
                bad += 1;
                if (bad <= 5) std.debug.print("round {d} ({d} rows) row {d}: got {d}, Python {d}\n", .{ ri, win.len, i, pick[i], x.integer });
            }
        }
        m.keepRows(tokens[0..win.len], @intCast(o.get("keep").?.integer));
    }
    std.debug.print("drafted windows: {d} rounds, {d}/{d} rows equal to Python's\n", .{ rounds.len, total_rows - bad, total_rows });

    const ref_tokens: []const u32 = got.items;
    // 4. the MTP head against the Python engine's drafts (absorb windows of 1-8 rows, then a chain)
    if (!own_prompt) {
        const mref = ref.object.get("mtp").?.object;
        var seq: std.ArrayList(u32) = .empty;
        for (prompt) |x| try seq.append(gpa, @intCast(x.integer));
        for (want[0..16]) |x| try seq.append(gpa, @intCast(x.integer));
        const all = try r.buffer(seq.items.len * WIDE * 2);
        m.reset();
        for (seq.items, 0..) |tok, i| {
            try m.window(&.{tok}, &pick);
            m.keepRows(&.{tok}, 1);
            @memcpy(all.contents()[i * WIDE * 2 .. (i + 1) * WIDE * 2], m.last.b.contents()[0 .. WIDE * 2]);
        }
        m.mtp.pos = 0;
        m.mtp.drafted = 0;
        var ok: usize = 0;
        var n: usize = 0;
        var d: u32 = 0;
        for (mref.get("absorb").?.array.items) |a| {
            const start: usize = @intCast(a.object.get("start").?.integer);
            const nx = a.object.get("next").?.array.items;
            var nexts: [MAXR]u32 = undefined;
            for (nx, 0..) |x, i| nexts[i] = @intCast(x.integer);
            d = try m.mtpRun(nexts[0..nx.len], .{ .b = all, .off = start * WIDE * 2 });
            n += 1;
            if (d == @as(u32, @intCast(a.object.get("draft").?.integer))) ok += 1 else std.debug.print("absorb of {d} rows: draft {d}, Python {d}\n", .{ nx.len, d, a.object.get("draft").?.integer });
        }
        for (mref.get("chain").?.array.items) |x| {
            d = try m.mtpChain(d);
            n += 1;
            if (d == @as(u32, @intCast(x.integer))) ok += 1 else std.debug.print("chained draft {d}, Python {d}\n", .{ d, x.integer });
        }
        std.debug.print("MTP drafts equal to Python's: {d}/{d}\n", .{ ok, n });
    }

    // 5. one stream with MTP chains of a fixed depth: tokens against the one-row reference, lanes landed, tok/s
    const depths5: []const usize = if (own_prompt) &.{} else &.{ 1, 2, 3, 4, 6 };
    for (depths5) |depth| {
        m.reset();
        m.mtp.pos = 0;
        m.mtp.drafted = 0;
        const all = try r.buffer(prompt.len * WIDE * 2);
        var ptoks: std.ArrayList(u32) = .empty;
        for (prompt, 0..) |x, i| {
            const tok: u32 = @intCast(x.integer);
            try ptoks.append(gpa, tok);
            try m.window(&.{tok}, &pick);
            m.keepRows(&.{tok}, 1);
            @memcpy(all.contents()[i * WIDE * 2 .. (i + 1) * WIDE * 2], m.last.b.contents()[0 .. WIDE * 2]);
        }
        var out: std.ArrayList(u32) = .empty;
        try out.append(gpa, pick[0]);
        const s0 = mtl.clock.seconds();
        var nexts: std.ArrayList(u32) = .empty;
        try nexts.appendSlice(gpa, ptoks.items[1..]);
        try nexts.append(gpa, pick[0]);
        var drafts: [MAXR]u32 = undefined;
        drafts[0] = try m.mtpAbsorb(nexts.items, .{ .b = all });
        for (1..depth) |j| drafts[j] = try m.mtpChain(drafts[j - 1]);
        var n_rounds: usize = 0;
        var landed: usize = 0;
        while (out.items.len < want.len) {
            var win: [MAXR]u32 = undefined;
            win[0] = out.items[out.items.len - 1];
            @memcpy(win[1 .. depth + 1], drafts[0..depth]);
            try m.window(win[0 .. depth + 1], &pick);
            var keep: usize = 1;
            while (keep <= depth and win[keep] == pick[keep - 1]) keep += 1;
            try out.appendSlice(gpa, pick[0..keep]);
            n_rounds += 1;
            landed += keep - 1;
            m.keepRows(win[0 .. depth + 1], keep);
            drafts[0] = try m.mtpAbsorb(pick[0..keep], m.last);
            for (1..depth) |j| drafts[j] = try m.mtpChain(drafts[j - 1]);
        }
        const wall = mtl.clock.seconds() - s0;
        var eq: usize = 0;
        while (eq < want.len and out.items[eq] == (if (r.xnew) ref_tokens[eq] else @as(u32, @intCast(want[eq].integer)))) eq += 1;
        std.debug.print("depth {d}: {d}/{d} tokens equal; {d} rounds, {d:.2} tokens a round, {d:.2} of {d} drafts landing; {d:.1} tok/s\n", .{ depth, eq, want.len, n_rounds, @as(f64, @floatFromInt(out.items.len - 1)) / @as(f64, @floatFromInt(n_rounds)), @as(f64, @floatFromInt(landed)) / @as(f64, @floatFromInt(n_rounds)), depth, @as(f64, @floatFromInt(out.items.len - 1)) / wall });
        if (eq < want.len) bad += 1;
    }

    // 6. one command buffer a round: the head absorbs the kept rows and chains its drafts into the next window's
    //    token slots on the GPU, the window hashes its n-grams on the GPU, and the host reads the picks once
    const wids: Buf = .{ .b = try r.buffer(64) };
    const adapt_story = [_][3]usize{ .{ 3, 3, 0 }, .{ 1, 4, 1 }, .{ 1, 5, 1 }, .{ 2, 5, 1 }, .{ 2, 5, 2 }, .{ 1, 6, 2 } };
    const adapt_own = [_][3]usize{ .{ 3, 3, 0 }, .{ 4, 4, 0 }, .{ 5, 5, 0 }, .{ 6, 6, 0 }, .{ 7, 7, 0 }, .{ 2, 7, 1 }, .{ 2, 7, 2 }, .{ 3, 7, 2 } };
    const adapt: []const [3]usize = if (own_prompt) &adapt_own else &adapt_story;
    const copy_min: usize = if (std.c.getenv("FZ_COPY_MIN")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 3;
    var hist: std.ArrayList(u32) = .empty;
    for (prompt) |x| try hist.append(gpa, @intCast(x.integer));
    const n_prompt = hist.items.len;
    for (0..if (r.copy) 2 else adapt.len) |run6| {
        const cfg_a = if (r.copy) adapt[0] else adapt[run6];
        const use_copy = r.copy and run6 == 1;
        var copy_rounds: usize = 0;
        var copy_landed: usize = 0;
        var depth: usize = cfg_a[0];
        m.reset();
        m.mtp.pos = 0;
        m.mtp.drafted = 0;
        const all = try r.buffer(prompt.len * WIDE * 2);
        var nexts: std.ArrayList(u32) = .empty;
        for (prompt, 0..) |x, i| {
            const tok: u32 = @intCast(x.integer);
            try m.window(&.{tok}, &pick);
            m.keepRows(&.{tok}, 1);
            @memcpy(all.contents()[i * WIDE * 2 .. (i + 1) * WIDE * 2], m.last.b.contents()[0 .. WIDE * 2]);
            if (i > 0) try nexts.append(gpa, tok);
        }
        try nexts.append(gpa, pick[0]);
        var out: std.ArrayList(u32) = .empty;
        try out.append(gpa, pick[0]);
        const w = wids.b.slice(u32, 16);
        var absorb_rows: []const u32 = nexts.items;
        var absorb_from: Buf = .{ .b = all };
        if (absorb_rows.len > MAXR) { // a long prompt: the head absorbs it before the rounds, a command buffer a chunk
            w[1] = try m.mtpAbsorb(absorb_rows, absorb_from);
            absorb_rows = &.{};
        }
        var n_rounds: usize = 0;
        var landed: usize = 0;
        const s0 = mtl.clock.seconds();
        m.gpu_seconds = 0;
        while (out.items.len < want.len) {
            w[0] = out.items[out.items.len - 1];
            var d = depth;
            var copied: usize = 0;
            if (use_copy) { // copy lanes: the window takes the tokens after an earlier match of its suffix
                hist.shrinkRetainingCapacity(n_prompt);
                try hist.appendSlice(gpa, out.items);
                copied = copyDrafts(hist.items, copy_min, w[1..MAXR]);
                if (copied > 0) d = copied;
            }
            const cb = r.queue.commandBuffer();
            r.enc = cb.compute(if (r.serial) .serial else .concurrent);
            // the head: absorb the kept rows (chunks of up to MAXR), its draft into slot 1, then chain into 2..depth
            m.mtp.pos -= m.mtp.drafted;
            m.mtp.drafted = 0;
            var at: usize = 0;
            var slot: usize = 0;
            while (at < absorb_rows.len) : (slot += 1) {
                const n = @min(MAXR, absorb_rows.len - at);
                const sl = &m.mtp.slots[slot];
                const ids = sl.ids8.b.slice(u32, 8);
                for (0..8) |i| ids[i] = if (i < n) absorb_rows[at + i] else 0;
                try m.mtpEncode(slot, n, sl.ids8, .{ .b = absorb_from.b, .off = absorb_from.off + at * WIDE * 2 }, .{ .b = wids.b, .off = if (copied > 0) 4 * 12 else 4 });
                m.mtp.pos += n;
                at += n;
            }
            if (copied == 0) for (1..depth) |j| {
                try m.mtpEncode(slot, 1, .{ .b = wids.b, .off = 4 * j }, m.mtp.last, .{ .b = wids.b, .off = 4 * (j + 1) });
                m.mtp.pos += 1;
                m.mtp.drafted += 1;
                slot += 1;
            };
            // the target window [pending, drafts]: with FZ_SPLIT the head's part is committed first and the window
            // is encoded while the GPU runs it (same queue, so the window still follows it)
            var wcb = cb;
            if (r.split) { // untracked buffers: the window waits on the head's signal, not on queue order alone
                r.enc.end();
                r.event_value += 1;
                cb.signal(r.event, r.event_value);
                cb.commit();
                wcb = r.queue.commandBuffer();
                wcb.waitFor(r.event, r.event_value);
                r.enc = wcb.compute(if (r.serial) .serial else .concurrent);
            }
            m.windowMeta(d + 1);
            m.pleIdsGpu(d + 1, wids);
            try m.windowEncode(d + 1, wids);
            try m.finish(wcb);
            if (r.split) m.gpu_seconds += cb.gpuSeconds();
            const picks = m.t.picks.b.slice(u32, d + 1);
            var keep: usize = 1;
            while (keep <= d and w[keep] == picks[keep - 1]) keep += 1;
            try out.appendSlice(gpa, picks[0..keep]);
            n_rounds += 1;
            landed += keep - 1;
            if (copied > 0) {
                copy_rounds += 1;
                copy_landed += keep - 1;
            }
            var win: [MAXR]u32 = undefined;
            @memcpy(win[0 .. d + 1], w[0 .. d + 1]);
            m.keepRows(win[0 .. d + 1], keep);
            @memcpy(pick[0..keep], picks[0..keep]);
            absorb_rows = pick[0..keep];
            absorb_from = m.last;
            if (cfg_a[0] != cfg_a[1]) depth = @max(cfg_a[0], @min(cfg_a[1], keep - 1 + cfg_a[2]));
        }
        const wall = mtl.clock.seconds() - s0;
        var eq: usize = 0;
        while (eq < want.len and out.items[eq] == (if (r.xnew) ref_tokens[eq] else @as(u32, @intCast(want[eq].integer)))) eq += 1;
        const made: f64 = @floatFromInt(out.items.len - 1);
        std.debug.print("one buffer a round, depth {d}-{d} (+{d}){s}: {d}/{d} tokens equal; {d:.2} tokens a round, {d:.2} drafts landing; {d:.1} tok/s (GPU busy {d:.0}%)\n", .{ cfg_a[0], cfg_a[1], cfg_a[2], if (use_copy) " + copy lanes" else "", eq, want.len, made / @as(f64, @floatFromInt(n_rounds)), @as(f64, @floatFromInt(landed)) / @as(f64, @floatFromInt(n_rounds)), made / wall, 100 * m.gpu_seconds / wall });
        if (use_copy) std.debug.print("  copy rounds {d} of {d}, {d:.2} copied lanes landing a copy round\n", .{ copy_rounds, n_rounds, @as(f64, @floatFromInt(copy_landed)) / @as(f64, @floatFromInt(@max(copy_rounds, 1))) });
        if (eq < want.len) bad += 1;
    }

    // 7. GPU-side rounds: the verdict, positions, history and kept states stay on the GPU; the host encodes round
    //    N+1 while round N runs and reads the emitted tokens from a ring
    if (r.xnew and !own_prompt) for ([_]usize{ 2, 3, 4 }) |depth| {
        const W = depth + 1;
        const n_lin: usize = 36;
        const g_cs = try r.buffer(n_lin * CS_ROW);
        const g_so = try r.buffer(n_lin * SO_ROW);
        const o_cs = try r.buffer(n_lin * MAXR * CS_ROW);
        const o_so = try r.buffer(n_lin * MAXR * SO_ROW);
        var saved: [LAYERS][4]Buf = undefined;
        var gi: usize = 0;
        for (&m.layers, 0..) |*L, i| if (L.linear) {
            saved[i] = .{ L.cs[0], L.cs[1], L.so[0], L.so[1] };
            L.cs[0] = .{ .b = g_cs, .off = gi * CS_ROW };
            L.cs[1] = .{ .b = o_cs, .off = gi * MAXR * CS_ROW };
            L.so[0] = .{ .b = g_so, .off = gi * SO_ROW };
            L.so[1] = .{ .b = o_so, .off = gi * MAXR * SO_ROW };
            gi += 1;
        };
        const cin_saved = m.ple.cin;
        const cins = [2]Buf{ m.ple.cin, .{ .b = try r.buffer((PLE_TAIL + MAXR) * WIDE * 2) } };
        r.ar = .{ .b = try r.buffer(4 * 256) };
        const ring = try r.buffer(9 * 4 * 512);
        m.mtp.mixsel = .{ .b = try r.buffer(D * 2) };
        m.mtp.hsel = .{ .b = try r.buffer(WIDE * 2) };
        const ar = r.ar.b.slice(i32, 256);
        const Copy = struct { // the kept row of every DeltaNet layer's window output into its state
            fn states(rr: *Run, gcs: mtl.Buffer, gso: mtl.Buffer, ocs: mtl.Buffer, oso: mtl.Buffer) void {
                rr.copyKept(.{ .b = oso }, .{ .b = gso }, SO_ROW / 4, SO_ROW / 4, MAXR * SO_ROW / 4, SO_ROW / 4, 36, -1);
                rr.copyKept(.{ .b = ocs }, .{ .b = gcs }, CS_ROW / 4, CS_ROW / 4, MAXR * CS_ROW / 4, CS_ROW / 4, 36, -1);
            }
        };
        m.reset();
        m.mtp.pos = 0;
        m.mtp.drafted = 0;
        const all = try r.buffer(prompt.len * WIDE * 2);
        var nexts: std.ArrayList(u32) = .empty;
        for (prompt, 0..) |x, i| {
            const tok: u32 = @intCast(x.integer);
            try m.window(&.{tok}, &pick);
            ar[0] = 1;
            const cb = r.queue.commandBuffer();
            r.enc = cb.compute(if (r.serial) .serial else .concurrent);
            Copy.states(&r, g_cs, g_so, o_cs, o_so);
            try m.finish(cb);
            const cin = m.ple.cin.b.contents()[m.ple.cin.off..];
            std.mem.copyForwards(u8, cin[0 .. PLE_TAIL * WIDE * 2], cin[WIDE * 2 .. (PLE_TAIL + 1) * WIDE * 2]);
            m.ple.hist = .{ m.ple.hist[1], tok };
            m.pos += 1;
            @memcpy(all.contents()[i * WIDE * 2 .. (i + 1) * WIDE * 2], m.last.b.contents()[0 .. WIDE * 2]);
            if (i > 0) try nexts.append(gpa, tok);
        }
        try nexts.append(gpa, pick[0]);
        var drafts: [MAXR]u32 = undefined;
        drafts[0] = try m.mtpAbsorb(nexts.items, .{ .b = all });
        for (1..depth) |j| drafts[j] = try m.mtpChain(drafts[j - 1]);
        const w = wids.b.slice(u32, 16);
        w[0] = pick[0];
        for (0..depth) |j| w[1 + j] = drafts[j];
        const T: i32 = @intCast(m.pos);
        @memset(ar, 0);
        ar[1] = T;
        for (0..8) |i| {
            ar[4 + i] = if (i < W) T + @as(i32, @intCast(i)) else 0;
            ar[12 + i] = if (i < W) T + @as(i32, @intCast(i)) + 1 else 0;
        }
        ar[20], ar[21], ar[22] = .{ T, CAP, @intCast(W) };
        const pm = m.t.ple_meta.b.slice(i64, 39);
        pm[0], pm[1], pm[2], pm[3] = .{ m.ple.hist[0], m.ple.hist[1], m.ple.eos, @intCast(W) };
        for (0..3) |k| pm[4 + k] = m.ple.mult[k];
        for (0..16) |k| {
            pm[7 + k] = m.ple.sizes[k];
            pm[23 + k] = m.ple.offsets[k];
        }
        const t_saved = .{ m.t.rows, m.t.mdims, m.t.pos8, m.t.nk8, m.t.kvmeta };
        m.t.rows = try i32Buf(&r, &.{@intCast(W)});
        m.t.mdims = try i32Buf(&r, &.{ @intCast(W), 16, 0, 0, 0, 0, 0, 0 });
        m.t.pos8 = .{ .b = r.ar.b, .off = 4 * 4 };
        m.t.nk8 = .{ .b = r.ar.b, .off = 12 * 4 };
        m.t.kvmeta = .{ .b = r.ar.b, .off = 20 * 4 };
        const slots_saved = m.mtp.slots;
        try m.mtpMeta(&m.mtp.slots[0], W);
        m.mtp.slots[0].pos8 = .{ .b = r.ar.b, .off = 24 * 4 };
        m.mtp.slots[0].nk8 = .{ .b = r.ar.b, .off = 32 * 4 };
        m.mtp.slots[0].kvmeta = .{ .b = r.ar.b, .off = 40 * 4 };
        for (1..depth) |j| {
            try m.mtpMeta(&m.mtp.slots[j], 1);
            const b = (44 + (j - 1) * 20) * 4;
            m.mtp.slots[j].pos8 = .{ .b = r.ar.b, .off = b };
            m.mtp.slots[j].nk8 = .{ .b = r.ar.b, .off = b + 32 };
            m.mtp.slots[j].kvmeta = .{ .b = r.ar.b, .off = b + 64 };
        }
        r.gpu_round = true;
        const cfg = [4]u32{ @intCast(W), @intCast(depth), CAP, 0 };
        const base = r.event_value;
        var cbs: std.ArrayList(mtl.CommandBuffer) = .empty;
        var out: std.ArrayList(u32) = .empty;
        try out.append(gpa, pick[0]);
        m.gpu_seconds = 0;
        const s0 = mtl.clock.seconds();
        var s1 = s0;
        var round: usize = 0;
        var done: usize = 0;
        const rg = ring.slice(u32, 9 * 512);
        while (true) {
            const cb = r.queue.commandBuffer();
            if (round > 0) cb.waitFor(r.event, base + round);
            r.enc = cb.compute(if (r.serial) .serial else .concurrent);
            if (round > 0) {
                Copy.states(&r, g_cs, g_so, o_cs, o_so);
                r.copyKept(cins[(round - 1) % 2], cins[round % 2], PLE_TAIL * WIDE / 2, WIDE / 2, 0, 0, 1, 0);
                try m.mtpEncode(0, W, m.t.picks, m.last, .{ .b = wids.b, .off = 4 });
                for (1..depth) |j| {
                    const streams = if (j == 1) m.mtp.hsel else Buf{ .b = m.mtp.h[1].b, .off = 0 };
                    try m.mtpEncode(j, 1, .{ .b = wids.b, .off = 4 * j }, streams, .{ .b = wids.b, .off = 4 * (j + 1) });
                }
            }
            m.ple.cin = cins[round % 2];
            m.pleIdsGpu(W, wids);
            try m.windowEncode(W, wids);
            r.enc.setPipeline(r.accept_pipe);
            for ([_]Buf{ wids, m.t.picks, r.ar, .{ .b = ring }, m.t.ple_meta }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
            r.enc.setBytes(std.mem.asBytes(&cfg), 5);
            r.enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
            r.enc.end();
            cb.signal(r.event, base + round + 1);
            cb.commit();
            try cbs.append(gpa, cb);
            round += 1;
            if (round >= 2) {
                const prev = cbs.items[done];
                prev.wait();
                if (prev.failure()) |msg| {
                    std.log.err("command buffer failed: {s}", .{msg});
                    return error.GpuFailed;
                }
                m.gpu_seconds += prev.gpuSeconds();
                const keep = rg[done * 9];
                try out.appendSlice(gpa, rg[done * 9 + 1 .. done * 9 + 1 + keep]);
                done += 1;
                if (out.items.len >= want.len) {
                    s1 = mtl.clock.seconds();
                    break;
                }
            }
        }
        for (cbs.items[done..]) |cb| cb.wait();
        r.event_value = base + round + 1;
        r.gpu_round = false;
        m.t.rows, m.t.mdims, m.t.pos8, m.t.nk8, m.t.kvmeta = t_saved;
        m.mtp.slots = slots_saved;
        m.ple.cin = cin_saved;
        for (&m.layers, 0..) |*L, i| if (L.linear) {
            L.cs[0], L.cs[1], L.so[0], L.so[1] = saved[i];
        };
        var eq: usize = 0;
        while (eq < want.len and out.items[eq] == ref_tokens[eq]) eq += 1;
        const made: f64 = @floatFromInt(out.items.len - 1);
        std.debug.print("GPU-side rounds, depth {d}: {d}/{d} tokens equal; {d:.2} tokens a round; {d:.1} tok/s (GPU busy {d:.0}%)\n", .{ depth, eq, want.len, made / @as(f64, @floatFromInt(done)), made / (s1 - s0), 100 * m.gpu_seconds / (s1 - s0) });
        if (eq < want.len) bad += 1;
    };

    // 3. each window size's cost
    for ([_]usize{ 1, 2, 3, 4, 6, 8 }) |rows| {
        var toks: [MAXR]u32 = undefined;
        for (0..rows) |i| toks[i] = got.items[i];
        m.gpu_seconds = 0;
        const s0 = mtl.clock.seconds();
        for (0..10) |_| try m.window(toks[0..rows], &pick);
        const wall = (mtl.clock.seconds() - s0) * 1e2;
        std.debug.print("window of {d} rows: {d:.2} ms (GPU {d:.2} ms)\n", .{ rows, wall, m.gpu_seconds * 1e2 });
    }
    if (same != want.len or bad != 0) std.process.exit(1);
}
