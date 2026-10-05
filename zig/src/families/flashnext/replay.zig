//! Flash Next on our Metal runtime: the Python engine's recorded kernels (a dump from tools/zig/flashnext_dump.py)
//! replayed from Zig, with our own kernels for wide experts, prompt chunks, block selection and GPU-side rounds.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");

pub const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

pub const D = 2560;
pub const WIDE = 4 * D;
pub const LAYERS = 48;
pub const VOCAB = 248320;
pub const CAP = 262144; // keys an attention layer holds in this runner
pub const PLE_TAIL = 9;
pub const GROUPS = 8;
pub const MAXR = 8;
pub const CS_ROW = 3 * WIDE * 2; // a DeltaNet conv state row (bytes)
pub const SO_ROW = 48 * 128 * 128 * 4; // a DeltaNet recurrent state row (bytes)

pub const Buf = struct { b: mtl.Buffer, off: usize = 0 };
pub const Entry = struct { fd: std.c.fd_t, at: usize, len: usize };
pub const Variant = struct { inputs: [][]const u8, outputs: [][]const u8, meta: [][]const u8, pipe: mtl.Pipeline, file: []const u8 = "", name: []const u8 = "" };
pub const Site = struct { v: *Variant, grid: mtl.Size, tg: mtl.Size };

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
    \\// NGramEmbedding.ids for a window: pm = history (2), eos, unused, multipliers (3), head sizes (16), offsets (16)
    \\kernel void fz_ple_ids(device const uint* tok [[buffer(0)]], device const long* pm [[buffer(1)]],
    \\    device uint* out [[buffer(2)]], constant uint& rows [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    \\  const uint row = i / 16, hh = i % 16;
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
    \\// cfg = this window's rows, the next window's rows, cache capacity.
    \\kernel void fz_accept(device uint* wids [[buffer(0)]], device const uint* picks [[buffer(1)]],
    \\    device int* ar [[buffer(2)]], device uint* ring [[buffer(3)]], device long* pm [[buffer(4)]],
    \\    constant uint4& cfg [[buffer(5)]], uint tid [[thread_position_in_grid]]) {
    \\  if (tid != 0) return;
    \\  const int W = int(cfg.x), Wn = int(cfg.y), cap = int(cfg.z);
    \\  int keep = 1;
    \\  while (keep < W && wids[keep] == picks[keep - 1]) keep++;
    \\  const int round = ar[2];
    \\  ring[round * 9] = uint(keep);
    \\  for (int i = 0; i < keep; i++) ring[round * 9 + 1 + i] = picks[i];
    \\  for (int i = 0; i < keep; i++) { pm[0] = pm[1]; pm[1] = long(wids[i]); }
    \\  const int t_old = ar[1], t_new = t_old + keep;
    \\  ar[0] = keep; ar[1] = t_new; ar[2] = round + 1;
    \\  for (int i = 0; i < 8; i++) {
    \\    ar[4 + i] = i < Wn ? t_new + i : 0;
    \\    ar[12 + i] = i < Wn ? t_new + i + 1 : 0;
    \\    ar[24 + i] = i < W ? t_old + i : 0;
    \\    ar[32 + i] = i < W ? t_old + i + 1 : 0;
    \\  }
    \\  ar[20] = t_new; ar[21] = cap; ar[22] = Wn;
    \\  ar[40] = t_old; ar[41] = cap; ar[42] = W;
    \\  for (int j = 1; j < Wn - 1; j++) {
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

pub fn readAll(fd: std.c.fd_t, dest: []u8, at: usize) !void {
    var done: usize = 0;
    while (done < dest.len) {
        const n = std.c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
}

pub const Run = struct {
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
    xnew_header: []const u8 = "",
    sel: ?Select = null, // long contexts: the indexer's pool, scores and selection
    gsel: ?GSelect = null, // the same in GPU-side rounds
    sel_meta_pipe: mtl.Pipeline = undefined,
    pool_abs_pipe: mtl.Pipeline = undefined,

    pub fn buffer(r: *Run, len: usize) !mtl.Buffer {
        const n = @max(len, 64);
        const b = try r.device.buffer(n, opts);
        @memset(b.contents()[0..n], 0);
        return b;
    }

    /// Every tensor of a safetensors file in the index, at its absolute offset.
    pub fn indexFile(r: *Run, path: [:0]const u8) !void {
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

    pub fn entry(r: *Run, name: []const u8) !Entry {
        return r.index.get(name) orelse {
            std.log.err("no tensor {s}", .{name});
            return error.MissingTensor;
        };
    }

    pub fn load(r: *Run, name: []const u8) !Buf {
        const e = try r.entry(name);
        const b = try r.device.buffer(@max(e.len, 64), opts);
        try readAll(e.fd, b.contents()[0..e.len], e.at);
        r.loaded += e.len;
        return .{ .b = b };
    }

    pub fn loadf(r: *Run, comptime fmt: []const u8, args: anytype) !Buf {
        var name: [160]u8 = undefined;
        return r.load(try std.fmt.bufPrint(&name, fmt, args));
    }

    /// Shards `first..first+count` of `suffix` concatenated into one buffer (PleTables' group).
    pub fn group(r: *Run, first: usize, count: usize, suffix: []const u8) !Buf {
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

    pub fn compile(r: *Run, dir: []const u8) !void {
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
            v.* = .{ .inputs = try strings(r.arena, o.get("inputs").?), .outputs = try strings(r.arena, o.get("outputs").?), .meta = try strings(r.arena, o.get("meta").?), .pipe = try mtl.Pipeline.init(r.device, lib, kv.key_ptr.*, false), .file = try std.fmt.allocPrint(r.arena, "{s}/{s}", .{ dir, o.get("file").?.string }), .name = kv.key_ptr.* };
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
        const slib = try mtl.Library.fromSource(r.device, ks.flashnext_select, mtl.CompileOptions.mlx());
        r.sel_meta_pipe = try mtl.Pipeline.init(r.device, slib, "fz_sel_meta", false);
        r.pool_abs_pipe = try mtl.Pipeline.init(r.device, slib, "fz_idx_pool_abs", false);
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
            r.xnew_header = try std.mem.concat(r.arena, u8, &.{ define, text[0..cut] });
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

    pub fn strings(a: std.mem.Allocator, v: std.json.Value) ![][]const u8 {
        const out = try a.alloc([]const u8, v.array.items.len);
        for (v.array.items, 0..) |s, i| out[i] = s.string;
        return out;
    }

    pub fn size3(v: std.json.Value) mtl.Size {
        const a = v.array.items;
        return mtl.Size.of(@intCast(a[0].integer), @intCast(a[1].integer), @intCast(a[2].integer));
    }




    /// FZ_XPACK: an expert set's rows (gate, up, shared gate, shared up, down, shared down) to the packed layout.
    pub fn repack(r: *Run, ex: []Buf) !void {
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
    pub fn touch(r: *Run, b: Buf) void {
        if (!r.prefetch) return;
        const n: u32 = @intCast((b.b.length() - b.off) / 16);
        r.enc.setPipeline(r.touch_pipe);
        r.enc.setBuffer(b.b, b.off, 0);
        r.enc.setBuffer(r.sink.b, 0, 1);
        r.enc.setBytes(std.mem.asBytes(&n), 2);
        r.enc.dispatchThreads(mtl.Size.of(256 * 64, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// fz_copy_kept: `layers` copies of `n` words from row (keep + base) of src (row and layer strides in words).
    pub fn copyKept(r: *Run, src: Buf, dst: Buf, n: usize, row: usize, src_stride: usize, dst_stride: usize, layers: usize, base: i32) void {
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
    pub fn denseRows(r: *Run, x: Buf, k: usize, l: Lane, rows: usize, y: Buf) void {
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
    pub fn experts(r: *Run, gu_role: []const u8, down_role: []const u8, x: Buf, lg: Buf, e: [18]Buf, act: Buf, pick: Buf, wts: Buf, rows_buf: Buf, y: Buf) !void {
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
    pub fn class(role: []const u8) u32 {
        const names = [_][]const u8{ "hc_", "lane_qmm", "expert", "router", "gdn", "attn", "ple", "head" };
        if (std.mem.indexOf(u8, role, "@head") != null) return 1 << 7;
        for (names, 0..) |n, i| if (std.mem.indexOf(u8, role, n) != null) return @as(u32, 1) << @intCast(i);
        return 0;
    }

    /// `role` at `rows` rows: the recorded one-row launch with its row dimension (the one that grows from one row to two)
    /// set to `rows`; `v` replaces the recorded variant when given.
    pub fn callRows(r: *Run, role: []const u8, rows: usize, ins: []const Buf, outs: []const Buf, v: ?*Variant) !void {
        var k1: [96]u8 = undefined;
        var k2: [96]u8 = undefined;
        const s1 = r.roles.get(try std.fmt.bufPrint(&k1, "{s}|1", .{role})) orelse return error.NoSite;
        const s2 = r.roles.get(try std.fmt.bufPrint(&k2, "{s}|2", .{role})) orelse return error.NoSite;
        var grid = s1.grid;
        if (s2.grid.height != s1.grid.height) grid.height = rows * s1.grid.height else if (s2.grid.depth != s1.grid.depth) grid.depth = rows * s1.grid.depth;
        try r.bindV(v orelse s1.v, ins, outs);
        r.enc.dispatchThreads(grid, s1.tg);
        if (!r.serial) r.enc.barrier();
    }

    pub fn bindV(r: *Run, v: *Variant, ins: []const Buf, outs: []const Buf) !void {
        if (ins.len != v.inputs.len or outs.len != v.outputs.len) return error.Arity;
        r.enc.setPipeline(v.pipe);
        var at: usize = 0;
        for (v.inputs, ins) |input, b| {
            r.enc.setBuffer(b.b, b.off, at);
            at += 1;
            for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suffix| {
                for (v.meta) |mm| {
                    if (mm.len == input.len + suffix.len and std.mem.startsWith(u8, mm, input) and std.mem.endsWith(u8, mm, suffix)) {
                        r.enc.setBuffer(r.shapes.get(mm) orelse return error.NoShape, 0, at);
                        at += 1;
                    }
                }
            }
        }
        for (outs) |b| {
            r.enc.setBuffer(b.b, b.off, at);
            at += 1;
        }
    }

    pub fn call(r: *Run, role: []const u8, ins: []const Buf, outs: []const Buf) !void {
        return r.callAs(role, r.rows, ins, outs);
    }

    /// `role` launched as recorded at `as_rows` rows (its kernel reads the row count at run time).
    pub fn callAs(r: *Run, role: []const u8, as_rows: usize, ins: []const Buf, outs: []const Buf) !void {
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


/// Past 512 complete 4-key blocks the attention reads each row's 512 best blocks and its tail: the recorded pool,
/// scores and selection kernels (dispatched at any context), with per-row metadata from the host.
pub const TOP = 512;
pub const KW = 4 * TOP + 3;
pub const Select = struct {
    pool: *Variant,
    scores: *Variant,
    select: *Variant,
    start: Buf,
    sc: Buf,
    keys: Buf,
    complete: Buf,
    ends: Buf,
    counts: Buf,
    sparse: Buf,
    pooled_shape: mtl.Buffer,
    q_shape: mtl.Buffer,
    sc_shape: mtl.Buffer,
    ids_shape: mtl.Buffer,

    pub fn init(r: *Run, rows_max: usize) !?Select {
        var found: [3]?*Variant = .{ null, null, null };
        var it = r.variants.iterator();
        while (it.next()) |kv| {
            const n = kv.key_ptr.*;
            if (std.mem.indexOf(u8, n, "q4_idx_pool_") != null and std.mem.indexOf(u8, n, "_rel") == null) found[0] = kv.value_ptr.*;
            if (std.mem.indexOf(u8, n, "q4_idx_scores_") != null) found[1] = kv.value_ptr.*;
            if (std.mem.indexOf(u8, n, "q4_idx_select_") != null) found[2] = kv.value_ptr.*;
        }
        if (found[0] == null or found[1] == null or found[2] == null) return null;
        const B = struct {
            fn of(rr: *Run, n: usize) !Buf {
                return .{ .b = try rr.buffer(n) };
            }
        };
        return .{
            .pool = found[0].?, .scores = found[1].?, .select = found[2].?,
            .start = try B.of(r, 16), .sc = try B.of(r, rows_max * (CAP / 4) * 4), .keys = try B.of(r, rows_max * KW * 4),
            .complete = try B.of(r, rows_max * 4), .ends = try B.of(r, rows_max * 4), .counts = try B.of(r, rows_max * 4),
            .sparse = try B.of(r, rows_max * 4),
            .pooled_shape = try r.buffer(16), .q_shape = try r.buffer(16), .sc_shape = try r.buffer(16), .ids_shape = try r.buffer(16),
        };
    }

    /// The window's per-row metadata from its first position; true when some row reads selected blocks.
    pub fn meta(s: *Select, pos: usize, rows: usize) bool {
        const cp = s.complete.b.slice(i32, rows);
        const en = s.ends.b.slice(i32, rows);
        const ct = s.counts.b.slice(i32, rows);
        const sp = s.sparse.b.slice(i32, rows);
        var any = false;
        for (0..rows) |i| {
            const e = pos + i + 1;
            const c = e / 4;
            const sparse = c > TOP;
            cp[i], en[i] = .{ @intCast(c), @intCast(e) };
            ct[i] = @intCast(if (sparse) 4 * TOP + e - 4 * c else e);
            sp[i] = @intFromBool(sparse);
            any = any or sparse;
        }
        return any;
    }

    /// Pool every complete block below `upto` the layer has not pooled (before GPU-side rounds take over).
    pub fn catchUp(s: *Select, r: *Run, L: anytype, eps: Buf, log2base: Buf, upto: usize) !void {
        if (upto <= L.pooled_n) return;
        s.start.b.slice(i32, 1)[0] = @intCast(L.pooled_n);
        try r.bindV(s.pool, &.{ L.raw, s.start, L.pool, eps, log2base }, &.{.{ .b = L.pooled.b, .off = L.pooled.off + L.pooled_n * 128 * 2 }});
        r.enc.dispatchThreads(mtl.Size.of(128, upto - L.pooled_n, 1), mtl.Size.of(128, 1, 1));
        if (!r.serial) r.enc.barrier();
        L.pooled_n = upto;
    }

    /// Pool the blocks the window completes, score every complete block for each row, select each row's keys.
    pub fn encode(s: *Select, r: *Run, L: anytype, iq: Buf, eps: Buf, log2base: Buf, pos: usize, rows: usize) !void {
        const last = (pos + rows) / 4;
        if (last > L.pooled_n) {
            s.start.b.slice(i32, 1)[0] = @intCast(L.pooled_n); // read at run time: one window a command buffer
            try r.bindV(s.pool, &.{ L.raw, s.start, L.pool, eps, log2base }, &.{.{ .b = L.pooled.b, .off = L.pooled.off + L.pooled_n * 128 * 2 }});
            r.enc.dispatchThreads(mtl.Size.of(128, last - L.pooled_n, 1), mtl.Size.of(128, 1, 1));
            if (!r.serial) r.enc.barrier();
            L.pooled_n = last;
        }
        const nb = L.pooled_n;
        @memcpy(s.pooled_shape.slice(i32, 2), &[2]i32{ @intCast(nb), 128 });
        @memcpy(s.q_shape.slice(i32, 3), &[3]i32{ @intCast(rows), 4, 128 });
        @memcpy(s.sc_shape.slice(i32, 2), &[2]i32{ @intCast(rows), @intCast(nb) });
        @memcpy(s.ids_shape.slice(i32, 2), &[2]i32{ @intCast(rows), KW });
        try r.shapes.put(r.arena, "POOLED_shape", s.pooled_shape);
        try r.shapes.put(r.arena, "Q_shape", s.q_shape);
        try r.shapes.put(r.arena, "SC_shape", s.sc_shape);
        try r.bindV(s.scores, &.{ iq, L.pooled, s.complete }, &.{s.sc});
        r.enc.dispatchThreads(mtl.Size.of(((nb + 7) / 8) * 256, (rows + 7) / 8, 1), mtl.Size.of(256, 1, 1));
        if (!r.serial) r.enc.barrier();
        try r.bindV(s.select, &.{ s.sc, s.complete, s.ends }, &.{s.keys});
        r.enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
        if (!r.serial) r.enc.barrier();
    }
};


/// Block selection in GPU-side rounds: fz_sel_meta writes each round's rows, pooling range and pooled count; the pool
/// runs at absolute blocks (up to 3 a round); scores cover an upper bound of the pooled blocks the host keeps.
pub const GSelect = struct {
    sel: Buf, // complete[8] ends[8] counts[8] sparse[8] start count pooled
    sc: Buf,
    keys: Buf,
    pooled_shape: mtl.Buffer,
    q_shape: [MAXR + 1]mtl.Buffer, // by window rows
    sc_shape: mtl.Buffer,
    ids_shape: [MAXR + 1]mtl.Buffer,
    nb_ub: usize = 0,

    pub fn init(r: *Run, pooled: usize) !GSelect {
        var g: GSelect = .{
            .sel = .{ .b = try r.buffer(64 * 4) }, .sc = .{ .b = try r.buffer(MAXR * (CAP / 4) * 4) },
            .keys = .{ .b = try r.buffer(MAXR * KW * 4) }, .pooled_shape = try r.buffer(16), .q_shape = undefined,
            .sc_shape = try r.buffer(16), .ids_shape = undefined,
        };
        g.sel.b.slice(i32, 64)[34] = @intCast(pooled);
        for (0..MAXR + 1) |w| {
            g.q_shape[w] = (try i32Buf(r, &.{ @intCast(w), 4, 128 })).b;
            g.ids_shape[w] = (try i32Buf(r, &.{ @intCast(w), KW })).b;
        }
        return g;
    }

    pub fn view(g: *GSelect, at: usize) Buf {
        return .{ .b = g.sel.b, .off = at * 4 };
    }
};

pub const Hc = struct { scale: Buf, dw: Buf, ds: Buf, db: Buf, uw: Buf, us: Buf, ub: Buf };
pub const Lane = struct { wq: Buf, sbt: Buf };
pub const Layer = struct {
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
    pool: Buf = undefined,
    pooled: Buf = undefined, // the indexer's pooled block keys [CAP / 4, 128]
    pooled_n: usize = 0,
};

pub fn hcOf(r: *Run, comptime fmt: []const u8, args: anytype) !Hc {
    var name: [96]u8 = undefined;
    const stem = try std.fmt.bufPrint(&name, fmt, args);
    var full: [128]u8 = undefined;
    const parts = [_][]const u8{ "scale", "down.w", "down.s", "down.b", "up.w", "up.s", "up.b" };
    var out: [7]Buf = undefined;
    for (parts, 0..) |p, i| out[i] = try r.load(try std.fmt.bufPrint(&full, "{s}.{s}", .{ stem, p }));
    return .{ .scale = out[0], .dw = out[1], .ds = out[2], .db = out[3], .uw = out[4], .us = out[5], .ub = out[6] };
}

pub fn laneOf(r: *Run, comptime fmt: []const u8, args: anytype) !Lane {
    var name: [96]u8 = undefined;
    const stem = try std.fmt.bufPrint(&name, fmt, args);
    var full: [128]u8 = undefined;
    return .{ .wq = try r.load(try std.fmt.bufPrint(&full, "{s}.wq", .{stem})), .sbt = try r.load(try std.fmt.bufPrint(&full, "{s}.sbt", .{stem})) };
}

pub const Ple = struct {
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
pub const Tmp = struct {
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

pub fn f32Buf(r: *Run, v: f32) !Buf {
    const b = try r.buffer(4);
    b.slice(f32, 1)[0] = v;
    return .{ .b = b };
}

pub fn i32Buf(r: *Run, vals: []const i32) !Buf {
    const b = try r.buffer(vals.len * 4);
    @memcpy(b.slice(i32, vals.len), vals);
    return .{ .b = b };
}

/// One MTP call's per-row values (every call in a command buffer reads its own).
pub const Slot = struct { rows: Buf, md: Buf, md4: Buf, ids8: Buf, pos8: Buf, nk8: Buf, kvmeta: Buf, n_add: Buf };

/// The MTP head: its decoder layer and mixer, the input projections, the cut head, its own attention cache.
pub const Mtp = struct {
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
    pool: Buf = undefined, // its indexer's block pooling (the head's layer is a sparse-attention layer)
    pooled: Buf = undefined,
    pooled_n: usize = 0,
    gsel: ?GSelect = null, // its block selection in GPU-side rounds
};

pub const Model = struct {
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

    pub fn reset(m: *Model) void {
        m.pos = 0;
        m.state = 0;
        m.state_row = 0;
        for (&m.layers) |*L| if (L.linear) {
            @memset(L.cs[0].b.contents()[L.cs[0].off .. L.cs[0].off + CS_ROW], 0);
            @memset(L.so[0].b.contents()[L.so[0].off .. L.so[0].off + SO_ROW], 0);
        } else {
            L.pooled_n = 0;
        };
        m.ple.hist = .{ m.ple.eos, m.ple.eos };
        @memset(m.ple.cin.b.contents()[m.ple.cin.off .. m.ple.cin.off + (PLE_TAIL + MAXR) * WIDE * 2], 0);
        m.mtp.pooled_n = 0;
    }

    pub fn lane(m: *Model, x: Buf, k: usize, l: Lane, role: []const u8, y: Buf) !void {
        if (m.r.dense_target) return m.r.denseRows(x, k, l, m.r.rows, y);
        if (!m.r.fused_xsum) try m.r.call(if (k == D) "lane_qmm_xsum#[2560]" else "lane_qmm_xsum#[6144]", &.{ x, m.t.mdims }, &.{m.t.xs});
        try m.r.call(role, &.{ x, m.t.xs, l.wq, l.sbt, m.t.mdims }, &.{y});
    }

    pub fn hcProject(m: *Model, hn: Buf, hc: Hc, down: []const u8, up: []const u8, inj: Buf) !void {
        const t = &m.t;
        const as_rows = if (m.r.hc_mma and m.r.rows > 1) 8 else m.r.rows;
        try m.r.callAs(down, as_rows, &.{ hn, t.ssp, hc.scale, hc.dw, hc.ds, hc.db, t.eps, t.rows }, &.{t.part});
        try m.r.callAs(up, as_rows, &.{ hn, t.ssp, hc.scale, t.part, hc.uw, hc.us, hc.ub, t.eps, t.rows }, &.{ t.mixed, inj });
    }

    pub fn grouped(m: *Model, h: Buf, out: Buf) !void {
        const t = &m.t;
        try m.r.call("q4_hc_norm_grouped#[10240]", &.{ h, t.inj_m, t.ydown, t.wts, t.lg }, &.{ out, t.ssp });
    }

    /// The window's n-gram row ids [rows, 16] after the history (NGramEmbedding.ids on [history, tokens]).
    pub fn pleIds(m: *Model, tokens: []const u32) void {
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
    pub fn windowMeta(m: *Model, rows: usize) void {
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
    pub fn window(m: *Model, tokens: []const u32, picks: []u32) !void {
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

    pub fn finish(m: *Model, cb: mtl.CommandBuffer) !void {
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
    pub fn pleIdsGpu(m: *Model, rows: usize, ids: Buf) void {
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
        const n: u32 = @intCast(rows);
        r.enc.setBytes(std.mem.asBytes(&n), 3);
        r.enc.dispatchThreads(mtl.Size.of(16 * rows, 1, 1), mtl.Size.of(16 * rows, 1, 1));
        if (!r.serial) r.enc.barrier();
    }

    /// Encode the window's forward (tokens read from `ids`, n-gram ids already in t.ple_ids) and its argmax.
    pub fn windowEncode(m: *Model, rows: usize, ids: Buf) !void {
        const r = m.r;
        const t = &m.t;
        r.rows = rows;
        if (r.gpu_round and r.gsel != null) {
            const g = &r.gsel.?;
            r.enc.setPipeline(r.sel_meta_pipe);
            r.enc.setBuffer(r.ar.b, r.ar.off, 0);
            const wu: u32 = @intCast(rows);
            r.enc.setBytes(std.mem.asBytes(&wu), 1);
            r.enc.setBuffer(g.sel.b, 0, 2);
            r.enc.setBuffer(g.pooled_shape, 0, 3);
            r.enc.setBuffer(g.sc_shape, 0, 4);
            r.enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
            if (!r.serial) r.enc.barrier();
        }
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
                if (r.gpu_round and r.gsel != null) { // the round's selection, all on the GPU
                    const g = &r.gsel.?;
                    const sl = &r.sel.?;
                    r.enc.setPipeline(r.pool_abs_pipe);
                    for ([_]Buf{ L.raw, g.sel, L.pool, t.eps, t.log2base, L.pooled }, 0..) |bb, j| r.enc.setBuffer(bb.b, bb.off, j);
                    r.enc.dispatchThreads(mtl.Size.of(128, 3, 1), mtl.Size.of(128, 1, 1));
                    if (!r.serial) r.enc.barrier();
                    try r.shapes.put(r.arena, "POOLED_shape", g.pooled_shape);
                    try r.shapes.put(r.arena, "Q_shape", g.q_shape[rows]);
                    try r.shapes.put(r.arena, "SC_shape", g.sc_shape);
                    try r.bindV(sl.scores, &.{ t.iq, L.pooled, g.view(0) }, &.{g.sc});
                    r.enc.dispatchThreads(mtl.Size.of(((g.nb_ub + 7) / 8) * 256, (rows + 7) / 8, 1), mtl.Size.of(256, 1, 1));
                    if (!r.serial) r.enc.barrier();
                    try r.bindV(sl.select, &.{ g.sc, g.view(0), g.view(8) }, &.{g.keys});
                    r.enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
                    if (!r.serial) r.enc.barrier();
                    const dense_ids = r.shapes.get("IDS_shape").?;
                    try r.shapes.put(r.arena, "IDS_shape", g.ids_shape[rows]);
                    try r.call("q4_attn_parts#[24, 256]", &.{ t.q, L.keys, L.vals, g.keys, g.view(16), g.view(24), t.scale }, &.{ t.po, t.pm });
                    try r.shapes.put(r.arena, "IDS_shape", dense_ids);
                } else if (r.sel != null and !r.gpu_round and r.sel.?.meta(m.pos, rows)) {
                    var sl = &r.sel.?;
                    try sl.encode(r, L, t.iq, t.eps, t.log2base, m.pos, rows);
                    const dense_ids = r.shapes.get("IDS_shape").?;
                    try r.shapes.put(r.arena, "IDS_shape", sl.ids_shape);
                    try r.call("q4_attn_parts#[24, 256]", &.{ t.q, L.keys, L.vals, sl.keys, sl.counts, sl.sparse, t.scale }, &.{ t.po, t.pm });
                    try r.shapes.put(r.arena, "IDS_shape", dense_ids);
                } else try r.call("q4_attn_parts#[24, 256]", &.{ t.q, L.keys, L.vals, t.ids81, t.nk8, t.zero8, t.scale }, &.{ t.po, t.pm });
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
    pub fn mtpRun(m: *Model, nexts: []const u32, streams: Buf) !u32 {
        const r = m.r;
        const h = &m.mtp;
        const rows = nexts.len;
        h.pos -= h.drafted;
        h.drafted = 0;
        h.pooled_n = @min(h.pooled_n, h.pos / 4); // blocks dropped rows completed are pooled again
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
    pub fn mtpEncode(m: *Model, slot: usize, rows: usize, ids: Buf, streams: Buf, out: Buf) !void {
        const r = m.r;
        const h = &m.mtp;
        const sl = &h.slots[slot];
        if (!r.gpu_round) try m.mtpMeta(sl, rows);
        try m.mtpLayer(sl, rows, ids, streams, out);
    }

    pub fn mtpMeta(m: *Model, sl: *Slot, rows: usize) !void {
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

    pub fn mtpLayer(m: *Model, sl: *Slot, rows: usize, ids: Buf, streams: Buf, out: Buf) !void {
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
        if (r.gpu_round and h.gsel != null) { // the call's selection on the GPU, from its own first position
            const g = &h.gsel.?;
            const ss = &r.sel.?;
            r.enc.setPipeline(r.sel_meta_pipe);
            r.enc.setBuffer(sl.pos8.b, sl.pos8.off - 4, 0); // fz_sel_meta reads the first position at [1]
            const wu: u32 = @intCast(rows);
            r.enc.setBytes(std.mem.asBytes(&wu), 1);
            r.enc.setBuffer(g.sel.b, 0, 2);
            r.enc.setBuffer(g.pooled_shape, 0, 3);
            r.enc.setBuffer(g.sc_shape, 0, 4);
            r.enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
            if (!r.serial) r.enc.barrier();
            r.enc.setPipeline(r.pool_abs_pipe);
            for ([_]Buf{ h.raw, g.sel, h.pool, t.eps, t.log2base, h.pooled }, 0..) |bb, j| r.enc.setBuffer(bb.b, bb.off, j);
            r.enc.dispatchThreads(mtl.Size.of(128, 3, 1), mtl.Size.of(128, 1, 1));
            if (!r.serial) r.enc.barrier();
            try r.shapes.put(r.arena, "POOLED_shape", g.pooled_shape);
            try r.shapes.put(r.arena, "Q_shape", g.q_shape[rows]);
            try r.shapes.put(r.arena, "SC_shape", g.sc_shape);
            try r.bindV(ss.scores, &.{ t.iq, h.pooled, g.view(0) }, &.{g.sc});
            r.enc.dispatchThreads(mtl.Size.of(((g.nb_ub + 7) / 8) * 256, (rows + 7) / 8, 1), mtl.Size.of(256, 1, 1));
            if (!r.serial) r.enc.barrier();
            try r.bindV(ss.select, &.{ g.sc, g.view(0), g.view(8) }, &.{g.keys});
            r.enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
            if (!r.serial) r.enc.barrier();
            const dense_ids = r.shapes.get("IDS_shape").?;
            try r.shapes.put(r.arena, "IDS_shape", g.ids_shape[rows]);
            try r.call("mtp:q4_attn_parts#[24, 256]", &.{ t.q, h.keys, h.vals, g.keys, g.view(16), g.view(24), t.scale }, &.{ t.po, t.pm });
            try r.shapes.put(r.arena, "IDS_shape", dense_ids);
        } else if (r.sel != null and !r.gpu_round and r.sel.?.meta(h.pos, rows)) { // one call a command buffer
            var ss = &r.sel.?;
            try ss.encode(r, h, t.iq, t.eps, t.log2base, h.pos, rows);
            const dense_ids = r.shapes.get("IDS_shape").?;
            try r.shapes.put(r.arena, "IDS_shape", ss.ids_shape);
            try r.call("mtp:q4_attn_parts#[24, 256]", &.{ t.q, h.keys, h.vals, ss.keys, ss.counts, ss.sparse, t.scale }, &.{ t.po, t.pm });
            try r.shapes.put(r.arena, "IDS_shape", dense_ids);
        } else try r.call("mtp:q4_attn_parts#[24, 256]", &.{ t.q, h.keys, h.vals, t.ids81, sl.nk8, t.zero8, t.scale }, &.{ t.po, t.pm });
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

    pub fn mtpProject(m: *Model, hn: Buf, hc: Hc, down: []const u8, up: []const u8, inj: Buf, rows: Buf) !void {
        const t = &m.t;
        try m.r.call(down, &.{ hn, t.ssp, hc.scale, hc.dw, hc.ds, hc.db, t.eps, rows }, &.{t.part});
        try m.r.call(up, &.{ hn, t.ssp, hc.scale, t.part, hc.uw, hc.us, hc.ub, t.eps, rows }, &.{ t.mixed, inj });
    }

    /// Absorb `nexts.len` rows (chunks of up to MAXR) from `streams`; returns the draft after the last row.
    pub fn mtpAbsorb(m: *Model, nexts: []const u32, streams: Buf) !u32 {
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
    pub fn mtpChain(m: *Model, draft: u32) !u32 {
        const prev = m.mtp.last;
        const rows_before = m.mtp.drafted;
        m.mtp.drafted = 0;
        const d = try m.mtpRun(&.{draft}, prev);
        m.mtp.drafted = rows_before + 1;
        return d;
    }

    /// Keep the window's first `keep` rows: the DeltaNet state of row keep-1, the n-gram history and conv tail.
    pub fn keepRows(m: *Model, tokens: []const u32, keep: usize) void {
        m.state = 1 - m.state;
        m.state_row = keep - 1;
        m.pos += keep;
        for (&m.layers) |*L| if (!L.linear) {
            L.pooled_n = @min(L.pooled_n, m.pos / 4); // a block a rejected row completed is pooled again
        };
        const cin = m.ple.cin.b.contents();
        std.mem.copyForwards(u8, cin[0 .. PLE_TAIL * WIDE * 2], cin[keep * WIDE * 2 .. (keep + PLE_TAIL) * WIDE * 2]);
        for (tokens[0..keep]) |tok| m.ple.hist = .{ m.ple.hist[1], tok };
    }
};

/// Drafts a round from recent landing: 6 while drafts land (code-like text), 3 otherwise (prose).
pub const DepthRule = struct {
    rate: f64 = 0.6, // moving average of drafts landed over drafts offered
    depth: usize = 3,

    pub fn pick(self: *const DepthRule) usize {
        return self.depth;
    }

    pub fn update(self: *DepthRule, depth: usize, landed: usize) void {
        self.rate = 0.7 * self.rate + 0.3 * @as(f64, @floatFromInt(landed)) / @as(f64, @floatFromInt(depth));
        if (self.depth == 3 and self.rate > 0.8) self.depth = 6;
        if (self.depth == 6 and self.rate < 0.65) self.depth = 3;
    }
};

/// Copy lanes: the longest suffix of `hist` (`min`..8 tokens) seen earlier; the tokens after its latest earlier
/// occurrence go into `out`. Returns how many (0 when nothing matches).
pub fn copyDrafts(hist: []const u32, min: usize, out: []u32) usize {
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


/// Prompt chunks of up to PMAX rows: projections on the 6-bit tensor-unit kernels, experts sorted by expert and
/// gathered, the decode's row kernels at the chunk's rows, and DeltaNet storing only its last row's state.
pub const PMAX = 2048;
pub const Prompt = struct {
    r: *Run,
    qmm6: mtl.Pipeline,
    gather64: mtl.Pipeline,
    gather32: mtl.Pipeline,
    router_mm: mtl.Pipeline,
    attn256: mtl.Pipeline,
    splitk: mtl.Pipeline,
    parts_sum: mtl.Pipeline,
    pl: [14]mtl.Pipeline, // normed, act, mix, router, route, offsets, sort, gather rows, act2, scatter, copy, DeltaNet pre/scan/post
    gdn: Variant,
    gdn_grid: mtl.Size,
    gdn_tg: mtl.Size,
    sel: ?Select = null, // long prompts: selection for the chunk's rows
    skip: u32 = 0, // timing knock-outs: 1 experts, 2 DeltaNet, 4 attention, 8 projections, 16 hyper-connections, 32 routing, 64 router, 128 sort, 256 row moves, 512 shared expert
    proj: [LAYERS][3]Buf,
    out: [LAYERS][3]Buf,
    ple_kv: [3]Buf,
    b: struct { ids: Buf, pids: Buf, h: [2]Buf, ssp: Buf, normed: Buf, dn: Buf, hact: Buf, inj_a: Buf, inj_m: Buf, up: Buf, mixed: Buf, p: Buf, gout: Buf, branch: Buf, cso: Buf, q: Buf, kout: Buf, iq: Buf, po: Buf, pm: Buf, aout: Buf, pos: Buf, nk: Buf, zeros: Buf, kvmeta: Buf, lg: Buf, pick: Buf, wts: Buf, cnt: Buf, off: Buf, cur: Buf, row_of: Buf, xs: Buf, g: Buf, u: Buf, a: Buf, ds: Buf, sg: Buf, su: Buf, sa: Buf, ydown: Buf, emb: Buf, kvp: Buf, gated: Buf, hout: Buf, cin: Buf, rows: Buf, part: Buf, qn: Buf, kn: Buf, v: Buf, gg: Buf, beta: Buf, ys: Buf, mids: Buf, n_add: Buf },

    pub fn init(r: *Run, dir: []const u8, header: []const u8) !Prompt {
        var p: Prompt = undefined; // every field is set below: `undefined` skips the declared defaults
        p.r = r;
        p.skip = 0;
        p.sel = null;
        const qsrc = try std.mem.replaceOwned(u8, r.arena, ks.flashnext_qmm6, "#include \"../nax.h\"", ks.nax);
        const qlib = try mtl.Library.fromSource(r.device, qsrc, mtl.CompileOptions.mlx());
        p.qmm6 = try mtl.Pipeline.init(r.device, qlib, "tf_qmm6_t_nax", false);
        p.gather64 = try mtl.Pipeline.init(r.device, qlib, "tf_gather_qmm6_nax_64", false);
        p.gather32 = try mtl.Pipeline.init(r.device, qlib, "tf_gather_qmm6_nax_32", false);
        p.router_mm = try mtl.Pipeline.init(r.device, qlib, "tf_mm_bf16_f32_t_nax", false);
        p.attn256 = try mtl.Pipeline.init(r.device, qlib, "tf_attn256_nax", false);
        p.splitk = try mtl.Pipeline.init(r.device, qlib, "tf_qmm6_splitk_nax", false);
        p.parts_sum = try mtl.Pipeline.init(r.device, qlib, "tf_parts_sum", false);
        const glib = try mtl.Library.fromSource(r.device, try std.mem.concat(r.arena, u8, &.{ header, ks.flashnext_prompt }), mtl.CompileOptions.mlx());
        const names = [_][:0]const u8{ "pf_hc_normed", "pf_hc_act", "pf_hc_mix", "pf_router", "pf_route", "pf_offsets", "pf_sort", "pf_gather_rows", "pf_act", "pf_scatter_y", "pf_copy", "pf_gdn_pre", "pf_gdn_scan", "pf_gdn_post" };
        for (names, 0..) |n, i| p.pl[i] = try mtl.Pipeline.init(r.device, glib, n, false);
        // DeltaNet at the chunk's rows, storing only the last row's recurrent state (in row 0)
        const gs = r.roles.get("q4_gdn@gdn|8") orelse return error.NoSite;
        const f = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(r.arena, "{s}", .{gs.v.file}, 0));
        const from = "SO[((size_t(r) * NV + hv)";
        if (std.mem.count(u8, f.bytes[0..f.size], from) != 1) return error.GdnPatch;
        const patched = try std.mem.replaceOwned(u8, r.arena, f.bytes[0..f.size], from, "if (r == R - 1) SO[((size_t(0) * NV + hv)");
        const dlib = try mtl.Library.fromSource(r.device, patched, mtl.CompileOptions.mlx());
        p.gdn = gs.v.*;
        p.gdn.pipe = try mtl.Pipeline.init(r.device, dlib, try std.fmt.allocPrintSentinel(r.arena, "{s}", .{gs.v.name}, 0), false);
        p.gdn_grid = gs.grid;
        p.gdn_tg = gs.tg;
        try r.indexFile(try std.fmt.allocPrintSentinel(r.arena, "{s}/pack_mlx.safetensors", .{dir}, 0));
        for (0..LAYERS) |i| {
            const kind = if (i % 4 != 3) "gdn" else "att";
            const o_name = if (i % 4 != 3) "out" else "o";
            const in_name = if (i % 4 != 3) "in" else "proj";
            for (0..3) |k| {
                const suf = [_][]const u8{ "mw", "ms", "mb" };
                p.proj[i][k] = try r.loadf("L{d}.{s}.{s}.{s}", .{ i, kind, in_name, suf[k] });
                p.out[i][k] = try r.loadf("L{d}.{s}.{s}.{s}", .{ i, kind, o_name, suf[k] });
            }
        }
        p.ple_kv = .{ try r.load("ple.kv.mw"), try r.load("ple.kv.ms"), try r.load("ple.kv.mb") };
        const B = struct {
            fn of(rr: *Run, n: usize) !Buf {
                return .{ .b = try rr.buffer(n) };
            }
        };
        const R = PMAX;
        p.b = .{
            .ids = try B.of(r, R * 4), .pids = try B.of(r, R * 16 * 4), .h = .{ try B.of(r, R * WIDE * 2), try B.of(r, R * WIDE * 2) },
            .ssp = try B.of(r, R * 10 * 4 * 4), .normed = try B.of(r, R * WIDE * 2), .dn = try B.of(r, R * 324 * 2),
            .hact = try B.of(r, R * 320 * 2), .inj_a = try B.of(r, R * 4 * 2), .inj_m = try B.of(r, R * 4 * 2),
            .up = try B.of(r, R * WIDE * 2), .mixed = try B.of(r, R * D * 2), .p = try B.of(r, R * 16480 * 2),
            .gout = try B.of(r, R * 6144 * 2), .branch = try B.of(r, R * D * 2), .cso = try B.of(r, CS_ROW),
            .q = try B.of(r, R * 24 * 256 * 2), .kout = try B.of(r, R * 2 * 256 * 2), .iq = try B.of(r, R * 4 * 128 * 2),
            .po = try B.of(r, R * 24 * 16 * 256 * 4), .pm = try B.of(r, R * 24 * 16 * 2 * 4), .aout = try B.of(r, R * 6144 * 2),
            .pos = try B.of(r, R * 4), .nk = try B.of(r, R * 4), .zeros = try B.of(r, R * 4), .kvmeta = try B.of(r, 16),
            .lg = try B.of(r, R * 513 * 4), .pick = try B.of(r, R * 10 * 4), .wts = try B.of(r, R * 10 * 4),
            .cnt = try B.of(r, 512 * 4), .off = try B.of(r, 513 * 4), .cur = try B.of(r, 512 * 4), .row_of = try B.of(r, R * 10 * 4),
            .xs = try B.of(r, R * 10 * D * 2), .g = try B.of(r, R * 10 * 640 * 2), .u = try B.of(r, R * 10 * 640 * 2),
            .a = try B.of(r, R * 10 * 640 * 2), .ds = try B.of(r, R * 10 * D * 2), .sg = try B.of(r, R * 640 * 2),
            .su = try B.of(r, R * 640 * 2), .sa = try B.of(r, R * 640 * 2), .ydown = try B.of(r, R * 11 * D * 2),
            .emb = try B.of(r, R * D * 2), .kvp = try B.of(r, R * 12800 * 2), .gated = try B.of(r, R * WIDE * 2),
            .hout = try B.of(r, R * WIDE * 2), .cin = try B.of(r, (PLE_TAIL + R) * WIDE * 2), .rows = try B.of(r, 16),
            .part = try B.of(r, 8 * R * 324 * 4), .qn = try B.of(r, R * 16 * 128 * 4), .kn = try B.of(r, R * 16 * 128 * 4),
            .v = try B.of(r, R * 48 * 128 * 4), .gg = try B.of(r, R * 48 * 4), .beta = try B.of(r, R * 48 * 4), .ys = try B.of(r, R * 6144 * 4),
            .mids = try B.of(r, R * 4), .n_add = try B.of(r, 16),
        };
        p.sel = try Select.init(r, PMAX);
        return p;
    }

    pub fn barrier(p: *Prompt) void {
        if (!p.r.serial) p.r.enc.barrier();
    }

    pub fn bind(p: *Prompt, pipe: mtl.Pipeline, bufs: []const Buf) void {
        p.r.enc.setPipeline(pipe);
        for (bufs, 0..) |b, j| p.r.enc.setBuffer(b.b, b.off, j);
    }

    /// y[rows, n] (row stride ldy, 0: n) = x[rows, k] W^T, W 6-bit g32 in MLX's layout.
    pub fn qmm(p: *Prompt, x: Buf, w: [3]Buf, k: usize, n: usize, rows: usize, y: Buf, ldy: usize) void {
        if (p.skip & 8 != 0) return;
        p.bind(p.qmm6, &.{ w[0], w[1], w[2], x });
        const prm = [4]i32{ @intCast(k), @intCast(n), @intCast(rows), @intCast(ldy) };
        p.r.enc.setBytes(std.mem.asBytes(&prm), 4);
        p.r.enc.setBuffer(y.b, y.off, 5);
        p.r.enc.dispatchThreads(mtl.Size.of(((n + 63) / 64) * 128, (rows + 63) / 64, 1), mtl.Size.of(128, 1, 1));
        p.barrier();
    }

    /// y[pairs, n] = x[slot] W_e^T over the sorted slots, the experts' first slots in b.off.
    pub fn gather(p: *Prompt, x: Buf, w: []const Buf, k: usize, n: usize, pairs: usize, y: Buf) void {
        if (p.skip & 1 != 0) return;
        const tall = pairs >= 512 * 32; // 64-row tiles once experts average 32 rows
        p.bind(if (tall) p.gather64 else p.gather32, &.{ x, w[0], w[1], w[2], p.b.off });
        const prm = [4]i32{ @intCast(pairs), @intCast(n), @intCast(k), 512 };
        p.r.enc.setBytes(std.mem.asBytes(&prm), 5);
        p.r.enc.setBuffer(y.b, y.off, 6);
        p.r.enc.dispatchThreads(mtl.Size.of(((n + 63) / 64) * 128, pairs / (if (tall) @as(usize, 64) else 32) + 512, 1), mtl.Size.of(128, 1, 1));
        p.barrier();
    }

    /// The block input from streams hn (after hc_norm wrote b.ssp): normed, down + inject, act, up, the stream mix.
    pub fn hc(p: *Prompt, m: *Model, hn: Buf, w: Hc, rows: usize, inj: Buf) void {
        if (p.skip & 16 != 0) return;
        const b = &p.b;
        const nd = w.dw.b.length() / (WIDE * 6 / 32 * 4);
        p.bind(p.pl[0], &.{ hn, b.ssp, w.scale, m.t.eps, b.normed });
        p.r.enc.dispatchThreads(mtl.Size.of(WIDE, rows, 1), mtl.Size.of(256, 1, 1));
        p.barrier();
        if (p.skip & 8 == 0) { // the down projection's 10,240-deep sum in 8 parts, added in order
            const parts: usize = 8;
            p.bind(p.splitk, &.{ w.dw, w.ds, w.db, b.normed });
            const prm = [4]i32{ WIDE, @intCast(nd), @intCast(rows), @intCast(parts) };
            p.r.enc.setBytes(std.mem.asBytes(&prm), 4);
            p.r.enc.setBuffer(b.part.b, b.part.off, 5);
            p.r.enc.dispatchThreads(mtl.Size.of(((nd + 63) / 64) * 128, (rows + 63) / 64, parts), mtl.Size.of(128, 1, 1));
            p.barrier();
            p.bind(p.parts_sum, &.{b.part});
            const ps = [2]i32{ @intCast(parts), @intCast(rows * nd) };
            p.r.enc.setBytes(std.mem.asBytes(&ps), 1);
            p.r.enc.setBuffer(b.dn.b, b.dn.off, 2);
            p.r.enc.dispatchThreads(mtl.Size.of(rows * nd, 1, 1), mtl.Size.of(256, 1, 1));
            p.barrier();
        }
        p.bind(p.pl[1], &.{b.dn});
        const ndi: i32 = @intCast(nd);
        p.r.enc.setBytes(std.mem.asBytes(&ndi), 1);
        p.r.enc.setBuffer(b.hact.b, b.hact.off, 2);
        p.r.enc.setBuffer(inj.b, inj.off, 3);
        p.r.enc.dispatchThreads(mtl.Size.of(nd, rows, 1), mtl.Size.of(256, 1, 1));
        p.barrier();
        p.qmm(b.hact, .{ w.uw, w.us, w.ub }, 320, WIDE, rows, b.up, 0);
        p.bind(p.pl[2], &.{ b.up, b.normed, b.mixed });
        p.r.enc.dispatchThreads(mtl.Size.of(D, rows, 1), mtl.Size.of(256, 1, 1));
        p.barrier();
    }

    /// Routed experts by expert over the chunk's pairs, the shared expert dense; outputs in the combine's layout.
    pub fn moe(p: *Prompt, L: *Layer, rows: usize) void {
        const b = &p.b;
        const r = p.r;
        const pairs = rows * 10;
        if (p.skip & 32 != 0) {
            p.gather(b.xs, L.ex[0..3], D, 640, pairs, b.g);
            p.gather(b.xs, L.ex[3..6], D, 640, pairs, b.u);
            p.gather(b.a, L.ex[12..15], 640, D, pairs, b.ds);
            return;
        }
        if (p.skip & 64 == 0) p.bind(p.router_mm, &.{ L.router, b.mixed }) else p.bind(p.pl[10], &.{ b.lg, b.lg });
        const rp = [3]i32{ D, 513, @intCast(rows) };
        r.enc.setBytes(std.mem.asBytes(&rp), 2);
        r.enc.setBuffer(b.lg.b, b.lg.off, 3);
        r.enc.dispatchThreads(if (p.skip & 64 == 0) mtl.Size.of(((513 + 63) / 64) * 128, (rows + 63) / 64, 1) else mtl.Size.of(1, 1, 1), mtl.Size.of(if (p.skip & 64 == 0) 128 else 1, 1, 1));
        p.barrier();
        if (p.skip & 128 == 0) {
        p.bind(p.pl[4], &.{ b.lg, b.pick, b.wts, b.cnt });
        r.enc.dispatchThreads(mtl.Size.of(32, rows, 1), mtl.Size.of(32, 1, 1));
        p.barrier();
        p.bind(p.pl[5], &.{ b.cnt, b.off, b.cur });
        r.enc.dispatchThreads(mtl.Size.of(512, 1, 1), mtl.Size.of(512, 1, 1));
        p.barrier();
        p.bind(p.pl[6], &.{ b.pick, b.cur, b.row_of });
        const np: i32 = @intCast(pairs);
        r.enc.setBytes(std.mem.asBytes(&np), 3);
        r.enc.dispatchThreads(mtl.Size.of(pairs, 1, 1), mtl.Size.of(256, 1, 1));
        p.barrier();
        }
        if (p.skip & 256 == 0) {
            p.bind(p.pl[7], &.{ b.mixed, b.row_of, b.xs });
            r.enc.dispatchThreads(mtl.Size.of(D / 8, pairs, 1), mtl.Size.of(64, 1, 1));
            p.barrier();
        }
        p.gather(b.xs, L.ex[0..3], D, 640, pairs, b.g);
        p.gather(b.xs, L.ex[3..6], D, 640, pairs, b.u);
        p.bind(p.pl[8], &.{ b.g, b.u, b.a });
        r.enc.dispatchThreads(mtl.Size.of(pairs * 640, 1, 1), mtl.Size.of(256, 1, 1));
        p.barrier();
        p.gather(b.a, L.ex[12..15], 640, D, pairs, b.ds);
        if (p.skip & 256 == 0) {
            p.bind(p.pl[9], &.{ b.ds, b.row_of, b.ydown });
            r.enc.dispatchThreads(mtl.Size.of(D / 8, pairs, 1), mtl.Size.of(64, 1, 1));
            p.barrier();
        }
        if (p.skip & 512 != 0) return;
        p.qmm(b.mixed, .{ L.ex[6], L.ex[7], L.ex[8] }, D, 640, rows, b.sg, 0);
        p.qmm(b.mixed, .{ L.ex[9], L.ex[10], L.ex[11] }, D, 640, rows, b.su, 0);
        p.bind(p.pl[8], &.{ b.sg, b.su, b.sa });
        r.enc.dispatchThreads(mtl.Size.of(rows * 640, 1, 1), mtl.Size.of(256, 1, 1));
        p.barrier();
        p.qmm(b.sa, .{ L.ex[15], L.ex[16], L.ex[17] }, 640, D, rows, .{ .b = b.ydown.b, .off = 10 * D * 2 }, 11 * D);
    }

    /// Each kernel class alone, 48 times in one command buffer, on the buffers the last chunk left (layer 0's and 3's
    /// weights): its GPU time a chunk.
    pub fn classes(p: *Prompt, m: *Model, rows: usize) !void {
        const r = p.r;
        const b = &p.b;
        const L0 = &m.layers[0];
        const L3 = &m.layers[3];
        const pairs = rows * 10;
        const names = [_][]const u8{ "router", "top-k+offsets+sort", "row gather+scatter", "expert gate+up gathers", "expert act", "expert down gather", "shared expert", "hyper-connection", "DeltaNet in+out projections", "DeltaNet pre", "DeltaNet scan", "DeltaNet post", "attention proj+o", "attention (tensor units)", "hc norms (3)" };
        for (names, 0..) |name, which| {
            const cb = r.queue.commandBuffer();
            r.enc = cb.compute(if (r.serial) .serial else .concurrent);
            for (0..48) |_| {
                switch (which) {
                    0 => {
                        p.bind(p.router_mm, &.{ L0.router, b.mixed });
                        const rp = [3]i32{ D, 513, @intCast(rows) };
                        r.enc.setBytes(std.mem.asBytes(&rp), 2);
                        r.enc.setBuffer(b.lg.b, b.lg.off, 3);
                        r.enc.dispatchThreads(mtl.Size.of(((513 + 63) / 64) * 128, (rows + 63) / 64, 1), mtl.Size.of(128, 1, 1));
                        p.barrier();
                    },
                    1 => {
                        p.bind(p.pl[4], &.{ b.lg, b.pick, b.wts, b.cnt });
                        r.enc.dispatchThreads(mtl.Size.of(32, rows, 1), mtl.Size.of(32, 1, 1));
                        p.barrier();
                        p.bind(p.pl[5], &.{ b.cnt, b.off, b.cur });
                        r.enc.dispatchThreads(mtl.Size.of(512, 1, 1), mtl.Size.of(512, 1, 1));
                        p.barrier();
                        p.bind(p.pl[6], &.{ b.pick, b.cur, b.row_of });
                        const np: i32 = @intCast(pairs);
                        r.enc.setBytes(std.mem.asBytes(&np), 3);
                        r.enc.dispatchThreads(mtl.Size.of(pairs, 1, 1), mtl.Size.of(256, 1, 1));
                        p.barrier();
                    },
                    2 => {
                        p.bind(p.pl[7], &.{ b.mixed, b.row_of, b.xs });
                        r.enc.dispatchThreads(mtl.Size.of(D / 8, pairs, 1), mtl.Size.of(64, 1, 1));
                        p.barrier();
                        p.bind(p.pl[9], &.{ b.ds, b.row_of, b.ydown });
                        r.enc.dispatchThreads(mtl.Size.of(D / 8, pairs, 1), mtl.Size.of(64, 1, 1));
                        p.barrier();
                    },
                    3 => {
                        p.gather(b.xs, L0.ex[0..3], D, 640, pairs, b.g);
                        p.gather(b.xs, L0.ex[3..6], D, 640, pairs, b.u);
                    },
                    4 => {
                        p.bind(p.pl[8], &.{ b.g, b.u, b.a });
                        r.enc.dispatchThreads(mtl.Size.of(pairs * 640, 1, 1), mtl.Size.of(256, 1, 1));
                        p.barrier();
                    },
                    5 => p.gather(b.a, L0.ex[12..15], 640, D, pairs, b.ds),
                    6 => {
                        p.qmm(b.mixed, .{ L0.ex[6], L0.ex[7], L0.ex[8] }, D, 640, rows, b.sg, 0);
                        p.qmm(b.mixed, .{ L0.ex[9], L0.ex[10], L0.ex[11] }, D, 640, rows, b.su, 0);
                        p.bind(p.pl[8], &.{ b.sg, b.su, b.sa });
                        r.enc.dispatchThreads(mtl.Size.of(rows * 640, 1, 1), mtl.Size.of(256, 1, 1));
                        p.barrier();
                        p.qmm(b.sa, .{ L0.ex[15], L0.ex[16], L0.ex[17] }, 640, D, rows, .{ .b = b.ydown.b, .off = 10 * D * 2 }, 11 * D);
                    },
                    7 => p.hc(m, b.h[0], L0.ahc, rows, b.inj_a),
                    8 => {
                        p.qmm(b.mixed, p.proj[0], D, 16480, rows, b.p, 0);
                        p.qmm(b.gout, p.out[0], 6144, D, rows, b.branch, 0);
                    },
                    9, 10, 11 => {
                        const ri: i32 = @intCast(rows);
                        if (which == 9) {
                            p.bind(p.pl[11], &.{ b.p, L0.cs[0], L0.conv, L0.alog, L0.dt });
                            r.enc.setBytes(std.mem.asBytes(&ri), 5);
                            for ([_]Buf{ b.qn, b.kn, b.v, b.gg, b.beta, L0.cs[1] }, 6..) |bb, j| r.enc.setBuffer(bb.b, bb.off, j);
                            r.enc.dispatchThreads(mtl.Size.of(80 * 128, rows, 1), mtl.Size.of(128, 1, 1));
                        } else if (which == 10) {
                            p.bind(p.pl[12], &.{ b.qn, b.kn, b.v, b.gg, b.beta, L0.so[0] });
                            r.enc.setBytes(std.mem.asBytes(&ri), 6);
                            r.enc.setBuffer(b.ys.b, b.ys.off, 7);
                            r.enc.setBuffer(L0.so[1].b, L0.so[1].off, 8);
                            r.enc.dispatchThreads(mtl.Size.of(48 * 4 * 1024, 1, 1), mtl.Size.of(1024, 1, 1));
                        } else {
                            p.bind(p.pl[13], &.{ b.ys, b.p, L0.norm, m.t.eps, b.gout });
                            r.enc.dispatchThreads(mtl.Size.of(48 * 128, rows, 1), mtl.Size.of(128, 1, 1));
                        }
                        p.barrier();
                    },
                    12 => {
                        p.qmm(b.mixed, p.proj[3], D, 13952, rows, b.p, 0);
                        p.qmm(b.aout, p.out[3], 6144, D, rows, b.branch, 0);
                    },
                    13 => {
                        p.bind(p.attn256, &.{ b.q, L3.keys, L3.vals, b.p });
                        const ap = [4]i32{ @intCast(rows), @intCast(rows), 0, CAP };
                        r.enc.setBytes(std.mem.asBytes(&ap), 4);
                        r.enc.setBuffer(m.t.scale.b, m.t.scale.off, 5);
                        r.enc.setBuffer(b.aout.b, b.aout.off, 6);
                        r.enc.dispatchThreads(mtl.Size.of(((rows + 63) / 64) * 128, 24, 1), mtl.Size.of(128, 1, 1));
                        p.barrier();
                    },
                    14 => {
                        try r.callRows("q4_hc_norm_none#[10240]", rows, &.{b.h[0]}, &.{ b.h[1], b.ssp }, null);
                        try r.callRows("q4_hc_norm_plain#[10240]", rows, &.{ b.h[1], b.inj_a, b.branch }, &.{ b.h[0], b.ssp }, null);
                        try r.callRows("q4_hc_norm_grouped#[10240]", rows, &.{ b.h[0], b.inj_m, b.ydown, b.wts, b.lg }, &.{ b.h[1], b.ssp }, null);
                    },
                    else => {},
                }
            }
            r.enc.end();
            cb.commit();
            cb.wait();
            std.debug.print("  {s:28} {d:7.2} ms a chunk (48 layers' worth)\n", .{ name, cb.gpuSeconds() * 1e3 });
        }
    }

    pub fn copyWords(p: *Prompt, src: Buf, dst: Buf, words: usize) void {
        p.bind(p.pl[10], &.{ src, dst });
        p.r.enc.dispatchThreads(mtl.Size.of(words, 1, 1), mtl.Size.of(256, 1, 1));
        p.barrier();
    }

    /// One prompt chunk from the model's position: caches, DeltaNet states, the n-gram tail and history move past it.
    /// Returns the greedy token after it; m.last holds the chunk's streams before the final mixer.
    pub fn chunk(p: *Prompt, m: *Model, gpa: std.mem.Allocator, tokens: []const u32) !u32 {
        const r = p.r;
        const b = &p.b;
        const t = &m.t;
        const rows = tokens.len;
        if (rows == 0 or rows > PMAX) return error.ChunkSize;
        @memcpy(b.ids.b.slice(u32, rows), tokens);
        for (0..rows) |i| {
            b.pos.b.slice(i32, PMAX)[i] = @intCast(m.pos + i);
            b.nk.b.slice(i32, PMAX)[i] = @intCast(m.pos + i + 1);
        }
        const kvm = b.kvmeta.b.slice(u32, 3);
        kvm[0], kvm[1], kvm[2] = .{ @intCast(m.pos), CAP, @intCast(rows) };
        b.rows.b.slice(i32, 1)[0] = @intCast(rows);
        { // the n-gram ids after the history (NGramEmbedding.ids)
            const pp = &m.ple;
            const seq = try gpa.alloc(i64, 2 + rows);
            defer gpa.free(seq);
            seq[0], seq[1] = .{ pp.hist[0], pp.hist[1] };
            for (tokens, 0..) |tok, i| seq[2 + i] = tok;
            const out = b.pids.b.slice(u32, 16 * PMAX);
            var last_eos: i64 = -1;
            if (seq[0] == pp.eos) last_eos = 0;
            if (seq[1] == pp.eos) last_eos = 1;
            for (0..rows) |row| {
                const at = 2 + row;
                const in_seg = @as(i64, @intCast(at)) - (last_eos + 1);
                var sh: [3]i64 = undefined;
                for (0..3) |s2| sh[s2] = if (in_seg >= @as(i64, @intCast(s2))) seq[at - s2] else pp.eos;
                for (2..4) |ng| {
                    var mixed: i64 = sh[0] *% pp.mult[0];
                    for (1..ng) |q2| mixed ^= sh[q2] *% pp.mult[q2];
                    for (0..8) |k| {
                        const hh = (ng - 2) * 8 + k;
                        out[row * 16 + hh] = @intCast(@mod(mixed, pp.sizes[hh]) + pp.offsets[hh]);
                    }
                }
                if (seq[at] == pp.eos) last_eos = @intCast(at);
            }
        }
        const cin_old = m.ple.cin.b.contents()[m.ple.cin.off..];
        @memcpy(b.cin.b.contents()[0 .. PLE_TAIL * WIDE * 2], cin_old[0 .. PLE_TAIL * WIDE * 2]);
        const cb = r.queue.commandBuffer();
        r.enc = cb.compute(if (r.serial) .serial else .concurrent);
        try r.callRows("qa_embed_rows@embed", rows, &.{ b.ids, m.embed[0], m.embed[1], m.embed[2] }, &.{b.h[0]}, null);
        var cur: usize = 0;
        var pending = false;
        const a = m.state;
        for (0..LAYERS) |i| {
            const L = &m.layers[i];
            if (i == 1) {
                if (pending) try r.callRows("q4_hc_norm_grouped#[10240]", rows, &.{ b.h[cur], b.inj_m, b.ydown, b.wts, b.lg }, &.{ b.h[1 - cur], b.ssp }, null);
                if (pending) cur = 1 - cur;
                pending = false;
                const pp = &m.ple;
                var tabs: [2 + 3 * GROUPS]Buf = undefined;
                tabs[0] = b.pids;
                tabs[1] = pp.starts;
                for (0..3 * GROUPS) |j| tabs[2 + j] = pp.tables[j];
                try r.callRows("qa_ple_lookup@ple", rows, &tabs, &.{b.emb}, null);
                p.qmm(b.emb, p.ple_kv, D, 12800, rows, b.kvp, 0);
                try r.callRows("q4_ple_gate@ple", rows, &.{ b.kvp, b.h[cur], pp.ks, pp.qs, pp.cs, t.eps }, &.{ b.gated, .{ .b = b.cin.b, .off = PLE_TAIL * WIDE * 2 } }, null);
                try r.callRows("q4_ple_conv@ple", rows, &.{ b.cin, pp.conv, b.gated, b.h[cur] }, &.{b.hout}, null);
                try r.callRows("q4_hc_norm_none#[10240]", rows, &.{b.hout}, &.{ b.h[1 - cur], b.ssp }, null);
            } else if (!pending) {
                try r.callRows("q4_hc_norm_none#[10240]", rows, &.{b.h[cur]}, &.{ b.h[1 - cur], b.ssp }, null);
            } else {
                try r.callRows("q4_hc_norm_grouped#[10240]", rows, &.{ b.h[cur], b.inj_m, b.ydown, b.wts, b.lg }, &.{ b.h[1 - cur], b.ssp }, null);
            }
            cur = 1 - cur;
            p.hc(m, b.h[cur], L.ahc, rows, b.inj_a);
            if (L.linear) {
                p.qmm(b.mixed, p.proj[i], D, 16480, rows, b.p, 0);
                const cs_in: Buf = .{ .b = L.cs[a].b, .off = L.cs[a].off + m.state_row * CS_ROW };
                const so_in: Buf = .{ .b = L.so[a].b, .off = L.so[a].off + m.state_row * SO_ROW };
                if (p.skip & 2 == 0) { // conv, norms and gates of every row; the recurrence; the gated norm of every row
                    const ri: i32 = @intCast(rows);
                    p.bind(p.pl[11], &.{ b.p, cs_in, L.conv, L.alog, L.dt });
                    r.enc.setBytes(std.mem.asBytes(&ri), 5);
                    for ([_]Buf{ b.qn, b.kn, b.v, b.gg, b.beta, L.cs[1 - a] }, 6..) |bb, j| r.enc.setBuffer(bb.b, bb.off, j);
                    r.enc.dispatchThreads(mtl.Size.of(80 * 128, rows, 1), mtl.Size.of(128, 1, 1));
                    p.barrier();
                    p.bind(p.pl[12], &.{ b.qn, b.kn, b.v, b.gg, b.beta, so_in });
                    r.enc.setBytes(std.mem.asBytes(&ri), 6);
                    r.enc.setBuffer(b.ys.b, b.ys.off, 7);
                    r.enc.setBuffer(L.so[1 - a].b, L.so[1 - a].off, 8);
                    r.enc.dispatchThreads(mtl.Size.of(48 * 4 * 1024, 1, 1), mtl.Size.of(1024, 1, 1));
                    p.barrier();
                    p.bind(p.pl[13], &.{ b.ys, b.p, L.norm, t.eps, b.gout });
                    r.enc.dispatchThreads(mtl.Size.of(48 * 128, rows, 1), mtl.Size.of(128, 1, 1));
                    p.barrier();
                }
                p.qmm(b.gout, p.out[i], 6144, D, rows, b.branch, 0);
            } else {
                p.qmm(b.mixed, p.proj[i], D, 13952, rows, b.p, 0);
                try r.callRows("q4_attn_prep@att", rows, &.{ b.p, b.pos, L.qn, L.kn, L.iqn, t.eps, t.log2base }, &.{ b.q, b.kout, b.iq }, null);
                p.bind(r.kv_pipe, &.{ b.kout, b.p, L.keys, L.vals, L.raw, b.kvmeta });
                r.enc.dispatchThreads(mtl.Size.of(512 * rows, 1, 1), mtl.Size.of(256, 1, 1));
                p.barrier();
                const sparse = p.sel != null and p.sel.?.meta(m.pos, rows);
                if (sparse) { // rows past the dense range: each row's selected blocks and tail (the decode's kernels)
                    var sl = &p.sel.?;
                    try sl.encode(r, L, b.iq, t.eps, t.log2base, m.pos, rows);
                    const dense_ids = r.shapes.get("IDS_shape").?;
                    try r.shapes.put(r.arena, "IDS_shape", sl.ids_shape);
                    try r.callRows("q4_attn_parts#[24, 256]", rows, &.{ b.q, L.keys, L.vals, sl.keys, sl.counts, sl.sparse, t.scale }, &.{ b.po, b.pm }, null);
                    try r.shapes.put(r.arena, "IDS_shape", dense_ids);
                    try r.callRows("q4_attn_merge_gate#[24, 16, 256]", rows, &.{ b.po, b.pm, b.p }, &.{b.aout}, null);
                } else if (p.skip & 4 == 0) { // causal attention over the cache and the chunk, gated on the way out
                    p.bind(p.attn256, &.{ b.q, L.keys, L.vals, b.p });
                    const ap = [4]i32{ @intCast(rows), @intCast(m.pos + rows), @intCast(m.pos), CAP };
                    r.enc.setBytes(std.mem.asBytes(&ap), 4);
                    r.enc.setBuffer(t.scale.b, t.scale.off, 5);
                    r.enc.setBuffer(b.aout.b, b.aout.off, 6);
                    r.enc.dispatchThreads(mtl.Size.of(((rows + 63) / 64) * 128, 24, 1), mtl.Size.of(128, 1, 1));
                    p.barrier();
                }
                p.qmm(b.aout, p.out[i], 6144, D, rows, b.branch, 0);
            }
            try r.callRows("q4_hc_norm_plain#[10240]", rows, &.{ b.h[cur], b.inj_a, b.branch }, &.{ b.h[1 - cur], b.ssp }, null);
            cur = 1 - cur;
            p.hc(m, b.h[cur], L.mhc, rows, b.inj_m);
            p.moe(L, rows);
            pending = true;
        }
        try r.callRows("q4_hc_norm_grouped#[10240]", rows, &.{ b.h[cur], b.inj_m, b.ydown, b.wts, b.lg }, &.{ b.h[1 - cur], b.ssp }, null);
        cur = 1 - cur;
        m.last = b.h[cur];
        p.hc(m, b.h[cur], m.mix, rows, b.inj_a);
        r.rows = 1;
        t.mdims.b.slice(i32, 2)[0] = 1;
        try m.lane(.{ .b = b.mixed.b, .off = b.mixed.off + (rows - 1) * D * 2 }, D, m.head, "lane_qmm_bytes_grouped@head", t.logits);
        r.enc.setPipeline(r.argmax_pipe);
        r.enc.setBuffer(t.logits.b, 0, 0);
        r.enc.setBuffer(t.picks.b, 0, 1);
        r.enc.setBuffer(t.vocab.b, 0, 2);
        r.enc.dispatchThreads(mtl.Size.of(1024, 1, 1), mtl.Size.of(1024, 1, 1));
        try m.finish(cb);
        m.state = 1 - a;
        m.state_row = 0;
        m.pos += rows;
        @memcpy(cin_old[0 .. PLE_TAIL * WIDE * 2], b.cin.b.contents()[rows * WIDE * 2 .. (rows + PLE_TAIL) * WIDE * 2]);
        for (tokens) |tok| m.ple.hist = .{ m.ple.hist[1], tok };
        return t.picks.b.slice(u32, 1)[0];
    }

    /// The MTP head's keys and values for prompt rows start .. start + n from their streams (n rows of `streams`)
    /// and next tokens: the head's layer up to its cache write, at prompt widths. The head's full layer runs later
    /// on the last prompt row, with the first generated token.
    pub fn mtpKeys(p: *Prompt, m: *Model, start: usize, nexts: []const u32, streams: Buf) !void {
        const r = p.r;
        const b = &p.b;
        const t = &m.t;
        const h = &m.mtp;
        const n = nexts.len;
        if (n == 0) return;
        if (n > PMAX) return error.ChunkSize;
        @memcpy(b.mids.b.slice(u32, n), nexts);
        for (0..n) |i| b.pos.b.slice(i32, PMAX)[i] = @intCast(start + i);
        const kvm = b.kvmeta.b.slice(u32, 3);
        kvm[0], kvm[1], kvm[2] = .{ @intCast(start), CAP, @intCast(n) };
        b.n_add.b.slice(u32, 1)[0] = @intCast(n * WIDE);
        const cb = r.queue.commandBuffer();
        r.enc = cb.compute(if (r.serial) .serial else .concurrent);
        try r.callRows("mtp:qa_embed_rows@embed", n, &.{ b.mids, m.embed[0], m.embed[1], m.embed[2] }, &.{b.emb}, null);
        try r.callRows("mtp:q4_rms_rows@mtp.enorm", n, &.{ b.emb, h.enorm, t.eps }, &.{b.branch}, null);
        r.denseRows(b.branch, D, h.fce, n, b.aout);
        try r.callRows("mtp:q4_rms_rows@mtp.hnorm", n, &.{ streams, h.hnorm, t.eps }, &.{b.normed}, null);
        r.denseRows(b.normed, D, h.fch, 4 * n, b.gated);
        p.bind(r.add_pipe, &.{ b.aout, b.gated, b.xs, b.n_add });
        r.enc.dispatchThreads(mtl.Size.of(n * WIDE, 1, 1), mtl.Size.of(256, 1, 1));
        p.barrier();
        try r.callRows("mtp:q4_hc_norm_none#[10240]", n, &.{b.xs}, &.{ b.hout, b.ssp }, null);
        p.hc(m, b.hout, h.ahc, n, b.inj_a);
        r.denseRows(b.mixed, D, h.proj, n, b.p);
        try r.callRows("mtp:q4_attn_prep@mtp.att", n, &.{ b.p, b.pos, h.qn, h.kn, h.iqn, t.eps, t.log2base }, &.{ b.q, b.kout, b.iq }, null);
        p.bind(r.kv_pipe, &.{ b.kout, b.p, h.keys, h.vals, h.raw, b.kvmeta });
        r.enc.dispatchThreads(mtl.Size.of(512 * n, 1, 1), mtl.Size.of(256, 1, 1));
        try m.finish(cb);
    }
};

pub fn jsonInt(v: std.json.Value) i64 {
    return switch (v) {
        .integer => |x| x,
        .float => |x| @intFromFloat(x),
        else => 0,
    };
}
