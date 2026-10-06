//! Flash Next on our Metal runtime: the Python engine's recorded kernels (a dump from tools/zig/flashnext_dump.py)
//! replayed from Zig, with our own kernels for wide experts, prompt chunks, block selection and GPU-side rounds.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const segments = @import("../../core/segments.zig");
const tpm = @import("tp.zig");
const mark_slots = @import("marks.zig");
pub const split = @import("split.zig");
pub const dense = @import("dense.zig");
pub const gdn_step = @import("gdn.zig");
pub const Tp2 = tpm.Tp2;
pub const frags = @import("../../core/frags.zig");

pub const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

pub const D = 2560;
pub const WIDE = 4 * D;
pub const LAYERS = 48;
pub const VOCAB = 248320;
pub const CAP = 262144; // keys an attention layer holds in this runner
pub const PLE_TAIL = 9;
pub const GROUPS = 8;
pub const MAXR = 16; // a decode window's rows at most
const HEAD_TILES = VOCAB / 32; // the vocabulary head's 32-column tiles
pub const CS_ROW = 3 * WIDE * 2; // a DeltaNet conv state row (bytes)
pub const SO_ROW = 48 * 128 * 128 * 4; // a DeltaNet recurrent state row (bytes)

pub const Buf = struct {
    b: mtl.Buffer,
    off: usize = 0,

    /// The same buffer `bytes` further on.
    pub fn at(x: Buf, bytes: usize) Buf {
        return .{ .b = x.b, .off = x.off + bytes };
    }
};
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
    \\  long seq[18];
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
    \\// The MTP head's kept row (keep[0] - 1) of its attention output, streams and inject gates into row 0, in place
    \\kernel void fz_mtp_take(device bfloat* aout [[buffer(0)]], device bfloat* h [[buffer(1)]], device bfloat* inj [[buffer(2)]],
    \\    device const int* keep [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    \\  const int k = keep[0] - 1;
    \\  if (k <= 0) return;
    \\  if (i < 6144) aout[i] = aout[k * 6144 + i];
    \\  h[i] = h[k * 10240 + i];
    \\  if (i < 4) inj[i] = inj[k * 4 + i];
    \\}
    \\// The round's verdict on the GPU: drafts kept, the emitted tokens to the ring, the next window's pending token,
    \\// positions and n-gram history; ar = keep, target length, round, then the meta blocks the next round reads:
    \\// the window's positions [4, 20), key counts [20, 36), cache meta [36, 39); the head's absorb [40, 56), [56, 72),
    \\// [72, 75); chained draft j at 76 + 36 (j - 1): positions, key counts (16 each), cache meta.
    \\// cfg = this window's rows, the next window's rows, cache capacity. The ring keeps 512 rounds of 20 words: keep,
    \\// the drafts' source (1: copied, from ar[801]), the copy's match length (ar[802]), the emitted tokens; H, the
    \\// token history (prompt, then every emitted token; its length in ar[800]), gets this round's.
    \\kernel void fz_accept(device uint* wids [[buffer(0)]], device const uint* picks [[buffer(1)]],
    \\    device int* ar [[buffer(2)]], device uint* ring [[buffer(3)]], device long* pm [[buffer(4)]],
    \\    constant uint4& cfg [[buffer(5)]], device uint* H [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
    \\  if (tid != 0) return;
    \\  const int W = int(cfg.x), Wn = int(cfg.y), cap = int(cfg.z);
    \\  int keep = 1;
    \\  while (keep < W && wids[keep] == picks[keep - 1]) keep++;
    \\  const int round = ar[2], slot = (round & 511) * 20;
    \\  ring[slot] = uint(keep);
    \\  ring[slot + 1] = uint(ar[801]);
    \\  ring[slot + 2] = uint(ar[802]);
    \\  for (int i = 0; i < keep; i++) ring[slot + 3 + i] = picks[i];
    \\  const int hl = ar[800];
    \\  for (int i = 0; i < keep; i++) H[hl + i] = picks[i];
    \\  ar[800] = hl + keep;
    \\  for (int i = 0; i < keep; i++) { pm[0] = pm[1]; pm[1] = long(wids[i]); }
    \\  const int t_old = ar[1], t_new = t_old + keep;
    \\  ar[0] = keep; ar[1] = t_new; ar[2] = round + 1;
    \\  for (int i = 0; i < 16; i++) {
    \\    ar[4 + i] = i < Wn ? t_new + i : 0;
    \\    ar[20 + i] = i < Wn ? t_new + i + 1 : 0;
    \\    ar[40 + i] = i < W ? t_old + i : 0;
    \\    ar[56 + i] = i < W ? t_old + i + 1 : 0;
    \\  }
    \\  ar[36] = t_new; ar[37] = cap; ar[38] = Wn;
    \\  ar[72] = t_old; ar[73] = cap; ar[74] = W;
    \\  for (int j = 1; j < Wn - 1; j++) {
    \\    const int b = 76 + (j - 1) * 36;
    \\    for (int i = 0; i < 16; i++) { ar[b + i] = 0; ar[b + 16 + i] = 0; }
    \\    ar[b] = t_new + j - 1; ar[b + 16] = t_new + j;
    \\    ar[b + 32] = t_new + j - 1; ar[b + 33] = cap; ar[b + 34] = 1;
    \\  }
    \\  wids[0] = picks[keep - 1];
    \\}
    \\// Copy drafts: the latest earlier occurrence of the longest suffix (cfg.y .. 8 tokens) of the history H[0, L);
    \\// when there is one and it is at least cfg.z long or the tokens after it start with the head's first cfg.w drafts,
    \\// they replace the drafts wids[1 .. cfg.x) (as many as follow it); ar[801] = 1 then, else 0; ar[802] the match
    \\// length. One threadgroup of 1024.
    \\kernel void fz_lookup(device const uint* H [[buffer(0)]], device int* ar [[buffer(1)]], device uint* wids [[buffer(2)]],
    \\    constant uint4& cfg [[buffer(3)]], uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint sg [[simdgroup_index_in_threadgroup]]) {
    \\  const int L = ar[800], W = int(cfg.x), MINN = int(cfg.y), LONGN = int(cfg.z), AGREE = int(cfg.w);
    \\  const uint last = H[L - 1];
    \\  uint best = 0;
    \\  for (int p = int(t); p < L - 1; p += 1024) {
    \\    if (H[p] != last) continue;
    \\    int n = 1;
    \\    while (n < 8 && p - n >= 0 && H[p - n] == H[L - 1 - n]) n++;
    \\    best = max(best, (uint(n) << 24) | uint(p + 1));
    \\  }
    \\  best = simd_max(best);
    \\  threadgroup uint part[32];
    \\  if (lane == 0) part[sg] = best;
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (t != 0) return;
    \\  for (int i = 0; i < 32; i++) best = max(best, part[i]);
    \\  const int n = int(best >> 24), p = int(best & 0xffffffu) - 1;
    \\  ar[801] = 0;
    \\  ar[802] = n;
    \\  if (n < MINN || p < 0) return;
    \\  if (n < LONGN)
    \\    for (int i = 0; i < AGREE && i < W - 1; i++) if (p + 1 + i >= L || H[p + 1 + i] != wids[1 + i]) return;
    \\  const int k = min(W - 1, L - 1 - p);
    \\  for (int i = 0; i < k; i++) wids[1 + i] = H[p + 1 + i];
    \\  ar[801] = 1;
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
    \\  for (int c = 8 * int(lane); c < D; c += 256) {
    \\    float wv[8];
    \\    for (int j = 0; j < 8; j++) wv[j] = float(w[c + j]);
    \\    for (int j = 0; j < 8; j++) a = fma(float(xr[c + j]), wv[j], a);
    \\  }
    \\  const float total = simd_sum(a);
    \\  if (lane == 0) OUT[r * NE + e] = total;
    \\}
;

/// The hyper-connection's up projection for windows of 3-8 rows (`Run.hc_up`), compiled after the recorded
/// qa_hc_down_row header: qa_hc_up's one-row sums (thread (group, output): one group's product sum; each output's
/// chain over the groups; the sigmoid times the normed stream; the streams' sum) for 4 rows and 16 dims of every
/// stream a threadgroup, each thread's codes read once for its rows. Each threadgroup adds the down projection's
/// split partials for its rows, as the recorded tiles do, but there are half as many threadgroups and no serial MMA
/// loop. The same bits.
const hc_up_source =
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
    lookup_pipe: mtl.Pipeline = undefined,
    take_pipe: mtl.Pipeline = undefined,
    gpu_round: bool = false,
    ar: Buf = undefined,
    probe: ?Buf = null,
    lg_probe: ?Buf = null, // FZ_DBENCH: every layer's router logits, MAXR * 513 floats a layer
    xnew: bool = false,
    xgu_pipe: mtl.Pipeline = undefined,
    xdown_pipe: mtl.Pipeline = undefined,
    dense: bool = false,
    dense_target: bool = false,
    skip: u32 = 0,
    hcskip: u32 = 0, // FZ_HCSKIP: hyper-connection knock-outs by kernel (1 norms, 2 downs, 4 ups)
    hc_up: bool = false, // fz_hc_up for the hyper-connection's up projection at 3-8 rows (the recorded kernels' bits)
    hc_up_pipe: mtl.Pipeline = undefined,
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
    router_pipe: mtl.Pipeline = undefined,
    ggu_pipe: mtl.Pipeline = undefined,
    gdown_pipe: mtl.Pipeline = undefined,
    ul: Buf = undefined,
    dense16_pipe: mtl.Pipeline = undefined,
    tp_gdn: std.AutoHashMapUnmanaged(*Variant, mtl.Pipeline) = .empty, // TP: DeltaNet steps from value head 24
    tp_lane: std.AutoHashMapUnmanaged(u64, mtl.Pipeline) = .empty, // TP: recorded lane kernels over tile maps
    lane_new: bool = false, // the target's lane projections on fz_lane (dense.zig): the recorded bits, weights read ahead
    lane_pf: usize = 1, // groups a simdgroup reads ahead in fz_lane
    lane_pipes: std.AutoHashMapUnmanaged(u64, mtl.Pipeline) = .empty,
    lane_shape: std.AutoHashMapUnmanaged(*Variant, [3]usize) = .empty, // recorded lane kernels' N, K, SK
    gdn_pipe: ?mtl.Pipeline = null, // the DeltaNet window step on fz_gdn (gdn.zig) when set
    gdn_kept: ?mtl.Pipeline = null, // GPU-side rounds: fz_gdn keeping one state a layer, kept rows replayed (gdn.zig)
    xnew_header: []const u8 = "",
    tp: ?*Tp2 = null, // TP=2 across two Macs (tp.zig): the target layers' experts split by id, outputs exchanged
    tp_layer: bool = false, // the experts being encoded are a target layer's (the MTP head keeps all of its own)
    sel: ?Select = null, // long contexts: the indexer's pool, scores and selection
    gsel: ?GSelect = null, // the same in GPU-side rounds
    sel_meta_pipe: mtl.Pipeline = undefined,
    pool_abs_pipe: mtl.Pipeline = undefined,
    ngram: []const u8 = "shard_", // how the checkpoint spells its n-gram table shards (index.ngramSpelling)

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
        errdefer b.deinit();
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
        const fmt = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.{s}{d}.{s}";
        for (first..first + count) |s| total += (try r.entry(try std.fmt.bufPrint(&name, fmt, .{ r.ngram, s, suffix }))).len;
        const b = try r.device.buffer(total, opts);
        errdefer b.deinit();
        var at: usize = 0;
        for (first..first + count) |s| {
            const e = try r.entry(try std.fmt.bufPrint(&name, fmt, .{ r.ngram, s, suffix }));
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
        r.lookup_pipe = try mtl.Pipeline.init(r.device, glue, "fz_lookup", false);
        r.take_pipe = try mtl.Pipeline.init(r.device, glue, "fz_mtp_take", false);
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
            r.router_pipe = try mtl.Pipeline.init(r.device, lib, "fz_router", false);
            r.xfused_pipe = try mtl.Pipeline.init(r.device, lib, "fz_xfused", false);
            r.xgu_sx_pipe = try mtl.Pipeline.init(r.device, lib, "fz_xgu_sx", false);
            r.xdown_sx_pipe = try mtl.Pipeline.init(r.device, lib, "fz_xdown_sx", false);
            r.xpart = .{ .b = try r.buffer(MAXR * 11 * 4 * D * 4) };
            r.xdone = .{ .b = try r.buffer(MAXR * 11 * 4) };
            r.ggu_pipe = try mtl.Pipeline.init(r.device, lib, "fz_ggu", false);
            r.gdown_pipe = try mtl.Pipeline.init(r.device, lib, "fz_gdown", false);
            r.ul = .{ .b = try r.buffer((1 + 2 * 320 + 320 * 32) * 4) };
        }
        if (r.hc_up) r.compileHc() catch |err| { // the recorded kernels then
            std.log.warn("fz_hc_up off: {s}", .{@errorName(err)});
            r.hc_up = false;
        };
    }

    /// fz_hc_up after the recorded qa_hc_down_row header, refused unless every recorded qa_hc_up variant has the
    /// constants it takes.
    fn compileHc(r: *Run) !void {
        const a = r.arena;
        var header: ?[]const u8 = null;
        var it = r.variants.iterator();
        while (it.next()) |kv| {
            const name = kv.key_ptr.*;
            if (std.mem.indexOf(u8, name, "qa_hc_down_row") != null) {
                const text = try fileText(a, kv.value_ptr.*.file);
                header = text[0 .. std.mem.indexOf(u8, text, "[[max_total_threads_per_threadgroup") orelse return error.NoKernel];
            }
            if (std.mem.indexOf(u8, name, "qa_hc_up_") == null) continue;
            const text = try fileText(a, kv.value_ptr.*.file);
            for ([_][]const u8{ "int S = 4;", "int D = 2560;", "int BITS = 6;", "int GS = 32;", "int LOW = 320;", "int KS = 10;" }) |c|
                if (std.mem.indexOf(u8, text, c) == null) return error.HcConstants;
        }
        const src = try std.mem.concat(a, u8, &.{ header orelse return error.NoHcHeader, hc_up_source });
        const lib = try mtl.Library.fromSource(r.device, src, mtl.CompileOptions.mlx());
        r.hc_up_pipe = try mtl.Pipeline.init(r.device, lib, "fz_hc_up", false);
    }

    fn fileText(a: std.mem.Allocator, path: []const u8) ![]const u8 {
        const f = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(a, "{s}", .{path}, 0));
        return f.bytes[0..f.size];
    }

    /// A hyper-connection's down then up projection at `as_rows` from normed streams `hn` (sums of squares in `ssp`):
    /// the mixed rows and inject gates; with `hc_up` the up projection of 3-8 rows on fz_hc_up (the same bits).
    pub fn hcProject(r: *Run, down: []const u8, up: []const u8, as_rows: usize, hn: Buf, ssp: Buf, hc: Hc, eps: Buf, rows: Buf, part: Buf, mixed: Buf, inj: Buf) !void {
        try r.callAs(down, as_rows, &.{ hn, ssp, hc.scale, hc.dw, hc.ds, hc.db, eps, rows }, &.{part});
        const ins = [_]Buf{ hn, ssp, hc.scale, part, hc.uw, hc.us, hc.ub, eps, rows };
        if (!r.hc_up or as_rows < 3 or as_rows > 8) return r.callAs(up, as_rows, &ins, &.{ mixed, inj });
        if (r.skip & class(up) != 0 or r.hcskip & 4 != 0) return;
        const nd: u32 = @intCast(hc.dw.b.length() / (WIDE * 6 / 8));
        r.enc.setPipeline(r.hc_up_pipe);
        for (ins ++ [_]Buf{ mixed, inj }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        r.enc.setBytes(std.mem.asBytes(&nd), 11);
        r.enc.dispatchThreads(mtl.Size.of(D / 16 * 640, (as_rows + 3) / 4, 1), mtl.Size.of(640, 1, 1));
        if (!r.serial) r.enc.barrier();
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
    /// One row's dense projection over output tiles [t0, t0 + tn) only (32 outputs a tile), in the full layout.
    pub fn denseTiles(r: *Run, x: Buf, k: usize, l: Lane, y: Buf, t0: usize, tn: usize) void {
        if (tn == 0) return;
        const n = l.wq.b.length() * 4 / (3 * k);
        const narrow = n <= 4096;
        const dims = [4]u32{ 1, @intCast(n), @intCast(k), @intCast(t0) };
        r.enc.setPipeline(if (narrow) r.dense16_pipe else r.dense8_pipe);
        r.enc.setBuffer(x.b, x.off, 0);
        r.enc.setBuffer(l.wq.b, l.wq.off, 1);
        r.enc.setBuffer(l.sbt.b, l.sbt.off, 2);
        r.enc.setBuffer(y.b, y.off, 3);
        r.enc.setBytes(std.mem.asBytes(&dims), 4);
        const sk: usize = if (narrow) 16 else 8;
        r.enc.dispatchThreads(mtl.Size.of(32 * sk * tn, 1, 1), mtl.Size.of(32 * sk, 1, 1));
        if (!r.serial) r.enc.barrier();
    }

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
    /// The router's logits for up to `rows` rows (the count itself read from `rows_buf`): fz_router, the recorded kernel's
    /// bits with its rows in parallel, under FZ_XNEW; the recorded launch otherwise.
    pub fn router(r: *Run, role: []const u8, x: Buf, w: Buf, rows_buf: Buf, out: Buf, rows: usize) !void {
        if (!r.xnew) return r.call(role, &.{ x, w, rows_buf }, &.{out});
        if (r.skip & class(role) != 0) return;
        r.enc.setPipeline(r.router_pipe);
        for ([_]Buf{ x, w, rows_buf, out }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        r.enc.dispatchGroups(mtl.Size.of((rows * 513 + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        if (!r.serial) r.enc.barrier();
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
        const rows_own = if (r.tp != null and r.tp_layer) r.tp.?.ownRows(r.rows) else [2]usize{ 0, r.rows };
        if (rows_own[1] == rows_own[0]) return; // TP: a one-row window's experts are rank 0's
        const own = [4]u32{ @intCast(rows_own[0]), @intCast(rows_own[1]), 0, 0 };
        const pairs = (rows_own[1] - rows_own[0]) * 11;
        if (r.tp != null and r.tp_layer and r.xsx) return error.TpNeedsPlainExperts;
        r.enc.setPipeline(if (r.xsx) r.xgu_sx_pipe else r.xgu_pipe);
        const gu = [_]Buf{ x, lg, e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11], act, pick, wts };
        for (gu, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        r.enc.setBytes(std.mem.asBytes(&own), 17);
        r.enc.dispatchThreads(mtl.Size.of(128, 80, pairs), mtl.Size.of(128, 1, 1));
        if (!r.serial) r.enc.barrier();
        r.enc.setPipeline(if (r.xsx) r.xdown_sx_pipe else r.xdown_pipe);
        const dn = [_]Buf{ act, pick, e[12], e[13], e[14], e[15], e[16], e[17], rows_buf, y };
        for (dn, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        r.enc.setBytes(std.mem.asBytes(&own), 10);
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
        if (r.hcskip != 0) for ([_][]const u8{ "hc_norm", "hc_down", "hc_up" }, 0..) |n, i| {
            if (r.hcskip & (@as(u32, 1) << @intCast(i)) != 0 and std.mem.indexOf(u8, role, n) != null) return;
        };
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
/// GPU-side rounds: a ring entry's words (keep, source, match length, up to 17 tokens), the arena's history length.
pub const RING_WORDS = 20;
pub const AR_HLEN = 800;
/// The arena's words, and where its metadata blocks start.
pub const AR_WORDS = 1024;
pub const AR_POS = 4;
pub const AR_NK = 20;
pub const AR_KV = 36;
pub const AR_ABS_POS = 40;
pub const AR_ABS_NK = 56;
pub const AR_ABS_KV = 72;
pub const AR_CHAIN = 76;
pub const AR_CHAIN_STRIDE = 36;

pub const TOP = 512;
pub const KW = 4 * TOP + 3;
/// catchUp's first-block slots: the target's attention layers, then the head.
pub const CATCH = LAYERS / 4 + 1;

/// Slot k's first block into `words` (CATCH slots of 64 words, 256 bytes apart); its byte offset.
pub fn catchSlot(words: []i32, k: usize, first: usize) usize {
    words[k * 64] = @intCast(first);
    return k * 256;
}

test {
    _ = tpm; // tp.zig's tests: the window layout and the host service's wraps
}

test "catch-up slots: every layer's dispatch reads its own first block after the later layers' writes" {
    var words: [CATCH * 64]i32 = undefined;
    var at: [CATCH]usize = undefined;
    for (0..CATCH) |k| at[k] = catchSlot(&words, k, if (k + 1 < CATCH) 648 + k else 0); // pair rank 0: the head pooled none
    for (0..CATCH) |k| try std.testing.expectEqual(@as(i32, if (k + 1 < CATCH) @intCast(648 + k) else 0), words[at[k] / 4]);
}

pub const Select = struct {
    pool: *Variant,
    scores: *Variant,
    select: *Variant,
    start: Buf, // encode's first block, read at run time: one window a command buffer
    starts: Buf, // catchUp's, one slot a layer (CATCH): one command buffer catches up every layer
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
    nax_scores: ?mtl.Pipeline = null, // prompt rows: the block scores on the tensor units

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
            .start = try B.of(r, 16), .starts = try B.of(r, CATCH * 256), .sc = try B.of(r, rows_max * (CAP / 4) * 4), .keys = try B.of(r, rows_max * KW * 4),
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

    /// Before GPU-side rounds: pool each attention layer's and the head's blocks below `upto`, each from its own slot.
    pub fn catchUp(s: *Select, r: *Run, m: *Model, upto: usize) !void {
        var k: usize = 0;
        for (&m.layers) |*L| if (!L.linear) {
            try s.catchLayer(r, L, m.t.eps, m.t.log2base, upto, k);
            k += 1;
        };
        try s.catchLayer(r, &m.mtp, m.t.eps, m.t.log2base, upto, k);
    }

    /// One layer's catch-up, its first block in slot k (the GPU reads it when it runs the dispatch).
    pub fn catchLayer(s: *Select, r: *Run, L: anytype, eps: Buf, log2base: Buf, upto: usize, k: usize) !void {
        if (upto <= L.pooled_n) return;
        const at = catchSlot(s.starts.b.slice(i32, CATCH * 64), k, L.pooled_n);
        try r.bindV(s.pool, &.{ L.raw, .{ .b = s.starts.b, .off = at }, L.pool, eps, log2base }, &.{.{ .b = L.pooled.b, .off = L.pooled.off + L.pooled_n * 128 * 2 }});
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
        if (s.nax_scores) |pipe| {
            r.enc.setPipeline(pipe);
            r.enc.setBuffer(iq.b, iq.off, 0);
            r.enc.setBuffer(L.pooled.b, L.pooled.off, 1);
            const prm = [2]i32{ @intCast(rows), @intCast(nb) };
            r.enc.setBytes(std.mem.asBytes(&prm), 2);
            r.enc.setBuffer(s.sc.b, s.sc.off, 3);
            r.enc.dispatchThreads(mtl.Size.of(((nb + 63) / 64) * 128, (rows + 31) / 32, 1), mtl.Size.of(128, 1, 1));
        } else {
            try r.bindV(s.scores, &.{ iq, L.pooled, s.complete }, &.{s.sc});
            r.enc.dispatchThreads(mtl.Size.of(((nb + 7) / 8) * 256, (rows + 7) / 8, 1), mtl.Size.of(256, 1, 1));
        }
        if (!r.serial) r.enc.barrier();
        try r.bindV(s.select, &.{ s.sc, s.complete, s.ends }, &.{s.keys});
        r.enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
        if (!r.serial) r.enc.barrier();
    }
};


/// Block selection in GPU-side rounds: fz_sel_meta writes each round's rows, pooling range and pooled count; the pool
/// runs at absolute blocks (up to POOL_BLOCKS a round); scores cover an upper bound of the pooled blocks the host keeps.
/// fz_sel_meta's layout: complete, ends, counts, sparse (MAXR each), then start, count, pooled.
pub const SEL_POOLED = 4 * MAXR + 2;
/// Blocks one round's pool can owe: a window of up to MAXR rows completes at most this many.
pub const POOL_BLOCKS = (MAXR + 3) / 4;

test "a round's pool covers every block its window completes, a copied window's 16 rows too" {
    for (0..64) |tn| for (1..MAXR + 1) |w| try std.testing.expect((tn + w) / 4 - tn / 4 <= POOL_BLOCKS);
    try std.testing.expectEqual(@as(usize, 4), (5 + MAXR) / 4 - 5 / 4);
}

pub const GSelect = struct {
    sel: Buf, // complete[16] ends[16] counts[16] sparse[16] start count pooled
    sc: Buf,
    keys: Buf,
    pooled_shape: mtl.Buffer,
    q_shape: [MAXR + 1]mtl.Buffer, // by window rows
    sc_shape: mtl.Buffer,
    ids_shape: [MAXR + 1]mtl.Buffer,
    nb_ub: usize = 0,

    pub fn init(r: *Run, pooled: usize) !GSelect {
        var g: GSelect = .{
            .sel = .{ .b = try r.buffer(128 * 4) }, .sc = .{ .b = try r.buffer(MAXR * (CAP / 4) * 4) },
            .keys = .{ .b = try r.buffer(MAXR * KW * 4) }, .pooled_shape = try r.buffer(16), .q_shape = undefined,
            .sc_shape = try r.buffer(16), .ids_shape = undefined,
        };
        g.sel.b.slice(i32, 128)[SEL_POOLED] = @intCast(pooled);
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

/// How many token ids the MTP head scores, from the pack's `mtp.draft_ids` bytes, refused unless whole 64-row tiles up to the vocabulary.
pub fn draftCount(bytes: usize) !usize {
    const n = bytes / 4;
    if (n == 0 or n % 64 != 0 or n > VOCAB) return error.DraftVocab;
    return n;
}

test "the MTP head takes any draft list of whole 64-row tiles up to the vocabulary" {
    try std.testing.expectEqual(@as(usize, 79_616), try draftCount(79_616 * 4)); // the shipped list, padded to 64
    try std.testing.expectEqual(@as(usize, 135_040), try draftCount(135_040 * 4)); // it and every CJK id
    try std.testing.expectError(error.DraftVocab, draftCount(79_591 * 4));
    try std.testing.expectError(error.DraftVocab, draftCount((VOCAB + 64) * 4));
    try std.testing.expectError(error.DraftVocab, draftCount(0));
}

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
    slots: [2 * MAXR]Slot, // 0: host calls; GPU-side rounds: chained draft j at j, an absorb of n rows at MAXR - 1 + n
    last: Buf = undefined, // the last call's output streams (its kept row)
    pool: Buf = undefined, // its indexer's block pooling (the head's layer is a sparse-attention layer)
    pooled: Buf = undefined,
    pooled_n: usize = 0,
    gsel: ?GSelect = null, // its block selection in GPU-side rounds
};

/// Marks a prompt call can take its DeltaNet states at, their slots, and a layer's index among the DeltaNet layers (marks.zig).
pub const MARKS = mark_slots.MARKS;
pub const Marks = mark_slots.Slots;
const linearIndex = mark_slots.linearIndex;

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
    recs: ?[2]Buf = null, // GPU-side rounds with r.gdn_kept: each window's DeltaNet replay records, by round parity
    rec_slot: usize = 0, // the record this window writes (the other holds the previous window's)
    marks: ?Marks = null, // DeltaNet states at marks inside a prompt chunk, as a chunk ending there would leave them

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

    /// The replay records when this window keeps one DeltaNet state a layer (GPU-side rounds with r.gdn_kept).
    pub fn keptState(m: *const Model) ?[2]Buf {
        return if (m.r.gpu_round and m.r.gdn_kept != null) m.recs else null;
    }

    pub fn lane(m: *Model, x: Buf, k: usize, l: Lane, role: []const u8, y: Buf) !void {
        if (m.r.dense_target) return m.r.denseRows(x, k, l, m.r.rows, y);
        if (m.r.lane_new) return dense.lane(m.r, role, x, l, m.t.mdims, y);
        if (!m.r.fused_xsum) try m.r.call(if (k == D) "lane_qmm_xsum#[2560]" else "lane_qmm_xsum#[6144]", &.{ x, m.t.mdims }, &.{m.t.xs});
        try m.r.call(role, &.{ x, m.t.xs, l.wq, l.sbt, m.t.mdims }, &.{y});
    }

    pub fn hcProject(m: *Model, hn: Buf, hc: Hc, down: []const u8, up: []const u8, inj: Buf) !void {
        const t = &m.t;
        const as_rows = if (m.r.hc_mma and m.r.rows > 1) 8 else m.r.rows;
        try m.r.hcProject(down, up, as_rows, hn, t.ssp, hc, t.eps, t.rows, t.part, t.mixed, inj);
    }

    pub fn grouped(m: *Model, h: Buf, out: Buf) !void {
        const t = &m.t;
        if (m.r.tp) |tp| return if (m.r.skip & TP_CLASS == 0) tp.combine(m.r.enc, h, t.inj_m, out, t.ssp, t.rows, m.r.rows); // TP: each rank's rows' branches
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
        const pos8 = t.pos8.b.slice(i32, MAXR);
        const nk8 = t.nk8.b.slice(i32, MAXR);
        for (0..MAXR) |i| {
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
        const ids = t.ids8.b.slice(u32, MAXR);
        for (0..MAXR) |i| ids[i] = if (i < rows) tokens[i] else 0;
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
        if (m.r.tp) |tp| if (tp.failed.load(.acquire)) return error.TpLinkFailed;
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
            var plain = false; // speed-up mode's DeltaNet exchange made the stream update
            if (L.linear and r.tp != null) { // TP: this Mac's 8 key heads and 24 value heads, then one partial-sum exchange
                const tp = r.tp.?;
                const k0: usize = tp.rank;
                if (!r.fused_xsum) try r.call("lane_qmm_xsum#[2560]", &.{ t.mixed, t.mdims }, &.{t.xs});
                // [q 64 tiles | k 64 | v 192 | z 192 | b, a 3]: this Mac's q, k, v and z heads, b and a whole, 8 K slices
                const cols = [_][2]usize{ .{ 32 * k0, 32 }, .{ 64 + 32 * k0, 32 }, .{ 128 + 96 * k0, 96 }, .{ 320 + 96 * k0, 96 }, .{ 512, 3 } };
                try split.laneTiles(r, "lane_qmm_bytes_grouped@gdn.in", &.{ t.mixed, t.xs, L.proj.wq, L.proj.sbt, t.mdims }, t.p, &cols, 8, null);
                const a = m.state;
                const cs_in: Buf = .{ .b = L.cs[a].b, .off = L.cs[a].off + m.state_row * CS_ROW };
                const so_in: Buf = .{ .b = L.so[a].b, .off = L.so[a].off + m.state_row * SO_ROW };
                if (m.keptState()) |recs| {
                    const li = i - i / 4;
                    gdn_step.stepKept(r, r.gdn_kept.?, &.{ t.p, cs_in, so_in, L.conv, L.alog, L.dt, L.norm, t.eps, t.rows }, &.{ t.gout, L.cs[1 - a] }, recs[1 - m.rec_slot].at(li * gdn_step.RECORD), recs[m.rec_slot].at(li * gdn_step.RECORD), r.ar, 24 * tp.rank, 24);
                } else try split.gdnHeads(r, "q4_gdn@gdn", if (r.gdn_step and rows > 1) 8 else rows, &.{ t.p, cs_in, so_in, L.conv, L.alog, L.dt, L.norm, t.eps, t.rows }, &.{ t.gout, L.cs[1 - a], L.so[1 - a] }, tp.rank);
                const pn = tp.partNext();
                const part: Buf = .{ .b = pn.b, .off = pn.off };
                try split.laneTiles(r, "lane_qmm_bytes_grouped@gdn.out", &.{ t.gout, t.xs, L.out.wq, L.out.sbt, t.mdims }, part, &.{.{ 0, 80 }}, 8, .{ 96 * k0, 96 }); // one kernel at every width: drafted == plain
                if (r.skip & TP_CLASS == 0) { // the partials' sum and the stream update in the exchange's launch
                    tp.plain(r.enc, t.h[cur], t.inj_a, t.h[1 - cur], t.ssp, t.rows, rows);
                    plain = true;
                }
            } else if (L.linear) {
                try m.lane(t.mixed, D, L.proj, "lane_qmm_bytes_grouped@gdn.in", t.p);
                const a = m.state;
                const cs_in: Buf = .{ .b = L.cs[a].b, .off = L.cs[a].off + m.state_row * CS_ROW };
                const so_in: Buf = .{ .b = L.so[a].b, .off = L.so[a].off + m.state_row * SO_ROW };
                r.touch(L.out.wq);
                r.touch(L.out.sbt);
                const gins = [_]Buf{ t.p, cs_in, so_in, L.conv, L.alog, L.dt, L.norm, t.eps, t.rows };
                const gouts = [_]Buf{ t.gout, L.cs[1 - a], L.so[1 - a] };
                if (m.keptState()) |recs| {
                    const li = i - i / 4;
                    gdn_step.stepKept(r, r.gdn_kept.?, &gins, gouts[0..2], recs[1 - m.rec_slot].at(li * gdn_step.RECORD), recs[m.rec_slot].at(li * gdn_step.RECORD), r.ar, 0, 48);
                } else if (r.gdn_pipe) |pipe| gdn_step.step(r, pipe, &gins, &gouts, 0, 48) else try r.callAs("q4_gdn@gdn", if (r.gdn_step and rows > 1) 8 else rows, &gins, &gouts);
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
                    r.enc.dispatchThreads(mtl.Size.of(128, POOL_BLOCKS, 1), mtl.Size.of(128, 1, 1));
                    if (!r.serial) r.enc.barrier();
                    try r.shapes.put(r.arena, "POOLED_shape", g.pooled_shape);
                    try r.shapes.put(r.arena, "Q_shape", g.q_shape[rows]);
                    try r.shapes.put(r.arena, "SC_shape", g.sc_shape);
                    try r.bindV(sl.scores, &.{ t.iq, L.pooled, g.view(0) }, &.{g.sc});
                    r.enc.dispatchThreads(mtl.Size.of(((g.nb_ub + 7) / 8) * 256, (rows + 7) / 8, 1), mtl.Size.of(256, 1, 1));
                    if (!r.serial) r.enc.barrier();
                    try r.bindV(sl.select, &.{ g.sc, g.view(0), g.view(MAXR) }, &.{g.keys});
                    r.enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
                    if (!r.serial) r.enc.barrier();
                    const dense_ids = r.shapes.get("IDS_shape").?;
                    try r.shapes.put(r.arena, "IDS_shape", g.ids_shape[rows]);
                    try r.call("q4_attn_parts#[24, 256]", &.{ t.q, L.keys, L.vals, g.keys, g.view(2 * MAXR), g.view(3 * MAXR), t.scale }, &.{ t.po, t.pm });
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
            if (!plain) try r.call("q4_hc_norm_plain#[10240]", &.{ t.h[cur], t.inj_a, t.branch }, &.{ t.h[1 - cur], t.ssp });
            cur = 1 - cur;
            r.touch(L.router);
            try m.hcProject(t.h[cur], L.mhc, "qa_hc_down@mhc", "qa_hc_up@mhc", t.inj_m);
            try r.router("q4_router_float@moe", t.mixed, L.router, t.rows, t.lg, rows);
            r.tp_layer = true;
            try r.experts("qa_expert_gateup@moe.gate", "qa_expert_down_y@moe.down", t.mixed, t.lg, L.ex, t.act, t.pick, t.wts, t.rows, t.ydown);
            r.tp_layer = false;
            if (r.tp) |tp| if (r.skip & TP_CLASS == 0) tp.exchange(r.enc, t.ydown, t.wts, t.lg, rows);
            if (r.probe) |pb| r.copyKept(t.pick, .{ .b = pb.b, .off = pb.off + i * MAXR * 10 * 4 }, rows * 10, 0, 0, 0, 1, -1);
            if (r.lg_probe) |pb| r.copyKept(t.lg, pb.at(i * MAXR * 513 * 4), rows * 513, 0, 0, 0, 1, 0);
            pending = .grouped;
        }
        try m.grouped(t.h[cur], t.h[1 - cur]);
        cur = 1 - cur;
        m.last = t.h[cur];
        try m.hcProject(t.h[cur], m.mix, "qa_hc_down@mix", "qa_hc_up@mix", t.inj_a);
        if (r.tp) |tp| { // TP: this Mac's half of the vocabulary (the same kernel, half its tiles), then one swap
            const half = HEAD_TILES / 2;
            if (!r.fused_xsum) try r.call("lane_qmm_xsum#[2560]", &.{ t.mixed, t.mdims }, &.{t.xs});
            try split.laneTiles(r, "lane_qmm_bytes_grouped@head", &.{ t.mixed, t.xs, m.head.wq, m.head.sbt, t.mdims }, t.logits, &.{.{ half * tp.rank, half }}, 1, null);
            if (r.skip & TP_CLASS == 0) tp.argmax(r.enc, t.logits, HEAD_TILES * 32, half * 32 * tp.rank, half * 32, t.picks, t.rows, rows);
            return;
        }
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
        const ids = h.slots[0].ids8.b.slice(u32, MAXR);
        for (0..MAXR) |i| ids[i] = if (i < rows) nexts[i] else 0;
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
        const pos8 = sl.pos8.b.slice(i32, MAXR);
        const nk8 = sl.nk8.b.slice(i32, MAXR);
        for (0..MAXR) |i| {
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
            r.enc.dispatchThreads(mtl.Size.of(128, POOL_BLOCKS, 1), mtl.Size.of(128, 1, 1));
            if (!r.serial) r.enc.barrier();
            try r.shapes.put(r.arena, "POOLED_shape", g.pooled_shape);
            try r.shapes.put(r.arena, "Q_shape", g.q_shape[rows]);
            try r.shapes.put(r.arena, "SC_shape", g.sc_shape);
            try r.bindV(ss.scores, &.{ t.iq, h.pooled, g.view(0) }, &.{g.sc});
            r.enc.dispatchThreads(mtl.Size.of(((g.nb_ub + 7) / 8) * 256, (rows + 7) / 8, 1), mtl.Size.of(256, 1, 1));
            if (!r.serial) r.enc.barrier();
            try r.bindV(ss.select, &.{ g.sc, g.view(0), g.view(MAXR) }, &.{g.keys});
            r.enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
            if (!r.serial) r.enc.barrier();
            const dense_ids = r.shapes.get("IDS_shape").?;
            try r.shapes.put(r.arena, "IDS_shape", g.ids_shape[rows]);
            try r.call("mtp:q4_attn_parts#[24, 256]", &.{ t.q, h.keys, h.vals, g.keys, g.view(2 * MAXR), g.view(3 * MAXR), t.scale }, &.{ t.po, t.pm });
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
        if (rows > 1) { // only the kept row goes on: the host's last row, or the row the GPU's verdict kept, into row 0
            const keep: Buf = if (r.gpu_round) r.ar else sl.rows;
            r.enc.setPipeline(r.take_pipe);
            for ([_]Buf{ t.aout, h.h[1], t.inj_a, keep }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
            r.enc.dispatchThreads(mtl.Size.of(WIDE, 1, 1), mtl.Size.of(256, 1, 1));
            if (!r.serial) r.enc.barrier();
        }
        r.rows = 1;
        if (!r.fused_xsum) try r.call("mtp:lane_qmm_xsum#[6144]", &.{ t.aout, h.md1 }, &.{t.xs});
        if (r.dense) r.denseRows(t.aout, 6144, h.out, 1, t.branch) else try r.call("mtp:lane_qmm_bytes_grouped@mtp.att.o", &.{ t.aout, t.xs, h.out.wq, h.out.sbt, h.md1 }, &.{t.branch});
        try r.call("mtp:q4_hc_norm_plain#[10240]", &.{ h.h[1], t.inj_a, t.branch }, &.{ h.h[0], t.ssp });
        try m.mtpProject(h.h[0], h.mhc, down[1], up[1], t.inj_m, h.md1);
        try r.router("mtp:q4_router_float@mtp.moe", t.mixed, h.router, h.md1, t.lg, 1);
        try r.experts("mtp:qa_expert_gateup@mtp.moe.gate", "mtp:qa_expert_down_y@mtp.moe.down", t.mixed, t.lg, h.ex, t.act, t.pick, t.wts, h.md1, t.ydown);
        try r.call("mtp:q4_hc_norm_grouped#[10240]", &.{ h.h[0], t.inj_m, t.ydown, t.wts, t.lg }, &.{ h.h[1], t.ssp });
        h.last = h.h[1];
        try m.mtpProject(h.h[1], h.mix, down[2], up[2], t.inj_a, h.md1);
        const x = t.mixed;
        if (!r.fused_xsum) try r.call("mtp:lane_qmm_xsum#[2560]", &.{ x, h.md1 }, &.{t.xs});
        if (r.tp != null and r.dense) { // TP: this Mac's half of the draft vocabulary's tiles, then one exact argmax swap
            const tp = r.tp.?;
            const n = h.draft.wq.b.length() * 4 / (3 * D);
            const tiles = n / 32;
            const half = (tiles + 1) / 2;
            const t0: usize = if (tp.rank == 0) 0 else half;
            const tn: usize = if (tp.rank == 0) half else tiles - half;
            r.denseTiles(x, D, h.draft, h.logits, t0, tn);
            const lo = @min(t0 * 32, h.ids_n);
            const hi = @min((t0 + tn) * 32, h.ids_n);
            tp.argmax(r.enc, h.logits, n, lo, hi - lo, Buf{ .b = tp.pick_tmp }, Buf{ .b = tp.one }, 1);
            tp.mapIds(r.enc, h.ids, out);
            return;
        }
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

/// Drafts a round from recent landing: 3 (prose), 6 while drafts land (code-like text), the widest window while
/// nearly all land (copied text).
/// FZ_PROFILE's knock-out bit for speed-up mode's exchanges: the bit after `Run.class`'s eight.
pub const TP_CLASS: u32 = 1 << 8;

pub const DepthRule = struct {
    rate: f64 = 0.6, // moving average of the head's drafts landed over drafts offered
    depth: usize = 3,
    pair: bool = false, // speed-up mode: even windows only (an odd window's last row costs a Mac a whole expert row)

    pub fn pick(self: *const DepthRule) usize {
        return self.depth;
    }

    /// A head round's drafts and how many landed: 3, 6 or 8 drafts on one Mac, 3 or 7 on the pair (copied rounds set
    /// their own width).
    pub fn update(self: *DepthRule, depth: usize, landed: usize) void {
        self.rate = 0.7 * self.rate + 0.3 * @as(f64, @floatFromInt(landed)) / @as(f64, @floatFromInt(depth));
        if (self.pair) {
            if (self.depth == 3 and self.rate > 0.95) self.depth = 7 else if (self.depth == 7 and self.rate < 0.8) self.depth = 3;
            return;
        }
        if (self.depth == 3 and self.rate > 0.8) {
            self.depth = 6;
        } else if (self.depth == 6 and self.rate > 0.74) {
            self.depth = 8;
        } else if (self.depth == 8 and self.rate < 0.62) {
            self.depth = 6;
        } else if (self.depth == 6 and self.rate < 0.65) self.depth = 3;
    }
};

test "the pair's depth rule keeps every window even" {
    var pair: DepthRule = .{ .pair = true };
    for (0..20) |_| pair.update(pair.pick(), pair.pick());
    try std.testing.expectEqual(@as(usize, 7), pair.pick());
    for (0..20) |_| pair.update(pair.pick(), 1);
    try std.testing.expectEqual(@as(usize, 3), pair.pick());
    var one: DepthRule = .{};
    for (0..20) |_| one.update(one.pick(), one.pick());
    try std.testing.expectEqual(@as(usize, 8), one.pick());
}

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
pub const PMAX = 8192; // the prompt buffers' rows; `step` cuts the chunks
pub const Prompt = struct {
    r: *Run,
    qmm6: mtl.Pipeline,
    gather64: mtl.Pipeline,
    gather32: mtl.Pipeline,
    qmm6_128: mtl.Pipeline,
    gather128: mtl.Pipeline,
    gu64: mtl.Pipeline,
    seg_queue: ?mtl.Queue = null, // a later staggered segment's queue (made at first use)
    last: Buf = undefined, // the last chunk's streams before the final mixer (each segment's own)
    gu32: mtl.Pipeline,
    fused_gu: bool = true, // the experts' gate and up products and activation in one pass (same bits)
    expert_bm: usize = 0, // expert tile rows (0: by the average rows an expert)
    mtp_w: ?[3][3]Buf = null, // the MTP head's attention projection, fce, fch in MLX layout (pack_mtp_mlx.safetensors)
    tall_tiles: bool = true, // 128-row tiles (a dequantized weight block serves twice the rows; the same sums)
    router_mm: mtl.Pipeline,
    attn256: mtl.Pipeline,
    splitk: mtl.Pipeline,
    parts_sum: mtl.Pipeline,
    pl: [16]mtl.Pipeline, // normed, act, mix, router, route, offsets, sort, gather rows, act2, scatter, copy, DeltaNet pre/scan/post, scan4, scan8
    scan8: bool = true, // scan4 with a state row over 16 lanes, 8 rows a simdgroup
    scan4: bool = true, // the DeltaNet recurrence with four state rows a simdgroup and both reductions at once
    gdn: Variant,
    gdn_grid: mtl.Size,
    gdn_tg: mtl.Size,
    sel: ?Select = null, // long prompts: selection for the chunk's rows
    step: usize = PMAX, // rows a chunk
    fast_attn: bool = true, // prompt rows' scores and attention on the tensor units (false: the decode's kernels)
    sattn: mtl.Pipeline,
    scores_nax: mtl.Pipeline,
    skip: u32 = 0, // timing knock-outs: 1 experts, 2 DeltaNet, 4 attention, 8 projections, 16 hyper-connections, 32 routing, 64 router, 128 sort, 256 row moves, 512 shared expert, 1024 sparse attention
    proj: [LAYERS][3]Buf,
    out: [LAYERS][3]Buf,
    ple_kv: [3]Buf,
    b: struct { ids: Buf, pids: Buf, h: [2]Buf, ssp: Buf, normed: Buf, dn: Buf, hact: Buf, inj_a: Buf, inj_m: Buf, up: Buf, mixed: Buf, p: Buf, gout: Buf, branch: Buf, cso: Buf, q: Buf, kout: Buf, iq: Buf, po: Buf, pm: Buf, aout: Buf, pos: Buf, nk: Buf, zeros: Buf, kvmeta: Buf, lg: Buf, pick: Buf, wts: Buf, cnt: Buf, off: Buf, cur: Buf, row_of: Buf, xs: Buf, g: Buf, u: Buf, a: Buf, ds: Buf, sg: Buf, su: Buf, sa: Buf, ydown: Buf, emb: Buf, kvp: Buf, gated: Buf, hout: Buf, cin: Buf, rows: Buf, part: Buf, qn: Buf, kn: Buf, v: Buf, gg: Buf, beta: Buf, ys: Buf, mids: Buf, n_add: Buf },

    pub fn init(r: *Run, dir: []const u8, header: []const u8) !Prompt {
        var p: Prompt = undefined; // every field is set below: `undefined` skips the declared defaults
        p.r = r;
        p.skip = 0;
        p.sel = null;
        p.step = PMAX;
        p.fast_attn = true;
        p.scan4 = true;
        p.scan8 = true;
        p.tall_tiles = false; // measured: no faster than 64-row tiles (same bits)
        p.fused_gu = true;
        const alib = try mtl.Library.fromSource(r.device, try frags.source(r.device, r.arena, ks.flashnext_attn), mtl.CompileOptions.mlx());
        p.sattn = try mtl.Pipeline.init(r.device, alib, "tf_sattn_nax", false);
        p.scores_nax = try mtl.Pipeline.init(r.device, alib, "tf_idx_scores_nax", false);
        const qlib = try mtl.Library.fromSource(r.device, try frags.source(r.device, r.arena, ks.flashnext_qmm6), mtl.CompileOptions.mlx());
        p.qmm6 = try mtl.Pipeline.init(r.device, qlib, "tf_qmm6_t_nax", false);
        p.gather64 = try mtl.Pipeline.init(r.device, qlib, "tf_gather_qmm6_nax_64", false);
        p.gather32 = try mtl.Pipeline.init(r.device, qlib, "tf_gather_qmm6_nax_32", false);
        p.qmm6_128 = try mtl.Pipeline.init(r.device, qlib, "tf_qmm6_t_nax_128", false);
        p.gather128 = try mtl.Pipeline.init(r.device, qlib, "tf_gather_qmm6_nax_128", false);
        p.gu64 = try mtl.Pipeline.init(r.device, qlib, "tf_gather_gu6_nax_64", false);
        p.gu32 = try mtl.Pipeline.init(r.device, qlib, "tf_gather_gu6_nax_32", false);
        p.router_mm = try mtl.Pipeline.init(r.device, qlib, "tf_mm_bf16_f32_t_nax", false);
        p.attn256 = try mtl.Pipeline.init(r.device, qlib, "tf_attn256_nax", false);
        p.splitk = try mtl.Pipeline.init(r.device, qlib, "tf_qmm6_splitk_nax", false);
        p.parts_sum = try mtl.Pipeline.init(r.device, qlib, "tf_parts_sum", false);
        try frags.check(r.device, r.queue, r.arena);
        const glib = try mtl.Library.fromSource(r.device, try std.mem.concat(r.arena, u8, &.{ header, ks.flashnext_prompt }), mtl.CompileOptions.mlx());
        const names = [_][:0]const u8{ "pf_hc_normed", "pf_hc_act", "pf_hc_mix", "pf_router", "pf_route", "pf_offsets", "pf_sort", "pf_gather_rows", "pf_act", "pf_scatter_y", "pf_copy", "pf_gdn_pre", "pf_gdn_scan", "pf_gdn_post", "pf_gdn_scan4", "pf_gdn_scan8" };
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
        p.mtp_w = null;
        p.expert_bm = 0;
        if (r.indexFile(try std.fmt.allocPrintSentinel(r.arena, "{s}/pack_mtp_mlx.safetensors", .{dir}, 0))) |_| {
            var w: [3][3]Buf = undefined;
            for ([_][]const u8{ "mtp.att.proj", "mtp.fce", "mtp.fch" }, 0..) |name, i| {
                for ([_][]const u8{ "mw", "ms", "mb" }, 0..) |suf, k| w[i][k] = try r.loadf("{s}.{s}", .{ name, suf });
            }
            p.mtp_w = w;
        } else |_| {}
        p.b = try buffers(r);
        p.sel = try Select.init(r, PMAX);
        p.seg_queue = null;
        p.last = p.b.h[0];
        return p;
    }

    /// Another staggered segment's prompt: the same kernels and weights, its own buffers, selection and queue.
    pub fn sibling(p: *const Prompt) !Prompt {
        var s = p.*;
        s.b = try buffers(p.r);
        s.sel = try Select.init(p.r, PMAX);
        s.seg_queue = null;
        s.last = s.b.h[0];
        return s;
    }

    /// A prompt's own buffers for chunks of up to PMAX rows (each staggered segment has its own set).
    pub fn buffers(r: *Run) !@FieldType(Prompt, "b") {
        const B = struct {
            fn of(rr: *Run, n: usize) !Buf {
                return .{ .b = try rr.buffer(n) };
            }
        };
        const R = PMAX;
        return .{
            .ids = try B.of(r, R * 4), .pids = try B.of(r, R * 16 * 4), .h = .{ try B.of(r, R * WIDE * 2), try B.of(r, R * WIDE * 2) },
            .ssp = try B.of(r, R * 10 * 4 * 4), .normed = try B.of(r, R * WIDE * 2), .dn = try B.of(r, R * 324 * 2),
            .hact = try B.of(r, R * 320 * 2), .inj_a = try B.of(r, R * 4 * 2), .inj_m = try B.of(r, R * 4 * 2),
            .up = try B.of(r, R * WIDE * 2), .mixed = try B.of(r, R * D * 2), .p = try B.of(r, R * 16480 * 2),
            .gout = try B.of(r, R * 6144 * 2), .branch = try B.of(r, R * D * 2), .cso = try B.of(r, CS_ROW),
            .q = try B.of(r, R * 24 * 256 * 2), .kout = try B.of(r, R * 2 * 256 * 2), .iq = try B.of(r, R * 4 * 128 * 2),
            .po = try B.of(r, 64), .pm = try B.of(r, 64), .aout = try B.of(r, R * 6144 * 2),
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
    }

    pub fn barrier(p: *Prompt) void {
        if (!p.r.serial) p.r.enc.barrier();
    }

    /// The decode kernels' attention partials (3.2 GB at PMAX rows), made at first use: the tensor-op path never reads them.
    fn partials(p: *Prompt) !void {
        if (p.b.po.b.length() >= PMAX * 24 * 16 * 256 * 4) return;
        p.b.po = .{ .b = try p.r.buffer(PMAX * 24 * 16 * 256 * 4) };
        p.b.pm = .{ .b = try p.r.buffer(PMAX * 24 * 16 * 2 * 4) };
    }

    pub fn bind(p: *Prompt, pipe: mtl.Pipeline, bufs: []const Buf) void {
        p.r.enc.setPipeline(pipe);
        for (bufs, 0..) |b, j| p.r.enc.setBuffer(b.b, b.off, j);
    }

    /// y[rows, n] (row stride ldy, 0: n) = x[rows, k] W^T, W 6-bit g32 in MLX's layout.
    pub fn qmm(p: *Prompt, x: Buf, w: [3]Buf, k: usize, n: usize, rows: usize, y: Buf, ldy: usize) void {
        if (p.skip & 8 != 0) return;
        const tall = p.tall_tiles and rows >= 512;
        p.bind(if (tall) p.qmm6_128 else p.qmm6, &.{ w[0], w[1], w[2], x });
        const prm = [4]i32{ @intCast(k), @intCast(n), @intCast(rows), @intCast(ldy) };
        p.r.enc.setBytes(std.mem.asBytes(&prm), 4);
        p.r.enc.setBuffer(y.b, y.off, 5);
        const bm: usize = if (tall) 128 else 64;
        p.r.enc.dispatchThreads(mtl.Size.of(((n + 63) / 64) * 128, (rows + bm - 1) / bm, 1), mtl.Size.of(128, 1, 1));
        p.barrier();
    }

    /// y[pairs, n] = x[slot] W_e^T over the sorted slots, the experts' first slots in b.off.
    pub fn gather(p: *Prompt, x: Buf, w: []const Buf, k: usize, n: usize, pairs: usize, y: Buf) void {
        if (p.skip & 1 != 0) return;
        // tiles of 32, 64 or 128 rows as experts average 32 and 128 rows (128: FZ prompt chunks of thousands of rows)
        const bm: usize = if (p.expert_bm != 0) p.expert_bm else if (p.tall_tiles and pairs >= 512 * 128) 128 else if (pairs >= 512 * 32) 64 else 32;
        p.bind(if (bm == 128) p.gather128 else if (bm == 64) p.gather64 else p.gather32, &.{ x, w[0], w[1], w[2], p.b.off });
        const prm = [4]i32{ @intCast(pairs), @intCast(n), @intCast(k), 512 };
        p.r.enc.setBytes(std.mem.asBytes(&prm), 5);
        p.r.enc.setBuffer(y.b, y.off, 6);
        p.r.enc.dispatchThreads(mtl.Size.of(((n + 63) / 64) * 128, pairs / bm + 512, 1), mtl.Size.of(128, 1, 1));
        p.barrier();
    }

    /// a[pairs, n] = act(x[slot] Wg_e^T, x[slot] Wu_e^T) over the sorted slots in one pass (gather's tiles and sums).
    pub fn gatherGU(p: *Prompt, x: Buf, wg: []const Buf, wu: []const Buf, k: usize, n: usize, pairs: usize, y: Buf) void {
        if (p.skip & 1 != 0) return;
        const bm: usize = if (p.expert_bm == 32 or p.expert_bm == 64) p.expert_bm else if (pairs >= 512 * 32) 64 else 32;
        p.bind(if (bm == 64) p.gu64 else p.gu32, &.{ x, wg[0], wg[1], wg[2], wu[0], wu[1], wu[2], p.b.off });
        const prm = [4]i32{ @intCast(pairs), @intCast(n), @intCast(k), 512 };
        p.r.enc.setBytes(std.mem.asBytes(&prm), 8);
        p.r.enc.setBuffer(y.b, y.off, 9);
        p.r.enc.dispatchThreads(mtl.Size.of(((n + 63) / 64) * 128, pairs / bm + 512, 1), mtl.Size.of(128, 1, 1));
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
        if (p.fused_gu) p.gatherGU(b.xs, L.ex[0..3], L.ex[3..6], D, 640, pairs, b.a) else {
            p.gather(b.xs, L.ex[0..3], D, 640, pairs, b.g);
            p.gather(b.xs, L.ex[3..6], D, 640, pairs, b.u);
            p.bind(p.pl[8], &.{ b.g, b.u, b.a });
            r.enc.dispatchThreads(mtl.Size.of(pairs * 640, 1, 1), mtl.Size.of(256, 1, 1));
            p.barrier();
        }
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
                            p.bindMarks(m, @splat(0), 0, 12, 13, false);
                            r.enc.dispatchThreads(mtl.Size.of(80 * 128, rows, 1), mtl.Size.of(128, 1, 1));
                        } else if (which == 10) {
                            p.bind(p.pl[if (p.scan4) 14 else 12], &.{ b.qn, b.kn, b.v, b.gg, b.beta, L0.so[0] });
                            r.enc.setBytes(std.mem.asBytes(&ri), 6);
                            r.enc.setBuffer(b.ys.b, b.ys.off, 7);
                            r.enc.setBuffer(L0.so[1].b, L0.so[1].off, 8);
                            p.bindMarks(m, @splat(0), 0, 9, 10, true);
                            if (p.scan4) r.enc.dispatchThreads(mtl.Size.of(48 * 4 * 256, 1, 1), mtl.Size.of(256, 1, 1)) else r.enc.dispatchThreads(mtl.Size.of(48 * 4 * 1024, 1, 1), mtl.Size.of(1024, 1, 1));
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
    /// Returns the greedy token after it; m.last and p.last hold the chunk's streams before the final mixer.
    pub fn chunk(p: *Prompt, m: *Model, gpa: std.mem.Allocator, tokens: []const u32) !u32 {
        return chunkN(&.{p}, m, gpa, tokens);
    }

    /// One staggered segment of a chunk: its prompt, rows and first position; the DeltaNet state slot and row it reads
    /// and the slot it writes (row 0); its stream index and the hyper-connection write still pending.
    const Seg = struct {
        p: *Prompt,
        rows: usize,
        pos: usize,
        ra: usize,
        rr: usize,
        wa: usize,
        cur: usize,
        pending: bool,
        mk: [MARKS]i32 = @splat(0), // segment rows after which the DeltaNet layers write their states to m.marks (0: none)
    };

    /// The DeltaNet pre and scan kernels' mark arguments: the segment's mark rows and layer `li`'s slots (any buffer when unmarked).
    fn bindMarks(p: *Prompt, m: *const Model, mk: [MARKS]i32, li: usize, rows_at: usize, buf_at: usize, scan: bool) void {
        const r = p.r;
        r.enc.setBytes(std.mem.asBytes(&mk), rows_at);
        const fallback = m.layers[0].cs[0];
        const b = if (m.marks) |k| (if (scan) k.so_at(li, 0) else k.cs_at(li, 0)) else fallback;
        r.enc.setBuffer(b.b, b.off, buf_at);
    }

    /// A chunk's host inputs at position `pos`: token ids, positions, the cache meta, the n-gram ids after `hist`.
    fn prep(p: *Prompt, m: *Model, gpa: std.mem.Allocator, tokens: []const u32, pos: usize, hist: [2]i64) !void {
        const b = &p.b;
        const rows = tokens.len;
        @memcpy(b.ids.b.slice(u32, rows), tokens);
        for (0..rows) |i| {
            b.pos.b.slice(i32, PMAX)[i] = @intCast(pos + i);
            b.nk.b.slice(i32, PMAX)[i] = @intCast(pos + i + 1);
        }
        const kvm = b.kvmeta.b.slice(u32, 3);
        kvm[0], kvm[1], kvm[2] = .{ @intCast(pos), CAP, @intCast(rows) };
        b.rows.b.slice(i32, 1)[0] = @intCast(rows);
        const pp = &m.ple;
        const seq = try gpa.alloc(i64, 2 + rows);
        defer gpa.free(seq);
        seq[0], seq[1] = .{ hist[0], hist[1] };
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

    /// A segment's layer i up to its mixer: the pending expert write (or the n-gram block at layer 1) and the attention
    /// hyper-connection.
    fn segPre(s: *Seg, m: *Model, i: usize) !void {
        const p = s.p;
        const r = p.r;
        const b = &p.b;
        const t = &m.t;
        const rows = s.rows;
        const L = &m.layers[i];
        if (i == 1) {
            if (s.pending) try r.callRows("q4_hc_norm_grouped#[10240]", rows, &.{ b.h[s.cur], b.inj_m, b.ydown, b.wts, b.lg }, &.{ b.h[1 - s.cur], b.ssp }, null);
            if (s.pending) s.cur = 1 - s.cur;
            s.pending = false;
            const pp = &m.ple;
            var tabs: [2 + 3 * GROUPS]Buf = undefined;
            tabs[0] = b.pids;
            tabs[1] = pp.starts;
            for (0..3 * GROUPS) |j| tabs[2 + j] = pp.tables[j];
            try r.callRows("qa_ple_lookup@ple", rows, &tabs, &.{b.emb}, null);
            p.qmm(b.emb, p.ple_kv, D, 12800, rows, b.kvp, 0);
            try r.callRows("q4_ple_gate@ple", rows, &.{ b.kvp, b.h[s.cur], pp.ks, pp.qs, pp.cs, t.eps }, &.{ b.gated, .{ .b = b.cin.b, .off = PLE_TAIL * WIDE * 2 } }, null);
            try r.callRows("q4_ple_conv@ple", rows, &.{ b.cin, pp.conv, b.gated, b.h[s.cur] }, &.{b.hout}, null);
            try r.callRows("q4_hc_norm_none#[10240]", rows, &.{b.hout}, &.{ b.h[1 - s.cur], b.ssp }, null);
        } else if (!s.pending) {
            try r.callRows("q4_hc_norm_none#[10240]", rows, &.{b.h[s.cur]}, &.{ b.h[1 - s.cur], b.ssp }, null);
        } else {
            try r.callRows("q4_hc_norm_grouped#[10240]", rows, &.{ b.h[s.cur], b.inj_m, b.ydown, b.wts, b.lg }, &.{ b.h[1 - s.cur], b.ssp }, null);
        }
        s.cur = 1 - s.cur;
        p.hc(m, b.h[s.cur], L.ahc, rows, b.inj_a);
    }

    /// A segment's layer-i mixer, DeltaNet or attention, at the segment's position and state slots.
    fn segMixer(s: *Seg, m: *Model, i: usize) !void {
        const p = s.p;
        const r = p.r;
        const b = &p.b;
        const t = &m.t;
        const rows = s.rows;
        const L = &m.layers[i];
        if (L.linear) {
            p.qmm(b.mixed, p.proj[i], D, 16480, rows, b.p, 0);
            const cs_in: Buf = .{ .b = L.cs[s.ra].b, .off = L.cs[s.ra].off + s.rr * CS_ROW };
            const so_in: Buf = .{ .b = L.so[s.ra].b, .off = L.so[s.ra].off + s.rr * SO_ROW };
            if (p.skip & 2 == 0) {
                const ri: i32 = @intCast(rows);
                const li = linearIndex(m, i);
                p.bind(p.pl[11], &.{ b.p, cs_in, L.conv, L.alog, L.dt });
                r.enc.setBytes(std.mem.asBytes(&ri), 5);
                for ([_]Buf{ b.qn, b.kn, b.v, b.gg, b.beta, L.cs[s.wa] }, 6..) |bb, j| r.enc.setBuffer(bb.b, bb.off, j);
                p.bindMarks(m, s.mk, li, 12, 13, false);
                r.enc.dispatchThreads(mtl.Size.of(80 * 128, rows, 1), mtl.Size.of(128, 1, 1));
                p.barrier();
                const scan: usize = if (p.scan4 and p.scan8) 15 else if (p.scan4) 14 else 12;
                p.bind(p.pl[scan], &.{ b.qn, b.kn, b.v, b.gg, b.beta, so_in });
                r.enc.setBytes(std.mem.asBytes(&ri), 6);
                r.enc.setBuffer(b.ys.b, b.ys.off, 7);
                r.enc.setBuffer(L.so[s.wa].b, L.so[s.wa].off, 8);
                p.bindMarks(m, s.mk, li, 9, 10, true);
                switch (scan) {
                    15 => r.enc.dispatchThreads(mtl.Size.of(48 * 2 * 256, 1, 1), mtl.Size.of(256, 1, 1)),
                    14 => r.enc.dispatchThreads(mtl.Size.of(48 * 4 * 256, 1, 1), mtl.Size.of(256, 1, 1)),
                    else => r.enc.dispatchThreads(mtl.Size.of(48 * 4 * 1024, 1, 1), mtl.Size.of(1024, 1, 1)),
                }
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
            const sparse = p.sel != null and p.sel.?.meta(s.pos, rows);
            if (sparse and p.skip & 1024 != 0) {
                // timing knock-out: no attention for chunks with rows past the dense range
            } else if (p.fast_attn and p.sel != null and p.skip & 4 == 0) {
                var sl = &p.sel.?;
                if (sparse) {
                    sl.nax_scores = p.scores_nax;
                    defer sl.nax_scores = null;
                    try sl.encode(r, L, b.iq, t.eps, t.log2base, s.pos, rows);
                }
                p.bind(p.sattn, &.{ b.q, L.keys, L.vals, sl.keys, sl.counts, sl.sparse, b.p, t.scale });
                const cap: i32 = CAP;
                r.enc.setBytes(std.mem.asBytes(&cap), 8);
                r.enc.setBuffer(b.aout.b, b.aout.off, 9);
                r.enc.dispatchThreads(mtl.Size.of(rows * 128, 2, 1), mtl.Size.of(128, 1, 1));
                p.barrier();
            } else if (sparse) {
                var sl = &p.sel.?;
                try sl.encode(r, L, b.iq, t.eps, t.log2base, s.pos, rows);
                try p.partials();
                const dense_ids = r.shapes.get("IDS_shape").?;
                try r.shapes.put(r.arena, "IDS_shape", sl.ids_shape);
                try r.callRows("q4_attn_parts#[24, 256]", rows, &.{ b.q, L.keys, L.vals, sl.keys, sl.counts, sl.sparse, t.scale }, &.{ b.po, b.pm }, null);
                try r.shapes.put(r.arena, "IDS_shape", dense_ids);
                try r.callRows("q4_attn_merge_gate#[24, 16, 256]", rows, &.{ b.po, b.pm, b.p }, &.{b.aout}, null);
            } else if (p.skip & 4 == 0) {
                p.bind(p.attn256, &.{ b.q, L.keys, L.vals, b.p });
                const ap = [4]i32{ @intCast(rows), @intCast(s.pos + rows), @intCast(s.pos), CAP };
                r.enc.setBytes(std.mem.asBytes(&ap), 4);
                r.enc.setBuffer(t.scale.b, t.scale.off, 5);
                r.enc.setBuffer(b.aout.b, b.aout.off, 6);
                r.enc.dispatchThreads(mtl.Size.of(((rows + 63) / 64) * 128, 24, 1), mtl.Size.of(128, 1, 1));
                p.barrier();
            }
            p.qmm(b.aout, p.out[i], 6144, D, rows, b.branch, 0);
        }
    }

    /// A segment's layer i after its mixer: the branch write, the MLP hyper-connection and the experts.
    fn segPost(s: *Seg, m: *Model, i: usize) !void {
        const p = s.p;
        const b = &p.b;
        const L = &m.layers[i];
        try p.r.callRows("q4_hc_norm_plain#[10240]", s.rows, &.{ b.h[s.cur], b.inj_a, b.branch }, &.{ b.h[1 - s.cur], b.ssp }, null);
        s.cur = 1 - s.cur;
        p.hc(m, b.h[s.cur], L.mhc, s.rows, b.inj_m);
        p.moe(L, s.rows);
        s.pending = true;
    }

    /// Speed-up mode's prefill (chunkPair): this Mac's segment of a chunk whose rows are split across two Macs.
    const Pair = struct {
        tp: *Tp2,
        call: u32,
        k: usize, // this Mac's segment: 0 hands each layer to the peer, 1 waits for it
        pos0: usize, // segment 0's first position and rows: what segment 1 receives
        rows0: usize,
        a: usize, // the DeltaNet slot the chunk starts from; segment 0 writes 1 - a, segment 1 writes a
    };

    fn bytesOf(b: Buf) [*]const u8 {
        return b.b.contents() + b.off;
    }

    /// Segment 0, after layer i's mixer: once the GPU gets here the host sends the state (DeltaNet) or new key, value and indexer rows (attention) of this layer to the peer's slot, with the n-gram tail at layer 1.
    fn pairSend(pr: *const Pair, m: *Model, s: *Seg, i: usize) void {
        const L = &m.layers[i];
        const slot = tpm.layerSlot(i);
        var ws: [8]tpm.Write = undefined;
        var n: usize = 0;
        if (L.linear) {
            ws[0] = .{ .src = bytesOf(L.cs[1 - pr.a]), .len = CS_ROW, .dst = slot };
            ws[1] = .{ .src = bytesOf(L.so[1 - pr.a]), .len = SO_ROW, .dst = slot + CS_ROW };
            n = 2;
        } else {
            for (0..2) |hd| {
                ws[n] = .{ .src = bytesOf(L.keys) + (hd * CAP + pr.pos0) * 512, .len = pr.rows0 * 512, .dst = slot + hd * tpm.KV_HEAD };
                ws[n + 1] = .{ .src = bytesOf(L.vals) + (hd * CAP + pr.pos0) * 512, .len = pr.rows0 * 512, .dst = slot + (2 + hd) * tpm.KV_HEAD };
                n += 2;
            }
            ws[n] = .{ .src = bytesOf(L.raw) + pr.pos0 * 256, .len = pr.rows0 * 256, .dst = slot + 4 * tpm.KV_HEAD };
            n += 1;
        }
        if (i == 1) {
            ws[n] = .{ .src = bytesOf(s.p.b.cin) + s.rows * WIDE * 2, .len = PLE_TAIL * WIDE * 2, .dst = tpm.TAIL };
            n += 1;
        }
        pr.tp.send(s.p.r.enc, ws[0..n], tpm.LAYER_FLAG + 8 * i, pr.call);
    }

    /// Segment 1, before layer i's mixer (its n-gram block at layer 1): the GPU waits for the layer's flag, then copies the peer's handoff into the slot it reads, the caches, and its n-gram tail.
    fn pairTake(pr: *const Pair, m: *Model, s: *Seg, i: usize) void {
        const L = &m.layers[i];
        const w = pr.tp.window();
        const slot = tpm.layerSlot(i);
        const p = s.p;
        pr.tp.waitWord(p.r.enc, tpm.LAYER_FLAG + 8 * i, pr.call);
        if (L.linear) {
            p.copyWords(.{ .b = w, .off = slot }, L.cs[1 - pr.a], CS_ROW / 4);
            p.copyWords(.{ .b = w, .off = slot + CS_ROW }, L.so[1 - pr.a], SO_ROW / 4);
        } else {
            for (0..2) |hd| {
                p.copyWords(.{ .b = w, .off = slot + hd * tpm.KV_HEAD }, .{ .b = L.keys.b, .off = L.keys.off + (hd * CAP + pr.pos0) * 512 }, pr.rows0 * 128);
                p.copyWords(.{ .b = w, .off = slot + (2 + hd) * tpm.KV_HEAD }, .{ .b = L.vals.b, .off = L.vals.off + (hd * CAP + pr.pos0) * 512 }, pr.rows0 * 128);
            }
            p.copyWords(.{ .b = w, .off = slot + 4 * tpm.KV_HEAD }, .{ .b = L.raw.b, .off = L.raw.off + pr.pos0 * 256 }, pr.rows0 * 64);
        }
        if (i == 1) p.copyWords(.{ .b = w, .off = tpm.TAIL }, p.b.cin, PLE_TAIL * WIDE * 2 / 4);
    }

    /// Flash Next's hooks for core/segments.zig: a segment's embedding, layer parts and final streams, encoded into its
    /// lane's encoder. Layer 1 waits ahead of the n-gram block, which reads the segment before's last gate rows.
    const Hooks = struct {
        m: *Model,
        segs: []Seg,
        pair: ?*const Pair = null,

        fn at(h: *Hooks, l: *const segments.Lane) *Seg {
            const s = &h.segs[l.k];
            s.p.r.enc = l.enc;
            return s;
        }
        pub fn begin(h: *Hooks, l: *segments.Lane) !void {
            const s = h.at(l);
            const b = &s.p.b;
            const m = h.m;
            try s.p.r.callRows("qa_embed_rows@embed", s.rows, &.{ b.ids, m.embed[0], m.embed[1], m.embed[2] }, &.{b.h[0]}, null);
        }
        pub fn wait(_: *Hooks, i: usize) segments.Wait {
            return if (i == 1) .pre else .mixer;
        }
        pub fn handoff(h: *Hooks, l: *segments.Lane, i: usize) !void {
            if (i != 1) return;
            const s = h.at(l);
            const prev = &h.segs[l.k - 1];
            s.p.copyWords(.{ .b = prev.p.b.cin.b, .off = prev.p.b.cin.off + prev.rows * WIDE * 2 }, s.p.b.cin, PLE_TAIL * WIDE * 2 / 4);
        }
        pub fn pre(h: *Hooks, l: *segments.Lane, i: usize) !void {
            const s = h.at(l);
            if (h.pair) |pr| if (pr.k == 1 and i == 1) pairTake(pr, h.m, s, i);
            try segPre(s, h.m, i);
        }
        pub fn mixer(h: *Hooks, l: *segments.Lane, i: usize) !void {
            const s = h.at(l);
            if (h.pair) |pr| if (pr.k == 1 and i != 1) pairTake(pr, h.m, s, i);
            try segMixer(s, h.m, i);
            if (h.pair) |pr| if (pr.k == 0) pairSend(pr, h.m, s, i);
        }
        pub fn post(h: *Hooks, l: *segments.Lane, i: usize) !void {
            try segPost(h.at(l), h.m, i);
        }
        /// Every segment's final streams (the MTP head's prompt keys read them); the head on the last one's last row.
        pub fn finish(h: *Hooks, l: *segments.Lane) !void {
            const s = h.at(l);
            const p = s.p;
            const r = p.r;
            const b = &p.b;
            const m = h.m;
            const t = &m.t;
            try r.callRows("q4_hc_norm_grouped#[10240]", s.rows, &.{ b.h[s.cur], b.inj_m, b.ydown, b.wts, b.lg }, &.{ b.h[1 - s.cur], b.ssp }, null);
            s.cur = 1 - s.cur;
            p.last = b.h[s.cur];
            if (l.k + 1 < h.segs.len) return;
            if (h.pair) |pr| if (pr.k == 0) return; // the peer's segment ends the chunk: it runs the head
            m.last = p.last;
            p.hc(m, b.h[s.cur], m.mix, s.rows, b.inj_a);
            r.rows = 1;
            t.mdims.b.slice(i32, 2)[0] = 1;
            try m.lane(.{ .b = b.mixed.b, .off = b.mixed.off + (s.rows - 1) * D * 2 }, D, m.head, "lane_qmm_bytes_grouped@head", t.logits);
            r.enc.setPipeline(r.argmax_pipe);
            r.enc.setBuffer(t.logits.b, 0, 0);
            r.enc.setBuffer(t.picks.b, 0, 1);
            r.enc.setBuffer(t.vocab.b, 0, 2);
            r.enc.dispatchThreads(mtl.Size.of(1024, 1, 1), mtl.Size.of(1024, 1, 1));
        }
    };

    /// A chunk as one to segments.MAX staggered segments, one queue each (core/segments.zig): segment k runs each
    /// layer's mixer after segment k-1's, so one segment's scan and glue run beside another's matrix work. Each
    /// segment does what a serial chunk of its rows does, so the output equals chunk calls in order. ps[0] is this
    /// prompt, on the run's queue; the others are siblings. Returns the greedy token after the chunk.
    pub fn chunkN(ps: []const *Prompt, m: *Model, gpa: std.mem.Allocator, tokens: []const u32) !u32 {
        return chunkMarked(ps, m, gpa, tokens, &.{});
    }

    /// chunkN, with each DeltaNet layer also writing its state after each of `marks` rows (ascending, 1..n) to m.marks.
    pub fn chunkMarked(ps: []const *Prompt, m: *Model, gpa: std.mem.Allocator, tokens: []const u32, marks: []const u32) !u32 {
        const N = ps.len;
        if (marks.len > MARKS or (marks.len > 0 and m.marks == null)) return error.Marks;
        if (N == 0 or N > segments.MAX) return error.Segments;
        const r = ps[0].r;
        const n = tokens.len;
        if (n < N or n > N * PMAX) return error.ChunkSize;
        const a = m.state;
        var queues: [segments.MAX]mtl.Queue = undefined;
        var segs: [segments.MAX]Seg = undefined;
        for (0..N) |k| {
            const at = segments.start(n, N, k);
            const rows = segments.rows(n, N, k);
            var hist = m.ple.hist; // the two tokens before the segment
            for (tokens[at - @min(at, 2) .. at]) |tok| hist = .{ hist[1], tok };
            try ps[k].prep(m, gpa, tokens[at .. at + rows], m.pos + at, hist);
            queues[k] = if (k == 0) r.queue else ps[k].seg_queue orelse blk: {
                const q = try r.device.queue();
                ps[k].seg_queue = q;
                break :blk q;
            };
            const ra = if (k % 2 == 0) a else 1 - a; // DeltaNet slots alternate as serial chunks would
            segs[k] = .{ .p = ps[k], .rows = rows, .pos = m.pos + at, .ra = ra, .rr = if (k == 0) m.state_row else 0, .wa = 1 - ra, .cur = 0, .pending = false };
            for (marks, 0..) |mk, j| if (mk > at and mk <= at + rows) {
                segs[k].mk[j] = @intCast(mk - at);
            };
        }
        const cin_old = m.ple.cin.b.contents()[m.ple.cin.off..];
        @memcpy(ps[0].b.cin.b.contents()[0 .. PLE_TAIL * WIDE * 2], cin_old[0 .. PLE_TAIL * WIDE * 2]);
        var hooks: Hooks = .{ .m = m, .segs = segs[0..N] };
        m.gpu_seconds += try segments.run(r.device, queues[0..N], LAYERS, if (r.serial) .serial else .concurrent, &hooks);
        const sl = &segs[N - 1];
        m.state = if (N % 2 == 0) a else 1 - a; // each segment flipped it once
        m.state_row = 0;
        m.pos += n;
        @memcpy(cin_old[0 .. PLE_TAIL * WIDE * 2], sl.p.b.cin.b.contents()[sl.rows * WIDE * 2 .. (sl.rows + PLE_TAIL) * WIDE * 2]);
        for (tokens) |tok| m.ple.hist = .{ m.ple.hist[1], tok };
        return m.t.picks.b.slice(u32, 1)[0];
    }

    /// Speed-up mode's prefill: a chunk's rows split across two Macs (tp.zig). This Mac runs its segment (rank 0 the first half, rank 1 the second, a layer behind); rank 0 hands each layer's state or new keys to rank 1, then rank 1 hands its final state, keys, n-gram tail, last row and first token back. Each segment does a serial chunk's arithmetic, so both Macs end with one Mac's bits. Returns the greedy token after the chunk; m.last points at the last row's streams.
    pub fn chunkPair(p: *Prompt, m: *Model, gpa: std.mem.Allocator, tokens: []const u32, tp: *Tp2, marks: []const u32) !u32 {
        const r = p.r;
        const n = tokens.len;
        if (n < 2 or n > 2 * PMAX) return error.ChunkSize;
        if (marks.len > MARKS or (marks.len > 0 and m.marks == null)) return error.Marks;
        const k: usize = tp.rank;
        const a = m.state;
        const at = segments.start(n, 2, k);
        const rows = segments.rows(n, 2, k);
        const rows0 = segments.rows(n, 2, 0);
        tp.call +%= 1;
        const pr: Pair = .{ .tp = tp, .call = tp.call, .k = k, .pos0 = m.pos, .rows0 = rows0, .a = a };
        var hist = m.ple.hist;
        for (tokens[at - @min(at, 2) .. at]) |tok| hist = .{ hist[1], tok };
        try p.prep(m, gpa, tokens[at .. at + rows], m.pos + at, hist);
        const ra = if (k == 0) a else 1 - a;
        var segs = [1]Seg{.{ .p = p, .rows = rows, .pos = m.pos + at, .ra = ra, .rr = if (k == 0) m.state_row else 0, .wa = 1 - ra, .cur = 0, .pending = false }};
        for (marks, 0..) |mk, j| if (mk > at and mk <= at + rows) {
            segs[0].mk[j] = @intCast(mk - at);
        };
        const cin_old = m.ple.cin.b.contents()[m.ple.cin.off..];
        if (k == 0) @memcpy(p.b.cin.b.contents()[0 .. PLE_TAIL * WIDE * 2], cin_old[0 .. PLE_TAIL * WIDE * 2]);
        var hooks: Hooks = .{ .m = m, .segs = &segs, .pair = &pr };
        if (tp.trace) std.debug.print("TP rank{d} chunk {d}: rows {d} at {d} (pos {d}), state slot {d}\n", .{ k, pr.call, rows, at, m.pos + at, a });
        m.gpu_seconds += try segments.run(r.device, &.{r.queue}, LAYERS, if (r.serial) .serial else .concurrent, &hooks);
        if (tp.trace) std.debug.print("TP rank{d} chunk {d}: GPU done, gave_up {d}\n", .{ k, pr.call, tp.gaveUp() });
        const t = &m.t;
        var pick: u32 = undefined;
        if (k == 1) { // the chunk's end: hand it back
            var ws: std.ArrayList(tpm.Write) = .empty;
            defer ws.deinit(gpa);
            const pos1 = m.pos + at;
            for (0..LAYERS) |i| {
                const L = &m.layers[i];
                const slot = tpm.layerSlot(i);
                if (L.linear) {
                    try ws.append(gpa, .{ .src = bytesOf(L.cs[a]), .len = CS_ROW, .dst = slot });
                    try ws.append(gpa, .{ .src = bytesOf(L.so[a]), .len = SO_ROW, .dst = slot + CS_ROW });
                } else {
                    for (0..2) |hd| {
                        try ws.append(gpa, .{ .src = bytesOf(L.keys) + (hd * CAP + pos1) * 512, .len = rows * 512, .dst = slot + hd * tpm.KV_HEAD });
                        try ws.append(gpa, .{ .src = bytesOf(L.vals) + (hd * CAP + pos1) * 512, .len = rows * 512, .dst = slot + (2 + hd) * tpm.KV_HEAD });
                    }
                    try ws.append(gpa, .{ .src = bytesOf(L.raw) + pos1 * 256, .len = rows * 256, .dst = slot + 4 * tpm.KV_HEAD });
                }
            }
            try ws.append(gpa, .{ .src = bytesOf(p.b.cin) + rows * WIDE * 2, .len = PLE_TAIL * WIDE * 2, .dst = tpm.TAIL });
            try ws.append(gpa, .{ .src = bytesOf(p.last) + (rows - 1) * WIDE * 2, .len = WIDE * 2, .dst = tpm.LAST });
            try ws.append(gpa, .{ .src = bytesOf(t.picks), .len = 16, .dst = tpm.LAST + WIDE * 2 });
            try tp.sendNow(ws.items, tpm.BACK_FLAG, pr.call);
            pick = t.picks.b.slice(u32, 1)[0];
            @memcpy(cin_old[0 .. PLE_TAIL * WIDE * 2], p.b.cin.b.contents()[p.b.cin.off + rows * WIDE * 2 ..][0 .. PLE_TAIL * WIDE * 2]);
            m.last = .{ .b = p.last.b, .off = p.last.off + (rows - 1) * WIDE * 2 };
        } else { // the peer's half of the chunk into place
            tp.hostWait(tpm.BACK_FLAG, pr.call);
            const w = tp.window();
            const rows1 = n - rows0;
            const pos1 = m.pos + rows0;
            const cb = r.queue.commandBuffer();
            r.enc = cb.compute(if (r.serial) .serial else .concurrent);
            for (0..LAYERS) |i| {
                const L = &m.layers[i];
                const slot = tpm.layerSlot(i);
                if (L.linear) {
                    p.copyWords(.{ .b = w, .off = slot }, L.cs[a], CS_ROW / 4);
                    p.copyWords(.{ .b = w, .off = slot + CS_ROW }, L.so[a], SO_ROW / 4);
                } else {
                    for (0..2) |hd| {
                        p.copyWords(.{ .b = w, .off = slot + hd * tpm.KV_HEAD }, .{ .b = L.keys.b, .off = L.keys.off + (hd * CAP + pos1) * 512 }, rows1 * 128);
                        p.copyWords(.{ .b = w, .off = slot + (2 + hd) * tpm.KV_HEAD }, .{ .b = L.vals.b, .off = L.vals.off + (hd * CAP + pos1) * 512 }, rows1 * 128);
                    }
                    p.copyWords(.{ .b = w, .off = slot + 4 * tpm.KV_HEAD }, .{ .b = L.raw.b, .off = L.raw.off + pos1 * 256 }, rows1 * 64);
                }
            }
            try m.finish(cb);
            @memcpy(cin_old[0 .. PLE_TAIL * WIDE * 2], tp.bytes(tpm.TAIL)[0 .. PLE_TAIL * WIDE * 2]);
            pick = std.mem.readInt(u32, tp.bytes(tpm.LAST + WIDE * 2)[0..4], .little);
            m.last = .{ .b = w, .off = tpm.LAST };
        }
        m.state = a; // two segments: each flipped it once
        m.state_row = 0;
        m.pos += n;
        for (tokens) |tok| m.ple.hist = .{ m.ple.hist[1], tok };
        return pick;
    }

    /// Speed-up mode: once each Mac has written the MTP head's prompt keys for its segment's rows (`mine`: first position, rows), each sends them to the other and copies the other's (`theirs`) into place.
    pub fn pairMtp(p: *Prompt, m: *Model, tp: *Tp2, mine: [2]usize, theirs: [2]usize) !void {
        const r = p.r;
        const h = &m.mtp;
        var ws: [5]tpm.Write = undefined;
        var n: usize = 0;
        if (mine[1] > 0) {
            for (0..2) |hd| {
                ws[n] = .{ .src = bytesOf(h.keys) + (hd * CAP + mine[0]) * 512, .len = mine[1] * 512, .dst = tpm.MTP + hd * tpm.KV_HEAD };
                ws[n + 1] = .{ .src = bytesOf(h.vals) + (hd * CAP + mine[0]) * 512, .len = mine[1] * 512, .dst = tpm.MTP + (2 + hd) * tpm.KV_HEAD };
                n += 2;
            }
            ws[n] = .{ .src = bytesOf(h.raw) + mine[0] * 256, .len = mine[1] * 256, .dst = tpm.MTP + 4 * tpm.KV_HEAD };
            n += 1;
        }
        try tp.sendNow(ws[0..n], tpm.MTP_FLAG, tp.call);
        tp.hostWait(tpm.MTP_FLAG, tp.call);
        if (theirs[1] == 0) return;
        const w = tp.window();
        const cb = r.queue.commandBuffer();
        r.enc = cb.compute(if (r.serial) .serial else .concurrent);
        for (0..2) |hd| {
            p.copyWords(.{ .b = w, .off = tpm.MTP + hd * tpm.KV_HEAD }, .{ .b = h.keys.b, .off = h.keys.off + (hd * CAP + theirs[0]) * 512 }, theirs[1] * 128);
            p.copyWords(.{ .b = w, .off = tpm.MTP + (2 + hd) * tpm.KV_HEAD }, .{ .b = h.vals.b, .off = h.vals.off + (hd * CAP + theirs[0]) * 512 }, theirs[1] * 128);
        }
        p.copyWords(.{ .b = w, .off = tpm.MTP + 4 * tpm.KV_HEAD }, .{ .b = h.raw.b, .off = h.raw.off + theirs[0] * 256 }, theirs[1] * 64);
        try m.finish(cb);
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
        if (p.mtp_w) |w| p.qmm(b.branch, w[1], D, D, n, b.aout, 0) else r.denseRows(b.branch, D, h.fce, n, b.aout);
        try r.callRows("mtp:q4_rms_rows@mtp.hnorm", n, &.{ streams, h.hnorm, t.eps }, &.{b.normed}, null);
        if (p.mtp_w) |w| p.qmm(b.normed, w[2], D, D, 4 * n, b.gated, 0) else r.denseRows(b.normed, D, h.fch, 4 * n, b.gated);
        p.bind(r.add_pipe, &.{ b.aout, b.gated, b.xs, b.n_add });
        r.enc.dispatchThreads(mtl.Size.of(n * WIDE, 1, 1), mtl.Size.of(256, 1, 1));
        p.barrier();
        try r.callRows("mtp:q4_hc_norm_none#[10240]", n, &.{b.xs}, &.{ b.hout, b.ssp }, null);
        p.hc(m, b.hout, h.ahc, n, b.inj_a);
        if (p.mtp_w) |w| p.qmm(b.mixed, w[0], D, 13952, n, b.p, 0) else r.denseRows(b.mixed, D, h.proj, n, b.p);
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
