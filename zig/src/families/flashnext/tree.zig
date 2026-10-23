//! Tree windows in the lone engine's GPU rounds: a root, the head's chain (rows 1..a) and a branch from the root.
const std = @import("std");
const mtl = @import("metal");
const replay = @import("replay.zig");
const gdn_step = @import("gdn.zig");
const Run = replay.Run;
const Buf = replay.Buf;
const Variant = replay.Variant;
const MAXR = replay.MAXR;
const PLE_TAIL = replay.PLE_TAIL;
const WIDE = replay.WIDE;

/// Ints a row takes in the table: parent (-1 at the root), depth, then its ancestor row at each depth 0..depth.
pub const STRIDE = 18;
/// Arena words the tree accept writes: the kept path, then path[keep - 1] + 1, the old cache length and keep.
pub const AR_PATH = 580;
pub const AR_LAST = 596;
pub const AR_TOLD = 597;
pub const AR_KEEP = 598;
/// A branch draft no reply holds (a vision marker), for rows without a guess.
pub const NONE: u32 = 248053;

/// The table of a root, a chain of `a` rows from it, then a branch of `b` rows from the root.
pub fn table(a: usize, b: usize) [MAXR * STRIDE]i32 {
    var t: [MAXR * STRIDE]i32 = @splat(0);
    t[0] = -1;
    for (1..a + 1) |r| {
        t[r * STRIDE] = @intCast(r - 1);
        t[r * STRIDE + 1] = @intCast(r);
        for (0..r + 1) |k| t[r * STRIDE + 2 + k] = @intCast(k);
    }
    for (1..b + 1) |k| {
        const r = a + k;
        t[r * STRIDE] = if (k == 1) 0 else @intCast(r - 1);
        t[r * STRIDE + 1] = @intCast(k);
        for (1..k + 1) |j| t[r * STRIDE + 2 + j] = @intCast(a + j);
    }
    return t;
}

/// The table of a root, a chain of `a` rows from it, then `k` siblings of the chain row at each depth 1..a.
pub fn siblingTable(a: usize, k: usize) [MAXR * STRIDE]i32 {
    var t = table(a, 0);
    for (1..a + 1) |d| for (1..k + 1) |i| {
        const r = a + (d - 1) * k + i;
        t[r * STRIDE] = @intCast(d - 1);
        t[r * STRIDE + 1] = @intCast(d);
        for (0..d) |j| t[r * STRIDE + 2 + j] = @intCast(j);
        t[r * STRIDE + 2 + d] = @intCast(r);
    };
    return t;
}

const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\// fz_accept for a tree window (cfg: rows, next rows, cache capacity, the next window's chain length): a row lands
    \\// when its parent did and its token is the parent's pick; the deepest landed row (lowest on ties) ends the path
    \\kernel void fz_tree_accept(device uint* wids [[buffer(0)]], device const uint* picks [[buffer(1)]],
    \\    device int* ar [[buffer(2)]], device uint* ring [[buffer(3)]], device long* pm [[buffer(4)]],
    \\    constant uint4& cfg [[buffer(5)]], device uint* H [[buffer(6)]], const device int* T [[buffer(7)]],
    \\    uint tid [[thread_position_in_grid]]) {
    \\  if (tid != 0) return;
    \\  const int W = int(cfg.x), Wn = int(cfg.y), cap = int(cfg.z);
    \\  bool ok[16];
    \\  ok[0] = true;
    \\  int best = 0;
    \\  for (int v = 1; v < W; v++) {
    \\    const int p = T[v * 18];
    \\    ok[v] = ok[p] && wids[v] == picks[p];
    \\    if (ok[v] && T[v * 18 + 1] > T[best * 18 + 1]) best = v;
    \\  }
    \\  const int keep = T[best * 18 + 1] + 1;
    \\  int path[16];
    \\  for (int k = 0; k < keep; k++) path[k] = T[best * 18 + 2 + k];
    \\  const int round = ar[2], slot = (round & 511) * 20;
    \\  ring[slot] = uint(keep);
    \\  ring[slot + 1] = 0;
    \\  ring[slot + 2] = 0;
    \\  for (int i = 0; i < keep; i++) ring[slot + 3 + i] = picks[path[i]];
    \\  const int hl = ar[800];
    \\  for (int i = 0; i < keep; i++) H[hl + i] = picks[path[i]];
    \\  ar[800] = hl + keep;
    \\  for (int i = 0; i < keep; i++) { pm[0] = pm[1]; pm[1] = long(wids[path[i]]); }
    \\  const int t_old = ar[1], t_new = t_old + keep;
    \\  ar[0] = keep; ar[1] = t_new; ar[2] = round + 1;
    \\  for (int i = 0; i < 16; i++) {
    \\    const int dn = i < Wn ? T[i * 18 + 1] : 0; // a row's position is its depth
    \\    ar[4 + i] = i < Wn ? t_new + dn : 0;
    \\    ar[20 + i] = i < Wn ? t_new + dn + 1 : 0;
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
    \\  for (int i = 0; i < 16; i++) ar[580 + i] = i < keep ? path[i] : i;
    \\  ar[596] = path[keep - 1] + 1; ar[597] = t_old; ar[598] = keep;
    \\  wids[0] = picks[path[keep - 1]];
    \\}
    \\// fz_ple_ids over a tree window: a row's n-grams read its ancestors' tokens instead of the rows before it
    \\kernel void fz_tree_ple_ids(device const uint* tok [[buffer(0)]], device const long* pm [[buffer(1)]],
    \\    device uint* out [[buffer(2)]], constant uint& rows [[buffer(3)]], const device int* T [[buffer(4)]],
    \\    uint i [[thread_position_in_grid]]) {
    \\  const uint row = i / 16, hh = i % 16;
    \\  if (row >= rows) return;
    \\  const int d = T[row * 18 + 1];
    \\  long seq[18];
    \\  seq[0] = pm[0]; seq[1] = pm[1];
    \\  for (int q = 0; q <= d; q++) seq[2 + q] = long(tok[T[row * 18 + 2 + q]]);
    \\  const long eos = pm[2];
    \\  const uint at = 2 + uint(d);
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
    \\// Rows of n words: row i takes row path[i] for i < keep, ascending; path[i] >= i, so no source is overwritten first
    \\kernel void fz_tree_rows(device uint* X [[buffer(0)]], device const int* ar [[buffer(1)]], constant uint& n [[buffer(2)]],
    \\    uint w [[thread_position_in_grid]]) {
    \\  if (w >= n) return;
    \\  const int keep = ar[598];
    \\  for (int i = 1; i < keep; i++) {
    \\    const int s = ar[580 + i];
    \\    if (s != i) X[size_t(i) * n + w] = X[size_t(s) * n + w];
    \\  }
    \\}
    \\// An attention layer's keys, values and raw rows (fz_kv_write's layout): slot t_old + i takes t_old + path[i]
    \\kernel void fz_tree_kv(device ushort* keys [[buffer(0)]], device ushort* vals [[buffer(1)]], device ushort* raw [[buffer(2)]],
    \\    device const int* ar [[buffer(3)]], constant uint& cap [[buffer(4)]], uint j [[thread_position_in_grid]]) {
    \\  if (j >= 512) return;
    \\  const int keep = ar[598], t0 = ar[597];
    \\  for (int i = 1; i < keep; i++) {
    \\    const int s = ar[580 + i];
    \\    if (s == i) continue;
    \\    const size_t from = (size_t(j >> 8) * cap + t0 + s) * 256 + (j & 255);
    \\    const size_t to = (size_t(j >> 8) * cap + t0 + i) * 256 + (j & 255);
    \\    keys[to] = keys[from];
    \\    vals[to] = vals[from];
    \\    if (j < 128) raw[size_t(t0 + i) * 128 + j] = raw[size_t(t0 + s) * 128 + j];
    \\  }
    \\}
    \\// The head's next k picks after its own (largest logit, lowest id on ties) into wids[c.y ..]: c = (k, first row)
    \\kernel void fz_tree_sib(device const bfloat* L [[buffer(0)]], device uint* wids [[buffer(1)]],
    \\    constant uint& n [[buffer(2)]], device const uint* ids [[buffer(3)]], constant uint2& c [[buffer(4)]],
    \\    uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint sg [[simdgroup_index_in_threadgroup]]) {
    \\  constexpr int KM = 5;
    \\  float v[KM]; uint at[KM];
    \\  for (int j = 0; j < KM; j++) { v[j] = -INFINITY; at[j] = 0xffffffffu; }
    \\  for (uint i = t; i < n; i += 1024) {
    \\    float x = float(L[i]); uint ix = i;
    \\    for (int j = 0; j < KM; j++) if (x > v[j]) { const float tv = v[j]; const uint ta = at[j]; v[j] = x; at[j] = ix; x = tv; ix = ta; }
    \\  }
    \\  threadgroup float vb[32]; threadgroup uint va[32];
    \\  int head = 0;
    \\  for (int round = 0; round <= int(c.x); round++) {
    \\    float best = head < KM ? v[head] : -INFINITY; uint bat = head < KM ? at[head] : 0xffffffffu;
    \\    for (ushort o = 16; o > 0; o >>= 1) {
    \\      const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(bat, o);
    \\      if (ob > best || (ob == best && oa < bat)) { best = ob; bat = oa; }
    \\    }
    \\    if (lane == 0) { vb[sg] = best; va[sg] = bat; }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    best = vb[lane]; bat = va[lane];
    \\    for (ushort o = 16; o > 0; o >>= 1) {
    \\      const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(bat, o);
    \\      if (ob > best || (ob == best && oa < bat)) { best = ob; bat = oa; }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (head < KM && at[head] == bat) head++;
    \\    if (round > 0 && t == 0) wids[c.y + round - 1] = bat < n ? ids[bat] : 248053u;
    \\  }
    \\}
    \\// The branch's drafts (rows a + 1 ..) from a guess a reply position: c = (a, b, prompt length, guesses)
    \\kernel void fz_tree_feed(device uint* wids [[buffer(0)]], device const int* ar [[buffer(1)]],
    \\    device const uint* F [[buffer(2)]], constant uint4& c [[buffer(3)]], uint t [[thread_position_in_grid]]) {
    \\  if (t != 0) return;
    \\  const int p = ar[1] - int(c.z);
    \\  for (int k = 0; k < int(c.y); k++) {
    \\    const int at = p + 1 + k;
    \\    wids[c.x + 1 + k] = at >= 0 && at < int(c.w) ? F[at] : 248053u;
    \\  }
    \\}
;

/// A recorded attention part: window keys (>= the window's first slot) read at their row's ancestor slot.
const attn_edits = [_][2][]const u8{
    .{ "  device float* PM [[buffer(10)]],\n", "  device float* PM [[buffer(10)]],\n  const device int* TREE [[buffer(11)]],\n  const device int* TBASE [[buffer(12)]],\n" },
    .{ "    const size_t key = size_t(sparse ? ids[j] : j) * D;\n", "    int kj = sparse ? ids[j] : j;\n    if (kj >= TBASE[0]) kj = TBASE[0] + TREE[r * 18 + 2 + (kj - TBASE[0])];\n    const size_t key = size_t(kj) * D;\n" },
};

/// The recorded PLE conv: tap j of a row at depth d reads the tail (d - 9 + 3 j < 0) or that ancestor's normed row.
const conv_edits = [_][2][]const u8{
    .{ "[[kernel]]", "inline int fz_tree_cin(const device int* T, int r, int j) {\n  const int pos = T[r * 18 + 1] - 9 + 3 * j;\n  return pos < 0 ? 9 + pos : 9 + T[r * 18 + 2 + pos];\n}\n[[kernel]]" },
    .{ "  device bfloat16_t* HOUT [[buffer(4)]],\n", "  device bfloat16_t* HOUT [[buffer(4)]],\n  const device int* TREE [[buffer(5)]],\n" },
    .{ "float(CIN[size_t(r + DIL * j) * W + c])", "float(CIN[size_t(fz_tree_cin(TREE, r, j)) * W + c])" },
};

/// `text` with each edit's old text, present exactly once, replaced.
pub fn patch(a: std.mem.Allocator, text: []const u8, edits: []const [2][]const u8) ![]const u8 {
    var out = text;
    for (edits) |e| {
        if (std.mem.count(u8, out, e[0]) != 1) return error.KernelPatch;
        out = try std.mem.replaceOwned(u8, a, out, e[0], e[1]);
    }
    return out;
}

/// The guesses for one prompt: the branch's drafts at each reply position.
const Entry = struct { prompt: []const u32, guesses: []const u32 };

pub const Tree = struct {
    a: usize,
    b: usize,
    table: Buf,
    accept: mtl.Pipeline,
    ids: mtl.Pipeline,
    rows: mtl.Pipeline,
    kv: mtl.Pipeline,
    feeder: mtl.Pipeline,
    gdn: mtl.Pipeline,
    attn: *Variant,
    conv: *Variant,
    feed: Buf,
    sib: usize = 0, // the head's siblings at each chain depth (siblingTable), in place of a fed branch
    sibling: mtl.Pipeline,
    entries: []const Entry,
    guesses: []const u32 = &.{},
    prompt_len: usize = 0,

    pub fn width(t: *const Tree) usize {
        return 1 + t.a + t.b;
    }

    pub fn depth(t: *const Tree, row: usize) usize {
        return @intCast(t.table.b.slice(i32, MAXR * STRIDE)[row * STRIDE + 1]);
    }

    /// The branch's guesses for `prompt` (none when no entry holds it), on the GPU for the rounds.
    pub fn select(t: *Tree, prompt: []const u32) void {
        t.guesses = &.{};
        t.prompt_len = prompt.len;
        for (t.entries) |e| if (std.mem.eql(u32, e.prompt, prompt)) {
            t.guesses = e.guesses;
        };
        const n = @min(t.guesses.len, t.feed.b.length() / 4);
        @memcpy(t.feed.b.slice(u32, n), t.guesses[0..n]);
    }

    /// The first window's branch drafts, the pick being the reply's first token.
    pub fn firstDrafts(t: *const Tree, w: []u32) void {
        for (0..t.b) |k| w[t.a + 1 + k] = if (1 + k < t.guesses.len) t.guesses[1 + k] else NONE;
    }

    /// The head's next picks at chain depth `d` from its draft logits, into that depth's sibling rows.
    pub fn siblingsEncode(t: *const Tree, r: *Run, logits: Buf, n_ids: Buf, ids: Buf, wids: Buf, d: usize) void {
        r.enc.setPipeline(t.sibling);
        for ([_]Buf{ logits, wids, n_ids, ids }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        const c = [2]u32{ @intCast(t.sib), @intCast(t.a + (d - 1) * t.sib + 1) };
        r.enc.setBytes(std.mem.asBytes(&c), 4);
        r.enc.dispatchThreads(mtl.Size.of(1024, 1, 1), mtl.Size.of(1024, 1, 1));
        if (!r.serial) r.enc.barrier();
    }

    pub fn feedEncode(t: *const Tree, r: *Run, wids: Buf) void {
        r.enc.setPipeline(t.feeder);
        r.enc.setBuffer(wids.b, wids.off, 0);
        r.enc.setBuffer(r.ar.b, r.ar.off, 1);
        r.enc.setBuffer(t.feed.b, t.feed.off, 2);
        const c = [4]u32{ @intCast(t.a), @intCast(t.b), @intCast(t.prompt_len), @intCast(t.guesses.len) };
        r.enc.setBytes(std.mem.asBytes(&c), 3);
        r.enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    }

    pub fn pleIds(t: *const Tree, r: *Run, rows: usize, wids: Buf, pm: Buf, out: Buf) void {
        r.enc.setPipeline(t.ids);
        for ([_]Buf{ wids, pm, out }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        const n: u32 = @intCast(rows);
        r.enc.setBytes(std.mem.asBytes(&n), 3);
        r.enc.setBuffer(t.table.b, t.table.off, 4);
        r.enc.dispatchThreads(mtl.Size.of(16 * rows, 1, 1), mtl.Size.of(16 * rows, 1, 1));
    }

    pub fn attnParts(t: *const Tree, r: *Run, ins: []const Buf, outs: []const Buf) !void {
        try callWith(r, "q4_attn_parts#[24, 256]", t.attn, ins, outs, &.{ t.table, .{ .b = r.ar.b, .off = r.ar.off + 36 * 4 } }, 11);
    }

    pub fn pleConv(t: *const Tree, r: *Run, ins: []const Buf, outs: []const Buf) !void {
        try callWith(r, "q4_ple_conv@ple", t.conv, ins, outs, &.{t.table}, 5);
    }

    /// The verdict: cfg as fz_accept's with the next window's chain length last.
    pub fn acceptEncode(t: *const Tree, r: *Run, bufs: [5]Buf, hist: mtl.Buffer, cfg: [4]u32) void {
        r.enc.setPipeline(t.accept);
        for (bufs, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        r.enc.setBytes(std.mem.asBytes(&cfg), 5);
        r.enc.setBuffer(hist, 0, 6);
        r.enc.setBuffer(t.table.b, t.table.off, 7);
        r.enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    }

    /// The kept path into rows 0..keep-1: attention cache slots, the rows the head absorbs and the PLE tail's rows.
    pub fn compact(t: *const Tree, r: *Run, layers: anytype, cap: usize, last: Buf, picks: Buf, cin: Buf) void {
        const c: u32 = @intCast(cap);
        for (layers) |*L| if (!L.linear) {
            r.enc.setPipeline(t.kv);
            for ([_]Buf{ L.keys, L.vals, L.raw, r.ar }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
            r.enc.setBytes(std.mem.asBytes(&c), 4);
            r.enc.dispatchThreads(mtl.Size.of(512, 1, 1), mtl.Size.of(256, 1, 1));
        };
        const rows = [_]struct { b: Buf, n: u32 }{
            .{ .b = last, .n = WIDE / 2 },
            .{ .b = picks, .n = 1 },
            .{ .b = .{ .b = cin.b, .off = cin.off + PLE_TAIL * WIDE * 2 }, .n = WIDE / 2 },
        };
        for (rows) |x| {
            r.enc.setPipeline(t.rows);
            r.enc.setBuffer(x.b.b, x.b.off, 0);
            r.enc.setBuffer(r.ar.b, r.ar.off, 1);
            r.enc.setBytes(std.mem.asBytes(&x.n), 2);
            r.enc.dispatchThreads(mtl.Size.of(x.n, 1, 1), mtl.Size.of(@min(x.n, 256), 1, 1));
        }
    }
};

/// `role` at the window's rows on variant `v`, `extra` bound from slot `at` on.
fn callWith(r: *Run, role: []const u8, v: *Variant, ins: []const Buf, outs: []const Buf, extra: []const Buf, at: usize) !void {
    var key: [96]u8 = undefined;
    const s = r.roles.get(try std.fmt.bufPrint(&key, "{s}|{d}", .{ role, r.rows })) orelse return error.NoSite;
    try r.bindV(v, ins, outs);
    for (extra, 0..) |b, j| r.enc.setBuffer(b.b, b.off, at + j);
    r.enc.dispatchThreads(s.grid, s.tg);
    if (!r.serial) r.enc.barrier();
}

/// `role`'s recorded one-row kernel, patched and compiled as a variant of its own.
fn patched(r: *Run, role: []const u8, edits: []const [2][]const u8) !*Variant {
    var key: [96]u8 = undefined;
    const s = r.roles.get(try std.fmt.bufPrint(&key, "{s}|1", .{role})) orelse return error.NoSite;
    const text = try patch(r.arena, try Run.variantText(r.arena, s.v), edits);
    const lib = try r.ownResource(try mtl.Library.fromSource(r.device, text, mtl.CompileOptions.mlx()));
    const nv = try r.arena.create(Variant);
    nv.* = s.v.*;
    nv.pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, try std.fmt.allocPrintSentinel(r.arena, "{s}", .{s.v.name}, 0), false));
    nv.text = text;
    return nv;
}

/// A tree of a head chain of `a` and a branch of `b` (1 + a + b <= MAXR) fed from `feed`, or `sib` head siblings a depth.
pub fn init(r: *Run, a: usize, b: usize, feed: ?[]const u8, sib: usize) !*Tree {
    if (a == 0 or b == 0 or 1 + a + b > MAXR or (sib > 0 and (b != a * sib or sib > 4))) return error.TreeShape;
    const lib = try r.ownResource(try mtl.Library.fromSource(r.device, source, mtl.CompileOptions.mlx()));
    const t = try r.arena.create(Tree);
    const tab = if (sib > 0) siblingTable(a, sib) else table(a, b);
    const tb: Buf = .{ .b = try r.buffer(@sizeOf(@TypeOf(tab))) };
    @memcpy(tb.b.slice(i32, tab.len), &tab);
    t.* = .{
        .a = a,
        .b = b,
        .table = tb,
        .accept = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_tree_accept", false)),
        .ids = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_tree_ple_ids", false)),
        .rows = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_tree_rows", false)),
        .kv = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_tree_kv", false)),
        .feeder = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_tree_feed", false)),
        .sib = sib,
        .sibling = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_tree_sib", false)),
        .gdn = try gdn_step.compileTree(r),
        .attn = try patched(r, "q4_attn_parts#[24, 256]", &attn_edits),
        .conv = try patched(r, "q4_ple_conv@ple", &conv_edits),
        .feed = .{ .b = try r.buffer(65536 * 4) },
        .entries = if (feed) |f| try parse(r.arena, f) else &.{},
    };
    return t;
}

/// Feed text: the number of prompts, then each prompt's ids and its guesses a reply position, as count-led lists.
fn parse(a: std.mem.Allocator, text: []const u8) ![]const Entry {
    var it = std.mem.tokenizeAny(u8, text, " \n\r\t");
    const n = try std.fmt.parseInt(usize, it.next() orelse return error.BadFeed, 10);
    const out = try a.alloc(Entry, n);
    for (out) |*e| {
        var lists: [2][]u32 = undefined;
        for (&lists) |*l| {
            const k = try std.fmt.parseInt(usize, it.next() orelse return error.BadFeed, 10);
            l.* = try a.alloc(u32, k);
            for (l.*) |*v| v.* = try std.fmt.parseInt(u32, it.next() orelse return error.BadFeed, 10);
        }
        e.* = .{ .prompt = lists[0], .guesses = lists[1] };
    }
    return out;
}

test "a two-branch tree: the chain's rows, then the branch from the root" {
    const t = table(2, 3);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 0 }, t[0..3]);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 0, 1, 2 }, t[2 * STRIDE ..][0..5]); // row 2: chain depth 2
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 0, 3 }, t[3 * STRIDE ..][0..4]); // row 3: the branch's first
    try std.testing.expectEqualSlices(i32, &.{ 4, 3, 0, 3, 4, 5 }, t[5 * STRIDE ..][0..6]); // row 5: its third
}

test "a sibling tree: each sibling hangs off the chain row one depth up, ancestors the chain's" {
    const t = siblingTable(5, 2);
    try std.testing.expectEqual(@as(i32, 0), t[6 * STRIDE]); // depth 1 siblings hang off the root
    try std.testing.expectEqual(@as(i32, 1), t[6 * STRIDE + 1]);
    try std.testing.expectEqual(@as(i32, 6), t[6 * STRIDE + 3]);
    const r = 5 + 4 * 2 + 2; // depth 5, second sibling
    try std.testing.expectEqual(@as(i32, 4), t[r * STRIDE]);
    try std.testing.expectEqual(@as(i32, 5), t[r * STRIDE + 1]);
    for (0..5) |j| try std.testing.expectEqual(@as(i32, @intCast(j)), t[r * STRIDE + 2 + j]);
    try std.testing.expectEqual(@as(i32, @intCast(r)), t[r * STRIDE + 7]);
}

test "the tree patches apply once to the recorded attention and PLE conv texts" {
    const ks = @import("kernel_sources");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var seen: usize = 0;
    for (ks.flashnext_gen.sources) |src| {
        const edits: []const [2][]const u8 = if (std.mem.startsWith(u8, src.name, "q4_attn_parts")) &attn_edits else if (std.mem.startsWith(u8, src.name, "q4_ple_conv")) &conv_edits else continue;
        seen += 1;
        const out = try patch(arena.allocator(), src.text, edits);
        try std.testing.expect(std.mem.indexOf(u8, out, "TREE") != null);
        try std.testing.expectError(error.KernelPatch, patch(arena.allocator(), out, edits[edits.len - 1 ..]));
    }
    try std.testing.expectEqual(@as(usize, 2), seen);
}
