//! TF_QWEN_PROMPT_SPLITK: a prompt chunk of at most 64 rows over an e4m3 cache, its keys spread over every SM.
const std = @import("std");
const cuda = @import("cuda");
const E = @import("cuda_engine.zig");
const geometry = @import("cuda_geometry.zig");
const Seg = @import("cuda_cache.zig").Seg;
const head_dim = geometry.head_dim;

/// prefill_attention_split.cuh's passes; index 0 a whole window, 1 pages.
pub const Fns = struct { maxima: [2]cuda.Function, weights: [2]cuda.Function, chain: [2]cuda.Function, scan: cuda.Function, diff: cuda.Function };

const geo_sig = "PK13__nv_bfloat16PKhS5_PKfS7_iiiiifiPKixiii";
pub const symbols = struct {
    pub const maxima = [2][:0]const u8{ "_ZN16tf_prefill_split13scores_kernelILi0ELb0EEEv" ++ geo_sig ++ "iiPfS7_PjSA_", "_ZN16tf_prefill_split13scores_kernelILi0ELb1EEEv" ++ geo_sig ++ "iiPfS7_PjSA_" };
    pub const weights = [2][:0]const u8{ "_ZN16tf_prefill_split13scores_kernelILi1ELb0EEEv" ++ geo_sig ++ "iiPfS7_PjSA_", "_ZN16tf_prefill_split13scores_kernelILi1ELb1EEEv" ++ geo_sig ++ "iiPfS7_PjSA_" };
    pub const chain = [2][:0]const u8{ "_ZN16tf_prefill_split12chain_kernelILb0EEEv" ++ geo_sig ++ "iPKjS7_S7_PS1_", "_ZN16tf_prefill_split12chain_kernelILb1EEEv" ++ geo_sig ++ "iPKjS7_S7_PS1_" };
    pub const scan = "_ZN16tf_prefill_split11scan_kernelEPKfPfS2_iii";
    pub const diff = "_ZN16tf_prefill_split11diff_kernelEPKtS1_iPj";
};

/// The tile kernel's one row block: the passes run once a group of 16 rows.
pub const most_rows = 64;
/// Query heads a key head the kernels are built for (the 27B's 24 over 4).
pub const group = 6;
pub const threads = 32 * group;
/// Two 32-key slots and six warps' staged queries of 256 dims (A_SMEM).
pub const scores_shared: u32 = (2 * 32 * 32 + group * 16 * 32) * 16;
/// 8-dim output fragments a chain warp (CW) and its four value tiles in flight (C_SMEM).
pub const chain_fragments = 2;
pub const chain_shared: u32 = 4 * 64 * chain_fragments * 16;
/// Key segments a key head for the score passes at most.
pub const most_segments = 88;

pub fn resolve(attention: cuda.Module) !Fns {
    var f: Fns = undefined;
    for (0..2) |i| {
        f.maxima[i] = try attention.function(symbols.maxima[i]);
        f.weights[i] = try attention.function(symbols.weights[i]);
        f.chain[i] = try attention.function(symbols.chain[i]);
        try f.maxima[i].allowDynamicShared(scores_shared);
        try f.weights[i].allowDynamicShared(scores_shared);
    }
    f.scan = try attention.function(symbols.scan);
    f.diff = try attention.function(symbols.diff);
    return f;
}

var state: std.atomic.Value(u8) = .init(0); // 0 unread, 1 off, 2 on, 3 on and checked against the tile kernel

fn mode() u8 {
    var v = state.load(.monotonic);
    if (v == 0) {
        const env = @import("sys").environment;
        const on = env.value(bool, "TF_QWEN_PROMPT_SPLITK", @import("profile").lookup("TF_QWEN_PROMPT_SPLITK"), false) catch false;
        const check = env.value(bool, "TF_QWEN_PROMPT_SPLITK_VERIFY", @import("profile").lookup("TF_QWEN_PROMPT_SPLITK_VERIFY"), false) catch false;
        v = if (!on) 1 else if (check) 3 else 2;
        state.store(v, .monotonic);
    }
    return v;
}

/// TF_QWEN_PROMPT_SPLITK_VERIFY: every split chunk is also run by the tile kernel and compared bit for bit.
pub fn verifies() bool {
    return mode() == 3;
}

/// Key tiles the tile kernel's block folds: those of its last row.
pub fn tilesOf(len: usize, rows: usize) usize {
    return (len + rows - 1) / 64 + 1;
}

/// Bytes of weights (e.po) and of maxima, factors and sums (e.logits) a chunk's tiles take.
pub fn scratchBytes(q_heads: usize, tiles: usize) [2]usize {
    return .{ q_heads * tiles * 2048, 3 * q_heads * tiles * 16 * 4 };
}

/// Whether this chunk takes the split passes: the switch, an e4m3 cache, the 27B's heads, few rows, scratch that fits.
pub fn admits(e: *const E.Engine, sg: Seg, rows: usize) bool {
    const g = e.g;
    if (mode() == 1 or !e.kv8 or rows == 0 or rows > most_rows or g.kv_heads == 0) return false;
    if (g.q_heads != group * g.kv_heads or !g.heads or e.logits == 0) return false;
    const need = scratchBytes(g.q_heads, tilesOf(@intCast(sg.len), rows));
    const partials = E.most_verify * g.q_heads * 64;
    if (verifies() and rows * g.q_heads * head_dim * 2 > partials * 4) return false;
    return need[0] <= partials * head_dim * 4 and need[1] <= E.most_verify * g.vocab * 2;
}

/// The arguments every pass takes first: the caches, the chunk, the pages, the group's first row and the last row.
fn geoArgs(a: *cuda.Args, e: *const E.Engine, sg: Seg, layer: usize, rows: usize, at: u64, r0: usize) void {
    const g = e.g;
    const paged = sg.pt != 0;
    const heads: u64 = @as(u64, g.kv_heads) * (if (paged) @as(u64, @intCast(sg.ptok)) else @as(u64, @intCast(sg.cap)));
    const keys = sg.kv + layer * 2 * heads * (head_dim + 4);
    const values = keys + heads * head_dim;
    const kscale = values + heads * head_dim;
    for ([_]u64{ e.qn + at, keys, values, kscale, kscale + heads * 4 }) |v| a.add(v);
    for ([_]usize{ @intCast(sg.len), rows, g.q_heads, g.kv_heads, group }) |v| a.add(@as(c_int, @intCast(v)));
    a.add(@as(f32, 1.0 / 16.0));
    a.add(@as(c_int, @intCast(sg.cap)));
    a.add(@as(u64, if (paged) sg.pt else 0));
    a.add(@as(i64, if (paged) @as(i64, @intCast(sg.ps)) else 0));
    a.add(@as(c_int, if (paged) sg.ptok else 1));
    a.add(@as(c_int, @intCast(r0)));
    a.add(@as(c_int, @intCast(rows - 1)));
}

/// Segments a key head and tiles a segment for `tiles` key tiles.
pub fn segments(tiles: usize) [2]usize {
    const tps = (tiles + @min(tiles, most_segments) - 1) / @min(tiles, most_segments);
    return .{ (tiles + tps - 1) / tps, tps };
}

/// The chunk's rows from byte `at` of e.qn into `out`: maxima, scan, weights and chain, once a group of 16 rows.
pub fn run(e: *E.Engine, sg: Seg, layer: usize, rows: usize, at: u64, out: u64) !void {
    const g = e.g;
    const f = e.prompt_attn.split;
    const pg: usize = @intFromBool(sg.pt != 0);
    const tiles = tilesOf(@intCast(sg.len), rows);
    const seg = segments(tiles);
    const stat = g.q_heads * tiles * 16 * 4;
    const n = e.logits; // the scan writes the running maxima over the tile maxima in place
    const factors = n + stat;
    const sums = factors + stat;
    const scores_cfg: cuda.Config = .{ .grid = .{ .x = 1, .y = @intCast(g.kv_heads), .z = @intCast(seg[0]) }, .block = .{ .x = threads }, .shared = scores_shared };
    const scan_cfg: cuda.Config = .{ .grid = .{ .x = @intCast((g.q_heads * 16 + 127) / 128) }, .block = .{ .x = 128 } };
    const chain_cfg: cuda.Config = .{ .grid = .{ .x = @intCast(g.kv_heads), .y = head_dim / 8 / chain_fragments }, .block = .{ .x = threads }, .shared = chain_shared };
    var r0: usize = 0;
    while (r0 < rows) : (r0 += 16) {
        inline for (.{ f.maxima[pg], f.weights[pg] }, 0..) |pass, i| {
            if (i == 1) {
                var s: cuda.Args = .{};
                for ([_]u64{ n, n, factors }) |v| s.add(v);
                for ([_]usize{ g.q_heads, tiles, tiles }) |v| s.add(@as(c_int, @intCast(v)));
                try cuda.launch.launch(f.scan, scan_cfg, e.stream, &s);
            }
            var a: cuda.Args = .{};
            geoArgs(&a, e, sg, layer, rows, at, r0);
            for ([_]usize{ seg[1], tiles }) |v| a.add(@as(c_int, @intCast(v)));
            for ([_]u64{ n, n, e.po, sums }) |v| a.add(v);
            try cuda.launch.launch(pass, scores_cfg, e.stream, &a);
        }
        var c: cuda.Args = .{};
        geoArgs(&c, e, sg, layer, rows, at, r0);
        c.add(@as(c_int, @intCast(tiles)));
        for ([_]u64{ e.po, sums, factors, out }) |v| c.add(v);
        try cuda.launch.launch(f.chain[pg], chain_cfg, e.stream, &c);
    }
}

/// Stops the process unless the `words` bf16 words at `got` equal those at `want` (TF_QWEN_PROMPT_SPLITK_VERIFY).
pub fn same(e: *E.Engine, got: u64, want: u64, words: usize, layer: usize, rows: usize) !void {
    try e.d.check(e.d.api.cuMemsetD8Async(e.pl, 0, 4, e.stream.handle), "cuMemsetD8Async");
    var a: cuda.Args = .{};
    for ([_]u64{ got, want }) |v| a.add(v);
    a.add(@as(c_int, @intCast(words)));
    a.add(e.pl);
    try cuda.launch.launch(e.prompt_attn.split.diff, .{ .grid = .{ .x = @intCast((words + 255) / 256) }, .block = .{ .x = 256 } }, e.stream, &a);
    try e.stream.synchronize();
    var bad: u32 = 0;
    try e.d.check(e.d.api.cuMemcpyDtoH_v2(@ptrCast(&bad), e.pl, 4), "cuMemcpyDtoH");
    if (bad == 0) return;
    std.log.err("TF_QWEN_PROMPT_SPLITK_VERIFY: {d} of {d} attention words differ (layer {d}, {d} rows)", .{ bad, words, layer, rows });
    @panic("TF_QWEN_PROMPT_SPLITK_VERIFY: the split prompt attention differs from the tile kernel");
}

test "split prompt attention: tiles, segments and scratch stay inside the decode partials" {
    const t = std.testing;
    try t.expectEqual(@as(usize, 1), tilesOf(0, 1));
    try t.expectEqual(@as(usize, 2), tilesOf(1, 64));
    try t.expectEqual(@as(usize, 4096), tilesOf(262080, 64));
    try t.expectEqual([2]usize{ 1, 1 }, segments(1));
    try t.expectEqual([2]usize{ 88, 1 }, segments(88));
    try t.expectEqual([2]usize{ 45, 2 }, segments(89));
    for ([_]usize{ 1, 87, 88, 89, 2188, 4097 }) |tiles| {
        const s = segments(tiles);
        try t.expect(s[0] <= most_segments and s[0] * s[1] >= tiles and (s[0] - 1) * s[1] < tiles);
    }
    // the decode partials hold 4,096 tiles of weights: a chunk ending at position 262,143 fits, the next does not
    const partials = E.most_verify * 24 * 64 * head_dim * 4;
    try t.expect(scratchBytes(24, 4096)[0] <= partials and scratchBytes(24, 4097)[0] > partials);
    try t.expectEqual(@as(usize, 3 * 24 * 4096 * 64), scratchBytes(24, 4096)[1]);
}
