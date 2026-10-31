//! A shared pass's lanes: what a lane may hold, what its rollback may keep, and attention in one launch a kernel.
const std = @import("std");
const cuda = @import("cuda");
const state = @import("cuda_batch_state.zig");
const glue = @import("cuda_glue.zig");
const wt = @import("cuda_prompt_wt.zig");

const Keep = state.Keep;
const HcKeep = state.HcKeep;
const LayerKeep = state.LayerKeep;
const Lane = state.Lane;
const att_rows = state.att_rows;
const lanes_n = state.attn_lanes;

/// Lanes whose rows add up to `n`, each window within its cache and the attention block, keeping a rollback at most.
pub fn checkLanes(lanes: []const Lane, n: usize) error{UnexpectedTensor}!void {
    var total: usize = 0;
    for (lanes) |l| {
        if (l.rows == 0 or l.rows > att_rows or l.pos + l.rows > l.seq.cap) return error.UnexpectedTensor;
        if (l.keep) |kp| if (!rollbackOnly(kp)) return error.UnexpectedTensor;
        total += l.rows;
    }
    if (total != n) return error.UnexpectedTensor;
}

/// A drafted window's rollback (the DeltaNet rows, history and end states alone), not a training pass's tensors.
pub fn rollbackOnly(k: *const Keep) bool {
    if (!std.meta.eql(k.mixer, HcKeep{})) return false;
    for (k.layers) |lk| {
        var rest = lk;
        rest.proj = 0;
        rest.hist = 0;
        rest.state_out = 0;
        if (!std.meta.eql(rest, LayerKeep{})) return false;
    }
    return true;
}

/// TF_FLASHNEXT_ATTN_LANES=1: a shared round's attention in one launch a kernel for every lane (else a lane at a time).
fn attnLanesOn() bool {
    const v = std.c.getenv("TF_FLASHNEXT_ATTN_LANES") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// Layer `ai`'s cache addresses for lane `i` in a lane table.
pub fn putCaches(tab: []u8, ai: usize, i: usize, kc: u64, vc: u64, ikc: u64, pooled: u64) void {
    const ptrs: []u64 = @alignCast(std.mem.bytesAsSlice(u64, tab[0..state.lanes_ptrs]));
    ptrs[(ai * lanes_n + i) * 4 ..][0..4].* = .{ kc, vc, ikc, pooled };
}

/// Lane `i`'s position, rows and first round row in a lane table.
pub fn putLane(tab: []u8, i: usize, row0: usize, pos: usize, rows: usize) void {
    const ints: []i32 = @alignCast(std.mem.bytesAsSlice(i32, tab[state.lanes_ptrs..]));
    ints[i] = @as(i32, @intCast(pos)) - @as(i32, @intCast(row0));
    ints[lanes_n + i] = @intCast(pos);
    ints[2 * lanes_n + i] = @intCast(rows);
    ints[3 * lanes_n + i] = @intCast(row0);
    for (0..rows) |r| ints[4 * lanes_n + row0 + r] = @intCast(i);
}

/// Whether a round's `n` rows take attn: two or more lanes of decode rows (the per-lane path's next-tile-ahead chunks),
/// text on contiguous caches, one block-score buffer, our glue owning every launch.
pub fn useAttn(pass: anytype, n: usize) bool {
    if (comptime !@hasDecl(@TypeOf(pass.tri), "lanesGlue")) return false;
    if (!attnLanesOn() or pass.lanes.len < 2 or pass.lanes.len > lanes_n or n > att_rows) return false;
    if (wt.index_heads != 4 or pass.tri.lanesGlue() == null) return false;
    for (pass.lanes) |l| {
        if (l.seq.media != null or l.seq.pg.pt != 0 or l.rows == 0 or l.rows > glue.ahead_rows) return false;
        if (l.seq.kc.len > wt.layers_n / 4 or l.seq.scores != pass.lanes[0].seq.scores or l.pos + l.rows > l.seq.cap) return false;
    }
    return true;
}

/// Uploads the lane table attn reads: each attention layer's caches a lane, the lanes' positions and rows.
pub fn stageAttn(pass: anytype) !void {
    const tab = pass.s.hostLanes();
    @memset(tab, 0);
    var row0: usize = 0;
    for (pass.lanes, 0..) |l, i| {
        for (0..l.seq.kc.len) |ai| putCaches(tab, ai, i, l.seq.kc[ai].ptr, l.seq.vc[ai].ptr, l.seq.ikc[ai].ptr, l.seq.pooled[ai].ptr);
        putLane(tab, i, row0, l.pos, l.rows);
        row0 += l.rows;
    }
    try pass.upload(pass.s.lanes_tab, tab);
}

/// The round's rows in one launch a kernel: each row's rope and cache writes, its lane's pooling and scores, then every
/// row's selection, attention and merge, the per-lane passes' programs over the round's rows.
pub fn attn(pass: anytype, lw: anytype, ai: usize, pa: u64) !void {
    if (comptime !@hasDecl(@TypeOf(pass.tri), "lanesGlue")) return error.NativeGlueRequired;
    const s = pass.s;
    const g = pass.tri.lanesGlue() orelse return error.NativeGlueRequired;
    const n = pass.rows;
    const lp = s.lanes_tab + ai * lanes_n * 4 * 8;
    const li = s.lanes_tab + state.lanes_ptrs;
    var max_rows: usize = 0;
    var max_blocks: usize = 0;
    var chunks: usize = 0;
    var cap: usize = 0;
    for (pass.lanes) |l| {
        const ends = l.pos + l.rows;
        max_rows = @max(max_rows, l.rows);
        if (ends / wt.index_ratio > wt.index_top) max_blocks = @max(max_blocks, @min(state.scoreStride(l.seq.cap), cdiv(ends, wt.index_ratio)));
        chunks = @max(chunks, cdiv(@min(ends, wt.dense_keys), 512));
        cap = @max(cap, l.seq.cap);
    }
    // one stride for every lane's rows of the engine's block scores (reserved for the longest cache)
    const nb = state.scoreStride(cap);
    const scores = pass.lanes[0].seq.scores;
    try g.attnPrepLanes(pass.stream, pa, lp, li, lw.q_scale.ptr, lw.k_scale.ptr, lw.i_scale.ptr, s.inv, s.q, s.iq, wt.eps, n);
    try g.qsaPoolLanes(pass.stream, lp, li, lw.ik_scale.ptr, s.inv, wt.eps, pass.lanes.len, max_rows);
    if (max_blocks > 0) try g.qsaScoresLanes(pass.stream, s.iq, lp, li, scores, nb, pass.lanes.len, max_rows, max_blocks);
    var a: cuda.Args = .{};
    inline for (.{ scores, li, s.qids, s.qnk, s.qspr, cint(nb), cint(wt.index_idw), cint(wt.index_top), cint(wt.index_ratio) }) |v| a.add(v);
    try cuda.launch.launch(pass.k.select_lanes, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 1024 } }, pass.stream, &a);
    try g.attnChunksLanes(pass.stream, s.q, lp, li, s.po, s.pm, s.pl, s.qids, s.qnk, s.qspr, n, wt.kv_heads, chunks);
    try g.attnMergeLanes(pass.stream, s.po, s.pm, s.pl, li, s.ao, s.qnk, s.qspr, n, wt.kv_heads);
}

fn cint(v: usize) c_int {
    return @intCast(v);
}
