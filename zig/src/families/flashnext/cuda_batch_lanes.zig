//! A shared pass's lanes: what a lane may hold, what its rollback may keep, attention and DeltaNet in one launch a kernel.
const std = @import("std");
const cuda = @import("cuda");
const state = @import("cuda_batch_state.zig");
const glue = @import("cuda_glue.zig");
const wt = @import("cuda_prompt_wt.zig");
const gdn_ = @import("cuda_gdn.zig");

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

fn switchOn(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// TF_FLASHNEXT_ATTN_LANES=1: a shared round's attention in one launch a kernel for every lane (else a lane at a time).
pub fn attnOn() bool {
    return switchOn("TF_FLASHNEXT_ATTN_LANES");
}

/// TF_FLASHNEXT_GDN_LANES=1: a shared round's DeltaNet in one launch a kernel for every lane (else a lane at a time).
pub fn gdnOn() bool {
    return switchOn("TF_FLASHNEXT_GDN_LANES");
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// Layer `ai`'s cache addresses for lane `i` in a lane table.
pub fn putCaches(tab: []u8, ai: usize, i: usize, kc: u64, vc: u64, ikc: u64, pooled: u64) void {
    const ptrs: []u64 = @alignCast(std.mem.bytesAsSlice(u64, tab[0..state.lanes_ptrs]));
    ptrs[(ai * lanes_n + i) * 4 ..][0..4].* = .{ kc, vc, ikc, pooled };
}

/// DeltaNet layer `li`'s addresses for lane `i` in a lane table; a state written in place has `state_out` equal to it.
pub fn putGdn(tab: []u8, li: usize, i: usize, conv: u64, st: u64, state_out: u64, proj: u64, hist: u64) void {
    const ptrs: []u64 = @alignCast(std.mem.bytesAsSlice(u64, tab[state.lanes_attn_ptrs..state.lanes_ptrs]));
    ptrs[(li * lanes_n + i) * 5 ..][0..5].* = .{ conv, st, if (state_out != 0) state_out else st, proj, hist };
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

/// Whether a round's `n` rows stage a lane table: two or more lanes of decode rows (the per-lane path's next-tile-ahead
/// chunks and small-row chain), text on contiguous caches, one block-score buffer, our glue owning every launch.
pub fn useTable(pass: anytype, n: usize) bool {
    if (comptime !@hasDecl(@TypeOf(pass.tri), "lanesGlue")) return false;
    if (!(attnOn() or gdnOn()) or pass.lanes.len < 2 or pass.lanes.len > lanes_n or n > att_rows) return false;
    if (wt.index_heads != 4 or pass.tri.lanesGlue() == null) return false;
    for (pass.lanes) |l| {
        if (l.seq.media != null or l.seq.pg.pt != 0 or l.rows == 0 or l.rows > glue.ahead_rows) return false;
        if (l.seq.kc.len > wt.attn_n or l.seq.conv.len > wt.linear_n or l.seq.state.len != l.seq.conv.len) return false;
        if (l.seq.scores != pass.lanes[0].seq.scores or l.pos + l.rows > l.seq.cap) return false;
    }
    return true;
}

/// Uploads the lane table: each attention layer's caches a lane, each DeltaNet layer's window, state and keeps, the
/// lanes' positions and rows.
pub fn stage(pass: anytype) !void {
    const tab = pass.s.hostLanes();
    @memset(tab, 0);
    var row0: usize = 0;
    for (pass.lanes, 0..) |l, i| {
        for (0..l.seq.kc.len) |ai| putCaches(tab, ai, i, l.seq.kc[ai].ptr, l.seq.vc[ai].ptr, l.seq.ikc[ai].ptr, l.seq.pooled[ai].ptr);
        for (0..wt.layers_n) |layer| {
            if (!wt.linearLayer(layer)) continue;
            const li = wt.linearIndex(layer);
            if (li >= l.seq.conv.len) continue;
            const lk: LayerKeep = if (l.keep) |kp| kp.layers[layer] else .{};
            putGdn(tab, li, i, l.seq.conv[li].ptr, l.seq.state[li].ptr, lk.state_out, lk.proj, lk.hist);
        }
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

/// What gdnRound launches on: a DeltaNet layer's table and weights, the round's projection rows and scratch.
pub const Gdn = struct { lp: u64, li: u64, proj: u64, cw: u64, a_log: u64, dt: u64, gq: u64, gk: u64, gv: u64, ggb: u64, ys: u64, rows: usize, lanes: usize };

/// The round's DeltaNet in four launches: each lane's kept rows and window, every row's front, every lane's chain on
/// its own state, then every lane's conv window past its rows (the per-lane loop's order).
pub fn gdnRound(k: anytype, stream: cuda.Stream, g: Gdn) !void {
    const lanes: u32 = @intCast(g.lanes);
    var a: cuda.Args = .{};
    inline for (.{ g.lp, g.li, g.proj, @as(c_longlong, gdn_.proj_width * 2 / 16), @as(c_longlong, (gdn_.taps - 1) * gdn_.conv_dim * 2 / 16) }) |v| a.add(v);
    try cuda.launch.launch(k.keep_lanes, .{ .grid = .{ .x = 16, .y = lanes, .z = 2 }, .block = .{ .x = 256 } }, stream, &a);
    a = .{};
    inline for (.{ g.proj, g.lp, g.li, g.cw, g.a_log, g.dt, g.gq, g.gk, g.gv, g.ggb }) |v| a.add(v);
    try cuda.launch.launch(k.front_lanes, .{ .grid = .{ .x = @intCast(g.rows), .y = gdn_.nk + gdn_.nv }, .block = .{ .x = 256 } }, stream, &a);
    a = .{};
    inline for (.{ g.gq, g.gk, g.gv, g.ggb, g.lp, g.li, g.ys }) |v| a.add(v);
    try cuda.launch.launch(k.chain_lanes, .{ .grid = .{ .x = gdn_.nv, .y = gdn_.dv / 16, .z = lanes }, .block = .{ .x = 128 } }, stream, &a);
    a = .{};
    inline for (.{ g.lp, g.li, g.proj, @as(c_longlong, gdn_.proj_width), cint(gdn_.taps - 1), cint(gdn_.conv_dim) }) |v| a.add(v);
    try cuda.launch.launch(k.shift_lanes, .{ .grid = .{ .x = @intCast(cdiv(gdn_.conv_dim, 256)), .y = lanes }, .block = .{ .x = 256 } }, stream, &a);
}

/// gdnRound on a pass's scratch, for its DeltaNet layer `li`.
pub fn gdn(pass: anytype, lw: anytype, li: usize) !void {
    const s = pass.s;
    return gdnRound(pass.k, pass.stream, .{
        .lp = s.lanes_tab + state.lanes_attn_ptrs + li * lanes_n * 5 * 8,
        .li = s.lanes_tab + state.lanes_ptrs,
        .proj = s.proj,
        .cw = lw.cw.ptr,
        .a_log = lw.a_log.ptr,
        .dt = lw.dt.ptr,
        .gq = s.gq,
        .gk = s.gk,
        .gv = s.gv,
        .ggb = s.ggb,
        .ys = s.ys,
        .rows = pass.rows,
        .lanes = pass.lanes.len,
    });
}

fn cint(v: usize) c_int {
    return @intCast(v);
}
