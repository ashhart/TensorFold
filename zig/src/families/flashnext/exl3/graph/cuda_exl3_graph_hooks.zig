//! The engine's hooks for EXL3 graphs: a decode round and a verify window prepare their host inputs and run.

const std = @import("std");
const wt = @import("../../cuda_prompt_wt.zig");
const state_binding = @import("../kv/exl3_state_binding.zig");
const dplan = @import("../dplan/cuda_exl3_dplan.zig");
const hub = @import("cuda_exl3_graph.zig");
const gkey = @import("cuda_exl3_graph_key.zig");
const grebind = @import("cuda_exl3_graph_rebind.zig");
const grun = @import("cuda_exl3_graph_run.zig");
const Key = gkey.Key;
const page_graph = @import("cuda_exl3_graph_pages.zig");

const dims = wt.dims;
const vocab = wt.vocab;
const att_rows = @import("../../cuda_batch_state.zig").att_rows;

/// A round's host inputs (what chunk's stage and gather would write), outside capture and replay.
pub fn prepareRound(pass: anytype, ids: []const u32) !void {
    const s = pass.s;
    try s.ready();
    for (s.hostIds(ids.len), ids) |*out, token| out.* = @intCast(token);
    const positions = s.hostPos(ids.len * 4);
    @memset(positions, 0);
    for (pass.lanes, 0..) |lane, i| {
        positions[i * 4] = @intCast(lane.pos);
        try pass.m.ngram.rows(lane.seq.history[0..], ids[i..][0..1], s.hostEmb(ids.len)[i * dims ..][0..dims], s.hostXs(ids.len)[i * (dims / 32) ..][0 .. dims / 32]);
    }
}

/// A window's host inputs: ids, one position a block of attention rows, the n-gram rows after the sequence's history.
pub fn prepareWindow(pass: anytype, ids: []const u32, pos0: usize) !void {
    const s = pass.s;
    const n = ids.len;
    try s.ready();
    for (s.hostIds(n), ids) |*out, token| out.* = @intCast(token);
    const blocks = (n + att_rows - 1) / att_rows;
    const positions = s.hostPos(blocks * 4);
    @memset(positions, 0);
    for (0..blocks) |b| positions[4 * b] = @intCast(pos0 + b * att_rows);
    try pass.m.ngram.rows(pass.seq.history[0..], ids, s.hostEmb(n), s.hostXs(n));
}

/// The device planner for `e`, or null when its decode steps keep the host cut.
pub fn planner(e: anytype, rows: usize) !?*dplan.DevPlan {
    if (!dplan.enabled()) return null;
    const dp = (try dplan.forLayers(e.gpa, e.driver, &e.exl3_dplan, e.layers[0..e.n_layers])) orelse return null;
    if (rows > dp.max_rows) {
        hub.stats.declined += 1;
        return null;
    }
    return dp;
}

/// Engine.round's hook: true when this module ran the round (captured or not); false leaves it to the plain pass.
pub fn round(e: anytype, p: anytype, seqs: anytype, ids: []const u32, scores: u64, out: u64) !bool {
    for (seqs) |q| if (q.media != null) return false;
    if (seqs.len > gkey.max_lanes) return false;
    const dp = (try planner(e, ids.len)) orelse return false;
    var staged: usize = 0;
    defer for (seqs[0..staged]) |q| if (q.sel) |selection| page_graph.finish(selection);
    for (seqs) |q| {
        if (q.sel) |selection| try page_graph.stage(selection, p.stream, 0, wt.attn_n, q.fed, 1);
        staged += 1;
    }
    try dp.toDevice(p.stream);
    e.exl3_dplan.active = dp;
    defer e.exl3_dplan.active = null;
    if (!hub.enabled()) {
        hub.stats.eager_dplan += 1;
        try p.chunk(ids, 0, null);
        return true;
    }
    var key: Key = .{ .owner = @intFromPtr(e), .kind = 1, .rows = ids.len, .scratch = p.s.ids, .scratch_cap = p.s.cap, .scores = scores, .output = out, .plan = dp.state.ptr };
    for (seqs, 0..) |q, i| {
        key.seq[i] = Key.seqOf(q);
        key.pages[i] = Key.pageOf(q.pg());
        if (q.sel) |selection| key.tiers[i] = Key.tierOf(selection, q.fed, 1);
        Key.seams(q.fed + 1, &key.chunks[i], &key.score_tiles[i]);
    }
    const tp0 = std.Io.Clock.awake.now(p.io).toNanoseconds();
    try prepareRound(p, ids);
    hub.stats.prepare_ns += @intCast(std.Io.Clock.awake.now(p.io).toNanoseconds() - tp0);
    hub.stats.prepares += 1;
    try grun.run(key, p, ids, 0);
    return true;
}

/// Engine.forward's hook for a window: every row's logits into hooks.logits_out (an MTP verify window, or a one-row
pub fn forward(e: anytype, p: anytype, seq: anytype, ids: []const u32) !bool {
    const hooks = p.hooks;
    if (hooks.logits != .all or hooks.logits_out == 0 or seq.media != null or ids.len == 0 or ids.len > att_rows) return false;
    if (seq.fed + ids.len > seq.cap or ids.len > p.s.cap) return false;
    const dp = (try planner(e, ids.len)) orelse return false;
    if (seq.sel) |selection| try page_graph.stage(selection, p.stream, 0, wt.attn_n, seq.fed, ids.len);
    defer if (seq.sel) |selection| page_graph.finish(selection);
    try dp.toDevice(p.stream);
    e.exl3_dplan.active = dp;
    defer e.exl3_dplan.active = null;
    const n = ids.len;
    if (!hub.enabled()) {
        hub.stats.eager_dplan += 1;
        try p.chunk(ids, seq.fed, null);
    } else {
        var key: Key = .{ .owner = @intFromPtr(e), .kind = 2, .rows = n, .scratch = p.s.ids, .scratch_cap = p.s.cap, .scores = p.seq.scores, .output = hooks.logits_out, .streams = hooks.streams, .plan = dp.state.ptr };
        key.seq[0] = Key.seqOf(seq);
        key.pages[0] = Key.pageOf(seq.pg());
        if (seq.sel) |selection| key.tiers[0] = Key.tierOf(selection, seq.fed, n);
        Key.seams(seq.fed + n, &key.chunks[0], &key.score_tiles[0]);
        if (hooks.keep) |kp| {
            key.keep = std.hash.Wyhash.hash(1, std.mem.asBytes(kp));
            if (grebind.stateRebindEnabled()) key.keep_nonstate = state_binding.keepHash(kp.*);
        }
        const tp0 = std.Io.Clock.awake.now(p.io).toNanoseconds();
        try prepareWindow(p, ids, seq.fed);
        hub.stats.prepare_ns += @intCast(std.Io.Clock.awake.now(p.io).toNanoseconds() - tp0);
        hub.stats.prepares += 1;
        try grun.run(key, p, ids, seq.fed);
    }
    if (hooks.output_tail) |tail| try tail.enqueue(tail.ptr, p.stream);
    // the pass's head(.all) copies the last row's scores to the host after the step (outside the graph)
    if (p.m.collective) |c| try c.finish(p.stream) else try p.stream.synchronize();
    const d = p.stream.d;
    try d.check(d.api.cuMemcpyDtoH_v2(seq.logits.ptr, hooks.logits_out + (n - 1) * vocab * 2, vocab * 2), "cuMemcpyDtoH");
    seq.fed += n;
    return true;
}
