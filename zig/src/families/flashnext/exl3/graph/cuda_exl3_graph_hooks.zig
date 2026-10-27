//! The engine's hooks for EXL3 graphs: a decode round and a verify window prepare their host inputs and run.

const std = @import("std");
const wt = @import("../../cuda_prompt_wt.zig");
const state_binding = @import("../kv/exl3_state_binding.zig");
const dplan = @import("../dplan/cuda_exl3_dplan.zig");
const hub = @import("cuda_exl3_graph.zig");
const gkey = @import("cuda_exl3_graph_key.zig");
const gcache = @import("cuda_exl3_graph_cache.zig");
const grebind = @import("cuda_exl3_graph_rebind.zig");
const grun = @import("cuda_exl3_graph_run.zig");
const Key = gkey.Key;

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
    const dp = (try dplan.forLayers(e.gpa, e.driver, e, e.layers[0..e.n_layers])) orelse return null;
    if (rows > dp.max_rows) {
        hub.stats.declined += 1;
        return null;
    }
    return dp;
}

/// Engine.round's hook: true when this module ran the round (device plan, captured or not); false leaves it to the
pub fn round(e: anytype, p: anytype, seqs: anytype, ids: []const u32, scores: u64, out: u64) !bool {
    for (seqs) |q| if (q.media != null) return false;
    if (seqs.len > gkey.max_lanes) return false;
    for (seqs) |q| if (q.paged) |pg| try pg.validateSharedLeaseOn(p.stream);
    const dp = (try planner(e, ids.len)) orelse return false;
    try dp.toDevice(p.stream);
    dplan.active = dp;
    defer dplan.active = null;
    if (hub.enabled() and gcache.broken) return error.GraphCaptureFailed;
    if (!hub.enabled()) {
        hub.stats.eager_dplan += 1;
        try p.chunk(ids, 0, null);
        return true;
    }
    for (seqs) |q| if (q.paged) |pg| for (0..wt.attn_n) |ai| try pg.validateRange(ai, q.fed, 1);
    defer for (seqs) |q| if (q.paged) |pg| {
        pg.graph_prepared = false;
    };
    for (seqs) |q| if (q.paged) |pg| try pg.prepareGraph(p.stream, q.fed, 1);
    var key: Key = .{ .kind = 1, .rows = ids.len, .scratch = p.s.ids, .scratch_cap = p.s.cap, .scores = scores, .output = out, .plan = dp.state.ptr };
    for (seqs, 0..) |q, i| {
        key.seq[i] = Key.seqOf(q);
        if (q.paged) |pg| {
            key.paging[i] = pg.graphKey();
            if (key.paging_logical == null) key.paging_logical = @splat(@splat(0));
            key.paging_logical.?[i] = pg.logicalGraphKey();
        }
        Key.seams(q.fed + 1, &key.chunks[i], &key.score_tiles[i]);
    }
    const tp0 = std.Io.Clock.awake.now(p.io).toNanoseconds();
    try prepareRound(p, ids);
    hub.stats.prepare_ns += @intCast(std.Io.Clock.awake.now(p.io).toNanoseconds() - tp0);
    hub.stats.prepares += 1;
    grun.run(key, p, ids, 0) catch |err| {
        if (p.m.collective) |c| {
            c.failGraph();
            return err;
        }
        gcache.broken = true;
        return err;
    };
    return true;
}

/// Engine.forward's hook for a window: every row's logits into hooks.logits_out (an MTP verify window, or a one-row
pub fn forward(e: anytype, p: anytype, seq: anytype, ids: []const u32) !bool {
    const hooks = p.hooks;
    if (hooks.logits != .all or hooks.logits_out == 0 or seq.media != null or ids.len == 0 or ids.len > att_rows) return false;
    if (seq.fed + ids.len > seq.cap or ids.len > p.s.cap) return false;
    if (seq.paged) |pg| try pg.validateSharedLeaseOn(p.stream);
    const dp = (try planner(e, ids.len)) orelse return false;
    try dp.toDevice(p.stream);
    dplan.active = dp;
    defer dplan.active = null;
    const n = ids.len;
    if (hub.enabled() and gcache.broken) return error.GraphCaptureFailed;
    if (!hub.enabled()) {
        hub.stats.eager_dplan += 1;
        try p.chunk(ids, seq.fed, null);
    } else {
        defer if (seq.paged) |pg| {
            pg.graph_prepared = false;
        };
        if (seq.paged) |pg| try pg.prepareGraph(p.stream, seq.fed, n);
        var key: Key = .{ .kind = 2, .rows = n, .scratch = p.s.ids, .scratch_cap = p.s.cap, .scores = p.seq.scores, .output = hooks.logits_out, .streams = hooks.streams, .plan = dp.state.ptr };
        key.seq[0] = Key.seqOf(seq);
        if (seq.paged) |pg| {
            key.paging[0] = pg.graphKey();
            key.paging_logical = @splat(@splat(0));
            key.paging_logical.?[0] = pg.logicalGraphKey();
        }
        Key.seams(seq.fed + n, &key.chunks[0], &key.score_tiles[0]);
        if (hooks.keep) |kp| {
            key.keep = std.hash.Wyhash.hash(1, std.mem.asBytes(kp));
            if (grebind.stateRebindEnabled()) key.keep_nonstate = state_binding.keepHash(kp.*);
        }
        const tp0 = std.Io.Clock.awake.now(p.io).toNanoseconds();
        try prepareWindow(p, ids, seq.fed);
        hub.stats.prepare_ns += @intCast(std.Io.Clock.awake.now(p.io).toNanoseconds() - tp0);
        hub.stats.prepares += 1;
        grun.run(key, p, ids, seq.fed) catch |err| {
            if (p.m.collective) |c| {
                c.failGraph();
                return err;
            }
            gcache.broken = true;
            return err;
        };
    }
    if (hooks.output_tail) |tail| try tail.enqueue(tail.ptr, p.stream);
    // the pass's head(.all) copies the last row's scores to the host after the step (outside the graph)
    if (p.m.collective) |c| try c.finish(p.stream) else try p.stream.synchronize();
    const d = p.stream.d;
    try d.check(d.api.cuMemcpyDtoH_v2(seq.logits.ptr, hooks.logits_out + (n - 1) * vocab * 2, vocab * 2), "cuMemcpyDtoH");
    seq.fed += n;
    return true;
}
