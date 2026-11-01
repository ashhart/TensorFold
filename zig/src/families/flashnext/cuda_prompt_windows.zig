//! Several streams' drafted windows in one batched pass (Engine.windows), each row's bits its lone pass's.
const std = @import("std");
const prompt = @import("cuda_prompt.zig");
const batch = prompt.batch;
const pocket = @import("cuda_pocket.zig");
const vocab = @import("cuda_prompt_wt.zig").vocab;
const window_graph = @import("cuda_window_graph.zig");
const Engine = prompt.Engine;
const Seq = prompt.Seq;

/// A drafted window on one GPU as a graph (one chunk, scores left on the device, no media or adds); true once run.
pub fn forwardGraph(e: *Engine, comptime Tri: type, p: *batch.Pass(Tri), seq: *Seq, ids: []const u32, hooks: batch.Hooks) !bool {
    if (!window_graph.enabled() or hooks.logits != .all or hooks.host_logits or hooks.moe_add != null) return false;
    if (ids.len > @import("cuda_mtp.zig").max_rows or ids.len > p.s.cap or e.collective != null or seq.media != null or p.m.pocket != null) return false;
    const key = window_graph.Key.init(seq, seq.fed, ids.len, p.s, p.seq.scores, @intFromPtr(p.m.layers.ptr), hooks);
    try window_graph.stageWindow(p, ids, seq.fed);
    try e.graphs.window.run(e.gpa, e.stream.*, key, window_graph.Chunk(batch.Pass(Tri)){ .p = p, .ids = ids, .pos0 = seq.fed });
    seq.fed += ids.len;
    return true;
}

/// A stream's window in a shared pass: its rows, and where its rollback rows, hidden rows and scores go (0: none).
pub const Window = struct { seq: *Seq, ids: []const u32, keep: ?*const batch.Keep = null, streams: u64 = 0, logits: u64 = 0 };

/// Several streams' windows in one batched pass, each row's bits its lone pass's; `out` holds every row's scores.
pub fn windows(e: *Engine, comptime Tri: type, tri: Tri, wins: []const Window, out: u64) !void {
    if (wins.len == 0) return error.UnexpectedTensor;
    var total: usize = 0;
    var cap: usize = 0;
    for (wins) |w| {
        if (w.ids.len == 0 or w.seq.fed + w.ids.len > w.seq.cap) return error.UnexpectedTensor;
        total += w.ids.len;
        cap = @max(cap, w.seq.cap);
    }
    if (total > e.chunk) return error.UnexpectedTensor;
    const lanes = try e.gpa.alloc(batch.Lane, wins.len);
    defer e.gpa.free(lanes);
    const ids = try e.gpa.alloc(u32, total);
    defer e.gpa.free(ids);
    // an untaught stream's rows read no pocket, as in a round
    const untaught = try e.gpa.alloc(bool, total);
    defer e.gpa.free(untaught);
    const scales = try e.gpa.alloc(f32, total);
    defer e.gpa.free(scales);
    const s = try e.reserve(total);
    const scores = try e.reserveScores(cap, s.cap);
    var at: usize = 0;
    for (wins, lanes) |w, *l| {
        l.* = .{ .seq = w.seq.view(scores), .pos = w.seq.fed, .rows = w.ids.len, .keep = w.keep, .streams = w.streams, .logits = w.logits };
        @memcpy(ids[at..][0..w.ids.len], w.ids);
        @memset(untaught[at..][0..w.ids.len], w.seq.untaught);
        at += w.ids.len;
    }
    const reads = pocket.reads(untaught, scales);
    defer e.taught_rows = 0;
    if (reads == .some and e.pockets.mask() != 0) e.taught_rows = try e.pockets.rowScales(e.driver, e.stream, scales);
    var p: batch.Pass(Tri) = .{
        .gpa = e.gpa,
        .io = e.io,
        .stream = e.stream.*,
        .tri = tri,
        .k = &e.batch_k,
        .s = s,
        .m = if (reads == .none) e.shipped() else e.model(),
        .seq = lanes[0].seq,
        .lanes = lanes,
        .hooks = .{ .logits = .all, .logits_out = out },
    };
    const media = for (wins) |w| {
        if (w.seq.media != null) break true;
    } else false;
    if (window_graph.enabled() and reads == .none and p.m.pocket == null and e.collective == null and !media) {
        // on one GPU: staged, then the pass as a graph keyed by every lane's buffers
        const key = window_graph.SharedKey.init(lanes, s, scores, @intFromPtr(p.m.layers.ptr), out);
        try window_graph.stageWindow(&p, ids, 0);
        try e.graphs.shared.run(e.gpa, e.stream.*, key, window_graph.Chunk(batch.Pass(Tri)){ .p = &p, .ids = ids, .pos0 = 0 });
    } else try p.chunk(ids, 0, null);
    try e.complete();
    // each stream's last row in its host scores only when asked: the drafted windows' draws read `out`
    const d = e.stream.d;
    const host = @import("cuda_batch_state.zig").verifyHostLogits();
    at = 0;
    for (wins) |w| {
        at += w.ids.len;
        if (host) try d.check(d.api.cuMemcpyDtoH_v2(w.seq.logits.ptr, out + (at - 1) * vocab * 2, vocab * 2), "cuMemcpyDtoH");
        w.seq.fed += w.ids.len;
    }
}
