//! Several streams' drafted windows in one batched pass (Engine.windows), each row's bits its lone pass's.
const std = @import("std");
const prompt = @import("cuda_prompt.zig");
const batch = prompt.batch;
const pocket = @import("cuda_pocket.zig");
const vocab = @import("cuda_prompt_wt.zig").vocab;
const Engine = prompt.Engine;
const Seq = prompt.Seq;

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
    try p.chunk(ids, 0, null);
    try e.complete();
    // each stream's last row in its host scores, as its lone pass leaves them
    const d = e.stream.d;
    at = 0;
    for (wins) |w| {
        at += w.ids.len;
        try d.check(d.api.cuMemcpyDtoH_v2(w.seq.logits.ptr, out + (at - 1) * vocab * 2, vocab * 2), "cuMemcpyDtoH");
        w.seq.fed += w.ids.len;
    }
}
