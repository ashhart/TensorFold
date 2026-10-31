//! A partial keep: the kept rows replayed over the DeltaNet layers (a graph on one GPU), or moved on with pending rows.
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const parts = @import("cuda_batch_state.zig");
const prompt = @import("cuda_prompt.zig");
const gdn = @import("cuda_gdn.zig");
const Draft = @import("cuda_mtp_draft.zig").Draft;
const window_graph = @import("cuda_window_graph.zig");
const std = @import("std");
const conv_bytes = (gdn.taps - 1) * gdn.conv_dim * 2;

fn cint(v: usize) c_int {
    return @intCast(v);
}

/// fn_late_roll: each DeltaNet layer's conv window and the n-gram tail from their saved copies, `keep` rows on.
pub fn lateRoll(o: anytype, seq: *prompt.Seq, d: *Draft, keep: usize) !void {
    var t: parts.RollJobs = .{};
    var n: usize = 0;
    for (o.e.layers[0..o.e.n_layers], 0..) |*lw, layer| {
        if (!lw.linear) continue;
        const lk = d.keep.layers[layer];
        t.j[n] = .{ .win = seq.conv[n].ptr, .hist = lk.hist, .rows = lk.proj, .stride = @intCast(gdn.proj_width), .taps = @intCast(gdn.taps - 1), .c = @intCast(gdn.conv_dim) };
        n += 1;
    }
    t.j[n] = .{ .win = seq.tail.ptr, .hist = d.saved.ptr + Draft.tail_at, .rows = d.saved.ptr + Draft.nrow_at, .stride = @intCast(wt.wide), .taps = @intCast(wt.ple_tail_n), .c = @intCast(wt.wide) };
    n += 1;
    t.n = @intCast(n);
    var a: cuda.Args = .{};
    a.add(t);
    a.add(cint(keep));
    const widest = @max(gdn.conv_dim, wt.wide);
    try cuda.launch.launch(o.e.batch_k.late_roll, .{ .grid = .{ .x = @intCast((widest + 255) / 256), .y = @intCast(n) }, .block = .{ .x = 256 } }, o.e.stream.*, &a);
}

/// The kept rows replayed on the window's start state (device work alone: a graph holds it).
const Replay = struct {
    e: *prompt.Engine,
    st: cuda.Stream,
    s: *parts.Scratch,
    seq: *prompt.Seq,
    d: *Draft,
    keep: usize,

    pub fn run(r: @This()) !void {
        const k = &r.e.batch_k;
        const rs = r.s;
        try copyAsync(r.st, r.seq.tail.ptr, r.d.saved.ptr + Draft.tail_at, wt.ple_tail_n * wt.wide * 2);
        try shift(k, r.st, r.seq.tail.ptr, r.d.saved.ptr + Draft.nrow_at, wt.wide, r.keep, wt.ple_tail_n, wt.wide);
        var li: usize = 0;
        for (r.e.layers[0..r.e.n_layers], 0..) |*lw, layer| {
            if (!lw.linear) continue;
            defer li += 1;
            // the sequence's state is still the window's start (the pass wrote its end state to the spare)
            const lk = r.d.keep.layers[layer];
            try copyAsync(r.st, r.seq.conv[li].ptr, lk.hist, conv_bytes);
            var f: cuda.Args = .{};
            for ([_]u64{ lk.proj, r.seq.conv[li].ptr, lw.cw.ptr, lw.a_log.ptr, lw.dt.ptr, rs.gq, rs.gk, rs.gv, rs.ggb }) |v| f.add(v);
            try cuda.launch.launch(k.front, .{ .grid = .{ .x = @intCast(r.keep), .y = gdn.nk + gdn.nv }, .block = .{ .x = 256 } }, r.st, &f);
            var a: cuda.Args = .{};
            for ([_]u64{ rs.gq, rs.gk, rs.gv, rs.ggb, r.seq.state[li].ptr }) |v| a.add(v);
            a.add(cint(r.keep));
            a.add(rs.ys);
            try cuda.launch.launch(k.chain, .{ .grid = .{ .x = gdn.nv, .y = gdn.dv / 16 }, .block = .{ .x = 128 } }, r.st, &a);
            try shift(k, r.st, r.seq.conv[li].ptr, lk.proj, gdn.proj_width, r.keep, gdn.taps - 1, gdn.conv_dim);
        }
    }
};

/// The replay of `keep` kept rows; on one GPU its ~146 launches as one graph, keyed by every buffer they touch.
pub fn replay(e: *prompt.Engine, s: *parts.Scratch, seq: *prompt.Seq, d: *Draft, keep: usize) !void {
    const r: Replay = .{ .e = e, .st = e.stream.*, .s = s, .seq = seq, .d = d, .keep = keep };
    if (!window_graph.enabled() or e.collective != null) return r.run();
    var h = std.hash.Wyhash.init(0);
    for ([_]u64{ seq.gen, d.gen, seq.tail.ptr, d.saved.ptr }) |v| h.update(std.mem.asBytes(&v));
    for (seq.conv) |b| h.update(std.mem.asBytes(&b.ptr));
    for (seq.state) |b| h.update(std.mem.asBytes(&b.ptr));
    h.update(std.mem.asBytes(&d.keep));
    const key: window_graph.RollbackKey = .{ .keep = @intCast(keep), .scratch = s.h, .scratch_cap = s.cap, .bufs = h.final() };
    try e.graphs.rollback.run(e.gpa, e.stream.*, key, r);
}

fn copyAsync(stream: cuda.Stream, dst: u64, src: u64, n: usize) !void {
    if (n == 0) return;
    try stream.d.check(stream.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, stream.handle), "cuMemcpyDtoDAsync");
}

fn shift(k: *const parts.Kernels, st: cuda.Stream, win: u64, rows_ptr: u64, stride: usize, n: usize, taps: usize, c: usize) !void {
    var a: cuda.Args = .{};
    a.add(win);
    a.add(rows_ptr);
    a.add(@as(c_longlong, @intCast(stride)));
    a.add(cint(n));
    a.add(cint(taps));
    a.add(cint(c));
    try cuda.launch.launch(k.shift, .{ .grid = .{ .x = @intCast((c + 255) / 256) }, .block = .{ .x = 256 } }, st, &a);
}
