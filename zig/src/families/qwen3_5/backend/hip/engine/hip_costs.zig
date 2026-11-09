//! The HIP lanes backend's own costs, timed at load on scratch streams: what the depth rule weighs drafts against.

const std = @import("std");
const hip = @import("hip");
const serve = @import("hip_serve");
const Engine = @import("engine.zig").Engine;
const state = @import("../forward/state.zig");
const draw = serve.draw;
const mtp = @import("mtp.zig");
const plan = @import("../forward/plan.zig");

const shared = serve.costs;
pub const Costs = shared.Costs;
pub const nowNs = serve.backend.nowNs;

/// A stream's window position in the timings (the caches are scratch).
const at = 16;

/// A timed forward's tokens: one repeated for a lone window, one a row for shared rows, which an MoE routes apart.
pub fn fillTokens(tokens: []u32, windows: usize, vocab: usize) void {
    const span = @min(vocab, 65536) - 1;
    // two primes whose product passes every span: at most one divides it, so the stride is coprime to the span
    const stride: usize = if (span % 7919 == 0) 7927 else 7919;
    for (tokens, 0..) |*t, k| t.* = if (windows > 1) @intCast(1 + (k * stride) % span) else 1;
}

/// A timed round over windows of `widths`: each row's greedy draw and its token, by the window count; returns its rows.
pub fn fillRound(tokens: []u32, reqs: []draw.Request, widths: []const usize, vocab: usize) usize {
    var total: usize = 0;
    for (widths) |w| {
        for (0..w) |j| reqs[total + j] = .{ .sampling = null, .position = at + j + 1 };
        total += w;
    }
    fillTokens(tokens[0..total], widths.len, vocab);
    return total;
}

const Rig = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    /// A stream's caches, rows and the tokens, requests and draws of a round's rows.
    caches: []state.Caches = &.{},
    made: usize = 0,
    rows: []Engine.Rows = &.{},
    tokens: []u32 = &.{},
    reqs: []draw.Request = &.{},
    out: []u32 = &.{},

    fn init(gpa: std.mem.Allocator, e: *Engine, streams: usize, most: usize) !Rig {
        var r: Rig = .{ .gpa = gpa, .e = e };
        errdefer r.deinit();
        r.caches = try gpa.alloc(state.Caches, streams);
        r.rows = try gpa.alloc(Engine.Rows, streams);
        r.tokens = try gpa.alloc(u32, most);
        r.reqs = try gpa.alloc(draw.Request, most);
        r.out = try gpa.alloc(u32, most);
        // a bucket's rows over the streams can outgrow a lone window when `most` passes 16 rows a stream
        const share = most / streams + @intFromBool(most % streams != 0);
        // the rig runs before any request, on the caches of streams the memory plan holds and none uses yet
        while (r.made < streams) : (r.made += 1) r.caches[r.made] = try e.newCaches(at + @max(2 * shared.window * 4, share));
        return r;
    }

    fn deinit(r: *Rig) void {
        r.e.drain();
        for (r.caches[0..r.made]) |*c| c.deinit(r.gpa);
        r.gpa.free(r.out);
        r.gpa.free(r.reqs);
        r.gpa.free(r.tokens);
        r.gpa.free(r.rows);
        r.gpa.free(r.caches);
    }

    /// Mean ms of a verify of `streams` windows of the given widths.
    pub fn time(r: *Rig, widths: []const usize) !f64 {
        const total = fillRound(r.tokens, r.reqs, widths, r.e.model().spec.vocab);
        var from: usize = 0;
        for (widths, 0..) |w, i| {
            r.rows[i] = .{ .caches = &r.caches[i], .pos = at, .tokens = r.tokens[from..][0..w] };
            from += w;
        }
        const n = shared.runs(total);
        var spent: u64 = 0;
        for (0..n.warm + n.reps) |i| {
            const t0 = nowNs();
            _ = try r.e.verify(r.rows[0..widths.len], r.reqs[0..total], r.out);
            if (i >= n.warm) spent += nowNs() - t0;
        }
        return @as(f64, @floatFromInt(spent)) / @as(f64, @floatFromInt(n.reps)) / 1e6;
    }
};

/// Time `h`'s forwards and head on scratch caches of its engine (nothing of a stream is touched).
pub fn measure(gpa: std.mem.Allocator, e: *Engine, head: ?*mtp.Head, out: *Costs) !void {
    var r: Rig = try .init(gpa, e, @max(1, e.o.streams), e.o.batch_rows);
    defer r.deinit();
    try forwards(&r, e.o.streams, e.o.batch_rows, out);
    if (head) |hd| out.step_ms = try step(e, hd);
}

/// The forwards the depth rule weighs: lone windows, then the row buckets shared rounds of `streams` windows pad to.
pub fn forwards(r: anytype, streams: usize, batch_rows: usize, out: *Costs) !void {
    try shared.windows(r, batch_rows, out);
    try shared.buckets(r, streams, batch_rows, plan.bucketOf, out);
}

test {
    _ = @import("hip_costs_test.zig");
}

/// One chained head step: a lone chain of the deepest depth, over a scratch row.
fn step(e: *Engine, hd: *mtp.Head) !f64 {
    const m = e.model();
    var row = try hip.DeviceBuffer.alloc(&e.driver, m.spec.hidden * m.act.size());
    defer row.free();
    try row.fill8(0);
    var held: [mtp.max_depth]u32 = undefined;
    var spent: u64 = 0;
    for (0..shared.warm + shared.reps) |i| {
        var jobs = [_]mtp.Job{.{ .hidden = row.base(), .token = 1, .position = at, .depth = mtp.max_depth, .stop_under = 0.0, .out = &held }};
        const t0 = nowNs();
        try hd.chains(&e.lib, e.stream, m, &jobs, e.headGraphs());
        if (i >= shared.warm) spent += nowNs() - t0;
    }
    return @as(f64, @floatFromInt(spent)) / shared.reps / 1e6 / mtp.max_depth;
}
