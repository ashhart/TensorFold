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
        @memset(r.tokens, 1);
        r.reqs = try gpa.alloc(draw.Request, most);
        r.out = try gpa.alloc(u32, most);
        // the rig runs before any request, on the caches of streams the memory plan holds and none uses yet
        while (r.made < streams) : (r.made += 1) r.caches[r.made] = try e.newCaches(at + 2 * shared.window * 4);
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
        var total: usize = 0;
        for (widths, 0..) |w, i| {
            r.rows[i] = .{ .caches = &r.caches[i], .pos = at, .tokens = r.tokens[total..][0..w] };
            for (0..w) |j| r.reqs[total + j] = .{ .sampling = null, .position = at + j + 1 };
            total += w;
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
pub fn measure(gpa: std.mem.Allocator, e: *Engine, head: ?*mtp.Head, width: usize, out: *Costs) !void {
    var r: Rig = try .init(gpa, e, @max(1, e.o.streams), e.o.batch_rows);
    defer r.deinit();
    try forwards(&r, e.o.streams, width, e.o.batch_rows, out);
    if (head) |hd| out.step_ms = try step(e, hd);
}

/// The forwards the depth rule weighs: lone windows, then the row buckets shared rounds of `streams` windows pad to.
pub fn forwards(r: anytype, streams: usize, width: usize, batch_rows: usize, out: *Costs) !void {
    try shared.windows(r, batch_rows, out);
    try shared.buckets(r, streams, width, batch_rows, plan.bucketOf, out);
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
        var jobs = [_]mtp.Job{.{ .hidden = row.base(), .token = 1, .position = at, .depth = mtp.max_depth, .sampling = null, .stop_under = 0.0, .out = &held }};
        const t0 = nowNs();
        try hd.chains(&e.lib, e.stream, &e.drawer, m, &jobs, e.headGraphs());
        if (i >= shared.warm) spent += nowNs() - t0;
    }
    return @as(f64, @floatFromInt(spent)) / shared.reps / 1e6 / mtp.max_depth;
}
