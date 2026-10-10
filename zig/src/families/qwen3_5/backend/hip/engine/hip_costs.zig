//! The HIP lanes backend's own costs, timed at load on scratch streams: what the depth rule weighs drafts against.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const Engine = @import("engine.zig").Engine;
const state = @import("../forward/state.zig");
const draw = @import("draw.zig");
const mtp = @import("mtp.zig");

const Cost = lanes.config.Cost;

/// Forwards run before the timed ones (the first captures the shape's graph) and the timed reps.
const warm = 3;
const reps = 5;
/// Rows a stream's window holds in the shared timings, and its position (the caches are scratch).
const window = 4;
const at = 16;
/// The total rows timed in shared forwards.
pub const shared_rows = [_]u32{ 2, 4, 8, 12, 16 };
pub const most_streams = 8;

pub const Costs = struct {
    window: [16]Cost = undefined,
    windows: usize = 0,
    shared: [shared_rows.len]Cost = undefined,
    shareds: usize = 0,
    step_ms: f64 = 0.0,
};

pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

const Rig = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    caches: [most_streams]state.Caches = undefined,
    made: usize = 0,
    tokens: [64]u32 = @splat(1),
    reqs: [64]draw.Request = undefined,
    out: [64]u32 = undefined,

    fn deinit(r: *Rig) void {
        r.e.drain();
        for (r.caches[0..r.made]) |*c| c.deinit(r.gpa);
    }

    /// Mean ms of a verify of `streams` windows of the given widths.
    fn time(r: *Rig, widths: []const usize) !f64 {
        var rows: [most_streams]Engine.Rows = undefined;
        var total: usize = 0;
        for (widths, 0..) |w, i| {
            rows[i] = .{ .caches = &r.caches[i], .pos = at, .tokens = r.tokens[total..][0..w] };
            for (0..w) |j| r.reqs[total + j] = .{ .sampling = null, .position = at + j + 1 };
            total += w;
        }
        var spent: u64 = 0;
        for (0..warm + reps) |i| {
            const t0 = nowNs();
            _ = try r.e.verify(rows[0..widths.len], r.reqs[0..total], &r.out);
            if (i >= warm) spent += nowNs() - t0;
        }
        return @as(f64, @floatFromInt(spent)) / reps / 1e6;
    }
};

/// Time `h`'s forwards and head on scratch caches of its engine (nothing of a stream is touched).
pub fn measure(gpa: std.mem.Allocator, e: *Engine, head: ?*mtp.Head, out: *Costs) !void {
    var r: Rig = .{ .gpa = gpa, .e = e };
    defer r.deinit();
    // the rig runs before any request, on the caches of streams the memory plan holds and none uses yet
    const streams = @min(most_streams, e.o.streams);
    while (r.made < streams) : (r.made += 1) r.caches[r.made] = try e.newCaches(at + 2 * window * 4);
    out.windows = 0;
    for (1..out.window.len + 1) |w| {
        if (w > e.o.batch_rows) break;
        out.window[out.windows] = .{ .width = @intCast(w), .ms = try r.time(&.{w}) };
        out.windows += 1;
    }
    out.shareds = 0;
    for (shared_rows) |total| {
        if (total > e.o.batch_rows) break;
        var widths: [most_streams]usize = undefined;
        var n: usize = 0;
        var left: usize = total;
        while (left > 0 and n < streams) : (n += 1) {
            widths[n] = @min(window, left);
            left -= widths[n];
        }
        if (left > 0) continue;
        out.shared[out.shareds] = .{ .width = total, .ms = try r.time(widths[0..n]) };
        out.shareds += 1;
    }
    if (head) |hd| out.step_ms = try step(e, hd);
}

/// One chained head step: a lone chain of the deepest depth, over a scratch row.
fn step(e: *Engine, hd: *mtp.Head) !f64 {
    const m = e.model();
    var row = try hip.DeviceBuffer.alloc(&e.driver, m.spec.hidden * m.act.size());
    defer row.free();
    try row.fill8(0);
    var held: [mtp.max_depth]u32 = undefined;
    var spent: u64 = 0;
    for (0..warm + reps) |i| {
        var jobs = [_]mtp.Job{.{ .hidden = row.base(), .token = 1, .position = at, .depth = mtp.max_depth, .sampling = null, .stop_under = 0.0, .out = &held }};
        const t0 = nowNs();
        try hd.chains(&e.lib, e.stream, &e.drawer, m, &jobs, e.headGraphs());
        if (i >= warm) spent += nowNs() - t0;
    }
    return @as(f64, @floatFromInt(spent)) / reps / 1e6 / mtp.max_depth;
}
