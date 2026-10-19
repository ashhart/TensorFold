//! Latency and bandwidth of each backend's collectives, eager and graphed; RESULT lines are JSON, busbw as nccl-tests.
const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const Ctx = @import("gpu_checks.zig").Ctx;
const Collective = tp.Collective;

pub const Kind = enum { all_reduce, all_gather };

fn issue(c: *Ctx, comm: Collective, kind: Kind, bytes: usize) !void {
    const count = bytes / 2;
    switch (kind) {
        .all_reduce => try comm.allReduce(c.in.ptr, c.out.ptr, count, .bf16, .sum, c.stream.handle),
        .all_gather => try comm.allGather(c.in.ptr, c.out.ptr, count, .bf16, c.stream.handle),
    }
}

/// GPU microseconds a call, `iters` calls issued back to back on the stream.
fn eager(c: *Ctx, comm: Collective, kind: Kind, bytes: usize, iters: u32) !f64 {
    var e0 = try cuda.Event.init(c.d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(c.d, true);
    defer e1.deinit();
    for (0..10) |_| try issue(c, comm, kind, bytes);
    try c.stream.synchronize();
    try comm.barrier();
    try e0.record(c.stream);
    for (0..iters) |_| try issue(c, comm, kind, bytes);
    try e1.record(c.stream);
    try e1.synchronize();
    return @as(f64, try cuda.Event.elapsedMs(e0, e1)) * 1000.0 / @as(f64, @floatFromInt(iters));
}

/// GPU microseconds a call inside one graph of `per_graph` calls, replayed `replays` times.
fn graphed(c: *Ctx, comm: Collective, kind: Kind, bytes: usize, per_graph: u32, replays: u32) !f64 {
    try cuda.graph.beginCapture(c.stream, .thread_local);
    const err: ?tp.collective.Error = blk: {
        for (0..per_graph) |_| issue(c, comm, kind, bytes) catch |e| break :blk e;
        break :blk null;
    };
    var g = try cuda.graph.endCapture(c.stream);
    defer g.deinit();
    if (err) |e| return e;
    var exec = try g.instantiate();
    defer exec.deinit();
    try exec.upload(c.stream);
    try exec.launchOn(c.stream);
    try c.stream.synchronize();
    try comm.barrier();
    var e0 = try cuda.Event.init(c.d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(c.d, true);
    defer e1.deinit();
    try e0.record(c.stream);
    for (0..replays) |_| try exec.launchOn(c.stream);
    try e1.record(c.stream);
    try e1.synchronize();
    try comm.check();
    return @as(f64, try cuda.Event.elapsedMs(e0, e1)) * 1000.0 / @as(f64, @floatFromInt(per_graph * replays));
}

/// Every power of two from `lo` to `hi` bytes a rank, both kinds, eager and graphed.
pub fn sweep(c: *Ctx, comm: Collective, label: []const u8, lo: usize, hi: usize) !void {
    for ([_]Kind{ .all_reduce, .all_gather }) |kind| {
        var bytes = lo;
        while (bytes <= hi) : (bytes *= 2) {
            const per_graph: u32 = if (bytes >= 1 << 20) 20 else 100;
            const e = try eager(c, comm, kind, bytes, if (bytes >= 1 << 20) 50 else 200);
            const g = try graphed(c, comm, kind, bytes, per_graph, 5);
            const moved: f64 = @floatFromInt(if (kind == .all_gather) 2 * bytes else bytes);
            const factor: f64 = if (kind == .all_gather) 0.5 else 1.0;
            const busbw = moved / (g * 1e-6) * factor / 1e9;
            if (c.rank == 0) std.debug.print("RESULT {{\"backend\":\"{s}\",\"op\":\"{t}\",\"bytes\":{d},\"eager_us\":{d:.2},\"graph_us\":{d:.2},\"busbw_GBps\":{d:.2}}}\n", .{ label, kind, bytes, e, g, busbw });
        }
    }
}

/// One decode window's exchanges: `n` all-gathers of one row's bf16 partials (`row_bytes` a rank) in one graph.
pub fn decodeWindow(c: *Ctx, comm: Collective, label: []const u8, n: u32, row_bytes: usize) !void {
    const per = try graphed(c, comm, .all_gather, row_bytes, n, 20);
    if (c.rank == 0) std.debug.print("RESULT {{\"backend\":\"{s}\",\"op\":\"decode_window\",\"exchanges\":{d},\"bytes\":{d},\"per_exchange_us\":{d:.2},\"window_us\":{d:.1}}}\n", .{ label, n, row_bytes, per, per * @as(f64, @floatFromInt(n)) });
}
