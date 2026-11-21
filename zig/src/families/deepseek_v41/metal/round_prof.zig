//! TF_DSV41_ROUND_PROF=1: decode rounds' host (encode to last wait) and GPU milliseconds by rows, logged now and then.
const std = @import("std");
const mtl = @import("metal");

const Sum = struct { n: u64 = 0, host: f64 = 0, gpu: f64 = 0, timed: u64 = 0 };
const every = 1024;

var state: enum { unread, off, on } = .unread;
var sums: [17]Sum = @splat(.{});
var rounds: u64 = 0;

/// The host clock at a round's start, or null when off.
pub fn begin() ?f64 {
    if (state == .unread) state = if (@import("sys").environment.read(bool, "TF_DSV41_ROUND_PROF", false) catch false) .on else .off;
    return if (state == .on) mtl.clock.seconds() else null;
}

/// Metal 4 reports a command buffer's GPU times after its wait returns: a round's are read at the next round's end.
var pending: ?struct { first: mtl.CommandBuffer, last: mtl.CommandBuffer, rows: u32 } = null;

/// A round of `rows` rows from `t0`, its command buffers' first and last.
pub fn end(t0: f64, rows: u32, first: mtl.CommandBuffer, last: mtl.CommandBuffer) void {
    const s = &sums[@min(rows, sums.len - 1)];
    s.n += 1;
    s.host += mtl.clock.seconds() - t0;
    if (pending) |p| gpu(p.rows, p.first, p.last);
    pending = null;
    if (first.m4 != null) pending = .{ .first = first, .last = last, .rows = rows } else gpu(rows, first, last);
    rounds += 1;
    if (rounds % every == 0) log();
}

/// Reads and drops the pending round while its queue lives: an engine calls this before closing the queue.
pub fn forget() void {
    if (pending) |p| gpu(p.rows, p.first, p.last);
    pending = null;
}

/// Whether a round's buffers wait to be read (tests).
pub fn holding() bool {
    return pending != null;
}

/// Turns the profiler on or off without the environment (tests).
pub fn force(on: bool) void {
    state = if (on) .on else .off;
}

fn gpu(rows: u32, first: mtl.CommandBuffer, last: mtl.CommandBuffer) void {
    const s = &sums[@min(rows, sums.len - 1)];
    const a = first.gpuTimes() orelse return;
    const b = last.gpuTimes() orelse return;
    s.gpu += b[1] - a[0];
    s.timed += 1;
}

fn log() void {
    var buf: [1024]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    for (sums, 0..) |s, rows| {
        if (s.n == 0) continue;
        const g = if (s.timed > 0) s.gpu / @as(f64, @floatFromInt(s.timed)) * 1e3 else 0;
        w.print(" {d} rows: {d:.2} host, {d:.2} gpu ({d});", .{ rows, s.host / @as(f64, @floatFromInt(s.n)) * 1e3, g, s.n }) catch break;
    }
    std.log.info("dsv41 rounds, ms a round over {d}:{s}", .{ rounds, w.buffered() });
}
