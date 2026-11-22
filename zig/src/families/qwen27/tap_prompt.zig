//! Contiguous prompt taps enter the ring once; only proven unobservable prefixes may omit their plane stores.
const std = @import("std");
pub const Args = extern struct { width: u32, capacity: u32, planes: u32, window: u32, first: u32, rows: u32, start: u64 };
pub const Plan = struct { args: Args, end: u64 };
var once: std.atomic.Value(u8) = .init(2);
pub fn on() bool {
    const got = once.load(.monotonic);
    if (got != 2) return got == 1;
    const enabled = @import("sys").environment.read(bool, "TF_Q27_DFLASH_TAP_LEAN", false) catch false;
    once.store(@intFromBool(enabled), .monotonic);
    return enabled;
}

/// A direct serial prefill has no read before horizon; native yield/cancel boundaries use their current end instead.
pub fn dead(end: u64, horizon: u64, window: u32) bool {
    return window != 0 and end <= horizon and horizon - end >= window;
}
pub fn plan(capacity: u32, written: u32, width: u32, planes: usize, window: u32, start: u64, rows: u32, source_bytes: usize, ring_bytes: usize, stride: usize) !Plan {
    if (rows == 0 or rows > written or written > capacity or width == 0 or planes == 0 or planes > 8 or window == 0) return error.BadTapGather;
    if (width > std.math.maxInt(u32) / planes) return error.BadTapGather;
    const row_bytes = try std.math.mul(usize, try std.math.mul(usize, width, planes), 2);
    if (stride != row_bytes or source_bytes < try std.math.mul(usize, capacity, row_bytes) or ring_bytes < try std.math.mul(usize, window, row_bytes)) return error.BadTapGather;
    const end = try std.math.add(u64, start, rows);
    const count = @min(rows, window);
    return .{ .args = .{ .width = width, .capacity = capacity, .planes = @intCast(planes), .window = window, .first = rows - count, .rows = count, .start = start }, .end = end };
}
/// write() completes the GPU command or its fake before the committed end can move; each written slot is unique.
pub fn absorb(ctx: anytype, rows: u32) !void {
    try ctx.taps.complete();
    const ring = ctx.ring;
    const p = try plan(ctx.taps.capacity, ctx.taps.rows, ctx.taps.width, ctx.taps.count, ring.window, ring.end.*, rows, ctx.taps.buffer.length(), ring.buf.length(), ring.stride);
    try ctx.write(p.args);
    ring.end.* = p.end;
}

/// The existing direct prefill loop has no intermediate observer; an omitted prefix is invalid on any later error.
pub fn prefill(ctx: anytype, total: usize, requested: usize) !void {
    var skipped = false;
    errdefer if (skipped) ctx.invalidate();
    var first: usize = 0;
    while (first < total) {
        const count = @min(total - first, ctx.chunk(requested));
        if (count == 0) return error.BadDraftContext;
        const omit = ctx.enabled() and dead(first + count, total, ctx.window());
        skipped = skipped or omit;
        try ctx.feed(first, count, omit, first + count == total);
        first += count;
    }
}
