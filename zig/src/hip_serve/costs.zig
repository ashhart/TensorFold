//! A lane backend's costs, timed at load on scratch streams: what the core's depth rule weighs drafts against.

const std = @import("std");
const lanes = @import("lanes");
const nowNs = @import("backend.zig").nowNs;

pub const Cost = lanes.config.Cost;

/// Forwards run before the timed ones (the first captures the shape's graph) and the timed reps.
pub const warm = 3;
pub const reps = 5;
/// Rows a stream's window holds in the shared timings.
pub const window = 4;
/// The total rows timed in shared forwards, and the streams they spread over at most.
pub const shared_rows = [_]u32{ 2, 4, 8, 12, 16 };
pub const most_streams = 8;

pub const Costs = struct {
    window: [16]Cost = undefined,
    windows: usize = 0,
    shared: [shared_rows.len]Cost = undefined,
    shareds: usize = 0,
    step_ms: f64 = 0.0,
};

/// Mean ms of `run(ctx)` over the timed reps, after the warm ones.
pub fn mean(ctx: anytype, comptime run: fn (@TypeOf(ctx)) anyerror!void) !f64 {
    var spent: u64 = 0;
    for (0..warm + reps) |i| {
        const t0 = nowNs();
        try run(ctx);
        if (i >= warm) spent += nowNs() - t0;
    }
    return @as(f64, @floatFromInt(spent)) / reps / 1e6;
}

/// `r.time(widths)` (one forward's mean ms) of a lone window to `widest` rows, then of shared rows on `streams` lanes.
pub fn forwards(r: anytype, widest: usize, streams: usize, batch_rows: usize, out: *Costs) !void {
    out.windows = 0;
    for (1..out.window.len + 1) |w| {
        if (w > widest) break;
        out.window[out.windows] = .{ .width = @intCast(w), .ms = try r.time(&.{w}) };
        out.windows += 1;
    }
    out.shareds = 0;
    for (shared_rows) |total| {
        if (total > batch_rows) break;
        var widths: [most_streams]usize = undefined;
        var n: usize = 0;
        var left: usize = total;
        while (left > 0 and n < @min(streams, most_streams)) : (n += 1) {
            widths[n] = @min(window, left);
            left -= widths[n];
        }
        if (left > 0) continue;
        out.shared[out.shareds] = .{ .width = total, .ms = try r.time(widths[0..n]) };
        out.shareds += 1;
    }
}

const Fake = struct {
    calls: std.ArrayList(usize) = .empty,

    fn time(f: *Fake, widths: []const usize) !f64 {
        var rows: usize = 0;
        for (widths) |w| rows += w;
        try f.calls.append(std.testing.allocator, widths.len);
        return @floatFromInt(rows);
    }
};

test "lone widths to the widest, shared totals the streams can hold" {
    var f: Fake = .{};
    defer f.calls.deinit(std.testing.allocator);
    var c: Costs = .{};
    try forwards(&f, 4, 3, 16, &c);
    try std.testing.expectEqual(@as(usize, 4), c.windows);
    try std.testing.expectEqual(@as(f64, 3), c.window[2].ms);
    // 16 rows need four streams of four: three streams time 2, 4, 8 and 12 rows
    try std.testing.expectEqual(@as(usize, 4), c.shareds);
    try std.testing.expectEqual(@as(u32, 12), c.shared[3].width);
    try std.testing.expectEqual(@as(usize, 3), f.calls.items[f.calls.items.len - 1]);
}
