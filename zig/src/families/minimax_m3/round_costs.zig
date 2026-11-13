//! MiniMax-M3 Metal's timed rounds: a window of 1-8 rows, and shared rounds of 12-32 rows as several streams' windows.
const std = @import("std");
const lanes = @import("lanes");
const Cost = lanes.config.Cost;

/// Shared rounds' total rows that measure() times beside the 1-8 row windows (32 is a round's most rows).
pub const shared_widths = [_]u32{ 12, 16, 24, 32 };

/// A shared round of `total` rows as streams' windows of `window` rows, the last the remainder; `out` holds them.
pub fn windows(total: usize, window: usize, out: []usize) []usize {
    var n: usize = 0;
    var left = total;
    while (left > 0) : (n += 1) {
        out[n] = @min(left, window);
        left -= out[n];
    }
    return out[0..n];
}

/// The lane facts: `window` prices one stream's window by rows, `shared` a shared round by its total rows.
pub fn facts(window_rows: u32, batch_rows: u32, streams: u32, drafts: bool, window: []const Cost, shared: []const Cost) lanes.Model {
    // rows are independent in every decode kernel, so any width, alone or beside other streams, gives one-row bits
    return .{ .exact_width = window_rows, .gpu_tokens = !drafts, .hidden_rows = true, .batch_rows = batch_rows, .mtp = false, .speculate = false, .speculate_early = false, .drafts = 1, .max_streams = streams, .streams_exact = true, .window_costs = window, .shared_costs = shared };
}

test "MiniMax-M3 round costs: a shared round's rows split into 8-row windows, the remainder last" {
    var buf: [4]usize = undefined;
    try std.testing.expectEqualSlices(usize, &.{ 8, 4 }, windows(12, 8, &buf));
    try std.testing.expectEqualSlices(usize, &.{ 8, 8 }, windows(16, 8, &buf));
    try std.testing.expectEqualSlices(usize, &.{ 8, 8, 8 }, windows(24, 8, &buf));
    try std.testing.expectEqualSlices(usize, &.{ 8, 8, 8, 8 }, windows(32, 8, &buf));
    for (shared_widths) |w| try std.testing.expect(windows(w, 8, &buf).len <= buf.len);
}

test "MiniMax-M3 round costs: 16 and 32 rows are priced from the timed rounds, not at 2x and 4x the 8-row window" {
    const a = std.testing.allocator;
    var window: [8]Cost = undefined;
    for (&window, 1..) |*c, r| c.* = .{ .width = @intCast(r), .ms = 10.0 + 0.5 * @as(f64, @floatFromInt(r)) }; // 8 rows: 14 ms
    // what wide rounds cost on the device, about 1.2-1.4x and 1.5-2x the 8-row window
    const shared = [_]Cost{ .{ .width = 12, .ms = 15.5 }, .{ .width = 16, .ms = 17.5 }, .{ .width = 24, .ms = 21.0 }, .{ .width = 32, .ms = 24.5 } };
    var timed = try lanes.Config.init(a, facts(16, 32, 4, false, &window, &shared), 16, 15);
    defer timed.deinit(a);
    try std.testing.expectEqual(@as(?f64, 14.0), timed.shared_costs.get(8));
    try std.testing.expectEqual(@as(?f64, 17.5), timed.shared_costs.get(16));
    try std.testing.expectEqual(@as(?f64, 24.5), timed.shared_costs.get(32));
    try std.testing.expectEqual(@as(?f64, 19.25), timed.shared_costs.get(20)); // between 16 and 24
    try std.testing.expectEqual(@as(?f64, 14.75), timed.shared_costs.get(10)); // between 8 and 12
    // main's facts timed windows alone: past 8 rows, the 8-row window's ms a row, so 16 rows at 2x and 32 at 4x
    var guessed = try lanes.Config.init(a, facts(16, 32, 4, false, &window, &.{}), 16, 15);
    defer guessed.deinit(a);
    try std.testing.expectEqual(@as(?f64, 28.0), guessed.shared_costs.get(16));
    try std.testing.expectEqual(@as(?f64, 56.0), guessed.shared_costs.get(32));
}
