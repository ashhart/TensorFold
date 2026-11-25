//! Main's DSpark lane snapshot helper: the unabsorbed prompt-end tap row a snapshot carries, checked.
const std = @import("std");
const mtl = @import("metal");
const state = @import("dspark_state.zig");
const st = @import("state.zig");

comptime {
    if (st.feature_row_bytes != state.tap_row_bytes) @compileError("DSpark tail tap row bytes mismatch");
}

/// A lane back at position zero with no held drafts: a failed restore or a reset.
pub fn clearLane(lane: *state.Lane) void {
    lane.from = 0;
    lane.end = 0;
    lane.held_len = 0;
    lane.sampled = false;
}

pub fn resolveTailTap(tap: st.Tap, at_pos: u32, lane_end: u64, tap_gen: u64) !?state.Tail {
    if (lane_end == at_pos) return null;
    if (lane_end + 1 != at_pos) return error.SnapshotOutOfStep;
    const buf = tap.buf orelse return error.SnapshotOutOfStep;
    if (tap.gen) |g| if (g != tap_gen) return error.StaleInkling2Taps;
    if (lane_end < tap.start) return error.SnapshotOutOfStep;
    const rows_before = std.math.cast(usize, lane_end - tap.start) orelse return error.SnapshotOutOfStep;
    if (rows_before >= tap.rows) return error.SnapshotOutOfStep;
    const delta = std.math.mul(usize, rows_before, state.tap_row_bytes) catch return error.SnapshotOutOfStep;
    const src_off = std.math.add(usize, tap.off, delta) catch return error.SnapshotOutOfStep;
    const src_end = std.math.add(usize, src_off, state.tap_row_bytes) catch return error.SnapshotOutOfStep;
    if (src_end > buf.length()) return error.SnapshotOutOfStep;
    return .{ .buf = buf, .off = src_off };
}

test "resolveTailTap preserves non-first-stream tap offset and validates generation and bounds" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    const buf = try device.buffer(6 * state.tap_row_bytes, mtl.ResourceOptions.shared);
    defer buf.deinit();
    const shared_tap: st.Tap = .{ .buf = buf, .off = 2 * state.tap_row_bytes, .start = 10, .rows = 3, .gen = 7 };
    try std.testing.expectEqual(@as(?state.Tail, null), try resolveTailTap(shared_tap, 13, 13, 7));
    const tail = (try resolveTailTap(shared_tap, 13, 12, 7)).?;
    try std.testing.expectEqual(buf.id, tail.buf.id);
    try std.testing.expectEqual(4 * state.tap_row_bytes, tail.off);
    const stable_tap: st.Tap = .{ .buf = buf, .off = 2 * state.tap_row_bytes, .start = 10, .rows = 3, .gen = null };
    try std.testing.expectEqual(4 * state.tap_row_bytes, (try resolveTailTap(stable_tap, 13, 12, 99)).?.off);
    try std.testing.expectError(error.StaleInkling2Taps, resolveTailTap(shared_tap, 13, 12, 8));
    try std.testing.expectError(error.SnapshotOutOfStep, resolveTailTap(shared_tap, 14, 13, 7));
    try std.testing.expectError(error.SnapshotOutOfStep, resolveTailTap(shared_tap, 13, 11, 7));
    try std.testing.expectError(error.SnapshotOutOfStep, resolveTailTap(shared_tap, 10, 9, 7));
    const truncated: st.Tap = .{ .buf = buf, .off = 5 * state.tap_row_bytes + 1, .start = 10, .rows = 1, .gen = 7 };
    try std.testing.expectError(error.SnapshotOutOfStep, resolveTailTap(truncated, 11, 10, 7));
}
