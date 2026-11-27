const std = @import("std");

pub const Frame = struct {
    source_row: usize,
    anchor: u32,
    query_rows: u32,
};

/// The target row survives rejection; the first draft query uses the target's bonus token.
pub fn prepare(positions: []const u32, rejected: usize, bonus: ?u32, context: u32, mask_token: u32, block_tokens: []u32, block_positions: []u32) !?Frame {
    if (block_tokens.len != block_positions.len or block_tokens.len < 2) return error.BadDraftBlock;
    const token = bonus orelse return null;
    if (rejected >= positions.len) return error.BadRejectionCount;
    const source_row = positions.len - rejected - 1;
    const anchor = std.math.add(u32, positions[source_row], 1) catch return error.BadDraftBlock;
    const rows = std.math.cast(u32, block_tokens.len) orelse return error.BadDraftBlock;
    if (anchor > context or rows > context - anchor) return null;
    @memset(block_tokens, mask_token);
    block_tokens[0] = token;
    for (block_positions, 0..) |*position, i| position.* = anchor + @as(u32, @intCast(i));
    return .{ .source_row = source_row, .anchor = anchor, .query_rows = rows };
}

pub const default_local_window: usize = 128;

/// Use target sparse indices, then an optional local prefix and each query's causal draft prefix.
pub fn reuseIndices(frame: Frame, target: []const i32, local_window: u32, query: u32, out: []i32) !void {
    const total = std.math.add(usize, target.len, local_window) catch return error.BadDraftBlock;
    const size = std.math.add(usize, total, frame.query_rows) catch return error.BadDraftBlock;
    if (query >= frame.query_rows or out.len < size or
        @as(u64, frame.anchor) + query > std.math.maxInt(i32)) return error.BadDraftBlock;
    const window_start = @as(i64, frame.anchor) - local_window;
    @memset(out, -1);
    for (out[0..target.len], target) |*entry, position| {
        if (position >= 0 and position < window_start) entry.* = position;
    }
    for (out[target.len..total], 0..) |*entry, i| {
        entry.* = @intCast(@max(-1, window_start + @as(i64, @intCast(i))));
    }
    for (out[total .. total + query + 1], 0..) |*entry, i| entry.* = @intCast(frame.anchor + i);
}

test "rejected target rows do not advance the DFly2 anchor" {
    var tokens: [6]u32 = undefined;
    var positions: [6]u32 = undefined;
    const frame = (try prepare(&.{ 10, 11, 12, 13 }, 1, 42, 20, 120009, &tokens, &positions)).?;
    try std.testing.expectEqual(@as(usize, 2), frame.source_row);
    try std.testing.expectEqualSlices(u32, &.{ 42, 120009, 120009, 120009, 120009, 120009 }, &tokens);
    try std.testing.expectEqualSlices(u32, &.{ 13, 14, 15, 16, 17, 18 }, &positions);
    try std.testing.expectEqual(@as(?Frame, null), try prepare(&.{ 10, 11, 12, 13 }, 1, 42, 18, 120009, &tokens, &positions));
}

test "the DFly2 draft row uses neither duplicate local keys nor future query keys" {
    const frame: Frame = .{ .source_row = 2, .anchor = 13, .query_rows = 6 };
    var out: [16]i32 = undefined;
    try reuseIndices(frame, &.{ 0, 8, 10, 12, 13, -1 }, 3, 2, &out);
    try std.testing.expectEqualSlices(i32, &.{ 0, 8, -1, -1, -1, -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1 }, &out);
}
