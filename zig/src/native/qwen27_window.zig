//! Target verification width is independent of the unchanged masked drafter block and its selected tree.
const std = @import("std");
pub const capacity = 64;
pub fn nodes(window: u32, copies: bool) u32 {
    return @min(window -| 1, if (copies) @as(u32, 15) else 31);
}
pub fn copyPolicy(tensor_units: bool, value: ?[]const u8) bool {
    _ = tensor_units;
    return if (value) |v| std.mem.eql(u8, v, "1") else false;
}
pub fn rows(tensor_units: bool) u32 {
    _ = tensor_units;
    return 16;
}
/// Wider pre-tensor windows are explicit until their served cost is qualified.
pub fn requested(tensor_units: bool, value: ?[]const u8) !u32 {
    const v = value orelse return rows(tensor_units);
    const n = std.fmt.parseInt(u32, v, 10) catch return error.BadQwenWindow;
    if (tensor_units) return if (n == 16) 16 else error.BadQwenWindow;
    return if (n == 16 or n == 32 or n == 64) n else error.BadQwenWindow;
}
test "qwen27 window: both device classes keep 16 until a wider pre-tensor window is requested" {
    try std.testing.expectEqual(@as(u32, 16), try requested(false, null));
    try std.testing.expectEqual(@as(u32, 32), try requested(false, "32"));
    try std.testing.expectEqual(@as(u32, 64), try requested(false, "64"));
    try std.testing.expectEqual(@as(u32, 16), try requested(true, null));
    try std.testing.expectError(error.BadQwenWindow, requested(false, "48"));
    try std.testing.expectError(error.BadQwenWindow, requested(true, "64"));
    try std.testing.expect(rows(false) <= capacity and rows(true) <= capacity);
}
test "qwen27 window: head plus copies matches the 64-row serial layout without enlarging the masked block" {
    try std.testing.expectEqual(@as(u32, 31), nodes(64, false));
    try std.testing.expectEqual(@as(u32, 31), nodes(32, false));
    try std.testing.expectEqual(@as(u32, 15), nodes(64, true));
    try std.testing.expectEqual(@as(u32, 15), nodes(32, true));
    try std.testing.expectEqual(@as(u32, 15), nodes(16, false));
    try std.testing.expectEqual(@as(u32, 15), nodes(16, true));
}

test "qwen27 window: copies remain off until explicitly requested" {
    try std.testing.expect(!copyPolicy(false, null));
    try std.testing.expect(!copyPolicy(true, null));
    try std.testing.expect(!copyPolicy(false, "0"));
    try std.testing.expect(copyPolicy(true, "1"));
}
