//! Target verification width is independent of the unchanged masked drafter block and its selected tree.
const std = @import("std");
pub const capacity = 64;
pub fn nodes(window: u32, copies: bool) u32 {
    return @min(window -| 1, if (copies) @as(u32, 15) else 31);
}
pub fn copyPolicy(tensor_units: bool, window: u32, value: ?[]const u8) !bool {
    return flag(value, tensor_units and window > 16, error.BadQwenCopies);
}
pub fn rows(tensor_units: bool) u32 {
    _ = tensor_units;
    return 16;
}
/// Both projection paths admit explicit wider windows with their measured costs.
pub fn requested(tensor_units: bool, value: ?[]const u8) !u32 {
    const v = value orelse return rows(tensor_units);
    const n = std.fmt.parseInt(u32, v, 10) catch return error.BadQwenWindow;
    return if (n == 16 or n == 32 or n == 64) n else error.BadQwenWindow;
}
test "qwen27 window: both device classes retain narrow defaults and admit explicit wide windows" {
    try std.testing.expectEqual(@as(u32, 16), try requested(false, null));
    try std.testing.expectEqual(@as(u32, 32), try requested(false, "32"));
    try std.testing.expectEqual(@as(u32, 64), try requested(false, "64"));
    try std.testing.expectEqual(@as(u32, 16), try requested(true, null));
    try std.testing.expectError(error.BadQwenWindow, requested(false, "48"));
    try std.testing.expectEqual(@as(u32, 64), try requested(true, "64"));
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
    try std.testing.expect(!try copyPolicy(false, 16, null));
    try std.testing.expect(!try copyPolicy(true, 16, null));
    try std.testing.expect(!try copyPolicy(false, 16, "0"));
    try std.testing.expect(try copyPolicy(true, 16, "1"));
}

fn flag(value: ?[]const u8, fallback: bool, comptime bad: anyerror) !bool {
    const v = value orelse return fallback;
    if (std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return bad;
}
/// Wide tensor-unit requests use lanes; explicit checked overrides select either loop.
pub fn serial(tensor_units: bool, window: u32, value: ?[]const u8) !bool {
    return !try flag(value, !(tensor_units and window == 16), error.BadQwenLanes);
}
pub fn serialPolicy(tensor_units: bool, value: ?[]const u8) !bool {
    return serial(tensor_units, 16, value);
}
pub fn problem(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.BadQwenWindow => "TF_QWEN27_WINDOW must be 16, 32 or 64",
        error.BadQwenCopies => "TF_QWEN27_COPIES must be 0 or 1",
        error.BadQwenLanes => "TF_QWEN27_LANES must be 0 or 1",
        else => null,
    };
}
test "qwen27 window: unknown switches refuse by their startup variable names" {
    for ([_][]const u8{ "true", "on", "yes", "", "2" }) |v| {
        try std.testing.expectError(error.BadQwenCopies, copyPolicy(false, 16, v));
        try std.testing.expectError(error.BadQwenLanes, serialPolicy(false, v));
    }
    try std.testing.expectEqualStrings("TF_QWEN27_COPIES must be 0 or 1", problem(error.BadQwenCopies).?);
    try std.testing.expect(std.mem.startsWith(u8, problem(error.BadQwenWindow).?, "TF_QWEN27_WINDOW"));
    try std.testing.expect(std.mem.startsWith(u8, problem(error.BadQwenLanes).?, "TF_QWEN27_LANES"));
    try std.testing.expect(try serialPolicy(true, null));
    try std.testing.expect(!try serialPolicy(false, null));
    try std.testing.expect(try serialPolicy(false, "0"));
    try std.testing.expect(!try serialPolicy(true, "1"));
}

pub fn describe(buf: []u8, serial_loop: bool, window: u32, drafting: bool, copies: bool) ![]const u8 {
    return std.fmt.bufPrint(buf, "lone loop={s} width={d}; lane window={d}, lane copies={s}", .{
        if (!drafting) "plain" else if (serial_loop) "serial" else "lanes",
        if (!drafting) @as(u32, 1) else if (serial_loop) @as(u32, 16) else window,
        window,
        if (copies) "on" else "off",
    });
}
test "qwen27 window: startup reports the effective lone width beside the lane policy" {
    var buf: [160]u8 = undefined;
    try std.testing.expectEqualStrings("lone loop=serial width=16; lane window=64, lane copies=on", try describe(&buf, true, 64, true, true));
    try std.testing.expectEqualStrings("lone loop=lanes width=32; lane window=32, lane copies=off", try describe(&buf, false, 32, true, false));
    try std.testing.expect(std.mem.startsWith(u8, try describe(&buf, false, 64, false, false), "lone loop=plain width=1;"));
}

test "qwen27 window: wide tensor lone rounds reach the checked lane policy" {
    for ([_]u32{ 16, 32, 64 }) |width| {
        try std.testing.expectEqual(width == 16, try serial(true, width, null));
        try std.testing.expect(!try serial(false, width, null));
        try std.testing.expect(!try serial(true, width, "1"));
        try std.testing.expect(try serial(true, width, "0"));
        try std.testing.expectError(error.BadQwenLanes, serial(true, width, "on"));
    }
}

test "qwen27 window: preserve each device class's qualified default and checked copy overrides" {
    for ([_]bool{ false, true }) |units| for ([_]u32{ 16, 32, 64 }) |width| {
        try std.testing.expectEqual(units and width > 16, try copyPolicy(units, width, null));
        try std.testing.expect(!try copyPolicy(units, width, "0"));
        try std.testing.expect(try copyPolicy(units, width, "1"));
        try std.testing.expectError(error.BadQwenCopies, copyPolicy(units, width, "on"));
    };
    try std.testing.expectEqual(@as(u32, 16), try requested(false, null));
    try std.testing.expectEqual(@as(u32, 16), try requested(true, null));
    try std.testing.expect(!try copyPolicy(false, try requested(false, null), null));
    try std.testing.expect(!try copyPolicy(true, try requested(true, null), null));
}
