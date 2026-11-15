//! Recording diagnostics for the release context host runner; no device or module is opened.
const std = @import("std");
pub var message: [2048]u8 = undefined;
pub var used: usize = 0;
pub fn log(comptime _: std.log.Level, comptime _: @EnumLiteral(), comptime fmt: []const u8, args: anytype) void {
    const text = std.fmt.bufPrint(&message, fmt, args) catch return;
    used = text.len;
}
pub fn expect(n: usize) !void {
    try std.testing.expectEqual(n, used);
}
