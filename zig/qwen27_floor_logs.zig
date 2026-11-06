//! The existing recording test runner checks that every expected floor diagnostic was consumed.
const std = @import("std");
var buffer: [320]u8 = undefined;
pub var message: []const u8 = "";
var count: usize = 0;
pub fn log(comptime _: std.log.Level, comptime _: @EnumLiteral(), comptime format: []const u8, args: anytype) void {
    count += 1;
    message = std.fmt.bufPrint(&buffer, format, args) catch "log did not fit";
}
pub fn expect(expected: usize) !void {
    const actual = count;
    count = 0;
    try std.testing.expectEqual(expected, actual);
}
