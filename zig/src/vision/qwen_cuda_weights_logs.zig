//! Expected checkpoint diagnostics are counted narrowly; every other error still fails the dedicated test runner.
const std = @import("std");
var count: usize = 0;
var unexpected: usize = 0;

pub fn log(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime format: []const u8, args: anytype) void {
    if (level == .err) {
        count += 1;
        const known = scope == .default and (std.mem.eql(u8, format, "checkpoint has no tensor {s}") or
            std.mem.eql(u8, format, "{s}: {t} {any}, expected {t} {any}") or
            std.mem.eql(u8, format, "unused checkpoint tensor {s}"));
        if (!known) unexpected += 1;
    }
    std.debug.print(format ++ "\n", args);
}

pub fn expect(want: usize) !void {
    const got = count;
    const other = unexpected;
    count = 0;
    unexpected = 0;
    try std.testing.expectEqual(want, got);
    try std.testing.expectEqual(@as(usize, 0), other);
}
