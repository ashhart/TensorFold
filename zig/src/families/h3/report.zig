//! The CUDA H3 library's stderr reports; a test build keeps the latest line for its test to check instead of printing.
const std = @import("std");
const builtin = @import("builtin");

var kept: [256]u8 = undefined;
var kept_len: usize = 0;

/// One report line (no trailing newline in ``format``).
pub fn say(comptime format: []const u8, args: anytype) void {
    if (!builtin.is_test) return std.debug.print(format ++ "\n", args);
    kept_len = if (std.fmt.bufPrint(&kept, format, args)) |line| line.len else |_| kept.len;
}

/// The latest report since ``clear`` (test builds only).
pub fn latest() []const u8 {
    comptime std.debug.assert(builtin.is_test);
    return kept[0..kept_len];
}

pub fn clear() void {
    kept_len = 0;
}

/// Fails unless the latest report is ``call: error``, then clears it.
pub fn expect(comptime call: []const u8, err: anyerror) !void {
    var buf: [128]u8 = undefined;
    try std.testing.expectEqualStrings(try std.fmt.bufPrint(&buf, call ++ ": {t}", .{err}), latest());
    clear();
}

test "a test build keeps the report line instead of printing it" {
    clear();
    say("tensor {s}: {t}", .{ "a.weight", error.MissingTensor });
    try expect("tensor a.weight", error.MissingTensor);
    try std.testing.expectEqualStrings("", latest());
}
