//! TP's log lines, silent under test: a test's stderr is the build runner's channel.
const std = @import("std");
const quiet = @import("builtin").is_test;

pub fn err(comptime format: []const u8, args: anytype) void {
    if (!quiet) std.log.err(format, args);
}

pub fn warn(comptime format: []const u8, args: anytype) void {
    if (!quiet) std.log.warn(format, args);
}

pub fn info(comptime format: []const u8, args: anytype) void {
    if (!quiet) std.log.info(format, args);
}
