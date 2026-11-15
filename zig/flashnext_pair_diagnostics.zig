//! Relay failure tests check the actual error logged by the handoff owner.
const std = @import("std");
pub var failure_name: []const u8 = "";
pub var failure_sequence: u64 = 0;
pub var unexpected: usize = 0;

pub fn capture(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime fmt: []const u8, args: anytype) void {
    _ = scope;
    if (comptime std.mem.eql(u8, fmt, "TP=2 rank {d}: sequence {d} failed: {s}")) {
        failure_sequence = args[1];
        failure_name = args[2];
    } else if (level == .err) {
        unexpected += 1;
        std.debug.print(fmt ++ "\n", args);
    }
}
