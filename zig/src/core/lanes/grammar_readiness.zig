//! Extended budget causes are optional; old libraries retain the original 0/1/2 readiness ABI.
const std = @import("std");
pub const Status = enum { ready, uncovered, over_work, over_edges, over_bytes, timeout, slots };
pub fn decode(code: c_int) !Status {
    return switch (code) {
        0 => .uncovered,
        1 => .ready,
        2 => .over_work,
        3 => .over_edges,
        4 => .over_bytes,
        5 => .timeout,
        6 => .slots,
        else => error.Grammar,
    };
}
test "grammar compile: budget status preserves every cause and rejects unknown failures" {
    for ([_]Status{ .uncovered, .ready, .over_work, .over_edges, .over_bytes, .timeout, .slots }, 0..) |want, code|
        try std.testing.expectEqual(want, try decode(@intCast(code)));
    for ([_]c_int{ -1, 7, 99 }) |bad| try std.testing.expectError(error.Grammar, decode(bad));
}
