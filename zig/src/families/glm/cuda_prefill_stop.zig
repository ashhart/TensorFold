//! Rank 0's word at GLM's chunk gate (cuda_lanes' cancel hook): a cancel, or a warm pass's yield to foreground work.
const std = @import("std");
const lanes = @import("lanes");

/// The ranks' all-reduce then stops every rank's pass before the same chunk, which keeps its completed prefix.
pub fn stop(s: *const lanes.Stream) bool {
    return s.isCancelled() or s.yields();
}

test "GLM warm turns: prefill: a foreground arrival stops a warm pass at its next chunk, as a cancel does" {
    const Flag = struct {
        value: bool = false,
        fn check(p: *anyopaque) bool {
            return @as(*@This(), @ptrCast(@alignCast(p))).value;
        }
    };
    var cancelled: Flag = .{};
    var arrival: Flag = .{};
    var stream: lanes.Stream = undefined;
    stream.cancel_check = .{ .ptr = &cancelled, .check = Flag.check };
    stream.yield_check = .{ .ptr = &arrival, .check = Flag.check };
    try std.testing.expect(!stop(&stream));
    arrival.value = true;
    try std.testing.expect(stop(&stream));
    arrival.value = false;
    cancelled.value = true;
    try std.testing.expect(stop(&stream));
    stream.cancel_check = null;
    stream.yield_check = null;
    try std.testing.expect(!stop(&stream));
}
