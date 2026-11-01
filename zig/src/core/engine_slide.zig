//! A native backend without a learner refuses the flag before opening a device or checkpoint.
const std = @import("std");

pub fn refusal(o: anytype, a: std.mem.Allocator, backend: []const u8) !?[]const u8 {
    if (!o.slide) return null;
    return try std.fmt.allocPrint(a, "--slide: {s} checkpoints do not learn on {s} yet", .{ o.model_type, backend });
}
