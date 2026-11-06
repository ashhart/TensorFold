//! Ling's group-limited Metal route (tf_ling_route): one simdgroup a row, no threadgroup memory.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "metal ling route tf_ling_route", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};

test "the Ling route launch fits every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
