//! GLM-5.3-Flash's DFlash2 tap mean+scatter kernel on Metal: 256 threads and 0 bytes of threadgroup storage.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "glm tap glm_tap_mean_ring", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
};

test "the GLM tap mean scatter launch fits every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
