//! GLM-5.3-Flash intact 8-bit/64 kv_b decode/verify absorb and unabsorb kernels on Metal.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "glm kvb8 glm_absorb_8b", .threads = 512, .tg_memory = 0, .reserved = 512 },
    .{ .name = "glm kvb8 glm_unabsorb_8b", .threads = 512, .tg_memory = 0, .reserved = 512 },
};

test "the GLM 8-bit kv_b launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
