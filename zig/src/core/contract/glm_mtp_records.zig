//! GLM-5.3-Flash MTP prep and 4-bit/64 greedy head+top1 kernels on Metal.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "glm mtp glm_mtp_prep_4b", .threads = 1024, .tg_memory = 512, .reserved = 1024 },
    .{ .name = "glm mtp glm_mtp_prep_8b", .threads = 1024, .tg_memory = 512, .reserved = 1024 },
    .{ .name = "glm mtp glm_head_top1_4b_one", .threads = 128, .tg_memory = 64, .reserved = 128 },
    .{ .name = "glm mtp glm_head_top1_4b", .threads = 512, .tg_memory = 0, .reserved = 512 },
    .{ .name = "glm mtp glm_reduce_top1", .threads = 1024, .tg_memory = 256, .reserved = 1024 },
};

test "the GLM MTP launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
