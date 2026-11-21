//! DeepSeek V4.1 prompt matvecs on simdgroup MMA: four 32-lane simdgroups per launch, 16 KiB shared bf16 tiles.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "dsv41 mv mma tf_dsv41_mv_mma_f32", .threads = 128, .tg_memory = 16384, .reserved = 128 },
    .{ .name = "dsv41 mv mma tf_dsv41_mv_mma_bf16", .threads = 128, .tg_memory = 16384, .reserved = 128 },
};

test "the DeepSeek V4.1 prompt MMA matvec launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
