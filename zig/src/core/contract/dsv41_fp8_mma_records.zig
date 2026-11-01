//! DeepSeek V4.1 FP8 prompt MMA: every launch is four 32-lane simdgroups sharing two 32x36 fp32 weight blocks.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "dsv41 fp8 mma tf_dsv41_fp8_mma_f32", .threads = 128, .tg_memory = 9216, .reserved = 128 },
    .{ .name = "dsv41 fp8 mma tf_dsv41_fp8_mma_bf16", .threads = 128, .tg_memory = 9216, .reserved = 128 },
};

test "the DeepSeek V4.1 FP8 MMA launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
