//! Inkling 2 Mini Metal tree-verify convolution and path-compaction launches.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "metal inkling2 tree conv_static tf_ink2_conv_static_tree", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 tree conv_dynamic tf_ink2_conv_dynamic_tree", .threads = 256, .tg_memory = 128, .reserved = 256 },
    .{ .name = "metal inkling2 tree keep_kv tf_ink2_keep_kv", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "metal inkling2 tree keep_conv tf_ink2_keep_conv", .threads = 256, .tg_memory = 0, .reserved = 256 },
};

test "Inkling 2 Metal tree launches respect their resource declarations" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
