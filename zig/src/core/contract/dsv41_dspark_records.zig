//! DeepSeek V4.1 DSpark Metal launches (every count reserved).
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "dsv41 dspark attn tf_dsv41_attn_dspark_bf16", .threads = 512, .tg_memory = 4096, .reserved = 512 },
};

test "the DeepSeek V4.1 DSpark launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
