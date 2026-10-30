//! DeepSeek V4.1's compressor launches on Metal: each reserves its thread count, so none depends on the M1/M2 compile.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "dsv41 compress pool tf_dsv41_c_pool2", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 compress fp4 round trip tf_dsv41_c_fp4rt", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 compress fp4 pack tf_dsv41_c_fp4pack", .threads = 32, .tg_memory = 0, .reserved = 32 },
};

test "the DeepSeek V4.1 compressor launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
