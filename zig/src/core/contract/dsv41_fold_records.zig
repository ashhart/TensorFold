//! DeepSeek V4.1's folded decode launches on Metal: each reserves its thread count, so none needs an M1/M2 receipt.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "dsv41 fold combpost tf_dsv41_o_combpost", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "dsv41 fold prenorm tf_dsv41_o_prenorm", .threads = 1024, .tg_memory = 20512, .reserved = 1024 },
    .{ .name = "dsv41 fold postnorm tf_dsv41_o_postnorm", .threads = 1024, .tg_memory = 20512, .reserved = 1024 },
};

test "the DeepSeek V4.1 folded launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
