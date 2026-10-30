//! DeepSeek V4.1's vision launches on Metal: each reserves its thread count, so none depends on M1/M2.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "dsv41 vision epilogue tf_dsv41_vis_epi", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 vision qkv rope tf_dsv41_vis_qkv", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 vision attention tf_dsv41_vis_attn", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 vision swiglu tf_dsv41_vis_swiglu", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 vision unshuffle tf_dsv41_vis_unshuffle", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 vision merge tf_dsv41_vis_merge", .threads = 32, .tg_memory = 0, .reserved = 32 },
};

test "the DeepSeek V4.1 vision launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
