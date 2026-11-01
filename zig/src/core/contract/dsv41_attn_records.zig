//! DeepSeek V4.1 CSA2 attention launches: each source reserves its group size, which the launcher uses whole.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "dsv41 attn decode f32 tf_dsv41_attn_dec_f32", .threads = 160, .tg_memory = 12352, .reserved = 160 },
    .{ .name = "dsv41 attn decode bf16 tf_dsv41_attn_dec_bf16", .threads = 160, .tg_memory = 12352, .reserved = 160 },
    .{ .name = "dsv41 attn decode prefetch f32 tf_dsv41_attn_dec_pf_f32", .threads = 160, .tg_memory = 12352, .reserved = 160 },
    .{ .name = "dsv41 attn decode prefetch bf16 tf_dsv41_attn_dec_pf_bf16", .threads = 160, .tg_memory = 12352, .reserved = 160 },
    .{ .name = "dsv41 attn prompt f32 tf_dsv41_attn_pre_f32", .threads = 256, .tg_memory = 16384, .reserved = 256 },
    .{ .name = "dsv41 attn prompt bf16 tf_dsv41_attn_pre_bf16", .threads = 256, .tg_memory = 16384, .reserved = 256 },
};

test "the DeepSeek V4.1 attention launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
