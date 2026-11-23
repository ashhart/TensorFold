//! DeepSeek V4.1 opt-in DSpark Metal launches (every count reserved).
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "dsv41 dspark_cert attn tf_dsv41_attn_dspark_bf16", .threads = 512, .tg_memory = 4096, .reserved = 512 },
    .{ .name = "dsv41 dspark_cert markov hid tf_dsv41_markov_hid", .threads = 1024, .tg_memory = 128, .reserved = 1024 },
    .{ .name = "dsv41 dspark_cert markov z 1 tf_dsv41_markov_z_1", .threads = 256, .tg_memory = 896, .reserved = 256 },
    .{ .name = "dsv41 dspark_cert markov z 2 tf_dsv41_markov_z_2", .threads = 256, .tg_memory = 1408, .reserved = 256 },
    .{ .name = "dsv41 dspark_cert markov z 4 tf_dsv41_markov_z_4", .threads = 256, .tg_memory = 2432, .reserved = 256 },
    .{ .name = "dsv41 dspark_cert markov z 8 tf_dsv41_markov_z_8", .threads = 256, .tg_memory = 4480, .reserved = 256 },
    .{ .name = "dsv41 dspark_cert markov pick 1 tf_dsv41_markov_pick_1", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 dspark_cert markov pick 2 tf_dsv41_markov_pick_2", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 dspark_cert markov pick 4 tf_dsv41_markov_pick_4", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 dspark_cert markov pick 8 tf_dsv41_markov_pick_8", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 dspark_cert markov topk part tf_dsv41_markov_topk_part", .threads = 256, .tg_memory = 1280, .reserved = 256 },
    .{ .name = "dsv41 dspark_cert markov topk merge tf_dsv41_markov_topk_merge", .threads = 256, .tg_memory = 3328, .reserved = 256 },
    .{ .name = "dsv41 dspark_cert markov chain 1 tf_dsv41_markov_chain_topk_1", .threads = 256, .tg_memory = 3584, .reserved = 256 },
    .{ .name = "dsv41 dspark_cert markov chain 2 tf_dsv41_markov_chain_topk_2", .threads = 256, .tg_memory = 3712, .reserved = 256 },
    .{ .name = "dsv41 dspark_cert markov chain 4 tf_dsv41_markov_chain_topk_4", .threads = 256, .tg_memory = 3968, .reserved = 256 },
    .{ .name = "dsv41 dspark_cert markov chain 8 tf_dsv41_markov_chain_topk_8", .threads = 256, .tg_memory = 4352, .reserved = 256 },
    .{ .name = "dsv41 dspark_cert early filtered tf_dsv41_early_filtered", .threads = 1024, .tg_memory = 5120, .reserved = 1024 },
    .{ .name = "dsv41 dspark_cert early accept tf_dsv41_early_accept_multi", .threads = 64, .tg_memory = 0, .reserved = 64 },
};

test "the DeepSeek V4.1 opt-in DSpark launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
