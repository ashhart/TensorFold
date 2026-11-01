//! LFM2-MoE's shared-round and greedy-pick launches (lfm2.metal, lfm2_segments.metal), as the encoders launch them.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
// The drafter kernels carry M1 and M2 limits too, recorded from the offline compile; no run covers those chips.
pub const records = [_]M{
    .{ .name = "metal lfm2 norm tap lfm2_norm_tap", .threads = 512, .tg_memory = 144, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal lfm2 norm tap first lfm2_norm_tap", .threads = 512, .tg_memory = 144, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal lfm2 combine norm tap lfm2_combine_norm_tap", .threads = 512, .tg_memory = 144, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal lfm2 argmax lfm2_argmax", .threads = 1024, .tg_memory = 384, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal lfm2 stream conv lfm2_conv_segs", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal lfm2 stream attention qk lfm2_qk_segs", .threads = 32, .tg_memory = 0, .m1 = 832, .m2 = 832 },
    .{ .name = "metal lfm2 stream attention lfm2_attn_segs", .threads = 128, .tg_memory = 18688, .reserved = 128 },
    .{ .name = "metal lfm2 stream attention merge lfm2_attn_merge_segs", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal lfm2 tap lfm2_tap", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal lfm2 markov scan lfm2_markov_scan", .threads = 256, .tg_memory = 1088, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal lfm2 markov scan bf16 lfm2_markov_scan_bf16", .threads = 256, .tg_memory = 1088, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal lfm2 bf16 rows lfm2_bf16_rows", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal lfm2 bf16 tile lfm2_bf16_mma", .threads = 128, .tg_memory = 24576, .m1 = 832, .m2 = 832 },
    .{ .name = "metal lfm2 markov pick lfm2_markov_pick", .threads = 256, .tg_memory = 3072, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal lfm2 draft ids lfm2_draft_ids", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal lfm2 block attention lfm2_attn_block", .threads = 128, .tg_memory = 18688, .reserved = 128 },
    .{ .name = "metal lfm2 block attention merge lfm2_attn_block_merge", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal lfm2 dspark confidence lfm2_dspark_confidence", .threads = 256, .tg_memory = 32, .m1 = 1024, .m2 = 1024 },
};

test "the LFM2 launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
