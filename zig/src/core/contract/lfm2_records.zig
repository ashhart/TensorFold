//! LFM2-MoE's shared-round and greedy-pick launches (lfm2.metal, lfm2_segments.metal), as the encoders launch them.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "metal lfm2 argmax lfm2_argmax", .threads = 1024, .tg_memory = 384, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal lfm2 stream conv lfm2_conv_segs", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal lfm2 stream attention qk lfm2_qk_segs", .threads = 32, .tg_memory = 0, .m1 = 832, .m2 = 832 },
    .{ .name = "metal lfm2 stream attention lfm2_attn_segs", .threads = 128, .tg_memory = 18688, .reserved = 128 },
    .{ .name = "metal lfm2 stream attention merge lfm2_attn_merge_segs", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal lfm2 tap lfm2_tap", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};

test "the LFM2 launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
