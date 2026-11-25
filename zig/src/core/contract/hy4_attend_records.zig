//! Hy4-flash's attend launches on Metal (tests/contract/metal_hy4_attend.zig), M1 and M2 limits from --offline.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
/// Attention.attend's 128-thread groups; scored stages 16 keys in threadgroup memory; controls compile their edits.
pub const records = [_]M{
    .{ .name = "hy4 mla_attention tf_hy4_attend_chunks", .threads = 128, .tg_memory = 0, .m1 = 640, .m2 = 640 },
    .{ .name = "hy4 mla_attention tf_hy4_attend_merge", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 mla_attention tf_hy4_attend_scores", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 mla_attention tf_hy4_attend_scored", .threads = 128, .tg_memory = 16384, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 mla_attention control drop-last tf_hy4_attend_chunks", .threads = 128, .tg_memory = 0, .m1 = 640, .m2 = 640 },
    .{ .name = "hy4 mla_attention control no-sink tf_hy4_attend_merge", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 mla_attention control half-sums tf_hy4_attend_chunks", .threads = 128, .tg_memory = 0, .m1 = 768, .m2 = 768 },
    .{ .name = "hy4 mla_attention control skip-nan tf_hy4_attend_merge", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 mla_attention control shuffle tf_hy4_attend_scored", .threads = 128, .tg_memory = 16384, .m1 = 1024, .m2 = 1024 },
};

test "Hy4-flash's attend launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
