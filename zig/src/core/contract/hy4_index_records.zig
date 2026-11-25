//! Hy4-flash's index launches on Metal with --offline's M1/M2 limits; a word before the function names a control.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "hy4 index_build tf_hy4_index_keys", .threads = 128, .tg_memory = 16, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 index_build unrounded tf_hy4_index_keys", .threads = 128, .tg_memory = 16, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 index_build tf_hy4_index_groups", .threads = 128, .tg_memory = 2064, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 index_build slot0-bias tf_hy4_index_groups", .threads = 128, .tg_memory = 2064, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 index_recall tf_hy4_group_scores", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 index_recall head15 tf_hy4_group_scores", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 index_recall tf_hy4_recall_groups", .threads = 256, .tg_memory = 3152, .m1 = 928, .m2 = 928 },
    .{ .name = "hy4 index_recall signed-zero tf_hy4_recall_groups", .threads = 256, .tg_memory = 3152, .m1 = 928, .m2 = 928 },
    .{ .name = "hy4 index_recall own-token tf_hy4_recall_groups", .threads = 256, .tg_memory = 3152, .m1 = 928, .m2 = 928 },
};

test "Hy4-flash's index launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
