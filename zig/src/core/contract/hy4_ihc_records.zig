//! Hy4-flash's ihc and DFly2 glue launches on Metal (tests/contract/metal_hy4_ihc.zig), M1/M2 limits from --offline.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "hy4 ihc tf_hy4_ihc_collapse_g4", .threads = 256, .tg_memory = 304, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc tf_hy4_ihc_collapse_g8", .threads = 256, .tg_memory = 304, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc tf_hy4_ihc_partials_g4", .threads = 32, .tg_memory = 0, .m1 = 928, .m2 = 928 },
    .{ .name = "hy4 ihc tf_hy4_ihc_partials_g8", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc tf_hy4_ihc_finish_g4", .threads = 256, .tg_memory = 16, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc tf_hy4_ihc_finish_g8", .threads = 256, .tg_memory = 16, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc tf_hy4_tail_g4", .threads = 256, .tg_memory = 640, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc tf_hy4_tail_g8", .threads = 256, .tg_memory = 640, .m1 = 832, .m2 = 832 },
    .{ .name = "hy4 ihc tf_hy4_head_g4", .threads = 256, .tg_memory = 64, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc tf_hy4_head_g8", .threads = 256, .tg_memory = 64, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc tf_hy4_seam_g4", .threads = 256, .tg_memory = 704, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc tf_hy4_seam_g8", .threads = 256, .tg_memory = 704, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc tf_hy4_ihc_add", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc tf_hy4_draft_fuse", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc tf_hy4_draft_residual", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    // the planted controls' pipelines, each compiled from its own edit of ihc.metal
    .{ .name = "hy4 ihc control flush tf_hy4_ihc_collapse_g8", .threads = 256, .tg_memory = 304, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc control seven tf_hy4_ihc_collapse_g8", .threads = 256, .tg_memory = 304, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc control half tf_hy4_ihc_collapse_g8", .threads = 256, .tg_memory = 304, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc control slot0 tf_hy4_ihc_finish_g8", .threads = 256, .tg_memory = 16, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc control unweighted tf_hy4_tail_g8", .threads = 256, .tg_memory = 640, .m1 = 832, .m2 = 832 },
    .{ .name = "hy4 ihc control once tf_hy4_seam_g8", .threads = 256, .tg_memory = 704, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc control onegate tf_hy4_ihc_add", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc control gain0 tf_hy4_draft_fuse", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 ihc control subtract tf_hy4_draft_residual", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};

test "Hy4-flash's ihc launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
