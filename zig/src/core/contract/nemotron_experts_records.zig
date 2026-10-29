//! Nemotron's routed-expert Metal launches: 32 threads a simdgroup, no threadgroup memory, M1/M2 limits compiled.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "nemotron experts up w3 tf_xup_w3", .threads = 128, .tg_memory = 0, .m1 = 576, .m2 = 576 },
    .{ .name = "nemotron experts up rows2 tf_xup_rows2", .threads = 64, .tg_memory = 0, .m1 = 448, .m2 = 448 },
    .{ .name = "nemotron experts up rows4 tf_xup_rows4", .threads = 64, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "nemotron experts down w3 tf_xdown_w3", .threads = 128, .tg_memory = 0, .m1 = 576, .m2 = 576 },
    .{ .name = "nemotron experts down rows2 tf_xdown_rows2", .threads = 64, .tg_memory = 0, .m1 = 448, .m2 = 448 },
    .{ .name = "nemotron experts down rows4 tf_xdown_rows4", .threads = 64, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "nemotron geometry up 2x2 tf_xup_2_2", .threads = 64, .tg_memory = 0, .m1 = 832, .m2 = 832 },
    .{ .name = "nemotron geometry up 8x2 tf_xup_8_2", .threads = 64, .tg_memory = 0, .m1 = 640, .m2 = 640 },
    .{ .name = "nemotron geometry up 4x4 tf_xup_4_4", .threads = 128, .tg_memory = 0, .m1 = 768, .m2 = 768 },
    .{ .name = "nemotron geometry up 8x4 tf_xup_8_4", .threads = 128, .tg_memory = 0, .m1 = 640, .m2 = 640 },
    .{ .name = "nemotron geometry up 4x1 tf_xup_4_1", .threads = 32, .tg_memory = 0, .m1 = 768, .m2 = 768 },
    .{ .name = "nemotron geometry up 2x4 tf_xup_2_4", .threads = 128, .tg_memory = 0, .m1 = 832, .m2 = 832 },
    .{ .name = "nemotron geometry down 2x2 tf_xdown_2_2", .threads = 64, .tg_memory = 0, .m1 = 832, .m2 = 832 },
    .{ .name = "nemotron geometry down 8x2 tf_xdown_8_2", .threads = 64, .tg_memory = 0, .m1 = 640, .m2 = 640 },
    .{ .name = "nemotron geometry down 4x4 tf_xdown_4_4", .threads = 128, .tg_memory = 0, .m1 = 768, .m2 = 768 },
    .{ .name = "nemotron geometry down 8x4 tf_xdown_8_4", .threads = 128, .tg_memory = 0, .m1 = 640, .m2 = 640 },
    .{ .name = "nemotron geometry down 4x1 tf_xdown_4_1", .threads = 32, .tg_memory = 0, .m1 = 768, .m2 = 768 },
    .{ .name = "nemotron geometry down 2x4 tf_xdown_2_4", .threads = 128, .tg_memory = 0, .m1 = 832, .m2 = 832 },
};

test "the Nemotron routed-expert launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
