//! Ornith-1.5-122B's routed block: the experts' decode rows (qwen27/moe_rows.metal) and qwen27/moe.metal's kernels.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "qwen27 moe rows q4 q27_rows_extra_q4", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "qwen27 moe rows q8 q27_rows_extra_q8", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "qwen27 moe route q27_route", .threads = 256, .tg_memory = 512, .m1 = 1024, .m2 = 1024 },
    .{ .name = "qwen27 moe sort q27_route_sort", .threads = 256, .tg_memory = 1024, .m1 = 1024, .m2 = 1024 },
    .{ .name = "qwen27 moe gather q27_route_gather", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "qwen27 moe SwiGLU q27_expert_glu", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "qwen27 moe combine q27_expert_combine", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};

test "the routed block's launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
