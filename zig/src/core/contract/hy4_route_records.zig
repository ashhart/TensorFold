//! Hy4-flash's route launches on Metal (tests/contract/metal_hy4_route*.zig), M1 and M2 limits from --offline.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
// The last four are the planted controls' texts; tf_add_bf16 at Nemotron's groups of 256 (Hy4 never dispatches it).
pub const records = [_]M{
    .{ .name = "hy4 moe_route tf_route_logits", .threads = 256, .tg_memory = 512, .m1 = 896, .m2 = 896 },
    .{ .name = "hy4 moe_route tf_route_select", .threads = 512, .tg_memory = 9408, .m1 = 576, .m2 = 576 },
    .{ .name = "hy4 route_rows tf_hy4_route_select_rows", .threads = 32, .tg_memory = 0, .m1 = 640, .m2 = 640 },
    .{ .name = "hy4 route_rows tf_hy4_route_select_rank", .threads = 512, .tg_memory = 2112, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 affine_experts tf_hy4_routed_sum", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 elementwise tf_add_bf16", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 moe_route half-sums tf_route_logits", .threads = 256, .tg_memory = 512, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 route_rows nan-picked tf_hy4_route_select_rows", .threads = 32, .tg_memory = 0, .m1 = 640, .m2 = 640 },
    .{ .name = "hy4 route_rows ties-high tf_hy4_route_select_rank", .threads = 512, .tg_memory = 2112, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 affine_experts half-sums tf_hy4_routed_sum", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};

test "Hy4-flash's route launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
