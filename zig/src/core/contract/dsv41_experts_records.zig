//! DeepSeek V4.1's routed-expert Metal launches (Kimi's plan and combine, the V4.1 act_quant, gate/up and down).
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "dsv41 experts plan k3_route_plan", .threads = 1024, .tg_memory = 12544, .m1 = 1024, .m2 = 1024 },
    .{ .name = "dsv41 experts up r1 tf_dsv41_xp_up_r1", .threads = 256, .tg_memory = 5888, .reserved = 256 },
    .{ .name = "dsv41 experts up r2 tf_dsv41_xp_up_r2", .threads = 256, .tg_memory = 11776, .reserved = 256 },
    .{ .name = "dsv41 experts up r4 tf_dsv41_xp_up_r4", .threads = 256, .tg_memory = 23552, .reserved = 256 },
    .{ .name = "dsv41 experts quant tf_dsv41_xp_quant", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "dsv41 experts up coded one tf_dsv41_xp_upc_one", .threads = 256, .tg_memory = 11008, .reserved = 256 },
    .{ .name = "dsv41 experts up coded multi tf_dsv41_xp_upc_multi", .threads = 256, .tg_memory = 19456, .reserved = 256 },
    .{ .name = "dsv41 experts down one tf_dsv41_xp_down_one", .threads = 256, .tg_memory = 6144, .reserved = 256 },
    .{ .name = "dsv41 experts down multi tf_dsv41_xp_down_multi", .threads = 256, .tg_memory = 24576, .reserved = 256 },
    .{ .name = "dsv41 experts combine k3_xp_combine", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "dsv41 experts combine f32 k3_xp_combine_f32", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};

test "the DeepSeek V4.1 routed-expert launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
