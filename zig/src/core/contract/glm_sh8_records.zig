//! GLM-5.3-Flash mixed 4/64 routed + 8/64 shared fused MoE launches on Metal.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "glm sh8 two glm_moe_gateup_0_sh8_one", .threads = 128, .tg_memory = 0, .reserved = 512, .m1 = 512, .m2 = 512 },
    .{ .name = "glm sh8 two glm_moe_down_0_sh8_one", .threads = 32, .tg_memory = 0, .m1 = 896, .m2 = 896 },
    .{ .name = "glm sh8 four glm_moe_gateup_0_sh8", .threads = 512, .tg_memory = 0, .reserved = 512, .m1 = 512, .m2 = 512 },
    .{ .name = "glm sh8 four glm_moe_down_0_sh8", .threads = 512, .tg_memory = 0, .m1 = 832, .m2 = 832 },
    .{ .name = "glm sh8 pick glm_moe_gateup_0_sh8_pick", .threads = 32, .tg_memory = 0, .reserved = 512, .m1 = 512, .m2 = 512 },
    .{ .name = "glm sh8 pick glm_moe_down_0_sh8_pick", .threads = 32, .tg_memory = 0, .m1 = 896, .m2 = 896 },
};

test "the GLM sh8 MoE launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
