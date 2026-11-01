//! DeepSeek V4.1's MoE route select launches (`tf_dsv41_route_select`), the backbone's and the drafter's.
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "dsv41 route backbone tf_dsv41_route_select", .threads = 32, .tg_memory = 0, .m1 = 832, .m2 = 832 },
    .{ .name = "dsv41 route drafter tf_dsv41_route_select", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};
