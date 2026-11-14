//! GLM's Metal handoff launches with no threadgroup scratch.
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "metal glm ep ep_pack", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal glm ep ep_unpack", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal glm ep ep_rcombine", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal glm ep ep_rfinal", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal glm ep ep_rsend", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal glm ep ep_rsum", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal glm ep ep_amax_send", .threads = 64, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal glm ep ep_amax_merge", .threads = 64, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};
