//! Full GLM-5.3's guarded fp32 pair exchange launches.
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "metal glm dsa pair pair_stage", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal glm dsa pair pair_add_rms", .threads = 256, .tg_memory = 32, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal glm dsa pair pair_argmax", .threads = 1024, .tg_memory = 256, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal glm dsa pair pair_pick", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};
