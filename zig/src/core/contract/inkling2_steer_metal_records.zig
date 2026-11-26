//! Inkling 2 Mini --glp launches (tests/contract/metal_inkling2_steer.zig): the steered visit output and its add.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "metal inkling2 steer mlp_partial_steer tf_ink2_mlp_partial_steer", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 steer steer_add tf_ink2_steer_add", .threads = 256, .tg_memory = 0, .reserved = 256 },
};
