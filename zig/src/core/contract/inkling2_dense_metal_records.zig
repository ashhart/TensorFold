//! Inkling 2 Mini fused six-projection attention-input launches (zig/tests/contract/metal_inkling2_dense.zig).
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "metal inkling2 dense mv6_1 tf_ink2_mv6_1", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 dense mv6_2 tf_ink2_mv6_2", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 dense mv6_4 tf_ink2_mv6_4", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 dense mv6_8 tf_ink2_mv6_8", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 dense mv6_16 tf_ink2_mv6_16", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 dense mv6_32 tf_ink2_mv6_32", .threads = 256, .tg_memory = 0, .reserved = 256 },
};
