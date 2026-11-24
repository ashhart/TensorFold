//! The offline compiler's launch limits for the in-place row penalties.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "flashnext penalties fz_penalize_rows", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};
