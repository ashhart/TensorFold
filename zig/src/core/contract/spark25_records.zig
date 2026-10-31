//! Spark-X2.6-4B's prompt attention launches (metal_spark25_attention.zig): DH / 64 simdgroups, 256 reserved.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    // tensor ops before the M5 too (nax.h's simdgroup fragments still multiply through mpp): no offline build
    .{ .name = "spark25 attention sliding sp_prompt_attention", .threads = 128, .tg_memory = 17920, .reserved = 256, .offline = false },
    .{ .name = "spark25 attention full sp_prompt_attention", .threads = 128, .tg_memory = 17920, .reserved = 256, .offline = false },
};
