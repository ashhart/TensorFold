//! Flash Next vision pipelines await compiled limits on the supported Apple families.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "flashnext vision fz_mask_rows", .threads = 256, .tg_memory = 0 },
    .{ .name = "flashnext vision fn_attn_prep_mrope", .threads = 256, .tg_memory = (256 + 8) * 4 },
};
