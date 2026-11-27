//! HC row sharing: source geometry only; compiler registers, spills and measured shared bytes remain pending.
const registry = @import("registry.zig");
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "glm_glue tf_glm_hc_partial8_rows", .symbol = "tf_glm_hc_partial8_rows", .image = "glm_glue", .threads = 1024, .min_sm = 80 }, .usage = registry.none },
    .{ .launch = .{ .name = "glm_glue tf_glm_hc_partial8_rows_control", .symbol = "tf_glm_hc_partial8_rows_control", .image = "glm_glue", .threads = 1024, .min_sm = 80 }, .usage = registry.none },
};
