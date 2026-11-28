//! MTP rank-scan launch declarations; measured nvcc usage is still pending on all five release SMs.
const registry = @import("registry.zig");

pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "glm MTP rank pick", .symbol = "tf_glm_mtp_pick", .image = "glm_glue", .threads = 32 }, .usage = registry.none },
    .{ .launch = .{ .name = "glm MTP rank pick tie control", .symbol = "tf_ctl_glm_mtp_pick_tie", .image = "contract_controls", .threads = 32 }, .usage = registry.none },
    .{ .launch = .{ .name = "glm MTP rank pick NaN control", .symbol = "tf_ctl_glm_mtp_pick_nan", .image = "contract_controls", .threads = 32 }, .usage = registry.none },
};
