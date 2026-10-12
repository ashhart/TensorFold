//! GLM-5.3 (model_type glm_moe_dsa) on EXLR8 weights: MLA with a DSA indexer, 256 experts, on the core's kernels.
pub const config = @import("config.zig");
pub const weights = @import("weights.zig");
pub const kernels = @import("kernels.zig");
pub const gpu = @import("gpu.zig");
pub const layer = @import("layer.zig");
pub const moe = @import("moe.zig");

test {
    _ = config;
    _ = weights;
    _ = kernels;
    _ = moe;
}
