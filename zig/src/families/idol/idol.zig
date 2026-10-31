//! Idol image to 3D: its denoiser and DINOv3 image encoder on Metal (beside the lane engine, not a token family).
pub const config = @import("config.zig");
pub const host = @import("host.zig");
pub const weights = @import("weights.zig");
pub const kernels = @import("kernels.zig");
pub const model = @import("model.zig");
pub const reference = @import("reference.zig");
/// Its DINOv3 ViT-H+/16 image encoder: an image to feature1.
pub const dino = @import("dino.zig");
pub const Model = model.Model;
pub const Condition = model.Condition;
pub const Timing = model.Timing;

test {
    _ = config;
    _ = host;
    _ = weights;
    _ = kernels;
    _ = model;
    _ = reference;
    _ = dino;
}
