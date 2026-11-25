//! Idol's DINOv3 ViT-H+/16 image encoder on Metal: an image to the denoiser's feature1 [4101, 1280].
pub const config = @import("dino_config.zig");
pub const host = @import("dino_host.zig");
pub const weights = @import("dino_weights.zig");
pub const kernels = @import("dino_kernels.zig");
pub const model = @import("dino_model.zig");
pub const reference = @import("dino_reference.zig");
pub const Encoder = model.Encoder;

test {
    _ = config;
    _ = host;
    _ = weights;
    _ = kernels;
    _ = model;
    _ = reference;
}
