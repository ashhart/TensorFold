//! Idol image to 3D: its denoiser, decoders and image encoders on Metal (beside the lane engine, not a token family).
pub const config = @import("config.zig");
pub const host = @import("host.zig");
pub const weights = @import("weights.zig");
pub const kernels = @import("kernels.zig");
pub const model = @import("model.zig");
pub const reference = @import("reference.zig");
pub const decoder_config = @import("decoder_config.zig");
pub const decoder_kernels = @import("decoder_kernels.zig");
pub const decoder_weights = @import("decoder_weights.zig");
pub const decoder = @import("decoder.zig");
pub const octree = @import("octree.zig");
pub const gaussian = @import("gaussian.zig");
pub const decode = @import("decode.zig");
pub const decode_reference = @import("decode_reference.zig");
pub const Decoder = decoder.Decoder;
pub const vae = @import("vae.zig");
pub const vae_kernels = @import("vae_kernels.zig");
pub const vae_weights = @import("vae_weights.zig");
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
    _ = decoder_config;
    _ = decoder_kernels;
    _ = decoder_weights;
    _ = decoder;
    _ = octree;
    _ = gaussian;
    _ = decode;
    _ = decode_reference;
    _ = vae;
    _ = vae_kernels;
    _ = vae_weights;
    _ = dino;
}
