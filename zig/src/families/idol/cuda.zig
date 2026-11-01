//! Idol on CUDA, each component driven by its tool as on Metal, and all by tf-idol-generate (idol_cuda_e2e.zig).
pub const config = @import("config.zig");
pub const host = @import("host.zig");
pub const reference = @import("reference.zig");
pub const kernels = @import("cuda_kernels.zig");
pub const Model = @import("cuda_model.zig").Model;
pub const Condition = @import("cuda_model.zig").Condition;
pub const Timing = @import("cuda_model.zig").Timing;
pub const seconds = @import("cuda_model.zig").seconds;
pub const admitted = @import("cuda_model.zig").admitted;
/// The DINOv3 image encoder (#1775) on CUDA.
pub const dino = @import("dino_cuda.zig");
pub const vae = @import("cuda_vae.zig");
pub const vae_kernels = @import("cuda_vae_kernels.zig");
pub const vae_config = @import("vae_config.zig");
pub const decoder_config = @import("decoder_config.zig");
pub const decoder_kernels = @import("cuda_decoder_kernels.zig");
pub const octree = @import("octree.zig");
pub const gaussian = @import("gaussian.zig");
pub const decode = @import("decode.zig");
pub const decode_reference = @import("decode_reference.zig");
pub const Decoder = @import("cuda_decoder.zig").Decoder;

test {
    _ = @import("cuda_kernels.zig");
    _ = @import("cuda_gemm.zig");
    _ = @import("cuda_model.zig");
    _ = @import("cuda_weights.zig");
    _ = dino;
    _ = @import("cuda_vae.zig");
    _ = @import("cuda_vae_kernels.zig");
    _ = @import("cuda_vae_weights.zig");
    _ = vae_config;
    _ = @import("cuda_decoder.zig");
    _ = @import("cuda_decoder_kernels.zig");
    _ = @import("cuda_decoder_weights.zig");
    _ = decoder_config;
    _ = octree;
    _ = gaussian;
    _ = decode;
    _ = decode_reference;
    _ = config;
    _ = host;
    _ = reference;
}
