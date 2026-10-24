//! Cosmos3-Edge's reasoner on CUDA (the Metal family's twin): dense GQA with interleaved M-RoPE and relu^2 on the MLX
//! 8-bit tiles, the SigLIP2 image tower on cuBLASLt.
pub const config = @import("config.zig");
pub const image = @import("cuda_image.zig");
pub const model = @import("cuda_model.zig");
pub const backend = @import("cuda_backend.zig");
pub const Model = model.Model;

test {
    _ = config;
    _ = image;
}
