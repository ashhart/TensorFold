//! Cosmos3-Edge's reasoner on CUDA, the Metal family's twin: GQA, M-RoPE, relu^2 on 8-bit tiles; SigLIP2 on cuBLASLt.
pub const config = @import("config.zig");
pub const image = @import("cuda_image.zig");
pub const model = @import("cuda_model.zig");
pub const backend = @import("cuda_backend.zig");
pub const Model = model.Model;

test {
    _ = config;
    _ = image;
}
