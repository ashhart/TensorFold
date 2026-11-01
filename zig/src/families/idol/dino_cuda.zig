//! Idol's DINOv3 ViT-H+/16 image encoder on CUDA: driven by tf-idol-dino on CUDA, as the Metal encoder is on Metal.
pub const config = @import("dino_config.zig");
pub const host = @import("dino_host.zig");
pub const reference = @import("dino_reference.zig");
pub const kernels = @import("dino_cuda_kernels.zig");
pub const weights = @import("dino_cuda_weights.zig");
pub const Encoder = @import("dino_cuda_model.zig").Encoder;
pub const Timing = @import("dino_cuda_model.zig").Timing;
pub const seconds = @import("dino_cuda_model.zig").seconds;
pub const admitted = @import("dino_cuda_model.zig").admitted;

test {
    _ = @import("dino_cuda_kernels.zig");
    _ = @import("dino_cuda_weights.zig");
    _ = @import("dino_cuda_model.zig");
    _ = config;
    _ = host;
    _ = reference;
}
