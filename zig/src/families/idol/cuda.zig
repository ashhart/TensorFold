//! Idol's denoiser on CUDA: driven by tf-idol-cuda beside the lane engine, as the Metal family is by tf-idol.
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

test {
    _ = @import("cuda_kernels.zig");
    _ = @import("cuda_gemm.zig");
    _ = @import("cuda_model.zig");
    _ = @import("cuda_weights.zig");
    _ = dino;
    _ = config;
    _ = host;
    _ = reference;
}
