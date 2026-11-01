//! Idol on CUDA: the denoiser (tf-idol) and the image encoder (tf-idol-vae), beside the lane engine as on Metal.
pub const config = @import("config.zig");
pub const host = @import("host.zig");
pub const reference = @import("reference.zig");
pub const kernels = @import("cuda_kernels.zig");
pub const Model = @import("cuda_model.zig").Model;
pub const Condition = @import("cuda_model.zig").Condition;
pub const Timing = @import("cuda_model.zig").Timing;
pub const seconds = @import("cuda_model.zig").seconds;
pub const admitted = @import("cuda_model.zig").admitted;
pub const vae = @import("cuda_vae.zig");
pub const vae_kernels = @import("cuda_vae_kernels.zig");

test {
    _ = @import("cuda_kernels.zig");
    _ = @import("cuda_gemm.zig");
    _ = @import("cuda_model.zig");
    _ = @import("cuda_weights.zig");
    _ = @import("cuda_vae.zig");
    _ = @import("cuda_vae_kernels.zig");
    _ = @import("cuda_vae_weights.zig");
    _ = @import("vae_config.zig");
    _ = config;
    _ = host;
    _ = reference;
}
