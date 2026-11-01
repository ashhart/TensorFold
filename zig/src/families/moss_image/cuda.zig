//! MOSS-Image's transformer on CUDA: a denoiser driven by tf-moss-image, outside the lane engine.
pub const config = @import("config.zig");
pub const host = @import("host.zig");
pub const Model = @import("cuda_model.zig").Model;
pub const Prompt = @import("cuda_model.zig").Prompt;
pub const seconds = @import("cuda_model.zig").seconds;

test {
    _ = @import("cuda_kernels.zig");
    _ = @import("cuda_model.zig");
    _ = config;
    _ = host;
}
