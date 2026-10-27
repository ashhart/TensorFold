//! LFM3 text on CUDA: the shared admission (config.zig, weights.zig), the shared CUDA operators, lfm3.cu's 7-tap conv.
pub const config = @import("config.zig");
pub const admission = @import("weights.zig");
pub const weights = @import("cuda_weights.zig");
pub const kernels = @import("cuda_kernels.zig");
pub const model = @import("cuda_model.zig");
pub const forward = @import("cuda_forward.zig");
pub const backend = @import("cuda_backend.zig");
pub const native = @import("cuda_native.zig");
pub const snap = @import("cuda_snap.zig");

test {
    _ = config;
    _ = admission;
    _ = weights;
    _ = kernels;
    _ = model;
    _ = forward;
    _ = backend;
    _ = native;
    _ = snap;
    _ = @import("config_test.zig");
}
