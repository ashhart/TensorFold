//! Ripple on CUDA: the family's kernels (glue, attention prep, attention), weights and forward, beside Metal's.

pub const kernels = @import("cuda_kernels.zig");
pub const prep = @import("cuda_prep.zig");
pub const attn = @import("cuda_attn.zig");
pub const weights = @import("cuda_weights.zig");
pub const model = @import("cuda_model.zig");
pub const backend = @import("cuda_backend.zig");
pub const native = @import("cuda_native.zig");

test {
    _ = kernels;
    _ = prep;
    _ = attn;
    _ = weights;
    _ = model;
    _ = backend;
    _ = native;
}
