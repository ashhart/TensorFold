//! Ripple on CUDA: the family's kernels (glue, attention prep, attention), weights and forward, beside Metal's.

pub const kernels = @import("cuda_kernels.zig");
pub const prep = @import("cuda_prep.zig");
pub const attn = @import("cuda_attn.zig");

test {
    _ = kernels;
    _ = prep;
    _ = attn;
}
