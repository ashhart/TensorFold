//! Shared draft primitive sources are embedded at build time.
pub const text = @embedFile("kernels/metal/core/draft_ops.metal");
