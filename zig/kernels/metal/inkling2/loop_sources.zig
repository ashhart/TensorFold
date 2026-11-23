//! Inkling 2 Mini's cross-pass message and rank-64 gate shaders.
pub const loop = @embedFile("loop.metal");
/// The blend on simdgroup matrices, for GPUs without tensor units: after the loop kernels, whose constants it uses.
pub const loop_mma = loop ++ @embedFile("loop_mma.metal");
