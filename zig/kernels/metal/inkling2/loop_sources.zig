//! Inkling 2 Mini's message and rank-64 gate shaders, then the split gates the contract compares bit for bit.
pub const loop = @embedFile("loop.metal") ++ "\n" ++ @embedFile("loop_split.metal");
/// The gates on simdgroup matrices, for GPUs without tensor units: after the loop kernels, whose constants it uses.
pub const loop_mma = loop ++ @embedFile("loop_mma.metal");
