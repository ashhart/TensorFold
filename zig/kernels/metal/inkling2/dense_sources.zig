//! Inkling 2 Mini's prompt projections on simdgroup matrices, for GPUs without tensor units.
pub const dense_mma = @embedFile("dense_mma.metal");
