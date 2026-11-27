//! Inkling 2 Mini's multi-row bf16 prompt projections: simdgroup matrices (pre-M5) and register-tiled (M5+).
pub const dense_mma = @embedFile("dense_mma.metal");
pub const dense_tiled = @embedFile("dense_tiled.metal");
