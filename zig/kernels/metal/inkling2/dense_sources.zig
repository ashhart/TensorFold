//! Inkling 2 Mini's multi-row bf16 prompt projections: simdgroup matrices (pre-M5) and register-tiled (M5+).
const guard = @import("../threadgroup_source.zig");
pub const dense_mma = guard.normalize(@embedFile("dense_mma.metal"));
pub const dense_tiled = guard.normalize(@embedFile("dense_tiled.metal"));
