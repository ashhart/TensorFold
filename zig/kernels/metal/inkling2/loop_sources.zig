//! Inkling 2 Mini's message and rank-64 gate shaders, then the split gates the contract compares bit for bit.
const guard = @import("../threadgroup_source.zig");
pub const loop = guard.normalize(@embedFile("loop.metal") ++ "\n" ++ @embedFile("loop_split.metal") ++ "\n" ++ @embedFile("loop_roles.metal"));
/// The gates on simdgroup matrices, for GPUs without tensor units: after the loop kernels, whose constants it uses.
pub const loop_mma = guard.normalize(loop ++ @embedFile("loop_mma.metal"));
