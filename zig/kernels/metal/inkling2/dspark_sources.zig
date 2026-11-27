//! Mini DSpark's arithmetic shader and its twelve Metal library entry points.
const guard = @import("../threadgroup_source.zig");

pub const dspark = guard.normalize(@embedFile("dspark.metal"));
pub const dspark_pre_m5 = guard.normalize("#define TF_SIMD_FRAGS 1\n" ++ dspark);
