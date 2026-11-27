//! Mini DSpark's arithmetic shader and its twelve Metal library entry points.

pub const dspark = @embedFile("dspark.metal");
pub const dspark_pre_m5 = "#define TF_SIMD_FRAGS 1\n" ++ dspark;
