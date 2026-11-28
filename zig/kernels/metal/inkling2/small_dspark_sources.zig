//! Small DSpark's fixed-shape arithmetic shader (7 Small-geometry Metal entry points).

pub const small_dspark = @embedFile("small_dspark.metal");
pub const small_dspark_pre_m5 = "#define TF_SIMD_FRAGS 1\n" ++ small_dspark;
