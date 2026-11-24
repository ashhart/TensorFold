//! Inkling 2 Mini's glue, split router, and multi-segment decode dense kernels.
pub const glue = @embedFile("glue.metal") ++ "\n" ++ @embedFile("glue_split.metal") ++ "\n" ++ @embedFile("glue_dense.metal");
