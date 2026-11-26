//! Inkling 2 Mini's glue, fused norm/AR-norm (glue_norm.metal), then split router (glue_split.metal).
pub const glue = @embedFile("glue.metal") ++ "\n" ++ @embedFile("glue_norm.metal") ++ "\n" ++ @embedFile("glue_split.metal");
