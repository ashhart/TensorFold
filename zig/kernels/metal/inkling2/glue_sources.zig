//! Inkling 2 Mini's glue, paired KV convolution, fused norms, and split router sources.
pub const glue = @embedFile("glue.metal") ++ "\n" ++ @embedFile("glue_kv_conv.metal") ++ "\n" ++ @embedFile("glue_norm.metal") ++ "\n" ++ @embedFile("glue_split.metal");
