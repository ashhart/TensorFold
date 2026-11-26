//! Inkling 2 Mini's glue, paired KV and segmented convs, fused norm/AR-norm, and split router.
pub const glue = @embedFile("glue.metal") ++ "\n" ++ @embedFile("glue_kv_conv.metal") ++ "\n" ++ @embedFile("glue_norm.metal") ++ "\n" ++ @embedFile("glue_split.metal") ++ "\n" ++ @embedFile("glue_conv_segs.metal");
