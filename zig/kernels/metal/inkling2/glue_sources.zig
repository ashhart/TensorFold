//! Inkling 2 Mini's glue, paired KV convs, in-place convs, and split router.
pub const glue = @embedFile("glue.metal") ++ "\n" ++ @embedFile("glue_kv_conv.metal") ++ "\n" ++ @embedFile("glue_inplace.metal") ++ "\n" ++ @embedFile("glue_split.metal");
