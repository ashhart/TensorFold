//! Inkling 2 Mini's glue, paired KV convs (glue_kv_conv.metal), and split router (glue_split.metal).
pub const glue = @embedFile("glue.metal") ++ "\n" ++ @embedFile("glue_kv_conv.metal") ++ "\n" ++ @embedFile("glue_split.metal");
