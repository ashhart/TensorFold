//! Inkling 2 Mini's glue, then the split router (glue_split.metal), which the contract compares bit for bit.
pub const glue = @embedFile("glue.metal") ++ "\n" ++ @embedFile("glue_split.metal");
