//! Inkling 2 Mini's glue, paired KV convolution, fused norms, segmented convs, and split router sources.
const guard = @import("../threadgroup_source.zig");
pub const glue = guard.normalize(@embedFile("glue.metal") ++ "\n" ++ @embedFile("glue_kv_conv.metal") ++ "\n" ++ @embedFile("glue_norm.metal") ++ "\n" ++ @embedFile("glue_split.metal") ++ "\n" ++ @embedFile("glue_conv_segs.metal"));
