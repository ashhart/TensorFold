//! Inkling 2 Mini's glue, paired KV convolution, fused norms, segmented convs, split router, and multi-segment decode dense kernels.
const guard = @import("../threadgroup_source.zig");
pub const glue = guard.normalize(@embedFile("glue.metal") ++ "\n" ++ @embedFile("glue_kv_conv.metal") ++ "\n" ++ @embedFile("glue_norm.metal") ++ "\n" ++ @embedFile("glue_split.metal") ++ "\n" ++ @embedFile("glue_conv_segs.metal") ++ "\n" ++ @embedFile("glue_dense.metal"));
