//! Inkling 2 Mini glue sources with convolution, fused norms, split routing, and multi-segment dense kernels.
const guard = @import("../threadgroup_source.zig");
pub const glue = guard.normalize(@embedFile("glue.metal") ++ "\n" ++ @embedFile("glue_kv_conv.metal") ++ "\n" ++ @embedFile("glue_norm.metal") ++ "\n" ++ @embedFile("glue_split.metal") ++ "\n" ++ @embedFile("glue_conv_segs.metal") ++ "\n" ++ @embedFile("glue_dense.metal"));
