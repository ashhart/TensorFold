//! Inkling 2 Mini glue, paired KV convolution, fused norms, in-place histories and segmented convs.
const guard = @import("../threadgroup_source.zig");
pub const glue = guard.normalize(@embedFile("glue.metal") ++ "\n" ++ @embedFile("glue_kv_conv.metal") ++ "\n" ++ @embedFile("glue_norm.metal") ++ "\n" ++ @embedFile("glue_split.metal") ++ "\n" ++ @embedFile("glue_conv_segs.metal") ++ "\n" ++ @embedFile("glue_inplace.metal"));
