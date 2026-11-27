//! Inkling 2 Mini's tree-verify convolution and surviving-path compaction shaders.
const guard = @import("../threadgroup_source.zig");
pub const tree = guard.normalize(@embedFile("tree.metal"));
