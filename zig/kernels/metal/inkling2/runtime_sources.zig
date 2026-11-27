//! Inkling 2 Mini's copy, widening and partial-residual shaders.
const guard = @import("../threadgroup_source.zig");
pub const runtime = guard.normalize(@embedFile("runtime.metal") ++ "\n" ++ @embedFile("runtime_conv.metal"));
