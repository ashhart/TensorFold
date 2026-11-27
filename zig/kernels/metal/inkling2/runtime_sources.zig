//! Inkling 2 Mini's copy, widening and partial-residual shaders.
const guard = @import("../threadgroup_source.zig");
pub const runtime = guard.normalize(@embedFile("runtime.metal") ++ "\n" ++ @embedFile("runtime_conv.metal"));
/// The steered visit output, its fusion with the next mix (over the glue's helpers) and the reference's add.
pub const steer = guard.normalize(@import("glue_sources.zig").glue ++ "\n" ++ @embedFile("steer.metal") ++ "\n" ++
    @embedFile("steer_fused.metal") ++ "\n" ++ @embedFile("steer_norm.metal"));
