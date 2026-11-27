//! Inkling 2 Mini's copy, widening and partial-residual shaders.
pub const runtime = @embedFile("runtime.metal") ++ "\n" ++ @embedFile("runtime_conv.metal");
