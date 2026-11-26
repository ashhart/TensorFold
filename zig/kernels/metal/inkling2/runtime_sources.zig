//! Inkling 2 Mini's copy, widening and partial-residual shaders.
pub const runtime = @embedFile("runtime.metal") ++ "\n" ++ @embedFile("runtime_conv.metal");
/// The steered visit output, and the separately ordered add its reference launches after mlp_partial.
pub const steer = @embedFile("steer.metal");
