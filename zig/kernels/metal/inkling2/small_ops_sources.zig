//! Inkling 2 Small (H=4608, E=384, 8 blocks/taps) Metal glue and runtime kernels.
pub const small_ops = @embedFile("small_ops.metal");
