//! The Qwen3 dense pass on CUDA alone, for a family that runs Qwen3 layers inside a larger model (Qwen3-ASR).
pub const config = @import("config.zig");
pub const lm = @import("cuda/lm.zig");
pub const weights = @import("cuda/weights.zig");
