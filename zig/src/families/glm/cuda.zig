//! GLM-5.3-Flash on CUDA in Zig at TP=2: the rank loader over the unsplit EXL3 checkpoint, the engine, greedy replies.

pub const Config = @import("config.zig").Config;
pub const load = @import("cuda_load.zig");
pub const Weights = @import("cuda_model.zig").Weights;
pub const Engine = @import("cuda_model.zig").Engine;

test {
    _ = @import("cuda_load.zig");
}
