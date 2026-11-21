//! Inkling 2 on CUDA: Small's kernels and Mini (inkling2_loop) at one rank, its loop and 48 visit slots a stream.

pub const kernels = @import("kernels.zig");
pub const ops = @import("ops.zig");
pub const config = @import("config.zig");
pub const plan = @import("plan.zig");
pub const names = @import("names.zig");
pub const loop = @import("loop.zig");
pub const mini_weights = @import("mini_weights.zig");
pub const mini_model = @import("mini_model.zig");
/// The float64 reference the CUDA forward's greedy tokens are checked against.
pub const mini_ref = @import("mini_ref.zig");

/// config.json's name in the release.
pub const model_type = "inkling2";

test {
    _ = kernels;
    _ = ops;
    _ = loop;
    _ = config;
    _ = plan;
    _ = mini_model;
    _ = @import("mini_visit.zig");
    _ = mini_weights;
    _ = mini_ref;
}
