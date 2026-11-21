//! Inkling 2 Small (model_type inkling2) on CUDA: config, chat, reply split, admission, kernels, EXL3 weights, forward.

pub const kernels = @import("kernels.zig");
pub const ops = @import("ops.zig");

/// config.json's name in the release.
pub const model_type = "inkling2";

test {
    _ = kernels;
    _ = ops;
}
