//! Inkling 2 Small (model_type inkling2) on CUDA: config, chat, reply split, admission, kernels, EXL3 weights, forward.

pub const image = @import("image.zig");
pub const vision = @import("vision.zig");
pub const kernels = @import("kernels.zig");
pub const ops = @import("ops.zig");
pub const experts = @import("experts.zig");

/// config.json's name in the release.
pub const model_type = "inkling2";

test {
    _ = image;
    _ = vision;
    _ = kernels;
    _ = ops;
    _ = experts;
}
