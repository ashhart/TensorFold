//! Inkling 2 Small (model_type inkling2) on CUDA: config, chat, reply split, admission, kernels, EXL3 weights, forward.

pub const image = @import("image.zig");
pub const vision = @import("vision.zig");
pub const kernels = @import("kernels.zig");
pub const ops = @import("ops.zig");
pub const proj_group = @import("proj_group.zig");
pub const experts = @import("experts.zig");
pub const modelopt = @import("modelopt.zig");
pub const loop = @import("loop.zig");
pub const loop_model = @import("loop_model.zig");
pub const loop_math_cuda = @import("loop_math_cuda.zig");
pub const native = @import("cuda_native.zig");
pub const dspark_ops = @import("dspark_ops.zig");

/// config.json's name in the release.
pub const model_type = "inkling2";

test {
    _ = image;
    _ = vision;
    _ = kernels;
    _ = ops;
    _ = proj_group;
    _ = experts;
    _ = modelopt;
    _ = native;
    _ = loop;
    _ = loop_model;
    _ = loop_math_cuda;
    _ = dspark_ops;
    _ = @import("shared_kernels_test.zig");
}
