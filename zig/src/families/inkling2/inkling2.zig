//! Inkling 2 Small (model_type inkling2) on CUDA: config, chat, reply split, admission, kernels, EXL3 weights, forward.

pub const image = @import("image.zig");
pub const vision = @import("vision.zig");
pub const shape = @import("shape.zig");
pub const kernels = @import("kernels.zig");
pub const ops = @import("ops.zig");
pub const shard = @import("shard.zig");
pub const exl3 = @import("exl3.zig");
pub const pack_weights = @import("pack_weights.zig");
pub const experts = @import("experts.zig");
pub const modelopt = @import("modelopt.zig");
pub const loop = @import("loop.zig");
pub const loop_model = @import("loop_model.zig");
pub const loop_math_cuda = @import("loop_math_cuda.zig");
pub const native = @import("cuda_native.zig");
pub const dspark_ops = @import("dspark_ops.zig");
pub const pictures = @import("pictures.zig");
pub const chat = @import("chat.zig");
pub const admit = @import("admit.zig");
pub const forward = @import("forward.zig");
pub const state = @import("state.zig");
pub const visits = @import("visits.zig");
pub const prompt_dense = @import("prompt_dense.zig");
pub const sampler = @import("sampler.zig");
pub const kept = @import("kept.zig");
pub const engine = @import("engine.zig");
pub const follow = @import("follow.zig");
pub const lanes = @import("lanes.zig");
pub const native_exl3 = @import("native.zig");
pub const loop_ops = @import("loop_ops.zig");

/// config.json's name in the release.
pub const model_type = "inkling2";

test {
    _ = image;
    _ = vision;
    _ = shape;
    _ = kernels;
    _ = ops;
    _ = shard;
    _ = exl3;
    _ = pack_weights;
    _ = experts;
    _ = modelopt;
    _ = native;
    _ = loop;
    _ = loop_model;
    _ = loop_math_cuda;
    _ = dspark_ops;
    _ = @import("shared_kernels_test.zig");
    _ = pictures;
    _ = chat;
    _ = admit;
    _ = forward;
    _ = prompt_dense;
    _ = state;
    _ = visits;
    _ = sampler;
    _ = kept;
    _ = engine;
    _ = follow;
    _ = lanes;
    _ = native_exl3;
    _ = loop_ops;
}
