//! The Qwen3.5 hybrids on CUDA (2B, Qwen3.8-27B): the engine and its native family entry; Metal's 2B is qwen3_5.zig.
pub const config = @import("config.zig");
pub const geometry = @import("cuda_geometry.zig");
pub const round = @import("cuda_round.zig");
pub const engine = @import("cuda_engine.zig");
pub const native = @import("cuda_native.zig");
pub const native_moe = @import("cuda_native_moe.zig");
pub const mtp = @import("cuda_mtp.zig");
pub const mtp_check = @import("cuda_mtp_check.zig");
pub const draft = @import("cuda_draft.zig");
pub const vision = @import("cuda_vision.zig");
pub const reuse = @import("cuda_reuse.zig");

test {
    _ = engine;
    _ = native;
    _ = native_moe;
    _ = mtp;
    _ = config;
    _ = geometry;
    _ = @import("draw.zig");
    _ = @import("cuda_draft_weights.zig");
    _ = draft;
    _ = vision;
    _ = @import("cuda_vision_prompt.zig");
    _ = reuse;
    _ = native;
}

pub const slide = @import("cuda_slide.zig");
test {
    _ = slide;
    _ = @import("cuda_pocket.zig");
    _ = @import("cuda_vision.zig");
}

test {
    _ = @import("cuda_wide.zig");
}

pub const train_ops = @import("cuda_train_ops.zig");
pub const train = @import("cuda_train.zig");
pub const Engine = engine.Engine;

test {
    _ = @import("cuda_completion.zig");
}

test {
    _ = train;
    _ = @import("cuda_read_cost_test.zig");
}
