//! The Qwen3.5 hybrids on CUDA (2B, Qwen3.8-27B): the engine and its native family entry; Metal's 2B is qwen3_5.zig.
pub const config = @import("config.zig");
pub const geometry = @import("cuda_geometry.zig");
pub const round = @import("cuda_round.zig");
pub const prompt_int8 = @import("cuda_prompt_int8.zig");
pub const modelopt = @import("cuda_modelopt.zig");
pub const engine = @import("cuda_engine.zig");
pub const prefill = @import("cuda_prefill.zig");
pub const cache = @import("cuda_cache.zig");
pub const signatures = @import("cuda_signatures.zig");
pub const native = @import("cuda_native.zig");
pub const native_moe = @import("cuda_native_moe.zig");
pub const mtp = @import("cuda_mtp.zig");
pub const mtp_check = @import("cuda_mtp_check.zig");
pub const draft = @import("cuda_draft.zig");
pub const vision = @import("cuda_vision.zig");
pub const reuse = @import("cuda_reuse.zig");
pub const joint = @import("cuda_joint_readout.zig");
/// The joint schema head on the device (cuda/joint_schema), for the runners.
pub const joint_schema = @import("cuda_joint_schema");
pub const stage = @import("cuda_stage.zig");

test {
    _ = @import("cuda_attention.zig");
    _ = @import("cuda_prompt.zig");
    _ = @import("cuda_prompt_tail.zig");
    _ = @import("cuda_forward.zig");
    _ = engine;
    _ = @import("cuda_modelopt.zig");
    _ = prompt_int8;
    _ = native;
    _ = native_moe;
    _ = mtp;
    _ = @import("cuda_mtp_bf16.zig");
    _ = @import("cuda_mtp_batch.zig");
    _ = config;
    _ = geometry;
    _ = @import("draw.zig");
    _ = @import("cuda_draft_weights.zig");
    _ = @import("cuda_draft_config.zig");
    _ = draft;
    _ = vision;
    _ = @import("cuda_vision_prompt.zig");
    _ = reuse;
    _ = @import("cuda_disk.zig");
    _ = native;
    _ = @import("cuda_pages.zig");
    _ = @import("cuda_penalties.zig");
    _ = @import("cuda_pages_init_test.zig");
    _ = @import("cuda_pages_lifetime_test.zig");
    _ = @import("cuda_pages_pieces_test.zig");
    _ = @import("cuda_state_pieces.zig");
    _ = @import("cuda_mtp_area_test.zig");
    _ = @import("cuda_prefix.zig");
    _ = @import("cuda_prefix_fault_test.zig");
    _ = @import("cuda_pend_test.zig");
    _ = @import("cuda_states.zig");
    _ = @import("cuda_reuse_pages_test.zig");
}

pub const slide = @import("cuda_slide.zig");
test {
    _ = slide;
    _ = @import("cuda_pocket.zig");
    _ = @import("cuda_vision.zig");
}

test {
    _ = @import("cuda_wide.zig");
    _ = @import("cuda_weights.zig");
    _ = @import("cuda_mt2.zig");
    _ = @import("cuda_stage_forward.zig");
    _ = @import("cuda_shapes.zig");
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

test "Qwen pages default: the learner's and the prefix share's refusals stop before the weights load" {
    try @import("cuda_admit_test.zig").refusals();
}

test {
    _ = @import("cuda_request_starts_test.zig");
}
