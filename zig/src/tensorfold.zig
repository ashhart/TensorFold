//! The native engine: model loading, families and the lane core, over our Metal runtime.
const std = @import("std");

pub const safetensors = @import("core/safetensors.zig");
pub const Checkpoint = checkpoint_host.Checkpoint;
pub const checkpoint_host = @import("core/checkpoint.zig");
pub const checkpoint_metal = checkpoint;
pub const qwen27 = @import("families/qwen27/qwen27.zig");
/// Qwen-VL image input: backend-neutral parts, and the tower on Metal.
pub const qwen_vl = @import("core/qwen_vl/qwen_vl.zig");
pub const qwen_vl_tower = @import("core/qwen_vl/metal.zig");
/// Qwen-Image-2.1's transformer on Metal: a denoiser driven by tf-qwen-image, outside the lane engine.
pub const qwen_image = @import("families/qwen_image/qwen_image.zig");
pub const checkpoint = @import("core/checkpoint_metal.zig");
pub const shard_edit = @import("core/shard_edit.zig");
pub const lora_merge = @import("core/lora_merge_dir.zig");
pub const npy = @import("core/npy.zig");
pub const ids_json = @import("core/ids_json.zig");
pub const lanes = @import("lanes");
pub const segments = @import("core/segments.zig");
pub const row_projection = @import("core/row_projection.zig");
pub const affine_mm = @import("core/affine_mm.zig");
/// The checkpoint formats (MLX affine and its fp64 reader for the device checks, exact widen).
pub const quant = @import("core/quant/quant.zig");
pub const exl3_format = @import("core/exl3_format.zig");
pub const exl3_projection = @import("core/exl3_projection.zig");
pub const exlr8 = @import("core/exlr8.zig");
pub const exlr8_projection = @import("core/exlr8_projection.zig");
pub const exlr8_dense_gpu = @import("core/exlr8_dense_gpu.zig");
pub const exlr8_prompt = @import("core/exlr8_prompt.zig");
pub const glm_dsa = @import("families/glm_dsa/glm_dsa.zig");
pub const nemotron = @import("families/nemotron/nemotron.zig");
pub const flashnext_replay = @import("families/flashnext/replay.zig");
pub const flashnext_rows = @import("families/flashnext/rows_load.zig");
pub const flashnext_engine = @import("families/flashnext/engine.zig");
pub const flashnext_snapshot = @import("families/flashnext/snapshot.zig");
pub const flashnext_learned = @import("families/flashnext/learned.zig");
pub const flashnext_think_check = @import("families/flashnext/think_check.zig");
pub const glm = @import("families/glm/glm.zig");
pub const qwen35 = @import("families/qwen3_5/qwen3_5.zig");
pub const lfm2_moe = @import("families/lfm2_moe/lfm2_moe.zig");
pub const flashnext_pack = @import("families/flashnext/pack.zig");

test {
    std.testing.refAllDecls(@This());
    _ = lora_merge;
    _ = @import("families/nemotron/prefill_kernels.zig"); // Compile the prompt-chunk sources in tests.
    _ = @import("families/nemotron/simd_attention.zig"); // The pre-M5 attention rewrite finds its lines.
    _ = @import("families/nemotron/weights.zig");
    _ = @import("families/nemotron/kernels.zig");
    _ = @import("families/nemotron/subspace.zig"); // a new block's directions
    _ = @import("families/nemotron/living_test.zig"); // saved lessons and their routers, beside the shards
    _ = @import("families/nemotron/pocket_shards_test.zig"); // learned neurons in the shards themselves
    _ = @import("families/nemotron/pocket_forget_test.zig"); // a fact deleted on request, and put back
    _ = @import("families/nemotron/pocket.zig"); // which row a fact's neurons fire at
    _ = @import("families/nemotron/pocket_fit.zig"); // where a fact's neurons fire, and stay silent
    _ = @import("families/nemotron/router.zig"); // when a lesson acts on a reply
    _ = @import("families/nemotron/lesson_route.zig");
    _ = @import("families/nemotron/learner.zig");
    _ = @import("families/flashnext/tp_settings.zig"); // speed-up settings read without a typed JSON parse
    _ = @import("families/flashnext/follow.zig"); // rank 1's reply hash
    _ = @import("families/flashnext/dense.zig"); // The core projection adapter keeps format, cuts and tiles.
    _ = @import("families/flashnext/marks.zig"); // a call's marks
    _ = @import("families/flashnext/think.zig"); // a thinking budget's close on the host
    _ = @import("families/flashnext/think_check.zig"); // and in the verdict kernels (Metal)
    _ = @import("families/flashnext/sample.zig"); // a sampled window's draws on the GPU (Metal)
    _ = @import("families/glm/glm.zig");
    _ = @import("families/qwen3_5/config.zig");
    _ = @import("families/qwen3_5/backend.zig");
    _ = @import("families/lfm2_moe/config.zig");
    _ = @import("families/lfm2_moe/backend.zig");
    _ = @import("families/lfm2_moe/state.zig");
}

pub const lane_projection = @import("core/lane_projection.zig");

pub const recurrent_forks = @import("core/recurrent_forks.zig");

pub const shared_kv = @import("core/shared_kv.zig");

pub const shared_attention = @import("core/shared_attention.zig");

pub const tile_qmm = @import("core/tile_qmm.zig");

pub const gpu_profile = @import("core/gpu_profile.zig");

pub const draft_ops = @import("draft_ops");

pub const affine4 = @import("core/affine4.zig");
pub const affine4_lane = @import("core/affine4_lane.zig");

pub const tree_round = @import("tree_round");
pub const tree_round_gpu = @import("tree_round_gpu");

pub const bf16_topk = @import("bf16_topk");
pub const bf16_topk_gpu = @import("bf16_topk_gpu");

pub const tree_commit_gpu = @import("tree_commit_gpu");
