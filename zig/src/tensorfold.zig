//! The native engine: model loading, families and the lane core, over our Metal runtime.
const std = @import("std");

pub const safetensors = @import("core/safetensors.zig");
pub const Checkpoint = checkpoint_host.Checkpoint;
pub const checkpoint_host = @import("core/checkpoint.zig");
pub const checkpoint_metal = checkpoint;
pub const qwen27 = @import("families/qwen27/qwen27.zig");
pub const checkpoint = @import("core/checkpoint_metal.zig");
pub const shard_edit = @import("core/shard_edit.zig");
pub const npy = @import("core/npy.zig");
pub const ids_json = @import("core/ids_json.zig");
pub const lanes = @import("lanes");
pub const segments = @import("core/segments.zig");
pub const row_projection = @import("core/row_projection.zig");
pub const nemotron = @import("families/nemotron/nemotron.zig");
pub const flashnext_replay = @import("families/flashnext/replay.zig");
pub const flashnext_engine = @import("families/flashnext/engine.zig");
pub const flashnext_snapshot = @import("families/flashnext/snapshot.zig");
pub const flashnext_learned = @import("families/flashnext/learned.zig");
pub const glm = @import("families/glm/glm.zig");
pub const qwen35 = @import("families/qwen3_5/qwen3_5.zig");
pub const flashnext_pack = @import("families/flashnext/pack.zig");

test {
    std.testing.refAllDecls(@This());
    _ = @import("families/nemotron/prefill_kernels.zig"); // Compile the prompt-chunk sources in tests.
    _ = @import("families/nemotron/simd_attention.zig"); // The pre-M5 attention rewrite finds its lines.
    _ = @import("families/nemotron/weights.zig");
    _ = @import("families/nemotron/kernels.zig");
    _ = @import("families/nemotron/subspace.zig"); // a new block's directions
    _ = @import("families/flashnext/tp_settings.zig"); // speed-up settings read without a typed JSON parse
    _ = @import("families/flashnext/follow.zig"); // rank 1's reply hash
    _ = @import("families/flashnext/dense.zig"); // The core projection adapter keeps format, cuts and tiles.
    _ = @import("families/flashnext/marks.zig"); // a call's marks
    _ = @import("families/glm/glm.zig");
    _ = @import("families/qwen3_5/config.zig");
    _ = @import("families/qwen3_5/backend.zig");
}

pub const lane_projection = @import("core/lane_projection.zig");

pub const recurrent_forks = @import("core/recurrent_forks.zig");

pub const shared_kv = @import("core/shared_kv.zig");

pub const shared_attention = @import("core/shared_attention.zig");

pub const gpu_profile = @import("core/gpu_profile.zig");

pub const draft_ops = @import("draft_ops");

pub const affine4 = @import("core/affine4.zig");
pub const affine4_lane = @import("core/affine4_lane.zig");

pub const tree_round = @import("tree_round");
pub const tree_round_gpu = @import("tree_round_gpu");

pub const bf16_topk = @import("bf16_topk");
pub const bf16_topk_gpu = @import("bf16_topk_gpu");

pub const tree_commit_gpu = @import("tree_commit_gpu");
