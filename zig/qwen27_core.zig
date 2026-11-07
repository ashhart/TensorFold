//! Qwen's view of shared checkpoint contracts and native tensor storage.
pub const checkpoint = @import("src/core/checkpoint.zig");
pub const gguf = @import("src/core/gguf.zig");
pub const checkpoint_metal = @import("src/core/checkpoint_metal.zig");
pub const safetensors = @import("src/core/safetensors.zig");
pub const Checkpoint = checkpoint.Checkpoint;
pub const npy = @import("src/core/npy.zig");
pub const tokenizer = @import("tokenizer");
/// Qwen-VL image input's backend-neutral parts (M-RoPE positions for the forward).
pub const qwen_vl = @import("src/core/qwen_vl/qwen_vl.zig");

pub const segments = @import("src/core/segments.zig");

pub const checkpoint_host = checkpoint;
pub const lane_projection = @import("core_lane");

/// A checkpoint's identity digest (config, index and shard headers), so two ranks can compare theirs.
pub const checkpoint_identity = @import("src/core/checkpoint_identity.zig");

pub const gpu_profile = @import("src/core/gpu_profile.zig");
/// Startup phase timing, as core/root.zig exports it for the served engine (TF_STARTUP_RECEIPT).
pub const startup_receipt = @import("profile").startup_receipt;

pub const row_projection = @import("core_row");
/// The tensor units' affine products by expert (the routed block's prompt path).
pub const affine_mm = @import("core_lane").affine_mm;
pub const quant = @import("src/core/quant/quant.zig");

pub const draft_ops = @import("draft_ops");
pub const shared_attention = @import("src/core/shared_attention.zig");
/// The core pool and its Metal pages, which TF_QWEN27_METAL_KV_PAGES keeps the 27B's keys and values in.
pub const kv_pool = @import("src/core/kv_pool.zig");
pub const metal_kv_pages = @import("src/core/metal_kv_pages.zig");

pub const affine4 = @import("src/core/affine4.zig");
pub const affine4_lane = @import("src/core/affine4_lane.zig");

pub const tree_round = @import("tree_round");
pub const tree_round_gpu = @import("tree_round_gpu");

pub const bf16_topk = @import("bf16_topk");
pub const bf16_topk_gpu = @import("bf16_topk_gpu");

pub const tree_commit_gpu = @import("tree_commit_gpu");

pub const tree_slots = @import("src/core/tree_slots.zig");
