//! Backend-neutral engine parts: checkpoint files, the CUDA draft-depth rule, the copy index, the tokenizer.

pub const safetensors = @import("safetensors.zig");
pub const exl3_format = @import("exl3_format.zig");
pub const exlr8 = @import("exlr8.zig");
pub const eaq_container = @import("eaq_container.zig");
pub const checkpoint = @import("checkpoint.zig");
pub const exl3_load = @import("exl3_load.zig");
pub const direct_io = @import("direct_io.zig");
pub const Checkpoint = checkpoint.Checkpoint;
pub const draft_depth = @import("draft_depth.zig");
pub const CopyIndex = @import("copy_index.zig").CopyIndex;
pub const tokenizer = @import("tokenizer"); // a module of its own, so the native server shares it
pub const ids_json = @import("ids_json.zig");
pub const affine4_host = @import("affine4_host.zig");
pub const affine4_fold = @import("affine4_fold.zig");
pub const shard_edit = @import("shard_edit.zig");
pub const lora_merge = @import("lora_merge_dir.zig");
pub const quant = @import("quant/quant.zig");
pub const nvfp4_rows = @import("nvfp4_rows.zig");
pub const route_ref = @import("route_ref.zig");
pub const nvfp4_csf = @import("nvfp4_csf.zig");
pub const nvfp4_slice = @import("nvfp4_slice.zig");
pub const registry = @import("registry.zig");
pub const tuning = @import("tuning.zig");
pub const norms_ref = @import("norms_ref.zig");
pub const act_ref = @import("act_ref.zig");
/// The kernel contract every GPU kernel's test registers with (zig/tests/contract, tf-cuda-test kernel-contract).
pub const contract = @import("contract.zig");
pub const nvfp4 = @import("nvfp4.zig");
pub const modelopt = @import("modelopt.zig");
/// Pictures as 0.6.6 loaded them: PNG, JPEG through the pinned libjpeg-turbo, orientation, the shared bicubic resizes.
pub const picture = @import("picture/picture.zig");
/// Qwen-VL image input every Qwen3.5-VL checkpoint shares: settings, geometry, M-RoPE positions, the tower's tables.
pub const qwen_vl = @import("qwen_vl/qwen_vl.zig");
pub const drafter_release = @import("drafter_release.zig");
pub const drafter_layout = @import("drafter_layout.zig");
pub const tree_round = @import("tree_round.zig");
pub const attention_ref = @import("attention_ref.zig");
pub const gdn_prep_ref = @import("gdn_prep_ref.zig");
pub const qk_ref = @import("qk_ref.zig");
pub const moe_ref = @import("moe_ref.zig");
pub const dense_ref = @import("dense_ref.zig");

pub const checkpoint_weights = @import("checkpoint_weights.zig");

test {
    _ = checkpoint_weights;
    _ = safetensors;
    _ = affine4_host;
    _ = affine4_fold;
    _ = shard_edit;
    _ = lora_merge;
    _ = quant;
    _ = nvfp4_rows;
    _ = route_ref;
    _ = nvfp4_csf;
    _ = nvfp4_slice;
    _ = registry;
    _ = tuning;
    _ = norms_ref;
    _ = act_ref;
    _ = contract;
    _ = tree_round;
    _ = attention_ref;
    _ = gdn_prep_ref;
    _ = qk_ref;
    _ = moe_ref;
    _ = dense_ref;
    _ = exl3_format;
    _ = exlr8;
    _ = eaq_container;
    _ = @import("eaq_container_test.zig");
    _ = exl3_load;
    _ = @import("exl3_rect_test.zig");
    _ = checkpoint;
    _ = direct_io;
    _ = draft_depth;
    _ = @import("copy_index.zig");
    _ = ids_json;
    _ = nvfp4;
    _ = modelopt;
    _ = picture;
    _ = qwen_vl;
    _ = @import("nvfp4_fixture_test.zig");
    _ = @import("drafter_release_test.zig");
}
