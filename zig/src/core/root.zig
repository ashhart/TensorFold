//! Backend-neutral engine parts: checkpoint files, the CUDA draft-depth rule, the copy index, the tokenizer.

pub const safetensors = @import("safetensors.zig");
pub const exl3_format = @import("exl3_format.zig");
pub const exlr8 = @import("exlr8.zig");
pub const checkpoint = @import("checkpoint.zig");
pub const direct_io = @import("direct_io.zig");
pub const Checkpoint = checkpoint.Checkpoint;
pub const draft_depth = @import("draft_depth.zig");
pub const CopyIndex = @import("copy_index.zig").CopyIndex;
pub const tokenizer = @import("tokenizer"); // a module of its own, so the native server shares it
pub const ids_json = @import("ids_json.zig");
pub const affine4_host = @import("affine4_host.zig");
pub const affine4_fold = @import("affine4_fold.zig");
pub const shard_edit = @import("shard_edit.zig");
pub const quant = @import("quant/quant.zig");
pub const registry = @import("registry.zig");
pub const tuning = @import("tuning.zig");
pub const norms_ref = @import("norms_ref.zig");
pub const act_ref = @import("act_ref.zig");
/// The kernel contract every GPU kernel's test registers with (zig/tests/contract, tf-cuda-test kernel-contract).
pub const contract = @import("contract.zig");
pub const nvfp4 = @import("nvfp4.zig");
pub const modelopt = @import("modelopt.zig");

test {
    _ = safetensors;
    _ = affine4_host;
    _ = affine4_fold;
    _ = shard_edit;
    _ = quant;
    _ = registry;
    _ = tuning;
    _ = norms_ref;
    _ = act_ref;
    _ = contract;
    _ = exl3_format;
    _ = exlr8;
    _ = @import("exl3_rect_test.zig");
    _ = checkpoint;
    _ = direct_io;
    _ = draft_depth;
    _ = @import("copy_index.zig");
    _ = ids_json;
    _ = nvfp4;
    _ = modelopt;
    _ = @import("nvfp4_fixture_test.zig");
}
