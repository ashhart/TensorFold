//! Backend-neutral engine parts: checkpoint files, the CUDA draft-depth rule, the copy index, the tokenizer.

pub const safetensors = @import("safetensors.zig");
pub const checkpoint = @import("checkpoint.zig");
pub const Checkpoint = checkpoint.Checkpoint;
pub const draft_depth = @import("draft_depth.zig");
pub const CopyIndex = @import("copy_index.zig").CopyIndex;
pub const tokenizer = @import("tokenizer"); // a module of its own, so the native server shares it
pub const ids_json = @import("ids_json.zig");

test {
    _ = safetensors;
    _ = checkpoint;
    _ = draft_depth;
    _ = @import("copy_index.zig");
    _ = ids_json;
}
