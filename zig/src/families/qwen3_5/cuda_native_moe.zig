//! The qwen3_5_moe model type (Qwen3.6-35B-A3B) on the same CUDA engine: routed experts in place of the dense MLP.
const native = @import("cuda_native.zig");

pub const model_type = "qwen3_5_moe";
pub const formats = native.formats;
pub const default_context = native.default_context;
pub const max_segments = native.max_segments;
pub const prompt_rows = native.prompt_rows;
/// --drafter names the checkpoint's own directory: its MTP layer (mtp-4bit.safetensors) drafts chains.
pub const takes_drafter = true;
pub const Options = native.Options;
pub const open = native.open;
pub const openDrafter = native.openMtp;
pub const explain = native.explain;
