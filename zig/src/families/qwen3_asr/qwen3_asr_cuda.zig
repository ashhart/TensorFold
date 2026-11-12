//! Qwen3-ASR on CUDA: cuda/ binds the tower schedule; the Qwen3 layers run qwen3's CUDA pass; tf-qwen3-asr drives it.
pub const host = @import("qwen3_asr.zig");
pub const Model = @import("cuda/model.zig").Model;
pub const Options = @import("cuda/model.zig").Options;
pub const Dump = @import("cuda/model.zig").Dump;
pub const Splice = @import("cuda/model.zig").Splice;

test {
    _ = @import("cuda/model.zig");
}
