//! Qwen3-ASR on the Intel GPU: xpu/ binds the shared schedules (tower.zig, lm.zig); tf-qwen3-asr-xpu drives it.
pub const host = @import("qwen3_asr.zig");
pub const Model = @import("xpu/model.zig").Model;
pub const Options = @import("xpu/model.zig").Options;
pub const Dump = @import("xpu/model.zig").Dump;
pub const Splice = @import("xpu/model.zig").Splice;
pub const view = @import("xpu/store.zig").view;

test {
    _ = @import("xpu/model.zig");
}
