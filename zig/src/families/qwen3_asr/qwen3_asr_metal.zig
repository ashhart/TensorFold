//! Qwen3-ASR on Metal: metal/ binds the shared schedules (tower.zig, lm.zig); tf-qwen3-asr-metal drives it.
pub const host = @import("qwen3_asr.zig");
pub const Model = @import("metal/model.zig").Model;
pub const Gpu = @import("metal/model.zig").Gpu;
pub const Options = @import("metal/model.zig").Options;
pub const Dump = @import("metal/model.zig").Dump;
pub const Splice = @import("metal/model.zig").Splice;

test {
    _ = @import("metal/model.zig");
}
