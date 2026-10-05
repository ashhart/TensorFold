//! The native engine: model loading, families and the lane core, over our Metal runtime.
const std = @import("std");

pub const checkpoint = @import("core/checkpoint_metal.zig");
pub const npy = @import("core/npy.zig");
pub const ids_json = @import("core/ids_json.zig");
pub const lanes = @import("lanes");
pub const nemotron = @import("families/nemotron/nemotron.zig");
pub const qwen35 = @import("families/qwen3_5/qwen3_5.zig");

test {
    std.testing.refAllDecls(@This());
    _ = @import("families/nemotron/prefill_kernels.zig"); // its sources compile at this macOS's Metal language
    _ = @import("families/qwen3_5/config.zig");
    _ = @import("families/qwen3_5/backend.zig");
}
