//! The native engine: model loading, families and the lane core, over our Metal runtime.
const std = @import("std");

pub const checkpoint = @import("core/checkpoint_metal.zig");
pub const npy = @import("core/npy.zig");
pub const ids_json = @import("core/ids_json.zig");
pub const lanes = @import("lanes");
pub const segments = @import("core/segments.zig");
pub const nemotron = @import("families/nemotron/nemotron.zig");
pub const flashnext_replay = @import("families/flashnext/replay.zig");
pub const flashnext_engine = @import("families/flashnext/engine.zig");
pub const flashnext_snapshot = @import("families/flashnext/snapshot.zig");

test {
    std.testing.refAllDecls(@This());
    _ = @import("families/nemotron/prefill_kernels.zig"); // Compile the prompt-chunk sources in tests.
    _ = @import("families/nemotron/simd_attention.zig"); // The pre-M5 attention rewrite finds its lines.
    _ = @import("families/nemotron/weights.zig");
    _ = @import("families/nemotron/kernels.zig");
    _ = @import("families/flashnext/tp_settings.zig"); // speed-up settings read without a typed JSON parse
    _ = @import("families/flashnext/follow.zig"); // rank 1's reply hash
    _ = @import("families/flashnext/marks.zig"); // a call's marks
}
