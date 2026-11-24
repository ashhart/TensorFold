//! Inkling 2 Mini loop launches (zig/tests/contract/metal_inkling2_loop.zig).
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "metal inkling2 loop messages tf_inkling2_loop_messages", .threads = 256, .tg_memory = 144, .reserved = 256 },
    .{ .name = "metal inkling2 loop gates tf_inkling2_loop_gates", .threads = 256, .tg_memory = 9552, .reserved = 256 },
};
