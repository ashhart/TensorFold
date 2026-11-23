//! Inkling 2 Mini loop launches (zig/tests/contract/metal_inkling2_loop.zig).
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "metal inkling2 loop messages tf_inkling2_loop_messages", .threads = 192, .tg_memory = 96, .reserved = 192 },
    .{ .name = "metal inkling2 loop gates tf_inkling2_loop_gates", .threads = 192, .tg_memory = 48, .reserved = 192 },
};
