//! Static threadgroup storage in kernels/metal/inkling2/loop.metal.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "inkling2 loop inkling2_loop_mix", .threads = 256, .tg_memory = std.mem.alignForward(u32, 4 * (8 * 4 * 9 + 9), 16), .reserved = 256 },
    .{ .name = "inkling2 loop inkling2_loop_gate_down", .threads = 256, .tg_memory = 4 * (8 + 8 * 128), .reserved = 256 },
    .{ .name = "inkling2 loop inkling2_loop_gate_mix", .threads = 256, .tg_memory = 4 * (8 + 128), .reserved = 256 },
};
