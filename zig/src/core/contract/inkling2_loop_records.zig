//! Static threadgroup storage in kernels/metal/inkling2/loop.metal.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "inkling2 loop inkling2_loop_mix", .threads = 256, .tg_memory = std.mem.alignForward(u32, 4 * (8 * 4 * 9 + 9), 16), .reserved = 256 },
    .{ .name = "inkling2 loop inkling2_loop_gate_down", .threads = 256, .tg_memory = 4 * (8 + 8 * 8), .reserved = 256 },
    .{ .name = "inkling2 loop inkling2_loop_gate_mix", .threads = 256, .tg_memory = 4 * (8 + 128), .reserved = 256 },
    .{ .name = "inkling2 loop inkling2_loop_mix_stats", .threads = 256, .tg_memory = 4 * 8 * 5, .reserved = 256 },
    .{ .name = "inkling2 loop inkling2_loop_mix_apply", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 loop inkling2_loop_gate_y", .threads = 256, .tg_memory = 4 * 128, .reserved = 256 },
    .{ .name = "inkling2 loop inkling2_loop_gate_norm", .threads = 256, .tg_memory = 4 * 8, .reserved = 256 },
    .{ .name = "inkling2 loop inkling2_loop_gate_prep", .threads = 256, .tg_memory = 4 * 8, .reserved = 256 },
    .{ .name = "inkling2 loop inkling2_loop_gate_dot", .threads = 256, .tg_memory = 4 * 8 * 2, .reserved = 256 },
};
