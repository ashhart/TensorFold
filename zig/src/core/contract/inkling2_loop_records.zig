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
    .{ .name = "inkling2 loop inkling2_loop_gate_y_mma", .threads = 32, .tg_memory = 0, .reserved = 32 },
    // tg_x and tg_w 4 x 8 x 2 x 8 x 4 words each, part 4 x 2 x 2 x 8 x 8 floats.
    .{ .name = "inkling2 loop inkling2_loop_gate_down_mma", .threads = 128, .tg_memory = 2 * 8192 + 4096, .reserved = 128 },
};
