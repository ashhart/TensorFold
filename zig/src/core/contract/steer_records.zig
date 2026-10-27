//! The steering launch's threads and threadgroup memory, for gate 5's limits.
const std = @import("std");
const limits = @import("limits.zig");
const steer = @import("steer.zig");

/// The launch as the engine makes it: a threadgroup a row, the fused norm's threads, its two sets of partials.
pub const metal = [_]limits.MetalLaunch{
    .{
        .name = "qwen27 norm steer custom_kernel_q27_norm_steer_bfloat16_t_bfloat16_t_bfloat16_t_floatc_int32_t_bfloat16_t_bfloat16_t_float_float_float_uint",
        .threads = steer.threads,
        .tg_memory = steer.group_memory,
        .reserved = steer.threads,
    },
};

test "the steering launch fits every Apple family" {
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    for (metal) |l| for (limits.apple) |f| try std.testing.expectEqual(@as(usize, 0), try limits.metal(l, f, &out, a));
}
