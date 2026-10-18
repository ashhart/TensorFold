//! The common EXL3 FP32 entry points refuse missing kernels and unsupported row strides before launch.
const std = @import("std");
const common = @import("qlinear.zig");
const exl3 = @import("exl3.zig");
const Weight = common.Weight;
const Linear = common.Linear;
test "EXL3 FP32 entries check loading and stride before launching" {
    const w: Weight = .{ .exl3 = .{ .trellis = 0, .suh = 0, .svh = 0, .k = 128, .n = 128, .halves = 8, .codebook = .mul1, .stride_k = 512, .stride_nb = 256 } };
    const unloaded: Linear = .{};
    try std.testing.expectError(error.FormatNotLoaded, unloaded.decodeF32(undefined, w, .{ .x = 0 }, 0, 1));
    try std.testing.expectError(error.FormatNotLoaded, unloaded.promptF32(undefined, w, .{ .x = 0 }, 0, 1));
    var dense: exl3.Dense = undefined;
    const loaded: Linear = .{ .exl3 = &dense };
    try std.testing.expectError(error.RowStride, loaded.decodeF32(undefined, w, .{ .x = 0, .ldx = 127 }, 0, 1));
    try std.testing.expectError(error.RowStride, loaded.promptF32(undefined, w, .{ .x = 0, .ldx = 129 }, 0, 1));
}
