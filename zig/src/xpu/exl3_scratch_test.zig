// SPDX-License-Identifier: Apache-2.0
const std = @import("std");
const exl3 = @import("exl3.zig");

test "descriptor layout and scratch" {
    const m = exl3.Matrix{ .trellis = 0x1000, .suh = 0x2000, .svh = 0x3000, .k = 2048, .n = 512, .halves = 8 };
    const d = try exl3.descriptor(.{ .m = m, .rows = 3, .x = 0x4000, .x_stride = 2048, .y = 0x5000, .y_stride = 512, .output = .f32, .scratch = 0x10000 });
    try std.testing.expectEqual(@as(u64, 0x10000 + 2048 * 3 * 4), d.sums);
    try std.testing.expectEqual(@as(u32, 1), d.out_type);
    try std.testing.expectEqual(@as(usize, 2048 * 3 * 4 + 16 * 512 * 3 * 4), exl3.scratchBytes(2048, 512, 3));
    try std.testing.expectEqual(@as(usize, 1), exl3.splitBound(248320));
    try std.testing.expectError(error.InvalidExl3Rows, exl3.descriptor(.{ .m = m, .rows = 1025, .x = 0, .x_stride = 2048, .y = 0, .y_stride = 512, .scratch = 0 }));
    try std.testing.expectEqual(@as(usize, 2048 * 512 * 4 + 512 * 512 * 4), exl3.promptScratchBytes(2048, 512, 512));
    try std.testing.expectError(error.UnsupportedExl3Shape, (exl3.Matrix{ .trellis = 0, .suh = 0, .svh = 0, .k = 100, .n = 512, .halves = 8 }).validate());
    try std.testing.expectEqual(@as(usize, 9 * 4 * 2 * 512 * 4 + 9 * 512 * 4), exl3.moeScratchBytes(2048, 512, 9));
}
