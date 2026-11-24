//! REV26: TF_QWEN_PDL changes only the launch attribute; the SASS rule covers the glue kernels' waits.
const std = @import("std");
const pdl = @import("cuda_pdl.zig");

test "TF_QWEN_PDL changes only the dependent-launch attribute" {
    const base: @import("cuda").Config = .{ .grid = .{ .x = 160, .y = 1, .z = 2 }, .block = .{ .x = 128 }, .shared = 8192 };
    const plain = pdl.config(base, false);
    const dependent = pdl.config(base, true);
    try std.testing.expect(!plain.pdl and dependent.pdl);
    try std.testing.expectEqual(plain.grid.x, dependent.grid.x);
    try std.testing.expectEqual(plain.grid.z, dependent.grid.z);
    try std.testing.expectEqual(plain.block.x, dependent.block.x);
    try std.testing.expectEqual(plain.shared, dependent.shared);
}
