//! A batch with a paged stream keys its captured graph apart; the kernels read each stream through its own layout.
const std = @import("std");

pub fn hasPages(segs: anytype) bool {
    for (segs) |sg| if (sg.pt != 0) return true;
    return false;
}

test "Qwen attention path: unpaged, paged and mixed segment tables" {
    const Seg = struct { pt: u64 };
    try std.testing.expect(!hasPages(@as([]const Seg, &.{})));
    try std.testing.expect(!hasPages(@as([]const Seg, &.{ .{ .pt = 0 }, .{ .pt = 0 } })));
    try std.testing.expect(hasPages(@as([]const Seg, &.{ .{ .pt = 0x4000 }, .{ .pt = 0 } })));
    try std.testing.expect(hasPages(@as([]const Seg, &.{ .{ .pt = 0 }, .{ .pt = 0x4000 } })));
}

test "Qwen attention path: layout is part of every captured shape key" {
    const Shape = @import("cuda_shapes.zig").Shape;
    for (1..129) |rows| for (1..rows + 1) |segments| for ([_]u8{ 1, 2, 4, 8, 16 }) |slots| {
        var plain = Shape.init(rows, segments, true, false, slots);
        for ([_]bool{ false, true }) |hidden| {
            plain.hidden = hidden;
            var pages = plain;
            pages.paged = true;
            try std.testing.expectEqual(plain.key() | (1 << 28), pages.key());
            try std.testing.expect(plain.key() != pages.key());
        }
    };
}
