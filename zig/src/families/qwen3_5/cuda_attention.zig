//! A batch with a paged stream keys its captured graph apart; the kernels read each stream through its own layout.
const std = @import("std");
const environment = @import("sys").environment;

const all_rows_name = "TF_QWEN35_ATTN_ALL_ROWS";

pub fn parseAllRows(raw: ?[]const u8) !bool {
    return environment.value(bool, all_rows_name, raw, false);
}

pub fn allRows() !bool {
    return environment.read(bool, all_rows_name, false);
}

/// Only a BF16 decode window wider than one row takes the single all-length rows launch.
pub fn rowsOnly(enabled: bool, prompt: bool, total: usize) bool {
    return enabled and !prompt and total > 1;
}

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

test "Qwen attention path: all-rows flag is strict and defaults off" {
    for ([_]?[]const u8{ null, "", " ", "\t", " \t\r\n" }) |raw| try std.testing.expect(!try parseAllRows(raw));
    try std.testing.expect(!try parseAllRows("0"));
    try std.testing.expect(try parseAllRows("1"));
    for ([_][]const u8{ "2", "yes", "on", "true", "00", "1 " }) |raw| try std.testing.expectError(error.InvalidEnvironment, parseAllRows(raw));
}

test "Qwen attention path: all-rows opt-in is limited to multi-row decode windows" {
    for ([_]bool{ false, true }) |prompt| for ([_]usize{ 1, 4, 8, 16 }) |total| {
        try std.testing.expect(!rowsOnly(false, prompt, total));
        try std.testing.expectEqual(!prompt and total > 1, rowsOnly(true, prompt, total));
    };
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
