//! A wrong pair ordinal must fail the exact integer-table comparison.
const std = @import("std");
const src = @import("inkling2_metal");
const from = "members[raw * a.cap + pos] = i;";
const to = "members[raw * a.cap + pos] = a.pairs - 1u - i;";

pub fn applyTo(a: std.mem.Allocator, text: []const u8) ![]u8 {
    if (std.mem.count(u8, text, from) != 1) return error.ControlEditCount;
    return std.mem.replaceOwned(u8, a, text, from, to);
}

pub fn apply(a: std.mem.Allocator) ![]u8 {
    return applyTo(a, src.experts_mma);
}

test "wide grouping ordinal control applies exactly once and refuses ambiguous anchors" {
    const a = std.testing.allocator;
    const edited = try apply(a);
    defer a.free(edited);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, edited, to));
    try std.testing.expectError(error.ControlEditCount, applyTo(a, edited));
    const doubled = try std.mem.concat(a, u8, &.{ src.experts_mma, from });
    defer a.free(doubled);
    try std.testing.expectError(error.ControlEditCount, applyTo(a, doubled));
}
