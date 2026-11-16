//! ``--compact-at`` given twice: each occurrence replaces both of its fields, so the last one wins.
const std = @import("std");
const cli = @import("cli.zig");

test "--compact-at: a later occurrence replaces an earlier one in full" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: cli.Usage = .{};
    const fraction_last = try cli.parse(a, &.{ "m", "--compact-at", "auto", "--compact-at=0.5" }, &u);
    try std.testing.expect(!fraction_last.compact_auto);
    try std.testing.expectEqual(@as(?f64, 0.5), fraction_last.compact_fraction);
    const auto_last = try cli.parse(a, &.{ "m", "--compact-at=0.5", "--compact-at", "auto" }, &u);
    try std.testing.expect(auto_last.compact_auto);
    try std.testing.expectEqual(@as(?f64, null), auto_last.compact_fraction);
    const auto = try cli.parse(a, &.{ "m", "--compact-at", "auto" }, &u);
    try std.testing.expect(auto.compact_auto and auto.compact_fraction == null);
    const fraction = try cli.parse(a, &.{ "m", "--compact-at", "0.25" }, &u);
    try std.testing.expect(!fraction.compact_auto);
    try std.testing.expectEqual(@as(?f64, 0.25), fraction.compact_fraction);
    try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--compact-at", "1.5" }, &u));
    try std.testing.expectEqualStrings("argument --compact-at: expected auto or a fraction in (0, 1]: '1.5'", u.message);
}
