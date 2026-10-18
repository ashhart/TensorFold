//! Key selection uses the same row metadata for dense and routed families.
const std = @import("std");
const pocket = @import("pocket.zig");
const Example = @import("lesson_examples.zig").Example;

pub fn log(p: anytype, lists: []const []const Example, answered: []const Example) void {
    var named: usize = 0;
    var asked: usize = 0;
    var valued: usize = 0;
    for (lists) |xs| for (xs) |ex| {
        asked += 1;
        named += @intFromBool(pocket.subjectRow(ex) != null);
    };
    for (answered) |ex| {
        valued += @intFromBool(pocket.valueRow(ex) != null);
    }
    std.log.info("slide pocket: key mode {t}, subject rows {d}/{d}, value rows {d}/{d}, value neurons {s}", .{ p.mode, named, asked, valued, answered.len, if (p.valuing) "enabled" else "disabled" });
}

test "pocket selection: dense and routed families share valid zero and nonzero key allocations" {
    const families = @import("families.zig");
    const ex: Example = .{ .ids = &.{ 1, 2, 3, 4, 5, 6, 7 }, .start = 4, .decide = 3, .subject = 2, .value = 4 };
    var cold = ex;
    cold.subject = 0;
    inline for (.{ families.nemotron_h, families.qwen4_exp, families.qwen3_5 }) |F| {
        var p = pocket.Pocket(struct {}, F).init(std.testing.allocator);
        defer p.deinit();
        p.mode = .auto;
        p.valuing = false;
        try std.testing.expectEqual([3]usize{ 10, 0, 0 }, p.pickSplit(&.{&.{cold}}, &.{ex}));
        try std.testing.expectEqual([3]usize{ 10, 0, 0 }, p.pickSplit(&.{&.{ ex, cold, cold }}, &.{ex}));
        try std.testing.expectEqual([3]usize{ 5, 5, 0 }, p.pickSplit(&.{&.{ ex, cold }}, &.{ex}));
        try std.testing.expectEqual([3]usize{ 0, 10, 0 }, p.pickSplit(&.{&.{ex}}, &.{ex}));
        p.valuing = true;
        try std.testing.expectEqual([3]usize{ 0, 6, 4 }, p.pickSplit(&.{&.{ex}}, &.{ex}));
        p.mode = .reply;
        try std.testing.expectEqual([3]usize{ 10, 0, 0 }, p.pickSplit(&.{&.{ex}}, &.{ex}));
        p.mode = .both;
        try std.testing.expectEqual([3]usize{ 5, 5, 0 }, p.pickSplit(&.{&.{ex}}, &.{ex}));
    }
}
