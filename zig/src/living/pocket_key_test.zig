//! Key selection and row roles through the generic pocket.
const std = @import("std");
const pocket = @import("pocket.zig");
const Example = @import("lesson_examples.zig").Example;
const Kind = @import("pocket_fit.zig").Kind;
const kindFor = pocket.kindFor;
test "a fact keys as many neurons on its subject as the share of its questions that name it, at least half, else none" {
    const P = pocket.Pocket(struct {}, @import("families.zig").nemotron_h);
    var p: P = .{ .gpa = std.testing.allocator, .mode = .auto };
    defer p.deinit();
    // ids: a question's tokens, the subject's last token at row 2, the answer from row 4
    const ids = [_]u32{ 1, 2, 7, 3, 4, 5 };
    const named: Example = .{ .ids = &ids, .start = 4, .decide = 3, .subject = 2 };
    const plain: Example = .{ .ids = &ids, .start = 4, .decide = 3 };
    try std.testing.expectEqual([3]usize{ 5, 5, 0 }, p.pickSplit(&.{ &.{ named, plain }, &.{} }, &.{}));
    try std.testing.expectEqual([3]usize{ 0, 10, 0 }, p.pickSplit(&.{ &.{ named, named }, &.{named} }, &.{}));
    try std.testing.expectEqual([3]usize{ 3, 7, 0 }, p.pickSplit(&.{&.{ named, named, plain }}, &.{}));
    try std.testing.expectEqual([3]usize{ 10, 0, 0 }, p.pickSplit(&.{&.{ named, plain, plain }}, &.{}));
    try std.testing.expectEqual([3]usize{ 10, 0, 0 }, p.pickSplit(&.{&.{}}, &.{}));
    // answers whose value comes after row 4: none there unless asked, then four, the rest shared as before
    const valued: Example = .{ .ids = &ids, .start = 4, .decide = 3, .subject = 2, .value = 4 };
    try std.testing.expectEqual([3]usize{ 0, 10, 0 }, p.pickSplit(&.{&.{ valued, valued }}, &.{ valued, valued }));
    p.valuing = true;
    try std.testing.expectEqual([3]usize{ 0, 6, 4 }, p.pickSplit(&.{&.{ valued, valued }}, &.{ valued, valued }));
    p.mode = .subject;
    try std.testing.expectEqual([3]usize{ 0, 10, 0 }, p.pickSplit(&.{&.{ named, plain, plain }}, &.{}));
    p.mode = .reply;
    try std.testing.expectEqual([3]usize{ 10, 0, 0 }, p.pickSplit(&.{&.{ named, named }}, &.{}));
}

test "a read row is a fact's row only to its own key's neurons; another prompt's mention of the subject is left free" {
    try std.testing.expectEqual(Kind.fact, kindFor(.{ .kind = .fact, .key = .subject, .ex = 0, .row = 2 }, .subject));
    try std.testing.expectEqual(Kind.other, kindFor(.{ .kind = .fact, .key = .subject, .ex = 0, .row = 2 }, .reply));
    try std.testing.expectEqual(Kind.held, kindFor(.{ .kind = .held, .key = .reply, .ex = 3, .row = 7 }, .reply));
    try std.testing.expectEqual(Kind.steady, kindFor(.{ .kind = .steady, .key = .reply, .ex = 1, .row = 9 }, .subject));
    try std.testing.expectEqual(Kind.none, kindFor(.{ .kind = .none, .key = .subject, .ex = 1, .row = 4 }, .subject));
    try std.testing.expectEqual(Kind.other, kindFor(.{ .kind = .none, .key = .subject, .ex = 1, .row = 4 }, .reply));
    try std.testing.expectEqual(Kind.other, kindFor(.{ .kind = .other, .key = null, .ex = 1, .row = 5 }, .subject));
}
