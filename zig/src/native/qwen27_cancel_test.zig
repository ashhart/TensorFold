//! A cancelled reply may retain only a completed prompt recurrence, never its advanced live state.
const std = @import("std");
const rules = @import("qwen27_resume.zig");
const t = std.testing;

test "qwen27 resume: cancellation keeps the landed prefix or a completed history rewind" {
    try t.expectEqual(@as(u32, 0), rules.cancelPoint(0, 0, 0));
    try t.expectEqual(@as(u32, 4), rules.cancelPoint(4, 4, 2));
    try t.expectEqual(@as(u32, 8), rules.cancelPoint(8, 8, 6));
    try t.expectEqual(@as(u32, 6), rules.cancelPoint(12, 8, 6));
    try t.expectEqual(@as(u32, 8), rules.cancelPoint(12, 8, 8));
    try t.expectEqual(@as(u32, 0), rules.cancelPoint(12, 8, 9));
    try t.expectEqual(@as(u32, 0), rules.cancelPoint(12, 8, 0));
}

test "qwen27 resume: a rewound cancelled prompt admits only matching tokens and spans" {
    var state: rules.Standing = .{ .rewind_at = 4 };
    defer state.deinit(t.allocator);
    const context = [_]u32{ 1, 2, 3, 4, 20, 21, 22, 23 };
    const at = rules.cancelPoint(7, 5, state.rewind_at);
    try t.expectEqual(@as(u32, 4), at);
    state.leave(t.allocator, .{ .at = at, .prompt_len = 5, .context_len = 8, .fill_at = at, .ended = false }, &context, &.{});
    const next = [_]u32{ 1, 2, 3, 4, 30, 31 };
    try t.expectEqual(rules.Resume{ .here = 4 }, state.plan(&next, &.{}, 0));
    try t.expectEqual(@as(rules.Resume, .none), state.plan(&next, &.{}, 4));
    try t.expectEqual(@as(rules.Resume, .none), state.plan(&.{ 1, 2, 9, 4, 30 }, &.{}, 0));
    try t.expectEqual(@as(rules.Resume, .none), state.plan(&next, &.{.{ 2, 3 }}, 0));
}
