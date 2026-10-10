//! Canonical decoded intervals are ordered, nonempty and strictly separated.
const std = @import("std");
/// A half-open decoded-row interval.
pub const Span = [2]u32;

/// Reject empty, reversed, overlapping, adjacent or unsorted intervals; the empty list is valid.
pub fn validate(spans: []const Span) error{InvalidSpans}!void {
    var previous: u32 = 0;
    for (spans, 0..) |span, i| {
        if (span[0] >= span[1] or (i > 0 and span[0] <= previous)) return error.InvalidSpans;
        previous = span[1];
    }
}

/// Compare canonical interval lists in linear time, clipped to consumed rows before at.
pub fn equal(a: []const Span, b: []const Span, at: u32) bool {
    var i: usize = 0;
    while (true) : (i += 1) {
        const left = i < a.len and a[i][0] < at;
        const right = i < b.len and b[i][0] < at;
        if (!left or !right) return left == right;
        if (a[i][0] != b[i][0] or @min(a[i][1], at) != @min(b[i][1], at)) return false;
    }
}

/// Copy a canonical map through at, without retaining later rows or borrowed storage.
pub fn prefix(a: std.mem.Allocator, spans: []const Span, at: u32) ![]Span {
    var n: usize = 0;
    while (n < spans.len and spans[n][0] < at) : (n += 1) {}
    const out = try a.dupe(Span, spans[0..n]);
    if (n > 0) out[n - 1][1] = @min(out[n - 1][1], at);
    return out;
}

/// Clip prompt modes, then append consumed reply rows and coalesce the one possible adjacent interval.
pub fn reply(a: std.mem.Allocator, spans: []const Span, prompt: u32, stands: u32) ![]Span {
    std.debug.assert(stands >= prompt);
    var n: usize = 0;
    while (n < spans.len and spans[n][0] < prompt) : (n += 1) {}
    const merge = n > 0 and spans[n - 1][1] >= prompt;
    const append = stands > prompt and !merge;
    const out = try a.alloc(Span, n + @intFromBool(append));
    @memcpy(out[0..n], spans[0..n]);
    if (n > 0) out[n - 1][1] = if (merge) stands else @min(out[n - 1][1], prompt);
    if (append) out[n] = .{ prompt, stands };
    return out;
}

test "canonical spans reject alternate spellings before storage" {
    try validate(&.{});
    try validate(&.{ .{ 0, 2 }, .{ 3, 100 } });
    const bad = [_][]const Span{ &.{.{ 2, 2 }}, &.{.{ 3, 2 }}, &.{ .{ 1, 4 }, .{ 3, 5 } }, &.{ .{ 1, 4 }, .{ 4, 5 } }, &.{ .{ 7, 8 }, .{ 1, 3 } } };
    for (bad) |spans| try std.testing.expectError(error.InvalidSpans, validate(spans));
}

test "canonical equality and owned prefixes ignore future rows and lookahead" {
    const spans = [_]Span{ .{ 2, 10 }, .{ 20, 30 } };
    const got = try prefix(std.testing.allocator, &spans, 7);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualDeep(@as([]const Span, &.{.{ 2, 7 }}), got);
    try std.testing.expect(equal(&spans, &.{.{ 2, 99 }}, 7));
    try std.testing.expect(!equal(&spans, &.{.{ 2, 6 }}, 7));
    try std.testing.expect(equal(&.{.{ 7, 20 }}, &.{}, 7));
    try std.testing.expect(equal(&.{.{ 0, 1 }}, &.{}, 0));
}

test "reply extension clips future prompt modes and coalesces only its boundary" {
    for ([_]u32{ 10, 11, 13 }) |stands| {
        const merged = try reply(std.testing.allocator, &.{ .{ 2, 10 }, .{ 12, 30 } }, 10, stands);
        defer std.testing.allocator.free(merged);
        try validate(merged);
        try std.testing.expectEqualDeep(@as([]const Span, &.{.{ 2, stands }}), merged);
        const extended = try reply(std.testing.allocator, &.{ .{ 2, 5 }, .{ 10, 30 } }, 10, stands);
        defer std.testing.allocator.free(extended);
        try validate(extended);
        try std.testing.expectEqual(@as(usize, if (stands == 10) 1 else 2), extended.len);
        if (stands > 10) try std.testing.expectEqualDeep(Span{ 10, stands }, extended[1]);
    }
}
