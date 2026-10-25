//! Batch preflight and host publication exercise the production batch owner before any device boundary.
const std = @import("std");
const T = std.testing;
const S = struct {
    pos: usize = 0,
    rows: usize = 0,
    cap: usize = 32,
    phase: enum { ready, working, verified, poisoned } = .ready,
    marked: bool = false,
    var writes: usize = 0;
    pub fn layers(_: *const S) usize {
        return 2;
    }
    pub fn canBegin(s: *const S, n: usize) !void {
        if (s.phase != .ready or n > s.cap - s.pos) return error.Refused;
    }
    pub fn begin(s: *S, n: usize) !void {
        try s.canBegin(n);
        writes += 1;
        s.rows = n;
        s.phase = .working;
    }
    pub fn admit(s: *const S, n: usize) !void {
        if (s.phase != .working or n != s.rows) return error.Refused;
    }
    pub fn poison(s: *S) void {
        s.phase = .poisoned;
    }
    pub fn captured(s: *S, layer: usize) !void {
        if (layer != 0 or s.phase != .working) return error.Refused;
        s.marked = true;
    }
    pub fn advanced(s: *S, tokens: []const u32) !void {
        try s.admit(tokens.len);
        if (!s.marked) return error.Unmarked;
        s.pos += tokens.len;
        s.phase = .verified;
    }
};
test "Qwen4 batch: every stream is admitted before the first write, duplicates refuse and token slices stay owned" {
    const B = @import("batch_state.zig").Batch(S);
    var a: S = .{};
    var b: S = .{ .cap = 1 };
    var batch: B = .{ .items = &.{ .{ .state = &a, .tokens = &.{ 1, 2 } }, .{ .state = &b, .tokens = &.{ 3, 4, 5 } } } };
    S.writes = 0;
    try T.expectError(error.Refused, batch.begin());
    try T.expectEqual(@as(usize, 0), S.writes);
    b.cap = 32;
    try batch.begin();
    try T.expectEqual(@as(usize, 2), S.writes);
    try batch.admit(5);
    try T.expectError(error.Qwen4BatchRows, batch.admit(4));
    try batch.captured(0);
    try T.expectError(error.Qwen4BatchTokens, batch.advanced(&.{ 1, 2, 3, 4, 6 }));
    try T.expectEqual(@as(usize, 0), a.pos);
    try T.expectEqual(@as(usize, 0), b.pos);
    try batch.advanced(&.{ 1, 2, 3, 4, 5 });
    try T.expectEqual(@as(usize, 2), a.pos);
    try T.expectEqual(@as(usize, 3), b.pos);
    var duplicate: B = .{ .items = &.{ .{ .state = &a, .tokens = &.{1} }, .{ .state = &a, .tokens = &.{2} } } };
    try T.expectError(error.Qwen4DuplicateStream, duplicate.count());
    batch.poison();
    try T.expectEqual(.poisoned, a.phase);
    try T.expectEqual(.poisoned, b.phase);
}
