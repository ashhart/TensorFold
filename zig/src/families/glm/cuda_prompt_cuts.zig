//! Where a GLM prompt pass cuts: every 2,048 rows, or sooner at request starts; a pass resumes only at a cold cut.
const std = @import("std");
const state = @import("cuda_state.zig");
const prefill_rows = state.prefill_rows;

/// Where a pass from `at` over `len` rows ends its chunk: prefill_rows on, or the next of `cuts` before that.
pub fn nextCut(len: usize, cuts: []const u32, at: usize) usize {
    var end: usize = @min(len, at + prefill_rows);
    for (cuts) |c| if (c > at) {
        end = @min(end, c);
        break;
    };
    return end;
}

/// Whether a cold pass over `len` rows (from 0) starts a chunk at `pos`.
pub fn coldCut(len: usize, cuts: []const u32, pos: usize) bool {
    var at: usize = 0;
    while (at < pos and at < len) at = nextCut(len, cuts, at);
    return at == pos;
}

/// A sequence's request chunk starts (TF_GLM_REQUEST_STARTS) into `list`: ascending and after 0, else error.BadCuts.
pub fn set(gpa: std.mem.Allocator, list: *std.ArrayList(u32), cuts: []const u32) !void {
    for (cuts, 0..) |c, i| if (c == 0 or (i > 0 and c <= cuts[i - 1])) return error.BadCuts;
    list.clearRetainingCapacity();
    try list.appendSlice(gpa, cuts);
}

/// A prompt step's rows from `start` with `part_len` ids: to the next cut, or a solo pass's when `part_len` holds more.
pub fn stepRows(len: usize, cuts: []const u32, start: usize, part_len: usize) !usize {
    if (start >= len or !coldCut(len, cuts, start)) return error.BadChunk;
    if (part_len <= prefill_rows + 1) return nextCut(len, cuts, start) - start;
    if (cuts.len > 0) return error.BadChunk; // a solo pass would step over a request start
    return @min(len - start, state.solo_rows);
}

/// Engine.prefill's chunks from `start`: each to its next cut, until the last row's step returns the first token.
pub fn pass(e: anytype, ids: []const u32, start: usize, cancel: anytype, marks: anytype) !u32 {
    var at = start;
    while (true) {
        const end = nextCut(ids.len, e.cur.cuts.items, at);
        if (try e.prefillChunk(ids[at..@min(ids.len, end + 1)], at, ids.len, null, cancel, marks)) |first| return first;
        at = end;
    }
}

test "glm request starts: with no cuts a pass cuts every prefill_rows rows from 0, as before" {
    const m = prefill_rows;
    const len = 5 * m + 7;
    try std.testing.expectEqual(@as(usize, m), nextCut(len, &.{}, 0));
    try std.testing.expectEqual(@as(usize, len), nextCut(len, &.{}, 5 * m));
    for (0..6) |k| try std.testing.expect(coldCut(len, &.{}, k * m));
    try std.testing.expect(!coldCut(len, &.{}, m + 1));
    try std.testing.expect(!coldCut(len, &.{}, m - 1));
}

test "glm request starts: a request start ends the chunk before it, and the grid restarts from that start" {
    const m = prefill_rows;
    const len = 4 * m;
    const s = m + 300; // a message start between the first two grid points
    const cuts = [_]u32{ m, s, s + m };
    try std.testing.expectEqual(@as(usize, m), nextCut(len, &cuts, 0));
    try std.testing.expectEqual(@as(usize, s), nextCut(len, &cuts, m));
    try std.testing.expectEqual(@as(usize, s + m), nextCut(len, &cuts, s));
    try std.testing.expectEqual(@as(usize, s + 2 * m), nextCut(len, &cuts, s + m)); // past the last start: the grid from it
    try std.testing.expectEqual(@as(usize, len), nextCut(len, &cuts, s + 2 * m));
    try std.testing.expect(coldCut(len, &cuts, s));
    try std.testing.expect(coldCut(len, &cuts, s + m));
    try std.testing.expect(!coldCut(len, &cuts, 2 * m)); // the old grid point inside the chunk from `s`
    try std.testing.expect(!coldCut(len, &cuts, s + 1));
}

test "glm request starts: a short chunk ends at a start inside the first grid step, and the prompt's end ends the last" {
    const cuts = [_]u32{ 500, 1700 };
    try std.testing.expectEqual(@as(usize, 500), nextCut(3000, &cuts, 0));
    try std.testing.expectEqual(@as(usize, 1700), nextCut(3000, &cuts, 500));
    try std.testing.expectEqual(@as(usize, 3000), nextCut(3000, &cuts, 1700));
    try std.testing.expectEqual(@as(usize, 1200), nextCut(1200, &cuts, 500));
    try std.testing.expect(coldCut(3000, &cuts, 500) and coldCut(3000, &cuts, 1700));
    try std.testing.expect(!coldCut(3000, &cuts, 0 + prefill_rows));
}

test "glm request starts: a step runs to the next cut, a solo step only without cuts, and only from a cold cut" {
    const m = prefill_rows;
    try std.testing.expectEqual(@as(usize, m), try stepRows(3 * m, &.{}, 0, m + 1));
    try std.testing.expectEqual(@as(usize, state.solo_rows), try stepRows(3 * m, &.{}, 0, state.solo_rows + 1));
    try std.testing.expectEqual(@as(usize, 500), try stepRows(3 * m, &.{500}, 0, 501));
    try std.testing.expectEqual(@as(usize, m), try stepRows(3 * m, &.{500}, 500, m + 1));
    try std.testing.expectError(error.BadChunk, stepRows(3 * m, &.{500}, 0, state.solo_rows + 1));
    try std.testing.expectError(error.BadChunk, stepRows(3 * m, &.{500}, m, m + 1)); // the old grid point, not a cut here
    try std.testing.expectError(error.BadChunk, stepRows(3 * m, &.{}, 3 * m, 1));
}

test "glm request starts: starts must ascend after 0, and a refused list leaves the last one in place" {
    var list: std.ArrayList(u32) = .empty;
    defer list.deinit(std.testing.allocator);
    try set(std.testing.allocator, &list, &.{ 300, 2348 });
    for ([_][]const u32{ &.{0}, &.{ 300, 300 }, &.{ 900, 300 } }) |bad| {
        try std.testing.expectError(error.BadCuts, set(std.testing.allocator, &list, bad));
        try std.testing.expectEqualSlices(u32, &.{ 300, 2348 }, list.items);
    }
    try set(std.testing.allocator, &list, &.{});
    try std.testing.expectEqual(@as(usize, 0), list.items.len);
}
