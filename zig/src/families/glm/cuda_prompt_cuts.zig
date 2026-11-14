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

/// A cold pass's chunk starts below `at` as a digest; 0 on the plain grid (a pass cut from `at` on chunks so too).
pub fn below(cuts: []const u32, at: u32) u64 {
    var h = std.hash.Wyhash.init(0);
    var grid = true;
    var p: u32 = 0;
    while (p < at) : (p = @intCast(nextCut(at, cuts, p))) {
        grid = grid and p % prefill_rows == 0;
        h.update(std.mem.asBytes(&p));
    }
    return if (grid) 0 else h.final() | 1;
}

/// A state kept at `at` with digest `kept_below` serves a pass over `len` rows: a cold cut there, over the same chunks.
pub fn resumes(len: usize, cuts: []const u32, at: u32, kept_below: u64) bool {
    return coldCut(len, cuts, at) and kept_below == below(cuts, at);
}

/// Rank 0's step rows from `start` (Leader.chunk): a solo or paired step's if `wide`, else to the next cut.
pub fn leaderRows(len: usize, cuts: []const u32, start: usize, wide: bool) usize {
    return if (wide) state.solo_rows else nextCut(len, cuts, start) - start;
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

test "glm request starts: starts 1 and 2 rows apart each end a chunk, and a step from each runs to the next" {
    const m = prefill_rows;
    const len = 3 * m;
    const cuts = [_]u32{ 1, 2, 1000, 1001, 1003 }; // starts a row or two apart (every message start): the cuts still hold
    const ends = [_]usize{ 1, 2, 1000, 1001, 1003, 1003 + m, 1003 + 2 * m, len };
    var at: usize = 0;
    for (ends) |end| {
        try std.testing.expectEqual(end, nextCut(len, &cuts, at));
        try std.testing.expect(coldCut(len, &cuts, at));
        try std.testing.expectEqual(end - at, try stepRows(len, &cuts, at, end - at + 1));
        at = end;
    }
    try std.testing.expect(!coldCut(len, &cuts, 1002));
    try std.testing.expectError(error.BadChunk, stepRows(len, &cuts, 1002, 2));
}

/// The prompts and starts the tests here cut: the grid alone, starts off and on it, past the prompt, and close starts.
const Pass = struct { len: usize, cuts: []const u32 };
const passes = [_]Pass{
    .{ .len = 5 * prefill_rows + 7, .cuts = &.{} },
    .{ .len = 4 * prefill_rows, .cuts = &.{ prefill_rows, prefill_rows + 300, 2 * prefill_rows + 300 } },
    .{ .len = 3000, .cuts = &.{ 500, 1700 } },
    .{ .len = 1200, .cuts = &.{ 500, 1700 } },
    .{ .len = 3 * prefill_rows, .cuts = &.{500} },
    .{ .len = 3 * prefill_rows, .cuts = &.{ 1, 2, 1000, 1001, 1003 } },
    .{ .len = 4 * prefill_rows, .cuts = &.{ 300, 2 * prefill_rows } },
    .{ .len = 4 * prefill_rows, .cuts = &.{ 700, 2 * prefill_rows } },
    .{ .len = 4 * prefill_rows, .cuts = &.{ 2 * prefill_rows, 2 * prefill_rows + 500 } },
};

/// A cold pass's chunk starts below `at` (from 0), written out.
fn startsBelow(cuts: []const u32, at: usize, buf: []usize) []usize {
    var n: usize = 0;
    var p: usize = 0;
    while (p < at) : (p = nextCut(at, cuts, p)) {
        buf[n] = p;
        n += 1;
    }
    return buf[0..n];
}

test "glm request starts: a kept state's digest below `at` matches another pass's exactly where their chunks below do" {
    const m = prefill_rows;
    try std.testing.expectEqual(@as(u64, 0), below(&.{}, 2 * m)); // the grid alone
    try std.testing.expectEqual(@as(u64, 0), below(&.{m}, 2 * m)); // a start on the grid cuts as the grid does
    try std.testing.expectEqual(@as(u64, 0), below(&.{ 2 * m, 2 * m + 500 }, 2 * m)); // starts from `at` on don't count
    try std.testing.expect(below(&.{ 300, 2 * m }, 2 * m) != 0);
    try std.testing.expect(below(&.{ 300, 2 * m }, 2 * m) != below(&.{ 700, 2 * m }, 2 * m));
    try std.testing.expectEqual(below(&.{ 300, 2 * m }, 2 * m), below(&.{ 300, 2 * m, 3 * m }, 2 * m));
    var a_buf: [64]usize = undefined;
    var b_buf: [64]usize = undefined;
    var compared: usize = 0;
    for (passes) |a| for (passes) |b| {
        var at: usize = 0;
        while (at < a.len) : (at = nextCut(a.len, a.cuts, at)) {
            if (at == 0 or at >= b.len or !coldCut(b.len, b.cuts, at)) continue; // resumeFrom refuses these before
            const same = std.mem.eql(usize, startsBelow(a.cuts, at, &a_buf), startsBelow(b.cuts, at, &b_buf));
            try std.testing.expectEqual(same, below(a.cuts, @intCast(at)) == below(b.cuts, @intCast(at)));
            compared += 1;
        }
    };
    try std.testing.expect(compared > 50);
}

test "glm request starts: rank 0's step rows (Leader.chunk) are the rows each other rank sizes from its part" {
    for (passes) |c| {
        var start: usize = 0;
        var steps: usize = 0;
        while (start < c.len) : (steps += 1) {
            const left = c.len - start;
            const n = leaderRows(c.len, c.cuts, start, false);
            try std.testing.expectEqual(@min(left, n), try stepRows(c.len, c.cuts, start, @min(left, n + 1)));
            const w = leaderRows(c.len, c.cuts, start, true); // a solo or paired step: two grid steps left
            if (start + prefill_rows + 1 < c.len) { // prompt_solo.takesTwo; with starts rank 0 never steps wide
                const wide = stepRows(c.len, c.cuts, start, @min(left, w + 1));
                if (c.cuts.len > 0) {
                    try std.testing.expectError(error.BadChunk, wide);
                } else try std.testing.expectEqual(@min(left, w), try wide);
            }
            start += @min(left, n);
        }
        try std.testing.expect(steps > 1);
    }
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
