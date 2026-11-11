//! A query's reuse identity (C6, #2444): the key's length, the request's chunk schedule, and where a pass may cut.
const std = @import("std");
const cuts = @import("prompt_cuts.zig");
const Rules = @import("prompt_rules.zig").Rules;

/// What the schedule rules read of a query: its key's length, the prompt's, and the request's own chunk starts.
pub fn Schedule(comptime Query: type) type {
    return struct {
        /// Whether the key names the prompt's rows one word each: otherwise nothing is reused or kept.
        pub fn keyed(q: Query) bool {
            return q.key.len == q.prompt_len;
        }

        /// The chunk start at or before `w`: a planned store's request starts if given, else its grid; unplanned, `w`.
        pub fn startAt(r: Rules, q: Query, w: u32) u32 {
            if (!r.planned) return w;
            if (q.starts.len > 0) return cuts.floorStart(q.starts, w);
            return w - w % r.grid;
        }

        /// Whether a pass may cut and resume at `at`: a page edge on the request's chunk schedule.
        pub fn usable(r: Rules, q: Query, at: u32) bool {
            return at > 0 and at % r.page == 0 and startAt(r, q, at) == at;
        }

        /// The mark a wanted position `w` gives: its chunk start on a page edge, or none when that is no start.
        pub fn markAt(r: Rules, q: Query, w: u32) ?u32 {
            const start = startAt(r, q, w);
            const m = start - start % r.page;
            return if (m > 0 and usable(r, q, m)) m else null;
        }

        /// The predicate the tree's walks take: whether a position is a resume point for `q`.
        pub const Resumable = struct {
            r: Rules,
            q: Query,
            pub fn ok(c: @This(), at: u32) bool {
                return usable(c.r, c.q, at);
            }
        };
    };
}

const testing = std.testing;
const Q = struct { key: []const u32, prompt_len: u32, starts: []const u32 = &.{} };
const S = Schedule(Q);

test "a planned store cuts on its grid, or on a request's own starts in place of it; an unplanned one anywhere" {
    const grid: Rules = .{ .page = 4, .planned = true, .grid = 8 };
    const none: Q = .{ .key = &.{}, .prompt_len = 0 };
    try testing.expectEqual(@as(?u32, 16), S.markAt(grid, none, 21));
    try testing.expect(S.usable(grid, none, 16) and !S.usable(grid, none, 12));
    const own: Q = .{ .key = &.{}, .prompt_len = 0, .starts = &.{ 12, 6, 20 } };
    try testing.expectEqual(@as(?u32, 12), S.markAt(grid, own, 15)); // a page-aligned start off the grid
    try testing.expect(S.usable(grid, own, 12) and !S.usable(grid, own, 16)); // a grid point inside its chunk is not
    try testing.expectEqual(@as(?u32, null), S.markAt(grid, own, 7)); // the start at 6 floors to 4, inside its chunk
    const pages: Rules = .{ .page = 4 };
    try testing.expectEqual(@as(?u32, 20), S.markAt(pages, own, 23)); // unplanned: starts do not bind
    try testing.expectEqual(@as(?u32, null), S.markAt(pages, none, 3));
}

test "a key of another length than the prompt is not keyed" {
    try testing.expect(S.keyed(.{ .key = &.{ 1, 2 }, .prompt_len = 2 }));
    try testing.expect(!S.keyed(.{ .key = &.{ 1, 2, 3 }, .prompt_len = 2 }));
}
