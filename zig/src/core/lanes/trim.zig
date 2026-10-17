//! An optional row policy in place of windows.allocate: a family's own depth rule trims drafts.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Stream = @import("stream.zig").Stream;

/// One planned window as the policy sees it.
pub const Window = struct {
    stream: *Stream,
    drafted: bool, // head drafts the policy may trim; every other window runs whole
    rows: u32, // rows as planned, the pending row included
    parents: ?[]const i32, // the drafts' parents for a tree (null: a chain)
};

/// What a window keeps: its first `count` drafts, or the listed drafts (ascending, each one's parent listed).
pub const Choice = struct { count: u32, nodes: ?[]const u32 = null };

/// A committed window: its rows, the accepted path (row 0 first) and the target's choice after the path's last row.
pub const Commit = struct {
    stream: *Stream,
    drafted: bool,
    rows: u32,
    parents: ?[]const i32, // each row's parent row (row 0: -1) for a tree; null for a chain
    path: []const u32,
    bonus: u32,
};

pub const Trim = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Each window's drafts for this round; `alone`: the round's only window. Slices in `out` live in `arena`.
        choose: *const fn (ptr: *anyopaque, windows: []const Window, alone: bool, arena: Allocator, out: []Choice) anyerror!void,
        /// Every window's commit, in plan order (a shared round's windows one after the other).
        commit: *const fn (ptr: *anyopaque, c: Commit) anyerror!void,
    };

    pub fn choose(t: Trim, windows: []const Window, alone: bool, arena: Allocator, out: []Choice) !void {
        return t.vtable.choose(t.ptr, windows, alone, arena, out);
    }
    pub fn commit(t: Trim, c: Commit) !void {
        return t.vtable.commit(t.ptr, c);
    }
};

pub const Sub = struct { tokens: []u32, parents: []i32 };

/// The listed drafts of a tree re-indexed among themselves (parents -1: the pending row); arena-owned.
pub fn select(arena: Allocator, tokens: []const u32, parents: []const i32, nodes: []const u32) !Sub {
    const at = try arena.alloc(i32, tokens.len);
    @memset(at, -1);
    const out_t = try arena.alloc(u32, nodes.len);
    const out_p = try arena.alloc(i32, nodes.len);
    for (nodes, out_t, out_p, 0..) |n, *t, *p, i| {
        const q = parents[n];
        if (q >= 0 and at[@intCast(q)] < 0) return error.ParentNotKept;
        at[n] = @intCast(i);
        t.* = tokens[n];
        p.* = if (q < 0) -1 else at[@intCast(q)];
    }
    return .{ .tokens = out_t, .parents = out_p };
}

test "select keeps listed drafts with their parents re-indexed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    // chain 10 11 12, sibling 20 (on the pending row) with child 21
    const s = try select(arena.allocator(), &.{ 10, 11, 12, 20, 21 }, &.{ -1, 0, 1, -1, 3 }, &.{ 0, 3, 4 });
    try std.testing.expectEqualSlices(u32, &.{ 10, 20, 21 }, s.tokens);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, 1 }, s.parents);
    try std.testing.expectError(error.ParentNotKept, select(arena.allocator(), &.{ 10, 11 }, &.{ -1, 0 }, &.{1}));
}
