//! Window rows as chains or trees, the accepted path and tree trimming (Python lane_tree, sanitize_tree).
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Each row's parent row (row 0 is the pending token): a chain when `parents` (drafts' parents) is null.
pub fn rowParents(gpa: Allocator, rows: usize, parents: ?[]const i32) ![]i32 {
    const out = try gpa.alloc(i32, rows);
    for (out, 0..) |*o, r| {
        if (r == 0) {
            o.* = -1;
        } else if (parents) |p| {
            const q = p[r - 1];
            o.* = if (q < 0) 0 else q + 1;
        } else o.* = @intCast(r - 1);
    }
    return out;
}

/// Each row's depth (its distance from row 0); parents must precede their children.
pub fn depths(gpa: Allocator, parents: []const i32) ![]u32 {
    const out = try gpa.alloc(u32, parents.len);
    errdefer gpa.free(out);
    for (parents, 0..) |q, row| {
        if (q < 0) {
            out[row] = 0;
        } else {
            if (q >= @as(i32, @intCast(row))) return error.ParentAfterChild;
            out[row] = out[@intCast(q)] + 1;
        }
    }
    return out;
}

pub fn isChain(parents: []const i32) bool {
    for (parents, 0..) |q, row| if (q != @as(i32, @intCast(row)) - 1) return false;
    return true;
}

/// Rows of the accepted path from row 0: follow the first child (by row) whose token is the target's pick.
pub fn acceptPath(gpa: Allocator, tokens: []const u32, parents: []const i32, picks: []const u32) ![]u32 {
    var path: std.ArrayList(u32) = .empty;
    errdefer path.deinit(gpa);
    try path.append(gpa, 0);
    while (true) {
        const at = path.items[path.items.len - 1];
        const want = picks[at];
        var next: ?u32 = null;
        for (parents, 0..) |q, row| {
            if (q == @as(i32, @intCast(at)) and tokens[row] == want) {
                next = @intCast(row);
                break;
            }
        }
        try path.append(gpa, next orelse break);
    }
    return path.toOwnedSlice(gpa);
}

pub const Tree = struct { tokens: []u32, parents: []i32 };

/// The first `budget` nodes, dropping orphaned subtrees and nodes whose parents do not precede them.
pub fn sanitizeTree(gpa: Allocator, tokens: []const u32, parents: []const i32, budget: usize) !Tree {
    const n = @min(budget, @min(tokens.len, parents.len));
    const kept = try gpa.alloc(i32, n);
    defer gpa.free(kept);
    @memset(kept, -1);
    var out_t: std.ArrayList(u32) = .empty;
    errdefer out_t.deinit(gpa);
    var out_p: std.ArrayList(i32) = .empty;
    errdefer out_p.deinit(gpa);
    for (0..n) |i| {
        const q = parents[i];
        if (q >= 0 and (q >= @as(i32, @intCast(i)) or kept[@intCast(q)] < 0)) continue;
        kept[i] = @intCast(out_t.items.len);
        try out_t.append(gpa, tokens[i]);
        try out_p.append(gpa, if (q < 0) -1 else kept[@intCast(q)]);
    }
    return .{ .tokens = try out_t.toOwnedSlice(gpa), .parents = try out_p.toOwnedSlice(gpa) };
}

test "accept path follows matching children" {
    const gpa = std.testing.allocator;
    const parents = [_]i32{ -1, 0, 0, 1 };
    const path = try acceptPath(gpa, &.{ 5, 6, 7, 8 }, &parents, &.{ 7, 9, 9, 9 });
    defer gpa.free(path);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2 }, path);
    const chain = try rowParents(gpa, 3, null);
    defer gpa.free(chain);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1 }, chain);
}

test "sanitize drops orphans" {
    const gpa = std.testing.allocator;
    const t = try sanitizeTree(gpa, &.{ 1, 2, 3, 4 }, &.{ -1, 0, 5, 2 }, 4);
    defer gpa.free(t.tokens);
    defer gpa.free(t.parents);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, t.tokens);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0 }, t.parents);
}
