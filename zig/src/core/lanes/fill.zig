//! More lanes for one stream: continuations of its context found by suffix match, grafted onto the head's chain.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// One proposed continuation of the pending token and the planner's trust in it.
pub const Proposal = struct { tokens: []const u32, weight: f64 };

/// Host drafts appended after a window's held drafts: each draft's token and its parent draft (-1: the pending row).
pub const Branches = struct {
    tokens: []u32,
    parents: []i32,

    pub fn deinit(b: Branches, gpa: Allocator) void {
        gpa.free(b.tokens);
        gpa.free(b.parents);
    }
};

pub fn freeProposals(gpa: Allocator, proposals: []Proposal) void {
    for (proposals) |p| gpa.free(p.tokens);
    gpa.free(proposals);
}

/// Distinct continuations of the context's last n tokens (n from `longest` down to `shortest`), most recent match first.
pub fn suffixCandidates(gpa: Allocator, context: []const u32, longest: usize, shortest: usize, span: usize, most: usize) ![]Proposal {
    var out: std.ArrayList(Proposal) = .empty;
    errdefer {
        for (out.items) |p| gpa.free(p.tokens);
        out.deinit(gpa);
    }
    if (context.len < shortest + 1 or shortest == 0) return out.toOwnedSlice(gpa);
    var n = @min(longest, context.len - 1);
    while (n >= shortest and out.items.len < most) : (n -= 1) {
        const tail = context[context.len - n ..];
        var i = context.len - n;
        while (i > 0 and out.items.len < most) {
            i -= 1;
            if (!std.mem.eql(u32, context[i .. i + n], tail)) continue;
            const next = context[i + n .. @min(i + n + span, context.len)];
            if (seen(out.items, next)) continue;
            try out.append(gpa, .{ .tokens = try gpa.dupe(u32, next), .weight = @floatFromInt(n) });
        }
    }
    return out.toOwnedSlice(gpa);
}

fn seen(have: []const Proposal, next: []const u32) bool {
    for (have) |p| if (std.mem.eql(u32, p.tokens, next)) return true;
    return false;
}

/// The head's chain the branches graft onto: its drafts (level by level) and the head's other best tokens at each level.
pub const Trunk = struct { tokens: []const u32 = &.{}, others: []const [3]u32 = &.{} };

/// Proposals grafted onto the trunk, heaviest first, leaving it past its end or at a token the head ranked there.
pub fn graft(gpa: Allocator, trunk: Trunk, proposals: []const Proposal, budget: usize) !Branches {
    var tokens: std.ArrayList(u32) = .empty;
    errdefer tokens.deinit(gpa);
    var parents: std.ArrayList(i32) = .empty;
    errdefer parents.deinit(gpa);
    const order = try gpa.alloc(usize, proposals.len);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, proposals, heavier);
    const first = trunk.tokens.len;
    for (order) |pi| {
        var at: i32 = -1;
        var on_trunk = true;
        for (proposals[pi].tokens, 0..) |t, level| {
            if (on_trunk and level < first and trunk.tokens[level] == t) {
                at = @intCast(level);
                continue;
            }
            if (child(tokens.items, parents.items, first, at, t)) |d| {
                at = d;
                on_trunk = false;
                continue;
            }
            if (on_trunk and level < first and !ranked(trunk, level, t)) break;
            if (tokens.items.len == budget) break;
            try tokens.append(gpa, t);
            try parents.append(gpa, at);
            at = @intCast(first + tokens.items.len - 1);
            on_trunk = false;
        }
    }
    return .{ .tokens = try tokens.toOwnedSlice(gpa), .parents = try parents.toOwnedSlice(gpa) };
}

fn ranked(trunk: Trunk, level: usize, t: u32) bool {
    if (level >= trunk.others.len) return false;
    return std.mem.indexOfScalar(u32, &trunk.others[level], t) != null;
}

fn heavier(ps: []const Proposal, a: usize, b: usize) bool {
    return ps[a].weight > ps[b].weight;
}

/// The draft under `parent` already carrying `token`, if the trie has one.
fn child(tokens: []const u32, parents: []const i32, first: usize, parent: i32, token: u32) ?i32 {
    for (tokens, parents, 0..) |t, p, k| if (p == parent and t == token) return @intCast(first + k);
    return null;
}

test "suffix candidates find what followed earlier occurrences, longest match first" {
    const gpa = std.testing.allocator;
    const ctx = [_]u32{ 1, 2, 3, 9, 8, 1, 2, 3, 7, 6, 2, 3 };
    const got = try suffixCandidates(gpa, &ctx, 4, 2, 3, 8);
    defer freeProposals(gpa, got);
    try std.testing.expectEqual(@as(usize, 2), got.len);
    try std.testing.expectEqualSlices(u32, &.{ 7, 6, 2 }, got[0].tokens);
    try std.testing.expectEqualSlices(u32, &.{ 9, 8, 1 }, got[1].tokens);
}

test "graft without a trunk keeps shared prefixes once and respects the budget" {
    const gpa = std.testing.allocator;
    const ps = [_]Proposal{ .{ .tokens = &.{ 5, 6, 7 }, .weight = 3 }, .{ .tokens = &.{ 5, 6, 9 }, .weight = 2 }, .{ .tokens = &.{4}, .weight = 1 } };
    const b = try graft(gpa, .{}, &ps, 8);
    defer b.deinit(gpa);
    try std.testing.expectEqualSlices(u32, &.{ 5, 6, 7, 9, 4 }, b.tokens);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1, 1, -1 }, b.parents);
    const small = try graft(gpa, .{}, &ps, 3);
    defer small.deinit(gpa);
    try std.testing.expectEqualSlices(u32, &.{ 5, 6, 7 }, small.tokens);
}

test "graft extends the head's chain and leaves it only at the head's own ranked tokens" {
    const gpa = std.testing.allocator;
    const trunk: Trunk = .{ .tokens = &.{ 5, 6 }, .others = &.{ .{ 9, 10, 11 }, .{ 12, 13, 14 } } };
    const ps = [_]Proposal{
        .{ .tokens = &.{ 5, 6, 7, 8 }, .weight = 4 }, // agrees with the chain, then extends it
        .{ .tokens = &.{ 5, 12, 3 }, .weight = 3 }, // leaves at level 1 for the head's second choice
        .{ .tokens = &.{ 9, 4 }, .weight = 2 }, // leaves at level 0 for the head's second choice
        .{ .tokens = &.{ 7, 1 }, .weight = 1 }, // a token the head did not rank: dropped
    };
    const b = try graft(gpa, trunk, &ps, 16);
    defer b.deinit(gpa);
    try std.testing.expectEqualSlices(u32, &.{ 7, 8, 12, 3, 9, 4 }, b.tokens);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 0, 4, -1, 6 }, b.parents);
}
