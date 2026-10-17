//! Keep a window's held prefix, host drafts and graft metadata in the same row numbering after a cut.
const std = @import("std");
const accept = @import("accept.zig");
pub fn prefix(a: std.mem.Allocator, p: anytype, count: u32) !void {
    if (count >= p.count()) return;
    if (count == 0) {
        p.kind = .none;
        p.held = 0;
        p.tokens = &.{};
        p.parents = null;
        p.lanes = null;
    } else if (p.parents) |parents| {
        const drafts = if (p.held == 0) p.tokens else blk: {
            const all = try a.alloc(u32, p.count());
            @memset(all[0..p.held], 0);
            @memcpy(all[p.held..], p.tokens);
            break :blk all;
        };
        const tree = try accept.sanitizeTree(a, drafts, parents, count);
        p.held = @intCast(@min(p.held, tree.tokens.len));
        p.tokens = tree.tokens[p.held..];
        p.parents = tree.parents;
    } else if (p.held > 0) {
        p.held = count;
    } else p.tokens = p.tokens[0..count];
    if (p.branch_from > p.count()) p.branch_from = 0;
    const grafts = if (p.branch_from > 0) p.count() + 1 - p.branch_from else 0;
    p.graft_recalled = p.graft_recalled[0..@min(grafts, p.graft_recalled.len)];
}
pub fn selected(a: std.mem.Allocator, p: anytype, nodes: []const u32) !void {
    if (p.branch_from == 0) return;
    var first: u32 = 0;
    var grafts: usize = 0;
    for (nodes, 0..) |n, i| if (n + 1 >= p.branch_from) {
        if (first == 0) first = @intCast(i + 1);
        grafts += 1;
    };
    if (p.graft_recalled.len > 0) {
        const flags = try a.alloc(bool, grafts);
        var i: usize = 0;
        for (nodes) |n| if (n + 1 >= p.branch_from) {
            flags[i] = p.recallRow(n + 1);
            i += 1;
        };
        p.graft_recalled = flags;
    }
    p.branch_from = first;
}
