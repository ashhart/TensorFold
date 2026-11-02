//! Prepare held tokens on the lane owner, walk one matcher, then apply its window on the lane owner.
const std = @import("std");
const accept = @import("accept.zig");
const grammar = @import("grammar.zig");
const timing = @import("grammar_time.zig");
pub const Prepared = struct {
    constraint: *grammar.Constraint,
    rows: []const u32,
    parents: []const i32,
    room: usize,
    forced: bool,
};
pub fn prepare(e: anytype, p: anytype) !Prepared {
    const a = e.arena.allocator();
    if (p.held > 0) {
        // the held drafts (a chain or a tree, in window order) come back as host tokens, grafts after them
        const read = e.backend.vtable.held orelse return error.HeldDraftsUnreadable;
        const held = try a.alloc(u32, p.held);
        try read(e.backend.ptr, p.stream, held);
        p.tokens = try std.mem.concat(a, u32, &.{ held, p.tokens });
        p.held = 0;
    }
    const rows = try a.alloc(u32, 1 + p.tokens.len);
    rows[0] = p.stream.pending.?;
    @memcpy(rows[1..], p.tokens);
    const grows = p.stream.drafts and p.kind != .forced and e.alone;
    const room = if (grows) @max(rows.len, e.cfg.family_width) else rows.len;
    const row_parents = try accept.rowParents(a, rows.len, p.parents);
    return .{ .constraint = p.stream.grammar.?, .rows = rows, .parents = row_parents, .room = room, .forced = p.kind == .forced };
}
pub fn walk(prepared: Prepared, a: std.mem.Allocator, profile: *timing.Profile) !grammar.Window {
    return timing.window(prepared.constraint, a, prepared.rows, prepared.parents, prepared.forced, prepared.room, profile);
}
pub fn apply(e: anytype, p: anytype, prepared: Prepared, w: grammar.Window) !void {
    const a = e.arena.allocator();
    p.masks = w.masks;
    if (w.rows.len == prepared.rows.len and w.extra.len == 0) return;
    if (w.rows.len < prepared.rows.len) p.lanes = null; // the tree shape no longer matches the rows
    const kept = try a.alloc(u32, w.rows.len - 1 + w.extra.len);
    for (kept[0 .. w.rows.len - 1], w.rows[1..]) |*t, r| t.* = prepared.rows[r];
    @memcpy(kept[w.rows.len - 1 ..], w.extra);
    p.tokens = kept;
    if (w.extra.len > 0) p.grammar_from = @intCast(w.rows.len);
    if (p.parents != null) {
        // the grammar drafts only past a chain's last kept row (Constraint.chain), so its drafts extend that chain
        const parents = try a.alloc(i32, kept.len);
        for (parents, 0..) |*q, i| q.* = if (i + 1 < w.parents.len) w.parents[i + 1] - 1 else @as(i32, @intCast(i)) - 1;
        p.parents = if (accept.isChain(parents)) null else parents;
    }
    if (p.branch_from > 0) {
        var below: u32 = 0;
        for (w.rows) |r| below += @intFromBool(r < p.branch_from);
        if (p.graft_recalled.len > 0) { // the kept rows are in window order: the grafts' sources follow them
            const flags = try a.alloc(bool, w.rows.len - below);
            for (flags, w.rows[below..]) |*fl, r| fl.* = p.recallRow(r);
            p.graft_recalled = flags;
        }
        p.branch_from = below;
    }
    if (p.kind != .forced and w.rows.len == 1) p.kind = if (w.extra.len > 0) .grammar else .none;
}
pub fn constrain(e: anytype, p: anytype) !void {
    if (p.stream.grammar == null) return;
    const prepared = try prepare(e, p);
    var profile: timing.Profile = .{ .clock = e.clock };
    defer timing.record(e, p.stream, profile);
    try apply(e, p, prepared, try walk(prepared, e.arena.allocator(), &profile));
}
