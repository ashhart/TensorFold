//! A verify tree's host plans: gdn.schedule's slots and visiting order, and each node's conv window and depth.

const std = @import("std");

/// The most nodes a stream's tree takes (forward._paths), and the most live states a schedule may hold.
pub const max_nodes = 128;
pub const max_slots = 32;

/// gdn.schedule's result: (node, source, dest) entries in visiting order and the slots they use.
pub const Schedule = struct { entries: []i32, slots: usize };

/// forward._paths: each row's depth from the root (parents topologically sorted, -1 at row 0 only).
pub fn depths(parents: []const i32, out: []usize) !void {
    if (parents.len == 0 or parents.len > max_nodes or parents[0] != -1) return error.InvalidQwenTree;
    for (parents, 0..) |p, row| {
        if (row == 0) {
            out[0] = 0;
            continue;
        }
        if (p < 0 or p >= row) return error.InvalidQwenTree;
        out[row] = out[@intCast(p)] + 1;
    }
}

/// Whether every row's parent is the row before it: a chain.
pub fn chain(parents: []const i32) bool {
    for (parents, 0..) |p, row| if (p != @as(i32, @intCast(row)) - 1) return false;
    return true;
}

/// gdn._order_slots: entries for one visiting order, and the slots it needs.
fn orderSlots(gpa: std.mem.Allocator, parents: []const i32, order: []const usize) !Schedule {
    const n = parents.len;
    const pos = try gpa.alloc(usize, n);
    defer gpa.free(pos);
    for (order, 0..) |node, i| pos[node] = i;
    // each node's last child in this order, and whether all its children follow it directly
    const last = try gpa.alloc(?usize, n);
    defer gpa.free(last);
    @memset(last, null);
    const direct = try gpa.alloc(bool, n);
    defer gpa.free(direct);
    @memset(direct, true);
    for (parents, 0..) |p, node| {
        if (p < 0) continue;
        const pu: usize = @intCast(p);
        if (last[pu] == null or pos[node] > pos[last[pu].?]) last[pu] = node;
        if (pos[node] != pos[pu] + 1) direct[pu] = false;
    }
    const slot_of = try gpa.alloc(?i32, n);
    defer gpa.free(slot_of);
    @memset(slot_of, null);
    var free: std.ArrayList(i32) = .empty;
    defer free.deinit(gpa);
    var entries = try gpa.alloc(i32, 3 * n);
    errdefer gpa.free(entries);
    var used: i32 = 0;
    for (order, 0..) |node, i| {
        const parent = parents[node];
        var source: i32 = -1;
        if (parent >= 0) {
            const pu: usize = @intCast(parent);
            source = if (i > 0 and order[i - 1] == pu) -2 else slot_of[pu] orelse return error.InvalidQwenTree;
            // its last child reads it now: the slot may be reused
            if (slot_of[pu] != null and pos[last[pu].?] == i) {
                try free.append(gpa, slot_of[pu].?);
                slot_of[pu] = null;
            }
        }
        var dest: i32 = -1;
        if (last[node] != null and !direct[node]) {
            if (free.items.len > 0) {
                std.mem.sort(i32, free.items, {}, std.sort.asc(i32));
                dest = free.orderedRemove(0);
            } else {
                dest = used;
                used += 1;
            }
            slot_of[node] = dest;
        }
        entries[3 * i ..][0..3].* = .{ @intCast(node), source, dest };
    }
    return .{ .entries = entries, .slots = @intCast(used) };
}

/// gdn.schedule: depth-first or level order, whichever needs fewer slots (depth-first on a tie).
pub fn schedule(gpa: std.mem.Allocator, parents: []const i32) !Schedule {
    const n = parents.len;
    const d = try gpa.alloc(usize, n);
    defer gpa.free(d);
    try depths(parents, d);
    // depth-first: children in row order, as the Python stack pushes them reversed
    const dfs = try gpa.alloc(usize, n);
    defer gpa.free(dfs);
    var stack: std.ArrayList(usize) = .empty;
    defer stack.deinit(gpa);
    try stack.append(gpa, 0);
    var at: usize = 0;
    while (stack.pop()) |node| {
        dfs[at] = node;
        at += 1;
        var child = n;
        while (child > node + 1) {
            child -= 1;
            if (parents[child] == @as(i32, @intCast(node))) try stack.append(gpa, child);
        }
    }
    const level = try gpa.alloc(usize, n);
    defer gpa.free(level);
    for (level, 0..) |*l, i| l.* = i;
    std.mem.sort(usize, level, d, struct {
        fn less(ds: []const usize, a: usize, b: usize) bool {
            return ds[a] < ds[b] or (ds[a] == ds[b] and a < b);
        }
    }.less);
    const first = try orderSlots(gpa, parents, dfs);
    const second = orderSlots(gpa, parents, level) catch |err| {
        gpa.free(first.entries);
        return err;
    };
    const best, const other = if (second.slots < first.slots) .{ second, first } else .{ first, second };
    gpa.free(other.entries);
    if (best.slots > max_slots) {
        gpa.free(best.entries);
        return error.QwenTreeTooWide;
    }
    return best;
}

/// forward._conv_windows: each node's last `keep` inputs along its path, then its own row (`keep + base + row`).
pub fn convWindows(parents: []const i32, keep: usize, base: usize, out: []i32) void {
    const w = keep + 1;
    for (parents, 0..) |p, row| {
        const dst = out[row * w ..][0..w];
        if (p < 0) {
            for (0..keep) |j| dst[j] = @intCast(j);
        } else {
            const src = out[@as(usize, @intCast(p)) * w ..][0..w];
            @memcpy(dst[0..keep], src[1..w]);
        }
        dst[keep] = @intCast(keep + base + row);
    }
}

test "a chain's schedule keeps no slot and reads each predecessor" {
    const gpa = std.testing.allocator;
    const s = try schedule(gpa, &.{ -1, 0, 1, 2 });
    defer gpa.free(s.entries);
    try std.testing.expectEqual(@as(usize, 0), s.slots);
    try std.testing.expectEqualSlices(i32, &.{ 0, -1, -1, 1, -2, -1, 2, -2, -1, 3, -2, -1 }, s.entries);
}

test "a fork saves the shared parent in a slot until its last child" {
    const gpa = std.testing.allocator;
    // root 0 with children 1 and 2, 1 with child 3: depth-first 0 1 3 2 needs the root kept for 2
    const s = try schedule(gpa, &.{ -1, 0, 0, 1 });
    defer gpa.free(s.entries);
    try std.testing.expectEqual(@as(usize, 1), s.slots);
    try std.testing.expectEqualSlices(i32, &.{ 0, -1, 0, 1, -2, -1, 3, -2, -1, 2, 0, -1 }, s.entries);
}

test "schedules match gdn.schedule on random trees" {
    const gpa = std.testing.allocator;
    const cases = .{
        .{ &[_]i32{ -1, 0, 1, 2, 0, 3 }, &[_]i32{ 0, -1, 0, 1, -2, -1, 2, -2, -1, 3, -2, -1, 5, -2, -1, 4, 0, -1 }, 1 },
        .{ &[_]i32{ -1, 0, 0, 0, 0, 2, 3, 6, 3 }, &[_]i32{ 0, -1, 0, 1, -2, -1, 2, 0, -1, 5, -2, -1, 3, 0, 1, 6, -2, -1, 7, -2, -1, 8, 1, -1, 4, 0, -1 }, 2 },
        .{ &[_]i32{ -1, 0, 0, 2, 1, 0, 5, 1, 6, 4, 2, 6 }, &[_]i32{ 0, -1, 0, 1, -2, 1, 4, -2, -1, 9, -2, -1, 7, 1, -1, 2, 0, 1, 3, -2, -1, 10, 1, -1, 5, 0, -1, 6, -2, 0, 8, -2, -1, 11, 0, -1 }, 2 },
    };
    inline for (cases) |case| {
        const s = try schedule(gpa, case[0]);
        defer gpa.free(s.entries);
        try std.testing.expectEqual(@as(usize, case[2]), s.slots);
        try std.testing.expectEqualSlices(i32, case[1], s.entries);
    }
}

test "conv windows follow each node's path" {
    var out: [4 * 4]i32 = undefined;
    convWindows(&.{ -1, 0, 0, 1 }, 3, 0, &out);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 5, 2, 3, 4, 6 }, &out);
}
