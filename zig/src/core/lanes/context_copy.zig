//! A lone head window's context copy, sharing existing nodes before appending its unmatched continuation.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const Plan = @import("windows.zig").Plan;

pub fn extend(e: *Engine, p: *Plan, copy: []const u32) !void {
    const first = p.count();
    const limit: usize = @intCast(@max(1, @min(e.cfg.family_width, e.cfg.batch_rows, p.stream.draftRoom())));
    const room = limit -| (first + 1);
    if (room == 0) return;
    const a = e.arena.allocator();
    var tokens = p.tokens;
    if (p.held > 0) {
        const read = e.backend.vtable.held orelse return;
        const held = try a.alloc(u32, p.held);
        try read(e.backend.ptr, p.stream, held);
        tokens = try std.mem.concat(a, u32, &.{ held, tokens });
    }
    var parent: i32 = -1;
    var shared: usize = 0;
    follow: for (copy) |token| {
        for (tokens, 0..) |t, i| {
            const up = if (p.parents) |ps| ps[i] else @as(i32, @intCast(i)) - 1;
            if (up == parent and t == token) {
                parent = @intCast(i);
                shared += 1;
                continue :follow;
            }
        }
        break;
    }
    const extra = copy[shared..@min(copy.len, shared + room)];
    if (extra.len == 0) return;
    const parents = try a.alloc(i32, first + extra.len);
    for (parents[0..first], 0..) |*up, i| up.* = if (p.parents) |ps| ps[i] else @as(i32, @intCast(i)) - 1;
    for (parents[first..], first..) |*up, i| up.* = if (i == first) parent else @intCast(i - 1);
    p.tokens = try std.mem.concat(a, u32, &.{ tokens, extra });
    p.parents = parents;
    p.held = 0;
    p.branch_from = @intCast(first + 1);
    p.context_copy = true;
}
