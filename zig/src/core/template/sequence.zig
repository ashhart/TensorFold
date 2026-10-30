//! Immutable sequence concatenations and contiguous views share their inputs until iteration needs a flat slice.
const std = @import("std");
const v = @import("value.zig");
const Allocator = std.mem.Allocator;

pub const List = struct {
    items: []const v.Value = &.{},
    seq: v.Seq = .list,
    range: [3]i64 = .{ 0, 0, 1 },
    left: ?*const List = null,
    right: ?*const List = null,
    start: usize = 0,
    len: usize = 0,
    flat: ?[]const v.Value = null,

    pub fn count(l: *const List) usize {
        return if (l.left == null) l.items.len else l.len;
    }

    pub fn at(l: *const List, index: usize) v.Value {
        std.debug.assert(index < l.count());
        var node = l;
        var i = index;
        while (true) {
            if (node.flat) |items| return items[i];
            const left = node.left orelse return node.items[i];
            if (node.right) |right| {
                if (i < left.count()) {
                    node = left;
                } else {
                    i -= left.count();
                    node = right;
                }
            } else {
                i += node.start;
                node = left;
            }
        }
    }

    pub fn concat(l: *const List, a: Allocator, r: *const List) Allocator.Error!*const List {
        const size = std.math.add(usize, l.count(), r.count()) catch return error.OutOfMemory;
        const result = try a.create(List);
        result.* = .{ .seq = l.seq, .left = l, .right = r, .len = size };
        return result;
    }

    pub fn slice(l: *const List, a: Allocator, start: usize, end: usize) Allocator.Error!*const List {
        std.debug.assert(start <= end and end <= l.count());
        const result = try a.create(List);
        result.* = .{ .seq = if (l.seq == .tuple) .tuple else .list, .left = l, .start = start, .len = end - start };
        return result;
    }

    pub fn values(l: *const List, a: Allocator) Allocator.Error![]const v.Value {
        if (l.flat) |items| return items;
        if (l.left == null) return l.items;
        const output = try a.alloc(v.Value, l.count());
        const Job = struct { node: *const List, start: usize, end: usize, dest: usize };
        var pending: std.ArrayList(Job) = .empty;
        defer pending.deinit(a);
        try pending.append(a, .{ .node = l, .start = 0, .end = l.count(), .dest = 0 });
        while (pending.pop()) |job| {
            if (job.start == job.end) continue;
            const node = job.node;
            if (node.flat orelse (if (node.left == null) node.items else null)) |items| {
                @memcpy(output[job.dest..][0 .. job.end - job.start], items[job.start..job.end]);
            } else if (node.right) |right| {
                const left = node.left.?;
                const split = left.count();
                if (job.end > split) try pending.append(a, .{
                    .node = right,
                    .start = job.start -| split,
                    .end = job.end - split,
                    .dest = job.dest + (split -| job.start),
                });
                if (job.start < split) try pending.append(a, .{
                    .node = left,
                    .start = job.start,
                    .end = @min(job.end, split),
                    .dest = job.dest,
                });
            } else try pending.append(a, .{
                .node = node.left.?,
                .start = node.start + job.start,
                .end = node.start + job.end,
                .dest = job.dest,
            });
        }
        // Every list is render-arena allocated; only this memo changes, never a visible element or prefix.
        @constCast(l).flat = output;
        return output;
    }
};
