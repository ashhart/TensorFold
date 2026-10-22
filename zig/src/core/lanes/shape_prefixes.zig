//! Bounded draft-tree prefixes share one traversal and copy only the selected shape to its owner's allocator.
const std = @import("std");
const shape = @import("shape.zig");
const Allocator = std.mem.Allocator;

pub fn Prefixes(comptime capacity: usize) type {
    return struct {
        parents: [capacity]i32 = undefined,
        ranks: [capacity]u8 = undefined,
        depths: [capacity]u8 = undefined,
        sums: [capacity]f64 = undefined,
        count: usize = 0,

        const Self = @This();

        pub fn init(gpa: Allocator, odds: shape.Odds, lanes: usize, depth_cap: usize) !Self {
            std.debug.assert(lanes <= capacity);
            var out: Self = .{};
            var cursor = try shape.Cursor.init(gpa, odds, if (lanes > 0) depth_cap else 0);
            defer cursor.deinit(gpa);
            while (out.count < lanes) {
                const lane = try cursor.next(gpa) orelse break;
                const i = out.count;
                out.parents[i] = lane.parent;
                out.ranks[i] = lane.rank;
                out.depths[i] = lane.depth;
                out.sums[i] = cursor.expected;
                out.count += 1;
            }
            return out;
        }

        pub fn view(p: *Self, lanes: usize) shape.Shape {
            const n = @min(lanes, p.count);
            return .{ .parents = p.parents[0..n], .ranks = p.ranks[0..n], .depths = p.depths[0..n], .expected = if (n == 0) 0 else p.sums[n - 1] };
        }

        pub fn copy(p: *Self, gpa: Allocator, lanes: usize) !shape.Shape {
            const s = p.view(lanes);
            const parents = try gpa.dupe(i32, s.parents);
            errdefer gpa.free(parents);
            const ranks = try gpa.dupe(u8, s.ranks);
            errdefer gpa.free(ranks);
            return .{ .parents = parents, .ranks = ranks, .depths = try gpa.dupe(u8, s.depths), .expected = s.expected };
        }
    };
}
