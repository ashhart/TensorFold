//! Detailed invariants: docs/NOTES-CODE-COMMENTS.md, section 15.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub fn Overlay(comptime T: type) type {
    return struct {
        const Self = @This();
        base: []const T,
        limit: usize,
        changed: std.AutoHashMapUnmanaged(u32, T) = .empty,
        tail: std.ArrayList(T) = .empty,

        pub fn init(base: []const T, limit: usize) Self {
            return .{ .base = base, .limit = limit };
        }

        pub fn deinit(self: *Self, a: Allocator) void {
            self.changed.deinit(a);
            self.tail.deinit(a);
        }

        pub fn len(self: *const Self) usize {
            return self.base.len + self.tail.items.len;
        }

        pub fn get(self: *const Self, at: u32) T {
            std.debug.assert(at < self.len());
            if (at >= self.base.len) return self.tail.items[at - self.base.len];
            return self.changed.get(at) orelse self.base[at];
        }

        pub fn put(self: *Self, a: Allocator, at: u32, value: T) (Allocator.Error || error{ Full, Bounds })!void {
            if (at >= self.len()) return error.Bounds;
            if (at >= self.base.len) {
                self.tail.items[at - self.base.len] = value;
                return;
            }
            if (!self.changed.contains(at) and self.changed.count() + self.tail.items.len >= self.limit) return error.Full;
            try self.changed.put(a, at, value);
        }

        pub fn append(self: *Self, a: Allocator, value: T) (Allocator.Error || error{Full})!void {
            if (self.changed.count() + self.tail.items.len >= self.limit) return error.Full;
            try self.tail.append(a, value);
        }

        pub fn fork(self: *const Self, a: Allocator) Allocator.Error!Self {
            var out = Self.init(self.base, self.limit);
            errdefer out.deinit(a);
            out.changed = try self.changed.clone(a);
            try out.tail.appendSlice(a, self.tail.items);
            return out;
        }
    };
}

fn exercise(a: Allocator) !void {
    const base = [_]u32{ 10, 20, 30 };
    var old = Overlay(u32).init(&base, 2);
    defer old.deinit(a);
    try old.put(a, 1, 21);
    var next = try old.fork(a);
    defer next.deinit(a);
    try next.put(a, 1, 22);
    try next.append(a, 40);
    try std.testing.expectEqual(@as(u32, 21), old.get(1));
    try std.testing.expectEqual(@as(u32, 22), next.get(1));
    try std.testing.expectEqual(@as(u32, 40), next.get(3));
    try std.testing.expectError(error.Full, next.put(a, 0, 11));
    try std.testing.expectError(error.Full, next.append(a, 50));
    try std.testing.expectEqual(@as(u32, 10), next.get(0));
    try std.testing.expectEqual(@as(usize, 3), old.len());
}

test "overlay fork isolates versions and bounds changed metadata" {
    try exercise(std.testing.allocator);
}

test "overlay allocation failures leave borrowed base and forks owned correctly" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, exercise, .{});
}

test "full overlay permits existing updates but rejects out of bounds" {
    const a = std.testing.allocator;
    const base = [_]u32{ 10, 20 };
    var view = Overlay(u32).init(&base, 2);
    defer view.deinit(a);
    try view.put(a, 0, 11);
    try view.append(a, 30);
    try view.put(a, 0, 12);
    try view.put(a, 2, 31);
    try std.testing.expectError(error.Bounds, view.put(a, 3, 40));
    try std.testing.expectError(error.Full, view.put(a, 1, 21));
    try std.testing.expectEqual(@as(u32, 12), view.get(0));
    try std.testing.expectEqual(@as(u32, 20), view.get(1));
    try std.testing.expectEqual(@as(u32, 31), view.get(2));
    try std.testing.expectEqual(@as(u32, 10), base[0]);
}
