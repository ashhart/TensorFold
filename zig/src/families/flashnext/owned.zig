//! Successfully acquired resources belong to a prefix immediately, before any later construction can fail.
const std = @import("std");

pub fn Owner(comptime Metal: type) type {
    return struct {
        const Self = @This();
        const Entry = union(enum) { object: Metal.objc.Id, queue: Metal.Queue };
        entries: std.ArrayList(Entry) = .empty,
        allocator: ?std.mem.Allocator = null,

        pub fn keep(o: *Self, a: std.mem.Allocator, value: anytype) !@TypeOf(value) {
            errdefer value.deinit();
            const item: Entry = if (@TypeOf(value) == Metal.Queue) .{ .queue = value } else .{ .object = value.id };
            try o.entries.append(a, item);
            o.allocator = a;
            return value;
        }

        pub fn drain(o: *const Self) void {
            for (o.entries.items) |item| if (item == .queue) {
                const cb = item.queue.commandBuffer();
                cb.commit();
                cb.wait();
            };
        }

        pub fn release(o: *Self) void {
            var n = o.entries.items.len;
            while (n > 0) {
                n -= 1;
                switch (o.entries.items[n]) {
                    .object => |id| Metal.objc.release(id),
                    .queue => |q| q.deinit(),
                }
            }
            if (o.allocator) |a| o.entries.deinit(a);
            o.* = .{};
        }

        pub fn deinit(o: *Self) void {
            o.drain();
            o.release();
        }
    };
}
