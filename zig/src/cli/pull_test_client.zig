//! Pull tests own their client I/O and leak-checking allocator without stack unwinds on worker frees.
const std = @import("std");

pub const Client = struct {
    heap: std.heap.SafeAllocator = undefined,
    threaded: std.Io.Threaded = undefined,

    pub fn init(c: *Client) void {
        c.heap = .init(std.heap.page_allocator, .{ .stack_trace_frames = 0 });
        c.threaded = .init(c.heap.allocator(), .{});
    }

    pub fn allocator(c: *Client) std.mem.Allocator {
        return c.heap.allocator();
    }

    pub fn io(c: *Client) std.Io {
        return c.threaded.io();
    }

    pub fn deinit(c: *Client) void {
        c.threaded.deinit();
        const leaks = c.heap.deinit();
        if (leaks != 0) std.log.err("pull client test leaked {d} allocations", .{leaks});
    }
};

test "pull client I/O: cancelled futures and completed worker groups release their allocations" {
    var client: Client = undefined;
    client.init();
    defer client.deinit();
    const io = client.io();
    const Work = struct {
        fn wait(i: std.Io, entered: *std.Io.Event, finish: *std.Io.Event) std.Io.Cancelable!void {
            entered.set(i);
            try finish.wait(i);
        }
    };
    var entered: std.Io.Event = .unset;
    var finish: std.Io.Event = .unset;
    var future = try io.concurrent(Work.wait, .{ io, &entered, &finish });
    entered.waitUncancelable(io);
    try std.testing.expectError(error.Canceled, future.cancel(io));
    entered = .unset;
    var group: std.Io.Group = .init;
    try group.concurrent(io, Work.wait, .{ io, &entered, &finish });
    entered.waitUncancelable(io);
    finish.set(io);
    try group.await(io);
}
