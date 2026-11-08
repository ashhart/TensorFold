//! A Flash Next host that closes with replies still queued tells each one it finished.
const std = @import("std");
const api = @import("engine_api");
const Host = @import("flashnext_host.zig").Host;

const Box = struct {
    finished: ?api.Reason = null,
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        if (e.* == .finished) b.finished = e.finished.reason;
    }
};

// A close that copied the queue and failed would drop every reply unfinished: each mailbox would wait on.
test "a drained Flash Next host finishes every queued reply, even when it cannot copy the queue" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    const gpa = failing.allocator();
    var host: Host = .{ .gpa = gpa, .io = std.testing.io, .eng = undefined, .info_ = .{} };
    const engine = host.engine();
    const request: api.Request = .{ .prompt = &.{1}, .max_tokens = 1 };
    var boxes: [3]Box = @splat(.{});
    for (&boxes, 0..) |*b, i| try engine.submit(i, &request, .{ .ctx = b, .event = Box.event });
    var jobs: [3]@TypeOf(host.queued.items[0]) = undefined;
    @memcpy(&jobs, host.queued.items);
    var ids: [3]usize = undefined;
    for (jobs, &ids) |j, *id| id.* = @intCast(j.id);
    engine.drain(); // closing before the thread starts: it takes the closing path at once, never touching the engine
    failing.fail_index = failing.alloc_index; // the close path's next allocation fails: it must need none
    try host.start();
    engine.drain();
    failing.fail_index = std.math.maxInt(usize);
    var unheard: usize = 0;
    for (jobs, ids) |j, id| if (boxes[id].finished == null) {
        unheard += 1;
        gpa.destroy(j); // a close that dropped it never finished it: freed here, as finish would have
    };
    host.stop();
    try std.testing.expectEqual(@as(usize, 0), unheard);
    for (boxes) |b| try std.testing.expectEqual(api.Reason.cancelled, b.finished.?);
}
