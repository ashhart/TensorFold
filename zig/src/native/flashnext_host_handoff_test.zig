//! A Flash Next host exports no prompt states, so it refuses a hand-off request (--mcdma-produce) up front.
const std = @import("std");
const api = @import("engine_api");
const Host = @import("flashnext_host.zig").Host;

test "a Flash Next host refuses a hand-off, which it cannot export, before it queues anything" {
    const gpa = std.testing.allocator;
    var host: Host = .{ .gpa = gpa, .io = std.testing.io, .eng = undefined, .info_ = .{} };
    const Sink = struct {
        fn event(_: *anyopaque, _: api.Id, _: *const api.Event) void {
            @panic("a refused hand-off must not be admitted");
        }
    };
    const request: api.Request = .{ .prompt = &.{ 1, 2 }, .max_tokens = 1, .handoff = .{ .id = @splat(7) } };
    const sink: api.Sink = .{ .ctx = &host, .event = Sink.event }; // never called: the hand-off is refused first
    try std.testing.expectError(error.InvalidRequest, host.engine().submit(1, &request, sink));
    try std.testing.expectEqual(@as(usize, 0), host.queued.items.len);
}
