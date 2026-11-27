//! A Flash Next host's status after a reply finishes, and its health once drained: the job it reads must be alive.
const std = @import("std");
const api = @import("engine_api");
const Host = @import("flashnext_host.zig").Host;
const Job = @import("flashnext_host.zig").Job;

test "status after a reply finishes reads no freed job" {
    const gpa = std.testing.allocator;
    var h: Host = .{ .gpa = gpa, .io = std.testing.io, .eng = undefined, .info_ = undefined };
    defer h.decoded.deinit(gpa);
    const Reader = struct { // the server's status poll, here inside the finished event and after it
        h: *Host,
        running: u32 = 9,
        fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
            const r: *@This() = @ptrCast(@alignCast(ctx));
            var st: api.Status = .{};
            var toks: [2]u32 = .{ 0, 0 };
            if (e.* == .finished) {
                Host.statusFn(r.h, &st, &toks);
                r.running = st.running;
            }
        }
    };
    var reader: Reader = .{ .h = &h };
    const prompt = [_]u32{ 1, 2, 3 };
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 4 };
    const job = try gpa.create(Job);
    job.* = .{ .id = 1, .request = &request, .sink = .{ .ctx = &reader, .event = Reader.event } };
    try job.emitted.append(gpa, 7);
    h.lock();
    h.running = job; // as run() holds it while serve runs
    h.live_prompt = prompt.len;
    h.live_generated = 1;
    h.unlock();
    var st: api.Status = .{};
    var toks: [2]u32 = .{ 0, 0 };
    Host.statusFn(&h, &st, &toks);
    try std.testing.expectEqual(@as(u32, 1), st.running);
    try std.testing.expectEqual(@as(u32, 4), toks[0]);
    h.finish(job, .stop, .{}, ""); // serve's completion: the job is freed, serve has not returned
    try std.testing.expectEqual(@as(u32, 0), reader.running);
    Host.statusFn(&h, &st, &toks); // a poll before run() clears its own pointer
    try std.testing.expectEqual(@as(u32, 0), st.running);
    try std.testing.expectEqual(@as(usize, 0), st.streams);
}

test "a drained Flash Next host is unhealthy before another submit" {
    const gpa = std.testing.allocator;
    var host: Host = .{ .gpa = gpa, .io = std.testing.io, .eng = undefined, .info_ = .{} };
    defer host.decoded.deinit(gpa);
    const engine = host.engine();
    var status: api.Status = .{};
    engine.status(&status, &.{});
    try std.testing.expect(status.healthy);
    engine.drain();
    engine.status(&status, &.{});
    const Sink = struct {
        fn event(_: *anyopaque, _: api.Id, _: *const api.Event) void {
            @panic("a drained host must refuse admission");
        }
    };
    var ctx: u8 = 0;
    const request: api.Request = .{ .prompt = &.{1}, .max_tokens = 1 };
    try std.testing.expectError(error.Closed, engine.submit(1, &request, .{ .ctx = &ctx, .event = Sink.event }));
    try std.testing.expectEqual(@as(usize, 0), host.queued.items.len);
    try std.testing.expect(!status.healthy);
}
