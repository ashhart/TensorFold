//! An engine thread cannot serve requests after its backend initialization fails.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");

test "Failure boundaries: failed thread context initialization fails queued work without backend execution" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.pieceBackend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, io, &core, .{ .lanes = 2 });
    defer host.stop();
    const Box = struct {
        done: std.atomic.Value(bool) = .init(false),
        reason: ?api.Reason = null,
        message_ok: bool = false,
        tokens: usize = 0,
        fn init(_: *anyopaque) anyerror!void {
            return error.NoContext;
        }
        fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            switch (e.*) {
                .tokens => |ts| b.tokens += ts.len,
                .finished => |f| {
                    b.reason = f.reason;
                    b.message_ok = std.mem.eql(u8, f.message, "NoContext");
                    b.done.store(true, .release);
                },
                else => {},
            }
        }
    };
    var box: Box = .{};
    host.on_thread = .{ .ctx = &box, .run = Box.init };
    const request: api.Request = .{ .prompt = &.{ 1, 2, 3 }, .max_tokens = 2 };
    try host.engine().submit(1, &request, .{ .ctx = &box, .event = Box.event });
    try host.start();
    for (0..2000) |_| {
        if (box.done.load(.acquire)) break;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expect(box.done.load(.acquire));
    try std.testing.expectEqual(api.Reason.failed, box.reason.?);
    try std.testing.expect(box.message_ok);
    try std.testing.expectEqual(@as(usize, 0), box.tokens);
}

test "Failure boundaries: a failed round retains every admitted job when allocation fails" {
    const a = std.testing.allocator;
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.pieceBackend(), clock.clock());
    defer core.deinit();
    var failing = std.testing.FailingAllocator.init(a, .{});
    var host = api.LaneHost.init(failing.allocator(), std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop();
    const Box = struct {
        n: usize = 0,
        fn event(p: *anyopaque, _: api.Id, e: *const api.Event) void {
            const b: *@This() = @ptrCast(@alignCast(p));
            if (e.* == .finished and e.finished.reason == .failed) b.n += 1;
        }
    };
    var box: Box = .{};
    const request: api.Request = .{ .prompt = &.{ 1, 2, 3 }, .max_tokens = 2 };
    try host.engine().submit(1, &request, .{ .ctx = &box, .event = Box.event });
    const job = host.queued.items[0];
    try host.admitted.append(host.gpa, job);
    _ = host.queued.pop();
    job.stream.finished = true;
    defer if (box.n == 0) host.finish(job, .failed, "test cleanup");
    failing.fail_index = failing.alloc_index;
    @import("lane_host_end.zig").failAll(&host, "OutOfMemory");
    try std.testing.expectEqual(@as(usize, 1), box.n);
}
