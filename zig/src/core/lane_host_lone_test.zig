//! The lone driver: reclaiming the lane core's last stream between rounds, and a halt during its prompt pass.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

const gpa = std.testing.allocator;

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?Reason = null,
    message: []const u8 = "",

    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| {
                b.done = f.reason;
                b.message = f.message;
            },
            else => {},
        }
    }

    fn wait(b: *Box) Reason {
        while (true) {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

/// Hands a new stream to the lane core after its first token; decodes a resumed one to its end by the fake's rule.
const Lone = struct {
    be: lanes.backend.Backend,
    opened: u32 = 0,
    resumed: u32 = 0,

    fn run(ctx: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const l: *Lone = @ptrCast(@alignCast(ctx));
        if (!hooks.resumed) {
            l.opened += 1;
            _ = try l.be.opening(gpa, s);
            hooks.committed(hooks.ctx);
            s.pending = s.context.items[s.context.items.len - 1];
            s.cache_len = s.prompt_len; // as a backend's hand-over leaves it: the first token not yet fed
            return true;
        }
        l.resumed += 1;
        while (!s.finished) {
            const t = lanes.fake.next(s.context.items, null, s.context.items.len);
            _ = try s.commit(gpa, &.{t}, &.{});
            hooks.committed(hooks.ctx);
        }
        l.be.release(s);
        return false;
    }
};

fn serve(request: *const Request, resumes: bool) !struct { opened: u32, resumed: u32 } {
    // a drafted family's windowed rounds, as the lone driver's backends run them (lanes/engine_test.zig's model)
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true }, 16, 15);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    var lone: Lone = .{ .be = target.backend() };
    host.lone = .{ .ctx = &lone, .run = Lone.run, .resumes = resumes };
    try host.start();
    defer host.stop();
    var box: Box = .{};
    defer box.tokens.deinit(gpa);
    try host.engine().submit(1, request, .{ .ctx = &box, .event = Box.event });
    try std.testing.expectEqual(Reason.length, box.wait());
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, request.prompt);
    for (box.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(usize, request.max_tokens), box.tokens.items.len);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    return .{ .opened = lone.opened, .resumed = lone.resumed };
}

const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };

test "the lane core's last stream goes back to the lone driver, its tokens served once each, in order" {
    const runs = try serve(&.{ .prompt = &prompt, .max_tokens = 24 }, true);
    try std.testing.expectEqual(@as(u32, 1), runs.opened);
    try std.testing.expectEqual(@as(u32, 1), runs.resumed);
}

test "a stream the lone driver does not take stays in the lane core to its end" {
    const runs = try serve(&.{ .prompt = &prompt, .max_tokens = 24, .drafts = false }, true);
    try std.testing.expectEqual(@as(u32, 0), runs.opened);
    try std.testing.expectEqual(@as(u32, 0), runs.resumed);
}

test "a lone driver that only opens streams never gets the lane core's last stream back" {
    const runs = try serve(&.{ .prompt = &prompt, .max_tokens = 24 }, false);
    try std.testing.expectEqual(@as(u32, 1), runs.opened);
    try std.testing.expectEqual(@as(u32, 0), runs.resumed);
}

test "halt fails a request in the lone driver's prompt pass with the reason" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    // the stream's lane is released once, by the driver's failed opening, not again when the halt ends the request
    const Releases = struct {
        n: u32 = 0,
        fn count(ctx: *anyopaque, _: *lanes.Stream) void {
            const r: *@This() = @ptrCast(@alignCast(ctx));
            r.n += 1;
        }
    };
    var releases: Releases = .{};
    target.release_hook = Releases.count;
    target.release_hook_ctx = &releases;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    // the driver holds its prompt pass until the halt is requested; the pass then finds the host closing
    const Held = struct {
        be: lanes.backend.Backend,
        entered: std.Io.Event = .unset,
        go: std.Io.Event = .unset,
        fn run(ctx: *anyopaque, s: *lanes.Stream, _: api.LoneHooks) anyerror!bool {
            const l: *@This() = @ptrCast(@alignCast(ctx));
            l.entered.set(std.testing.io);
            l.go.waitUncancelable(std.testing.io);
            _ = try l.be.opening(gpa, s); // error.Cancelled, the stream released
            return error.NotCancelled;
        }
    };
    var held: Held = .{ .be = target.backend() };
    host.lone = .{ .ctx = &held, .run = Held.run };
    try host.start();
    defer host.stop();
    var box: Box = .{};
    defer box.tokens.deinit(gpa);
    const request: Request = .{ .prompt = &prompt, .max_tokens = 24 };
    try host.engine().submit(1, &request, .{ .ctx = &box, .event = Box.event });
    held.entered.waitUncancelable(std.testing.io);
    const halter = try std.Thread.spawn(.{}, LaneHost.halt, .{ &host, "server restarting: retry shortly" });
    host.mutex.lockUncancelable(std.testing.io);
    while (host.halt_reason == null) host.wake.waitUncancelable(std.testing.io, &host.mutex);
    host.mutex.unlock(std.testing.io);
    held.go.set(std.testing.io);
    halter.join();
    try std.testing.expectEqual(@as(?Reason, .failed), box.done);
    try std.testing.expectEqualStrings("server restarting: retry shortly", box.message);
    try std.testing.expectEqual(@as(usize, 0), box.tokens.items.len);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    try std.testing.expectEqual(@as(u32, 1), releases.n);
}
