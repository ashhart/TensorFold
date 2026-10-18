//! A round the GPU fails ends the requests it held; a queued request and later ones still run.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;

const Box = struct {
    mutex: std.Io.Mutex = .init,
    done: ?Reason = null,
    message: []const u8 = "",
    tokens: usize = 0,
    finishes: usize = 0,
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| b.tokens += t.len,
            .finished => |f| {
                b.finishes += 1;
                b.done = f.reason;
                b.message = f.message;
            },
            else => {},
        }
    }
    fn waitFor(b: *@This(), ms: usize) ?Reason {
        for (0..ms) |_| {
            b.mutex.lockUncancelable(std.testing.io);
            const done = b.done;
            b.mutex.unlock(std.testing.io);
            if (done != null) return done;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return null;
    }
    fn wait(b: *@This()) Reason {
        while (true) {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

test "a failed round ends the request it held; the queued request and a later one run" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7); // every round verifies
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .fail_round = 3 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try host.start();
    defer host.stop();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 48 };
    const e = host.engine();
    var held: Box = .{};
    var queued: Box = .{};
    try e.submit(1, &request, .{ .ctx = &held, .event = Box.event });
    try e.submit(2, &request, .{ .ctx = &queued, .event = Box.event });
    try std.testing.expectEqual(Reason.failed, held.wait());
    try std.testing.expectEqualStrings("GpuFailed", held.message);
    try std.testing.expectEqual(Reason.length, queued.wait());
    try std.testing.expectEqual(@as(usize, 48), queued.tokens);
    var later: Box = .{};
    try e.submit(3, &request, .{ .ctx = &later, .event = Box.event });
    try std.testing.expectEqual(Reason.length, later.wait());
    try std.testing.expectEqual(@as(usize, 48), later.tokens);
}

/// The last running stream frees a session, but moving its paused peer into it fails.
fn cancelledResumeFailure(in_prefill: bool) !void {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Ctx = struct {
        engine: api.Engine,
        waiting: ?*lanes.Stream = null,
        in_prefill: bool,
        fn prefill(ptr: *anyopaque, stream: *lanes.Stream, _: usize) void {
            const ctx: *@This() = @ptrCast(@alignCast(ptr));
            if (stream.prompt()[0] == 2) {
                ctx.waiting = stream;
                stream.paused = true;
            } else if (ctx.in_prefill) ctx.engine.cancel(2);
        }
        fn release(ptr: *anyopaque, stream: *lanes.Stream) void {
            const ctx: *@This() = @ptrCast(@alignCast(ptr));
            const waiting = ctx.waiting orelse return;
            if (stream == waiting) return;
            waiting.paused = false;
            waiting.finished = true;
            waiting.reason = .@"error";
            waiting.problem = "migration failed";
        }
    };
    var ctx: Ctx = .{ .engine = host.engine(), .in_prefill = in_prefill };
    target.prefill_hook = Ctx.prefill;
    target.prefill_hook_ctx = &ctx;
    target.release_hook = Ctx.release;
    target.release_hook_ctx = &ctx;
    const CancelBox = struct {
        box: Box = .{},
        engine: api.Engine,
        cancelled: bool = false,
        fn event(ptr: *anyopaque, id: api.Id, event_: *const api.Event) void {
            const b: *@This() = @ptrCast(@alignCast(ptr));
            Box.event(&b.box, id, event_);
            if (event_.* == .tokens and !b.cancelled) {
                b.cancelled = true;
                b.engine.cancel(id);
            }
        }
    };
    var waiting: Box = .{};
    var cancelled: CancelBox = .{ .engine = ctx.engine };
    const request: api.Request = .{ .prompt = &.{ 2, 7, 1, 8 }, .max_tokens = 100000, .drafts = false };
    try ctx.engine.submit(1, &request, .{ .ctx = &waiting, .event = Box.event });
    var second = request;
    second.prompt = &.{ 8, 1, 2, 3 };
    try ctx.engine.submit(2, &second, .{ .ctx = &cancelled, .event = CancelBox.event });
    try host.start();
    var stopped = false;
    defer if (!stopped) host.stop();
    try std.testing.expectEqual(@as(?Reason, .cancelled), cancelled.box.waitFor(2000));
    try std.testing.expectEqual(@as(?Reason, .failed), waiting.waitFor(2000));
    host.stop();
    stopped = true;
    try std.testing.expectEqualStrings("migration failed", waiting.message);
    try std.testing.expectEqual(@as(usize, 1), waiting.finishes);
    try std.testing.expectEqual(@as(usize, 1), cancelled.box.finishes);
    try std.testing.expectEqual(@as(usize, 0), core.live.items.len);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
}

test "cancel of the last running stream delivers its paused peer's migration failure" {
    try cancelledResumeFailure(false);
}

test "cancel during prefill delivers its paused peer's migration failure after reaping it" {
    try cancelledResumeFailure(true);
}

test "a failed shared keep reaps an already finished stream before freeing its job" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true, .streams_exact = true, .max_streams = 2 }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    const Fault = struct {
        fn keep(ptr: *anyopaque, windows: []const lanes.backend.Window, paths: []const []const u32) !void {
            if (windows.len == 2) for (windows) |window| {
                if (window.stream.finished) return error.GpuFailed;
            };
            const fake: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            return fake.backend().vtable.keep(ptr, windows, paths);
        }
    };
    var vtable = target.backend().vtable.*;
    vtable.keep = Fault.keep;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, .{ .ptr = &target, .vtable = &vtable }, clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    var short: Box = .{};
    var long: Box = .{};
    const a: api.Request = .{ .prompt = &.{ 2, 7, 1, 8 }, .max_tokens = 2, .drafts = false };
    var b = a;
    b.max_tokens = 40;
    const engine = host.engine();
    try engine.submit(1, &a, .{ .ctx = &short, .event = Box.event });
    try engine.submit(2, &b, .{ .ctx = &long, .event = Box.event });
    try host.start();
    var stopped = false;
    defer if (!stopped) host.stop();
    try std.testing.expectEqual(@as(?Reason, .failed), short.waitFor(2000));
    try std.testing.expectEqual(@as(?Reason, .failed), long.waitFor(2000));
    host.stop();
    stopped = true;
    try std.testing.expectEqualStrings("GpuFailed", short.message);
    try std.testing.expectEqualStrings("GpuFailed", long.message);
    try std.testing.expectEqual(@as(usize, 1), short.finishes);
    try std.testing.expectEqual(@as(usize, 1), long.finishes);
    try std.testing.expectEqual(@as(usize, 0), core.live.items.len);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
}
