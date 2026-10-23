//! Admission under memory: requests behind one that waits for room still run, the waiter goes next, nothing spins.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;
const Request = api.Request;

const Box = struct {
    mutex: std.Io.Mutex = .init,
    done: ?Reason = null,
    message: []const u8 = "",
    tokens: usize = 0,
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| b.tokens += t.len,
            .finished => |f| {
                b.done = f.reason;
                b.message = f.message;
            },
            else => {},
        }
    }
    fn state(b: *@This()) struct { usize, ?Reason } {
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        return .{ b.tokens, b.done };
    }
    /// Its end within `ms` milliseconds, else null.
    fn waitFor(b: *@This(), ms: usize) ?Reason {
        for (0..ms) |_| {
            if (b.state()[1]) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return b.state()[1];
    }
    /// Whether it has tokens within `ms` milliseconds.
    fn startsWithin(b: *@This(), ms: usize) bool {
        for (0..ms) |_| {
            if (b.state()[0] > 0) return true;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return b.state()[0] > 0;
    }
};

/// A pool where prompts of 64 tokens or more fit only when nothing else runs, and shorter ones always fit.
const Pool = struct {
    fn fits(_: *anyopaque, prompt: u32, _: u32) bool {
        return prompt < 64;
    }
};

const short_prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
const long_prompt: [64]u32 = @splat(7);

test "a short request behind one that waits for pool room runs beside the running stream" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var unused: u8 = 0;
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 3, .admit = .{ .ctx = &unused, .fits = Pool.fits } });
    try host.start();
    defer host.stop();
    const e = host.engine();
    var a: Box = .{};
    var long: Box = .{};
    var short: Box = .{};
    const running: Request = .{ .prompt = &short_prompt, .max_tokens = 100000 };
    const big: Request = .{ .prompt = &long_prompt, .max_tokens = 4 };
    const small: Request = .{ .prompt = &short_prompt, .max_tokens = 4 };
    try e.submit(1, &running, .{ .ctx = &a, .event = Box.event });
    try std.testing.expect(a.startsWithin(2000));
    try e.submit(2, &big, .{ .ctx = &long, .event = Box.event });
    try e.submit(3, &small, .{ .ctx = &short, .event = Box.event });
    try std.testing.expectEqual(@as(?Reason, .length), short.waitFor(2000)); // it fits now: it does not wait behind the long one
    try std.testing.expectEqual(@as(usize, 4), short.state()[0]);
    try std.testing.expectEqual(@as(usize, 0), long.state()[0]); // the long one still waits for the room the first holds
    e.cancel(1);
    try std.testing.expectEqual(@as(?Reason, .cancelled), a.waitFor(2000));
    try std.testing.expectEqual(@as(?Reason, .length), long.waitFor(2000));
}

test "a request that waited for pool room goes next once the streams it waited on end" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var unused: u8 = 0;
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 3, .admit = .{ .ctx = &unused, .fits = Pool.fits } });
    try host.start();
    defer host.stop();
    const e = host.engine();
    var a: Box = .{};
    var long: Box = .{};
    var b: Box = .{};
    var late: Box = .{};
    const running: Request = .{ .prompt = &short_prompt, .max_tokens = 100000 };
    const big: Request = .{ .prompt = &long_prompt, .max_tokens = 4 };
    const small: Request = .{ .prompt = &short_prompt, .max_tokens = 4 };
    try e.submit(1, &running, .{ .ctx = &a, .event = Box.event });
    try std.testing.expect(a.startsWithin(2000));
    try e.submit(2, &big, .{ .ctx = &long, .event = Box.event });
    try e.submit(3, &running, .{ .ctx = &b, .event = Box.event });
    try std.testing.expect(b.startsWithin(2000)); // admitted past the waiter while the stream it waits on runs
    e.cancel(1);
    try std.testing.expectEqual(@as(?Reason, .cancelled), a.waitFor(2000));
    try e.submit(4, &small, .{ .ctx = &late, .event = Box.event });
    std.Io.sleep(std.testing.io, .fromMilliseconds(100), .awake) catch {};
    try std.testing.expectEqual(@as(usize, 0), late.state()[0]); // what it waited on has ended: no one passes it now
    try std.testing.expectEqual(@as(usize, 0), long.state()[0]);
    e.cancel(3);
    try std.testing.expectEqual(@as(?Reason, .cancelled), b.waitFor(2000));
    try std.testing.expectEqual(@as(?Reason, .length), long.waitFor(2000));
    try std.testing.expectEqual(@as(?Reason, .length), late.waitFor(2000));
}

test "a lane the backend had no memory for is tried again once a lane frees, not given up for good" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .memory_lanes = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const e = host.engine();
    const running: Request = .{ .prompt = &short_prompt, .max_tokens = 100000 };
    const small: Request = .{ .prompt = &short_prompt, .max_tokens = 4 };
    var a: Box = .{};
    var b: Box = .{};
    try e.submit(1, &running, .{ .ctx = &a, .event = Box.event });
    try std.testing.expect(a.startsWithin(2000));
    try e.submit(2, &small, .{ .ctx = &b, .event = Box.event }); // no memory for a second lane yet: it waits
    std.Io.sleep(std.testing.io, .fromMilliseconds(50), .awake) catch {};
    try std.testing.expectEqual(@as(usize, 0), b.state()[0]);
    e.cancel(1);
    try std.testing.expectEqual(@as(?Reason, .length), b.waitFor(2000));
    target.memory_lanes = null; // the memory is back (the host is idle: its next submit orders this write)
    var c: Box = .{};
    var d: Box = .{};
    try e.submit(3, &running, .{ .ctx = &c, .event = Box.event });
    try e.submit(4, &running, .{ .ctx = &d, .event = Box.event });
    try std.testing.expect(c.startsWithin(2000));
    try std.testing.expect(d.startsWithin(2000)); // both lanes again
    e.cancel(3);
    e.cancel(4);
    try std.testing.expectEqual(@as(?Reason, .cancelled), c.waitFor(2000));
    try std.testing.expectEqual(@as(?Reason, .cancelled), d.waitFor(2000));
}

test "a request the backend has no memory for with nothing else running fails in words instead of spinning" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .memory_lanes = 0 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const e = host.engine();
    const small: Request = .{ .prompt = &short_prompt, .max_tokens = 4 };
    var a: Box = .{};
    try e.submit(1, &small, .{ .ctx = &a, .event = Box.event });
    try std.testing.expectEqual(@as(?Reason, .failed), a.waitFor(2000));
    try std.testing.expect(std.mem.indexOf(u8, a.message, "memory") != null);
    var b: Box = .{};
    target.memory_lanes = null;
    try e.submit(2, &small, .{ .ctx = &b, .event = Box.event });
    try std.testing.expectEqual(@as(?Reason, .length), b.waitFor(2000)); // the host still serves
}

test "a prompt requeued after memory exhaustion can be cancelled before it fills" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .memory_lanes = 1, .prefill_chunks = 12 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Watch = struct {
        engine: api.Engine,
        saw_waiter: bool = false,

        fn chunk(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            var status: api.Status = .{};
            w.engine.status(&status, &.{});
            if (status.waiting != 1 or status.fill_prompt_tokens != @as(?u32, short_prompt.len) or w.saw_waiter) return;
            w.saw_waiter = true;
            w.engine.cancel(2);
        }
    };
    const e = host.engine();
    var watch: Watch = .{ .engine = e };
    target.prefill_hook = Watch.chunk;
    target.prefill_hook_ctx = &watch;
    const requests = [_]Request{
        .{ .prompt = &short_prompt, .max_tokens = 4, .drafts = false },
        .{ .prompt = &long_prompt, .max_tokens = 4, .drafts = false },
    };
    var boxes: [2]Box = .{ .{}, .{} };
    for (&requests, &boxes, 1..) |*request, *box, id|
        try e.submit(id, request, .{ .ctx = box, .event = Box.event });
    try host.start();
    var stopped = false;
    defer if (!stopped) host.stop();
    try std.testing.expectEqual(@as(?Reason, .length), boxes[0].waitFor(2000));
    try std.testing.expectEqual(@as(?Reason, .cancelled), boxes[1].waitFor(2000));
    host.stop();
    stopped = true;
    try std.testing.expect(watch.saw_waiter);
    try std.testing.expectEqual(@as(usize, 0), boxes[1].state()[0]);
    try std.testing.expectEqual(@as(u32, 0), target.lanes.count());
}
