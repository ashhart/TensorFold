//! Admission against a byte ledger: requests wait for bytes, none is refused for room, every lease comes back.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;
const Request = api.Request;
const gpa = std.testing.allocator;

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
    fn waitFor(b: *@This(), ms: usize) ?Reason {
        for (0..ms) |_| {
            if (b.state()[1]) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return b.state()[1];
    }
    fn startsWithin(b: *@This(), ms: usize) bool {
        for (0..ms) |_| {
            if (b.state()[0] > 0) return true;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return b.state()[0] > 0;
    }
};

/// A request costs its prompt's length in bytes; the last reply limit it was asked about is kept.
const Cost = struct {
    last_max: std.atomic.Value(u32) = .init(0),
    fn cost(ctx: *anyopaque, prompt: u32, max_tokens: u32) u64 {
        const c: *Cost = @ptrCast(@alignCast(ctx));
        c.last_max.store(max_tokens, .monotonic);
        return prompt;
    }
};

const eight = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
const sixteen: [16]u32 = @splat(7);
const three = [_]u32{ 2, 7, 1 };

fn idle(host: *LaneHost) !void {
    try std.testing.expect(host.kv.?.idle());
}

fn promised(host: *LaneHost) u64 {
    host.lock();
    defer host.unlock();
    return host.kv.?.promised;
}

const Rig = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine = undefined,

    fn init(r: *Rig, memory_lanes: ?usize) !void {
        r.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        r.target = .{ .gpa = gpa, .memory_lanes = memory_lanes };
        r.clock = .{};
        r.core = lanes.Engine.init(gpa, &r.cfg, r.target.backend(), r.clock.clock());
    }
    fn deinit(r: *Rig) void {
        r.core.deinit();
        r.target.deinit();
        r.cfg.deinit(gpa);
    }
};

test "ledger: requests past the budget wait for bytes, then run as leases come back" {
    var r: Rig = undefined;
    try r.init(null);
    defer r.deinit();
    var c: Cost = .{};
    var host = LaneHost.init(gpa, std.testing.io, &r.core, .{ .lanes = 4, .ledger = .{ .ctx = &c, .total = 16, .cost = Cost.cost } });
    try host.start();
    var stopped = false;
    defer if (!stopped) host.stop(); // a failed check must not free the engine under a running thread
    const e = host.engine();
    const running: Request = .{ .prompt = &eight, .max_tokens = 100000 };
    var boxes: [3]Box = .{ .{}, .{}, .{} };
    try e.submit(1, &running, .{ .ctx = &boxes[0], .event = Box.event });
    try e.submit(2, &running, .{ .ctx = &boxes[1], .event = Box.event });
    try std.testing.expect(boxes[0].startsWithin(2000) and boxes[1].startsWithin(2000));
    try e.submit(3, &running, .{ .ctx = &boxes[2], .event = Box.event }); // 8 more bytes than the 16 the ledger holds
    std.Io.sleep(std.testing.io, .fromMilliseconds(100), .awake) catch {};
    try std.testing.expectEqual(@as(usize, 0), boxes[2].state()[0]);
    try std.testing.expectEqual(@as(?Reason, null), boxes[2].state()[1]); // waiting, not refused
    try std.testing.expectEqual(@as(u64, 16), promised(&host));
    e.cancel(1);
    try std.testing.expectEqual(@as(?Reason, .cancelled), boxes[0].waitFor(2000));
    try std.testing.expect(boxes[2].startsWithin(2000)); // its bytes came back
    try std.testing.expectEqual(@as(u32, 100000), c.last_max.load(.monotonic)); // the cost hears the reply limit
    e.cancel(2);
    e.cancel(3);
    try std.testing.expectEqual(@as(?Reason, .cancelled), boxes[2].waitFor(2000));
    host.stop();
    stopped = true;
    try idle(&host);
}

test "ledger: a waiting head is passed only while streams admitted before its wait still run" {
    var r: Rig = undefined;
    try r.init(null);
    defer r.deinit();
    var c: Cost = .{};
    var host = LaneHost.init(gpa, std.testing.io, &r.core, .{ .lanes = 4, .ledger = .{ .ctx = &c, .total = 16, .cost = Cost.cost } });
    try host.start();
    var stopped = false;
    defer if (!stopped) host.stop(); // a failed check must not free the engine under a running thread
    const e = host.engine();
    const running: Request = .{ .prompt = &eight, .max_tokens = 100000 };
    const big: Request = .{ .prompt = &sixteen, .max_tokens = 4 };
    const small: Request = .{ .prompt = &eight, .max_tokens = 4 };
    var a: Box = .{};
    var head: Box = .{};
    var b: Box = .{};
    var late: Box = .{};
    try e.submit(1, &running, .{ .ctx = &a, .event = Box.event });
    try std.testing.expect(a.startsWithin(2000));
    try e.submit(2, &big, .{ .ctx = &head, .event = Box.event }); // needs the whole budget
    try e.submit(3, &running, .{ .ctx = &b, .event = Box.event });
    try std.testing.expect(b.startsWithin(2000)); // passes the head while the stream it waits on runs
    e.cancel(1);
    try std.testing.expectEqual(@as(?Reason, .cancelled), a.waitFor(2000));
    try e.submit(4, &small, .{ .ctx = &late, .event = Box.event });
    std.Io.sleep(std.testing.io, .fromMilliseconds(100), .awake) catch {};
    try std.testing.expectEqual(@as(usize, 0), late.state()[0]); // only later streams run now: nothing passes the head
    try std.testing.expectEqual(@as(usize, 0), head.state()[0]);
    e.cancel(3);
    try std.testing.expectEqual(@as(?Reason, .cancelled), b.waitFor(2000));
    try std.testing.expectEqual(@as(?Reason, .length), head.waitFor(2000));
    try std.testing.expectEqual(@as(?Reason, .length), late.waitFor(2000));
    host.stop();
    stopped = true;
    try idle(&host);
}

test "ledger: a request past the whole budget fails in words instead of waiting forever" {
    var r: Rig = undefined;
    try r.init(null);
    defer r.deinit();
    var c: Cost = .{};
    var host = LaneHost.init(gpa, std.testing.io, &r.core, .{ .lanes = 2, .ledger = .{ .ctx = &c, .total = 4, .cost = Cost.cost } });
    try host.start();
    var stopped = false;
    defer if (!stopped) host.stop(); // a failed check must not free the engine under a running thread
    const e = host.engine();
    const too_big: Request = .{ .prompt = &eight, .max_tokens = 4 };
    const fits: Request = .{ .prompt = &three, .max_tokens = 4 };
    var a: Box = .{};
    var b: Box = .{};
    try e.submit(1, &too_big, .{ .ctx = &a, .event = Box.event });
    try std.testing.expectEqual(@as(?Reason, .failed), a.waitFor(2000));
    try std.testing.expect(std.mem.indexOf(u8, a.message, "KV budget") != null);
    try e.submit(2, &fits, .{ .ctx = &b, .event = Box.event });
    try std.testing.expectEqual(@as(?Reason, .length), b.waitFor(2000)); // the host still serves
    host.stop();
    stopped = true;
    try idle(&host);
}

test "ledger: a lane the backend has no memory for returns its lease and waits" {
    var r: Rig = undefined;
    try r.init(1);
    defer r.deinit();
    var c: Cost = .{};
    var host = LaneHost.init(gpa, std.testing.io, &r.core, .{ .lanes = 2, .ledger = .{ .ctx = &c, .total = 64, .cost = Cost.cost } });
    try host.start();
    var stopped = false;
    defer if (!stopped) host.stop(); // a failed check must not free the engine under a running thread
    const e = host.engine();
    const running: Request = .{ .prompt = &eight, .max_tokens = 100000 };
    const small: Request = .{ .prompt = &sixteen, .max_tokens = 4 };
    var a: Box = .{};
    var b: Box = .{};
    try e.submit(1, &running, .{ .ctx = &a, .event = Box.event });
    try std.testing.expect(a.startsWithin(2000));
    try e.submit(2, &small, .{ .ctx = &b, .event = Box.event }); // the ledger fits it; the backend has no lane
    std.Io.sleep(std.testing.io, .fromMilliseconds(50), .awake) catch {};
    try std.testing.expectEqual(@as(usize, 0), b.state()[0]);
    try std.testing.expectEqual(@as(u64, 8), promised(&host)); // the requeued job holds nothing
    e.cancel(1);
    try std.testing.expectEqual(@as(?Reason, .length), b.waitFor(2000));
    host.stop();
    stopped = true;
    try idle(&host);
}

test "ledger: cancelled while queued or running, every lease comes back" {
    var r: Rig = undefined;
    try r.init(null);
    defer r.deinit();
    var c: Cost = .{};
    var host = LaneHost.init(gpa, std.testing.io, &r.core, .{ .lanes = 4, .ledger = .{ .ctx = &c, .total = 8, .cost = Cost.cost } });
    try host.start();
    var stopped = false;
    defer if (!stopped) host.stop(); // a failed check must not free the engine under a running thread
    const e = host.engine();
    const running: Request = .{ .prompt = &eight, .max_tokens = 100000 };
    var a: Box = .{};
    var b: Box = .{};
    try e.submit(1, &running, .{ .ctx = &a, .event = Box.event });
    try std.testing.expect(a.startsWithin(2000));
    try e.submit(2, &running, .{ .ctx = &b, .event = Box.event });
    e.cancel(2); // still queued: it never promised
    try std.testing.expectEqual(@as(?Reason, .cancelled), b.waitFor(2000));
    e.cancel(1);
    try std.testing.expectEqual(@as(?Reason, .cancelled), a.waitFor(2000));
    try std.testing.expectEqual(@as(u64, 0), promised(&host));
    host.stop();
    stopped = true;
    try idle(&host);
}

test "ledger: prompts batched in one pass each hold a lease and never exceed the budget" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.batched(), clock.clock());
    defer core.deinit();
    var c: Cost = .{};
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4, .ledger = .{ .ctx = &c, .total = 16, .cost = Cost.cost } });
    const Watch = struct {
        host: *LaneHost,
        most: u64 = 0,
        fn chunk(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            w.most = @max(w.most, promised(w.host));
        }
    };
    var watch: Watch = .{ .host = &host };
    target.prefill_hook = Watch.chunk;
    target.prefill_hook_ctx = &watch;
    const small: Request = .{ .prompt = &eight, .max_tokens = 4 };
    var boxes: [3]Box = .{ .{}, .{}, .{} };
    for (&boxes, 1..) |*box, id| try host.engine().submit(id, &small, .{ .ctx = box, .event = Box.event });
    try host.start(); // all three queued before the first pass: it takes what fits
    var stopped = false;
    defer if (!stopped) host.stop(); // a failed check must not free the engine under a running thread
    for (&boxes) |*box| try std.testing.expectEqual(@as(?Reason, .length), box.waitFor(2000));
    host.stop();
    stopped = true;
    try std.testing.expect(watch.most > 0 and watch.most <= 16);
    try idle(&host);
}

test "ledger: a round planner, which admits past takeJob's promise, is refused at start" {
    var r: Rig = undefined;
    try r.init(null);
    defer r.deinit();
    var c: Cost = .{};
    var host = LaneHost.init(gpa, std.testing.io, &r.core, .{ .lanes = 2, .ledger = .{ .ctx = &c, .total = 16, .cost = Cost.cost } });
    host.rounds = .{ .ctx = &c, .vtable = undefined }; // never called: start refuses first
    try std.testing.expectError(error.LedgerWithRounds, host.start());
}
