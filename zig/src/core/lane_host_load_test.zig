//! A lane host's `requests` group in the load ledger: queue counts, requeues, background yields, detach.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Ledger = api.load_ledger.Ledger;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Engine = api.Engine;

/// How a request finished, as the host reports it.
const Box = struct {
    mutex: std.Io.Mutex = .init,
    done: ?Reason = null,
    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        if (e.* == .finished) b.done = e.finished.reason;
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
    fn sink(b: *@This()) api.Sink {
        return .{ .ctx = b, .event = event };
    }
};

const Rig = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine = undefined,
    host: LaneHost = undefined,

    fn init(r: *Rig, info: api.Info) !void {
        const gpa = std.testing.allocator;
        r.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        r.target = .{ .gpa = gpa, .prefill_chunks = 3 };
        r.clock = .{};
        r.core = lanes.Engine.init(gpa, &r.cfg, r.target.backend(), r.clock.clock());
        r.host = LaneHost.init(gpa, std.testing.io, &r.core, info);
    }
    fn deinit(r: *Rig) void {
        r.host.stop();
        r.core.deinit();
        r.target.deinit();
        r.cfg.deinit(std.testing.allocator);
    }
};

test "queued requests show in the ledger as they arrive, before the engine thread takes them" {
    var rig: Rig = undefined;
    try rig.init(.{ .lanes = 2 });
    defer rig.deinit();
    var ledger: Ledger = .init(std.testing.io);
    ledger.attach(rig.host.engine());
    const short = [_]u32{ 2, 7, 1 };
    const long = [_]u32{ 2, 7, 1, 8, 2, 8 };
    var a: Box = .{};
    var b: Box = .{};
    const ra: Request = .{ .prompt = &short, .max_tokens = 4 };
    const rb: Request = .{ .prompt = &long, .max_tokens = 4 };
    try rig.host.engine().submit(1, &ra, a.sink());
    try rig.host.engine().submit(2, &rb, b.sink());
    var started = false;
    errdefer if (!started) { // a failed check still runs the queued jobs out, so the host can stop
        rig.host.start() catch {};
        _ = a.wait();
        _ = b.wait();
    };
    const queued = ledger.read().requests.?;
    try std.testing.expectEqual(@as(u32, 2), queued.waiting);
    try std.testing.expectEqual(@as(u32, 0), queued.running);
    try std.testing.expectEqual(@as(u64, 9), queued.waiting_prompt_tokens);
    try std.testing.expectEqual(@as(?u64, 0), queued.filling_left); // nothing fills: a zero, not unknown
    try std.testing.expectEqual(@as(u64, 2), queued.started);
    started = true;
    try rig.host.start();
    _ = a.wait();
    _ = b.wait();
    rig.host.lock(); // the finish's unlock has published by the time the reply's last event lands
    rig.host.unlock();
    const done = ledger.read().requests.?;
    try std.testing.expectEqual(@as(u32, 0), done.waiting);
    try std.testing.expectEqual(@as(u64, 0), done.waiting_prompt_tokens);
    var status: api.Status = .{};
    rig.host.engine().status(&status, &.{});
    try std.testing.expect(status.preemptions == null); // the ledger counts give-ups; no constant zero
    ledger.detach(false);
}

test "a request that finds no lane memory queues again and counts one requeue" {
    var rig: Rig = undefined;
    try rig.init(.{ .lanes = 2 });
    defer rig.deinit();
    rig.target.memory_lanes = 1; // the second lane's prefill fails with LaneMemory until the first ends
    var ledger: Ledger = .init(std.testing.io);
    ledger.attach(rig.host.engine());
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var a: Box = .{};
    var b: Box = .{};
    const ra: Request = .{ .prompt = &prompt, .max_tokens = 32 };
    const rb: Request = .{ .prompt = &prompt, .max_tokens = 4 };
    try rig.host.engine().submit(1, &ra, a.sink());
    try rig.host.engine().submit(2, &rb, b.sink()); // both queued first: the second meets the first's held lane
    try rig.host.start();
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(Reason.length, b.wait());
    rig.host.lock();
    rig.host.unlock();
    const s = ledger.read().requests.?;
    try std.testing.expectEqual(@as(u64, 1), s.requeued);
    try std.testing.expectEqual(@as(u64, 0), s.yields);
    ledger.detach(false);
}

test "a warm pass that steps aside for a request counts one yield; its cancel counts none" {
    var rig: Rig = undefined;
    try rig.init(.{ .lanes = 2, .warm_turns = true });
    defer rig.deinit();
    rig.host.warm_quiet = 30 * std.time.ns_per_s; // it waits for quiet: the request ends the wait
    var ledger: Ledger = .init(std.testing.io);
    ledger.attach(rig.host.engine());
    try rig.host.start();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var warm: Box = .{};
    var real: Box = .{};
    const warm_request: Request = .{ .prompt = &prompt, .max_tokens = 0, .background = true };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 8 };
    try rig.host.engine().submit(1, &warm_request, warm.sink());
    while (true) { // the engine thread has taken the pass
        rig.host.lock();
        const taken = rig.host.queued.items.len == 0;
        rig.host.unlock();
        if (taken) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try rig.host.engine().submit(2, &request, real.sink());
    try std.testing.expectEqual(Reason.cancelled, warm.wait());
    _ = real.wait();
    try std.testing.expectEqual(@as(u64, 1), ledger.read().requests.?.yields);
    var cancelled: Box = .{};
    try rig.host.engine().submit(3, &warm_request, cancelled.sink());
    while (true) {
        rig.host.lock();
        const taken = rig.host.queued.items.len == 0;
        rig.host.unlock();
        if (taken) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    rig.host.engine().cancel(3);
    try std.testing.expectEqual(Reason.cancelled, cancelled.wait());
    rig.host.lock();
    rig.host.unlock();
    try std.testing.expectEqual(@as(u64, 1), ledger.read().requests.?.yields);
    ledger.detach(false);
}

test "after detach the host's work reaches the ledger no more" {
    var rig: Rig = undefined;
    try rig.init(.{ .lanes = 2 });
    defer rig.deinit();
    var ledger: Ledger = .init(std.testing.io);
    ledger.attach(rig.host.engine());
    ledger.detach(true);
    try rig.host.start();
    const prompt = [_]u32{ 2, 7, 1 };
    var a: Box = .{};
    const ra: Request = .{ .prompt = &prompt, .max_tokens = 4 };
    try rig.host.engine().submit(1, &ra, a.sink());
    _ = a.wait();
    const s = ledger.read();
    try std.testing.expect(s.parked and s.requests == null);
}
