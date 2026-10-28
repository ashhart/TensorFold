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

    const Kind = enum { plain, batched, stepped };

    fn init(r: *Rig, info: api.Info) !void {
        return r.initWith(info, std.testing.io, .plain);
    }

    fn initWith(r: *Rig, info: api.Info, io: std.Io, kind: Kind) !void {
        const gpa = std.testing.allocator;
        r.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        r.target = .{ .gpa = gpa, .prefill_chunks = 3 };
        r.clock = .{};
        const backend = switch (kind) {
            .plain => r.target.backend(),
            .batched => r.target.batched(),
            .stepped => r.target.stepped(),
        };
        r.core = lanes.Engine.init(gpa, &r.cfg, backend, r.clock.clock());
        r.host = LaneHost.init(gpa, io, &r.core, info);
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

/// An awake clock only prompt chunks move: each chunk the fake runs takes `chunk_ns`, all else takes no time.
const Frozen = struct {
    const chunk_ns = 1000;
    var ns: std.atomic.Value(i64) = .init(1);
    var chunks: std.atomic.Value(u64) = .init(0);
    var table: std.Io.VTable = undefined;
    fn now(ctx: ?*anyopaque, kind: std.Io.Clock) std.Io.Timestamp {
        if (kind == .awake) return .{ .nanoseconds = ns.load(.monotonic) };
        return std.testing.io.vtable.now(ctx, kind);
    }
    fn io() std.Io {
        table = std.testing.io.vtable.*;
        table.now = now;
        chunks.store(0, .monotonic);
        return .{ .userdata = std.testing.io.userdata, .vtable = &table };
    }
    fn tick(_: *anyopaque, _: *lanes.Stream, _: usize) void {
        _ = ns.fetchAdd(chunk_ns, .monotonic);
        _ = chunks.fetchAdd(1, .monotonic);
    }
    /// Busy time is exactly the chunks' time: no waiting between calls, nothing twice.
    fn expectBusy(work: api.load_ledger.Work) !void {
        try std.testing.expectEqual(chunks.load(.monotonic) * chunk_ns, work.prefill_busy_ns);
    }
};

fn settled(rig: *Rig, ledger: *Ledger) api.load_ledger.Work {
    rig.host.lock(); // the last unlock has published
    rig.host.unlock();
    return ledger.read().work.?;
}

fn runTwo(kind: Rig.Kind, memory_lanes: ?usize) !api.load_ledger.Work {
    var rig: Rig = undefined;
    try rig.initWith(.{ .lanes = 2 }, Frozen.io(), kind);
    defer rig.deinit();
    rig.target.memory_lanes = memory_lanes;
    rig.target.prefill_hook = Frozen.tick;
    rig.target.prefill_hook_ctx = &rig;
    var ledger: Ledger = .init(std.testing.io);
    defer ledger.detach(false);
    ledger.attach(rig.host.engine());
    const one = [_]u32{ 2, 7, 1, 8, 2, 8 };
    const two = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5 };
    var a: Box = .{};
    var b: Box = .{};
    const ra: Request = .{ .prompt = &one, .max_tokens = 8 };
    const rb: Request = .{ .prompt = &two, .max_tokens = 4 };
    try rig.host.engine().submit(1, &ra, a.sink());
    try rig.host.engine().submit(2, &rb, b.sink()); // both queued: a batching backend takes them in one call
    try rig.host.start();
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(Reason.length, b.wait());
    if (kind == .batched) try std.testing.expectEqual(@as(usize, 1), rig.target.many_passes);
    return settled(&rig, &ledger);
}

test "a request counts its computed prompt once, its output after the first token, and its prompt calls' time" {
    var rig: Rig = undefined;
    try rig.initWith(.{ .lanes = 2 }, Frozen.io(), .plain);
    defer rig.deinit();
    rig.target.prefill_hook = Frozen.tick;
    rig.target.prefill_hook_ctx = &rig;
    var ledger: Ledger = .init(std.testing.io);
    defer ledger.detach(false);
    ledger.attach(rig.host.engine());
    try rig.host.start();
    const prompt = [_]u32{ 2, 7, 1, 8, 2, 8 };
    var a: Box = .{};
    const ra: Request = .{ .prompt = &prompt, .max_tokens = 8 };
    try rig.host.engine().submit(1, &ra, a.sink());
    _ = a.wait();
    const w = settled(&rig, &ledger);
    try std.testing.expectEqual(@as(u64, 6), w.prefill_compute);
    try std.testing.expectEqual(@as(u64, 0), w.prefill_cache);
    try std.testing.expectEqual(@as(u64, 7), w.decode); // the first of 8 came from the prompt pass
    try std.testing.expect(w.prefill_busy_ns > 0);
    try Frozen.expectBusy(w);
}

test "chunked passes count each chunk once and only their own chunks' time" {
    const w = try runTwo(.stepped, null);
    try std.testing.expectEqual(@as(u64, 15), w.prefill_compute);
    try Frozen.expectBusy(w);
}

test "a batched call counts its prompts and its time once" {
    const w = try runTwo(.batched, null);
    try std.testing.expectEqual(@as(u64, 15), w.prefill_compute);
    try Frozen.expectBusy(w);
}

test "a requeued request counts the work of each attempt" {
    const w = try runTwo(.plain, 1); // the second lane's pass fails for memory, then runs once the first ends
    try std.testing.expectEqual(@as(u64, 15), w.prefill_compute); // the failed attempt computed nothing
    try Frozen.expectBusy(w);
}

/// Two queued prompts on the batching backend where the first one's first-token read fails after its prompt ran.
fn failedFirstToken() !api.load_ledger.Work {
    var rig: Rig = undefined;
    try rig.initWith(.{ .lanes = 2 }, Frozen.io(), .batched);
    defer rig.deinit();
    rig.target.fail_read = 0; // the first draw's read: the first stream's first token, after its prompt pass
    rig.target.prefill_hook = Frozen.tick;
    rig.target.prefill_hook_ctx = &rig;
    var ledger: Ledger = .init(std.testing.io);
    defer ledger.detach(false);
    ledger.attach(rig.host.engine());
    const one = [_]u32{ 2, 7, 1, 8, 2, 8 };
    const two = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5 };
    var a: Box = .{};
    var b: Box = .{};
    const ra: Request = .{ .prompt = &one, .max_tokens = 8 };
    const rb: Request = .{ .prompt = &two, .max_tokens = 4 };
    try rig.host.engine().submit(1, &ra, a.sink());
    try rig.host.engine().submit(2, &rb, b.sink());
    try rig.host.start();
    try std.testing.expectEqual(Reason.failed, a.wait());
    try std.testing.expectEqual(Reason.length, b.wait());
    return settled(&rig, &ledger);
}

test "a batched prompt whose first token fails after its pass still counts its computed prompt" {
    const w = try failedFirstToken();
    try std.testing.expectEqual(@as(u64, 15), w.prefill_compute); // 9 if `done` waited for the first token
    try Frozen.expectBusy(w);
}

test "the one-at-a-time fallback counts a prompt whose first token fails after its pass" {
    var rig: Rig = undefined;
    try rig.initWith(.{ .lanes = 3 }, Frozen.io(), .batched);
    defer rig.deinit();
    rig.target.fail_read = 0; // the first stream's first token fails after its prompt pass
    rig.target.memory_lanes = 2; // prefill_many refuses three, so each goes alone; the third waits for a lane
    rig.target.prefill_hook = Frozen.tick;
    rig.target.prefill_hook_ctx = &rig;
    var ledger: Ledger = .init(std.testing.io);
    defer ledger.detach(false);
    ledger.attach(rig.host.engine());
    const one = [_]u32{ 2, 7, 1, 8, 2, 8 };
    const two = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5 };
    const three = [_]u32{ 1, 4, 1, 4 };
    var a: Box = .{};
    var b: Box = .{};
    var c: Box = .{};
    const ra: Request = .{ .prompt = &one, .max_tokens = 8 };
    const rb: Request = .{ .prompt = &two, .max_tokens = 16 };
    const rc: Request = .{ .prompt = &three, .max_tokens = 4 };
    try rig.host.engine().submit(1, &ra, a.sink());
    try rig.host.engine().submit(2, &rb, b.sink());
    try rig.host.engine().submit(3, &rc, c.sink());
    try rig.host.start();
    try std.testing.expectEqual(Reason.failed, a.wait());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(Reason.length, c.wait());
    try std.testing.expectEqual(@as(usize, 0), rig.target.many_passes);
    const w = settled(&rig, &ledger);
    try std.testing.expectEqual(@as(u64, 19), w.prefill_compute); // 13 if `done` waited for the first token
    try Frozen.expectBusy(w);
}

test "a window engine shows its mode and free lanes; a paged one shows its pool as soon as it attaches" {
    var rig: Rig = undefined;
    try rig.init(.{ .lanes = 3 });
    defer rig.deinit();
    var ledger: Ledger = .init(std.testing.io);
    defer ledger.detach(false);
    rig.host.capacity = .{ .mode = .window };
    ledger.attach(rig.host.engine());
    const window = ledger.read();
    try std.testing.expectEqual(api.load_ledger.Mode.window, window.capacity.?.mode);
    try std.testing.expectEqual(@as(u32, 3), window.capacity.?.lanes_free);
    try std.testing.expect(window.pool == null);
    ledger.detach(true);
    rig.host.capacity = .{ .mode = .paged };
    LaneHost.Load.poolChanged(&rig.host, .{ .capacity = 4096, .available = 4096 }); // the census before the thread starts
    try std.testing.expect(ledger.read().capacity == null); // detached: nothing reaches the ledger
    ledger.attach(rig.host.engine()); // idle, no request: its capacity shows anyway
    const paged = ledger.read();
    try std.testing.expectEqual(api.load_ledger.Mode.paged, paged.capacity.?.mode);
    try std.testing.expectEqual(@as(u64, 4096), paged.pool.?.available);
    LaneHost.Load.poolChanged(&rig.host, .{ .capacity = 4096, .available = 1024, .promised = 512 });
    try std.testing.expectEqual(@as(u64, 1024), ledger.read().pool.?.available);
}

test "a request holding a lane lowers the free lanes" {
    var rig: Rig = undefined;
    try rig.init(.{ .lanes = 2 });
    defer rig.deinit();
    rig.host.capacity = .{ .mode = .window };
    var ledger: Ledger = .init(std.testing.io);
    defer ledger.detach(false);
    ledger.attach(rig.host.engine());
    const Watch = struct { // the reply's first tokens: the request holds its lane now
        ledger: *Ledger,
        free: ?u32 = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            if (e.* == .tokens and w.free == null) w.free = w.ledger.read().capacity.?.lanes_free;
        }
    };
    var watch: Watch = .{ .ledger = &ledger };
    try rig.host.start();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 32 };
    try rig.host.engine().submit(1, &request, .{ .ctx = &watch, .event = Watch.event });
    while (true) {
        rig.host.lock();
        const left = rig.host.admitted.items.len + rig.host.queued.items.len;
        rig.host.unlock();
        if (left == 0) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expectEqual(@as(?u32, 1), watch.free);
    try std.testing.expectEqual(@as(u32, 2), ledger.read().capacity.?.lanes_free);
}
