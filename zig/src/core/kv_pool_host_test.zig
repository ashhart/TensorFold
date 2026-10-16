//! Competing reservations through a lane host: a request whose pages do not fit waits, then runs; nothing stays leased.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api");
const kv = @import("kv_pool");
const LaneHost = api.LaneHost;
const Reason = api.Reason;
const Request = api.Request;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

const layout: kv.Layout = .{ .fixed_bytes = 0, .token_bytes = 1, .page_tokens = 4, .snap_bytes = 1 };
const window = 1;

/// A backend's pool as the fake target reserves from it: a stream's lease at its prefill, back at its release.
const Pages = struct {
    pool: kv.Pool,
    leases: std.AutoHashMapUnmanaged(*const lanes.Stream, kv.Lease) = .empty,
    most: usize = 0, // the most leases live at once

    fn of(ctx: *anyopaque) *Pages {
        return @ptrCast(@alignCast(ctx));
    }

    fn fits(ctx: *anyopaque, prompt: u32, max_tokens: u32) bool {
        return of(ctx).pool.fits(layout.need(prompt, max_tokens, window));
    }

    fn take(ctx: *anyopaque, s: *lanes.Stream) anyerror!void {
        const p = of(ctx);
        var l: kv.Lease = .{};
        errdefer p.pool.finish(&l);
        const need = layout.need(s.prompt().len, s.max_new, window);
        try p.pool.promise(&l, need);
        try p.pool.cover(&l, need, null);
        try p.leases.put(std.testing.allocator, s, l);
        p.most = @max(p.most, p.leases.count());
    }

    fn give(ctx: *anyopaque, s: *lanes.Stream) void {
        const p = of(ctx);
        var kv_ = p.leases.fetchRemove(s) orelse return;
        p.pool.finish(&kv_.value);
    }

    fn reserve(p: *Pages) lanes.fake.Reserve {
        return .{ .ctx = p, .take = take, .give = give };
    }

    fn deinit(p: *Pages) void {
        p.leases.deinit(std.testing.allocator);
        p.pool.deinit();
    }
};

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

const tokens = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
const long_reply = 100000; // a stream that runs until it is cancelled: pages(8 + 100000 + 1) = 25003

/// A host over the fake target whose lanes reserve from `pages`; `check`: the host admits by the pool's counts.
const Rig = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine,
    host: LaneHost,

    fn start(r: *Rig, pages: *Pages, check: bool) !void {
        const gpa = std.testing.allocator;
        r.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        r.target = .{ .gpa = gpa, .reserve = pages.reserve() };
        r.clock = .{};
        r.core = lanes.Engine.init(gpa, &r.cfg, r.target.backend(), r.clock.clock());
        r.host = LaneHost.init(gpa, std.testing.io, &r.core, .{ .lanes = 3, .admit = if (check) .{ .ctx = pages, .fits = Pages.fits } else null });
        try r.host.start();
    }

    fn stop(r: *Rig) void {
        r.host.stop();
        r.core.deinit();
        r.target.deinit();
        r.cfg.deinit(std.testing.allocator);
    }
};

fn run(check: bool) !void {
    var pages: Pages = .{ .pool = try kv.Pool.init(std.testing.allocator, 25003 + 3, 4) };
    defer pages.deinit();
    var rig: Rig = undefined;
    try rig.start(&pages, check);
    const e = rig.host.engine();
    var a: Box = .{};
    var b: Box = .{};
    var c: Box = .{};
    const running: Request = .{ .prompt = &tokens, .max_tokens = long_reply };
    const four: Request = .{ .prompt = &tokens, .max_tokens = 7 }; // pages(16) = 4: three are left beside the first
    const two: Request = .{ .prompt = tokens[0..3], .max_tokens = 4 }; // pages(8) = 2
    try e.submit(1, &running, .{ .ctx = &a, .event = Box.event });
    try expect(a.startsWithin(2000));
    try e.submit(2, &four, .{ .ctx = &b, .event = Box.event });
    std.Io.sleep(std.testing.io, .fromMilliseconds(50), .awake) catch {};
    try expectEqual(@as(usize, 0), b.state()[0]); // it waits for the first stream's pages
    if (check) {
        try e.submit(3, &two, .{ .ctx = &c, .event = Box.event });
        try expectEqual(@as(?Reason, .length), c.waitFor(2000)); // a request that fits runs beside the first meanwhile
    }
    e.cancel(1);
    try expectEqual(@as(?Reason, .cancelled), a.waitFor(2000));
    try expectEqual(@as(?Reason, .length), b.waitFor(2000)); // then it runs
    try expectEqual(@as(usize, 7), b.state()[0]);
    rig.stop();
    try expectEqual(@as(usize, if (check) 2 else 1), pages.most);
    try expectEqual(@as(usize, 0), pages.leases.count());
    try expect(pages.pool.idle());
    try expectEqual(pages.pool.total(), pages.pool.nfree);
}

test "a request whose pages do not fit beside a running stream waits for it, then runs; one that fits goes meanwhile" {
    try run(true);
}

test "without an admission check the backend's LaneMemory requeues the request, and it runs once the stream ends" {
    try run(false);
}

test "a request past the whole pool with nothing else running fails in words, and the host still serves" {
    var pages: Pages = .{ .pool = try kv.Pool.init(std.testing.allocator, 3, 4) };
    defer pages.deinit();
    var rig: Rig = undefined;
    try rig.start(&pages, true);
    const e = rig.host.engine();
    var a: Box = .{};
    var b: Box = .{};
    const big: Request = .{ .prompt = &tokens, .max_tokens = 8 }; // pages(17) = 5
    const small: Request = .{ .prompt = &tokens, .max_tokens = 3 }; // pages(12) = 3
    try e.submit(1, &big, .{ .ctx = &a, .event = Box.event });
    try expectEqual(@as(?Reason, .failed), a.waitFor(2000));
    try expect(std.mem.indexOf(u8, a.message, "memory") != null);
    try e.submit(2, &small, .{ .ctx = &b, .event = Box.event });
    try expectEqual(@as(?Reason, .length), b.waitFor(2000));
    rig.stop();
    try expect(pages.pool.idle());
}

test "a cancel during the prompt pass gives the stream's pages back" {
    var pages: Pages = .{ .pool = try kv.Pool.init(std.testing.allocator, 8, 4) };
    defer pages.deinit();
    var rig: Rig = undefined;
    try rig.start(&pages, true);
    rig.target.prefill_chunks = 10;
    const Breaker = struct {
        engine: api.Engine,
        fn call(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
            const x: *@This() = @ptrCast(@alignCast(ctx));
            if (chunk == 2) x.engine.cancel(1);
        }
    };
    var breaker: Breaker = .{ .engine = rig.host.engine() };
    rig.target.prefill_hook = Breaker.call;
    rig.target.prefill_hook_ctx = &breaker;
    var a: Box = .{};
    const r: Request = .{ .prompt = &tokens, .max_tokens = 4 };
    try rig.host.engine().submit(1, &r, .{ .ctx = &a, .event = Box.event });
    try expectEqual(@as(?Reason, .cancelled), a.waitFor(2000));
    rig.stop();
    try expectEqual(@as(usize, 1), pages.most);
    try expect(pages.pool.idle());
}
