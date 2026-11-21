//! Admission by owner class through a lane host (#2444 C1, tests 3 and 5): refused by name, or passed over in order.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api");
const kv = @import("kv_pool");
const LaneHost = api.LaneHost;
const Reason = api.Reason;
const Request = api.Request;
const Need = api.kv_ledger.classes.Need;
const expect = std.testing.expect;
const equal = std.testing.expectEqual;

/// A family's pool in owner classes: a prompt's first token 100 or 101 fixes its class (a resumed prefix's), else none.
const Pages = struct {
    pool: kv.Pool,
    leases: std.AutoHashMapUnmanaged(*const lanes.Stream, kv.Lease) = .empty,
    refused: usize = 0, // reservations that failed: admission let a request through that its class lacked

    fn of(ctx: *anyopaque) *Pages {
        return @ptrCast(@alignCast(ctx));
    }

    /// A page every four prompt tokens; the reply takes none (a stream may run long on a small lease).
    fn needOf(prompt: []const u32) Need {
        const in: ?u32 = if (prompt[0] == 100) 0 else if (prompt[0] == 101) 1 else null;
        return .{ .one = .{ .n = (prompt.len + 3) / 4, .in = in } };
    }

    fn fits(ctx: *anyopaque, _: u32, _: u32) bool {
        return of(ctx).pool.available() > 0; // the family's own test: the class test is the host's
    }

    var prompts: [8][]const u32 = undefined; // need's view of a request by its length (lengths differ in a test)
    fn need(_: *anyopaque, prompt: u32, _: u32) Need {
        for (prompts) |p| if (p.len == prompt) return needOf(p);
        unreachable;
    }

    fn take(ctx: *anyopaque, s: *lanes.Stream) anyerror!void {
        const p = of(ctx);
        var l: kv.Lease = .{};
        errdefer p.pool.finish(&l);
        const d = needOf(s.prompt()).one;
        p.pool.reserve(&l, d.n, d.in) catch |e| {
            p.refused += 1;
            return e;
        };
        try p.pool.cover(&l, d.n, null);
        try p.leases.put(std.testing.allocator, s, l);
    }

    fn give(ctx: *anyopaque, s: *lanes.Stream) void {
        const p = of(ctx);
        var e = p.leases.fetchRemove(s) orelse return;
        p.pool.finish(&e.value);
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

/// A host over the fake target whose lanes reserve from `pages`, its pool counted in the host's ledger.
const Rig = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine,
    host: LaneHost,

    fn start(r: *Rig, pages: *Pages, lanes_n: u32) !void {
        const gpa = std.testing.allocator;
        r.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        errdefer r.cfg.deinit(gpa);
        r.target = .{ .gpa = gpa, .reserve = .{ .ctx = pages, .take = Pages.take, .give = Pages.give } };
        errdefer r.target.deinit();
        r.clock = .{};
        r.core = lanes.Engine.init(gpa, &r.cfg, r.target.backend(), r.clock.clock());
        errdefer r.core.deinit();
        const admit: api.Admit = .{ .ctx = pages, .fits = Pages.fits, .need = Pages.need };
        r.host = LaneHost.init(gpa, std.testing.io, &r.core, .{ .lanes = lanes_n, .admit = admit });
        try r.host.countPool(&pages.pool);
        try r.host.start();
    }

    fn stop(r: *Rig) void {
        r.host.stop();
        r.core.deinit();
        r.target.deinit();
        r.cfg.deinit(std.testing.allocator);
    }
};

const run_long = 100000; // a reply that runs until it is cancelled
const small: [8]u32 = .{ 3, 1, 4, 1, 5, 9, 2, 6 }; // 2 pages, any class
const nine: [36]u32 = @splat(7); // 9 pages: more than a class of 8 holds

/// Test 3 on `classes` classes of 16 pages, `n` lanes: a running stream, then a request of 9 pages; its end, or null.
fn pastClass(classes: u32, n: u32, box: *Box) !struct { ?Reason, usize } {
    var pages: Pages = .{ .pool = try kv.Pool.initClasses(std.testing.allocator, 16, 4, classes) };
    defer pages.deinit();
    Pages.prompts = .{ &small, &nine, &small, &small, &small, &small, &small, &small };
    var rig: Rig = undefined;
    try rig.start(&pages, n);
    defer rig.stop();
    const e = rig.host.engine();
    var a: Box = .{};
    try e.submit(1, &.{ .prompt = &small, .max_tokens = run_long }, .{ .ctx = &a, .event = Box.event });
    try expect(a.startsWithin(2000));
    try e.submit(2, &.{ .prompt = &nine, .max_tokens = 4 }, .{ .ctx = box, .event = Box.event });
    const end = box.waitFor(1000);
    const promised = pages.pool.census().promised + pages.pool.census().leased; // a's 2 only, if 2 took nothing
    e.cancel(1);
    try equal(@as(?Reason, .cancelled), a.waitFor(2000));
    _ = box.waitFor(2000);
    try equal(@as(usize, 0), pages.refused);
    return .{ end, promised };
}

test "kv classes 3: a request past a whole class is refused at once by name, while another stream runs" {
    var b: Box = .{};
    const end, const held = try pastClass(2, 3, &b);
    try equal(@as(?Reason, .failed), end); // not left queued for room no release can give
    try expect(std.mem.indexOf(u8, b.message, "KV owner class") != null); // the refusal names the class
    try equal(@as(usize, 0), b.state()[0]);
    try equal(@as(usize, 2), held); // nothing promised or taken for it before the refusal
    var one: Box = .{};
    const end1, _ = try pastClass(1, 3, &one); // one class of 16: the same request runs, as on #2415
    try equal(@as(?Reason, .length), end1);
}

test "kv classes 3: refused at once with every lane busy, not when the running stream ends" {
    var b: Box = .{};
    const end, const held = try pastClass(2, 1, &b); // one lane, held by the long stream until it is cancelled
    try equal(@as(?Reason, .failed), end);
    try expect(std.mem.indexOf(u8, b.message, "KV owner class") != null);
    try equal(@as(usize, 2), held);
}

test "kv classes 5: a head waiting on class 0 lets a later request for class 1 go first, then runs" {
    var pages: Pages = .{ .pool = try kv.Pool.initClasses(std.testing.allocator, 16, 4, 2) };
    defer pages.deinit();
    const six: [24]u32 = .{100} ++ @as([23]u32, @splat(7)); // class 0, 6 pages
    const four: [16]u32 = .{100} ++ @as([15]u32, @splat(7)); // class 0, 4 pages: 2 left there
    const three: [12]u32 = .{101} ++ @as([11]u32, @splat(7)); // class 1, 3 pages
    Pages.prompts = .{ &six, &four, &three, &small, &small, &small, &small, &small };
    var rig: Rig = undefined;
    try rig.start(&pages, 3);
    defer rig.stop();
    const e = rig.host.engine();
    var a: Box = .{};
    var head: Box = .{};
    var late: Box = .{};
    try e.submit(1, &.{ .prompt = &six, .max_tokens = run_long }, .{ .ctx = &a, .event = Box.event });
    try expect(a.startsWithin(2000));
    try e.submit(2, &.{ .prompt = &four, .max_tokens = 4 }, .{ .ctx = &head, .event = Box.event });
    try e.submit(3, &.{ .prompt = &three, .max_tokens = 4 }, .{ .ctx = &late, .event = Box.event });
    try equal(@as(?Reason, .length), late.waitFor(2000)); // class 1 has room: it does not wait behind the head
    try equal(@as(usize, 0), head.state()[0]); // 10 pages free in all, 2 in class 0: the head still waits
    try equal(@as(usize, 0), pages.refused); // and was never let through to a reservation that fails
    e.cancel(1);
    try equal(@as(?Reason, .cancelled), a.waitFor(2000));
    try equal(@as(?Reason, .length), head.waitFor(2000));
    try equal(@as(usize, 0), pages.refused);
    try expect(pages.pool.idle());
}
