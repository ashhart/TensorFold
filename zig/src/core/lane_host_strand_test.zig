//! Requests that end without finishing, 200 in a row down each path: every lane, page and slot comes back.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const kv = @import("kv_pool.zig");
const mp = @import("memory_pressure.zig");
const LaneHost = api.LaneHost;
const Ledger = api.load_ledger.Ledger;
const Reason = api.Reason;
const Request = api.Request;
const t = std.testing;
const a = t.allocator;
const io = t.io;

const runs = 200;
const lanes_n = 2;
const pool_pages = 8192;
const long_reply = 4096; // pages(8 + 4096 + 1) = 1027: runs until cancelled or failed, fits the pool twice over
const huge_reply = 40000; // pages(8 + 40000 + 1) = 10003: more than the whole pool
const layout: kv.Layout = .{ .fixed_bytes = 0, .token_bytes = 1, .page_tokens = 4, .snap_bytes = 1 };
const window = 1;
const pause_word = 2; // a prompt starting with it pauses in its prompt pass and holds its lane until cancelled

const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
const paused_prompt = [_]u32{ pause_word, 7, 1, 8, 2, 8, 1, 8 };
const bad_spans = [_][2]u32{.{ 3, 1 }};
const plain: Request = .{ .prompt = &prompt, .max_tokens = 12 };
const running: Request = .{ .prompt = &prompt, .max_tokens = long_reply };
const huge: Request = .{ .prompt = &prompt, .max_tokens = huge_reply };
const holder: Request = .{ .prompt = &paused_prompt, .max_tokens = long_reply };
const spanned: Request = .{ .prompt = &prompt, .max_tokens = 12, .decode_spans = &bad_spans };

/// The host's allocator: it refuses the test thread's allocations while shut, as a host out of memory does.
const Gate = struct {
    owner: std.Thread.Id,
    shut: std.atomic.Value(bool) = .init(false),

    fn allocator(g: *Gate) std.mem.Allocator {
        return .{ .ptr = g, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(ctx: *anyopaque, n: usize, al: std.mem.Alignment, ra: usize) ?[*]u8 {
        const g: *Gate = @ptrCast(@alignCast(ctx));
        if (g.shut.load(.acquire) and std.Thread.getCurrentId() == g.owner) return null;
        return a.rawAlloc(n, al, ra);
    }
    fn resize(_: *anyopaque, bytes: []u8, al: std.mem.Alignment, n: usize, ra: usize) bool {
        return a.rawResize(bytes, al, n, ra);
    }
    fn remap(_: *anyopaque, bytes: []u8, al: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        return a.rawRemap(bytes, al, n, ra);
    }
    fn free(_: *anyopaque, bytes: []u8, al: std.mem.Alignment, ra: usize) void {
        a.rawFree(bytes, al, ra);
    }
};

/// A backend's pool as the fake target reserves from it: a stream's lease at its prefill, back at its release.
const Pages = struct {
    pool: kv.Pool,
    leases: std.AutoHashMapUnmanaged(*const lanes.Stream, kv.Lease) = .empty,

    fn of(ctx: *anyopaque) *Pages {
        return @ptrCast(@alignCast(ctx));
    }
    fn take(ctx: *anyopaque, s: *lanes.Stream) anyerror!void {
        const p = of(ctx);
        var l: kv.Lease = .{};
        errdefer p.pool.finish(&l);
        const need = layout.need(s.prompt().len, s.max_new, window);
        try p.pool.promise(&l, need);
        try p.pool.cover(&l, need, null);
        try p.leases.put(a, s, l);
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
        p.leases.deinit(a);
        p.pool.deinit();
    }
};

/// A pressure source the test thread sets while the engine thread reads it.
const Dial = struct {
    level: std.atomic.Value(u8) = .init(@backingInt(mp.Level.normal)),
    fn set(d: *Dial, l: mp.Level) void {
        d.level.store(@backingInt(l), .release);
    }
    fn read(ctx: ?*anyopaque) ?mp.Level {
        const d: *Dial = @ptrCast(@alignCast(ctx.?));
        return @fromBackingInt(@intCast(d.level.load(.acquire)));
    }
};

/// The backend's readiness, which the test thread turns off for a refusal.
const Health = struct {
    up: std.atomic.Value(bool) = .init(true),
    fn read(ctx: *anyopaque) bool {
        const h: *Health = @ptrCast(@alignCast(ctx));
        return h.up.load(.acquire);
    }
};

const Mode = enum(u8) { serve, cancel_prompt, fail_prompt };

/// The fake target's prompt calls, which cancel the request or fail after the lane and its pages are taken.
const Driver = struct {
    base: *const lanes.backend.Backend.VTable,
    vt: lanes.backend.Backend.VTable,
    engine: api.Engine,
    mode: std.atomic.Value(u8) = .init(0),
    id: std.atomic.Value(api.Id) = .init(0),

    fn of(ptr: *anyopaque) *Driver {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return @ptrCast(@alignCast(f.prefill_hook_ctx.?));
    }
    fn is(d: *const Driver, m: Mode) bool {
        return d.mode.load(.acquire) == @backingInt(m);
    }
    fn set(d: *Driver, m: Mode) void {
        d.mode.store(@backingInt(m), .release);
    }
    fn hook(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
        const d: *Driver = @ptrCast(@alignCast(ctx));
        if (chunk == 0 and s.prompt()[0] == pause_word) s.paused = true;
        if (chunk == 0 and d.is(.cancel_prompt)) d.engine.cancel(d.id.load(.acquire));
    }
    fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const d = of(ptr);
        try d.base.prefill(ptr, s);
        if (d.is(.fail_prompt)) return error.GpuFailed;
    }
    fn step(ptr: *anyopaque, s: *lanes.Stream, max_rows: u32) anyerror!bool {
        const d = of(ptr);
        const done = try d.base.prefill_step.?(ptr, s, max_rows);
        if (d.is(.fail_prompt)) return error.GpuFailed;
        return done;
    }
};

/// One request's events; `cancel` cancels it at its first tokens.
const Box = struct {
    mutex: std.Io.Mutex = .init,
    done: ?Reason = null,
    message: []const u8 = "",
    tokens: usize = 0,
    events: usize = 0,
    cancel: ?api.Engine = null,

    fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        var stop: ?api.Engine = null;
        b.mutex.lockUncancelable(io);
        b.events += 1;
        switch (e.*) {
            .tokens => |ids| {
                b.tokens += ids.len;
                stop = b.cancel;
                b.cancel = null;
            },
            .finished => |f| {
                b.done = f.reason;
                b.message = f.message;
            },
            else => {},
        }
        b.mutex.unlock(io);
        if (stop) |engine| engine.cancel(id);
    }
    fn state(b: *Box) struct { usize, ?Reason } {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return .{ b.tokens, b.done };
    }
    fn waitFor(b: *Box, ms: usize) ?Reason {
        for (0..ms) |_| {
            if (b.state()[1]) |r| return r;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return b.state()[1];
    }
    fn startsWithin(b: *Box, ms: usize) bool {
        for (0..ms) |_| {
            if (b.state()[0] > 0) return true;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return b.state()[0] > 0;
    }
    fn sink(b: *Box) api.Sink {
        return .{ .ctx = b, .event = event };
    }
};

/// The host's lists and its load-ledger groups, read under its lock.
const HostSide = struct {
    queued: usize,
    admitted: usize,
    filling: usize,
    cancels: usize,
    running: u32,
    waiting: u32,
    filling_left: ?u64,
    lanes_free: u32,
};

/// The core's live streams, the fake's lanes, the pool's pages and leases, and the host's KV ledger.
const BackendSide = struct {
    live: usize,
    lanes: usize,
    leases: usize,
    pages_free: usize,
    pages_promised: usize,
    pages_leased: usize,
    kv_free: u64,
};

const Counts = struct { host: HostSide, backend: ?BackendSide };

const Kind = enum { whole, stepped };

const Path = enum { served, busy, closed, stale, pressure, invalid, spans, cancel_at_once, cancel_prompt, cancel_decode, fail_prompt, fail_round, no_memory };

/// A lane host on the fake target whose lanes reserve pages from a pool its ledger counts.
const Rig = struct {
    gate: Gate,
    cfg: lanes.Config,
    pages: Pages,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock,
    driver: Driver,
    core: lanes.Engine,
    host: LaneHost,
    ledger: Ledger,
    health: Health,
    dial: Dial,
    box: Box,
    blockers: [lanes_n]Box,
    next: api.Id,
    base: Counts,
    stopped: bool,

    fn init(r: *Rig, kind: Kind, governed: bool) !void {
        r.gate = .{ .owner = std.Thread.getCurrentId() };
        r.cfg = try lanes.Config.init(a, .{ .exact_width = 8, .hidden_rows = true }, 8, 7); // every round verifies
        errdefer r.cfg.deinit(a);
        r.pages = .{ .pool = try kv.Pool.init(a, pool_pages, 4) };
        errdefer r.pages.deinit();
        r.target = .{ .gpa = a, .prefill_chunks = 2, .reserve = r.pages.reserve() };
        errdefer r.target.deinit();
        const base = switch (kind) {
            .whole => r.target.backend(),
            .stepped => r.target.stepped(),
        };
        r.driver = .{ .base = base.vtable, .vt = base.vtable.*, .engine = undefined };
        r.driver.vt.prefill = Driver.prefill;
        if (kind == .stepped) r.driver.vt.prefill_step = Driver.step;
        r.target.prefill_hook = Driver.hook;
        r.target.prefill_hook_ctx = &r.driver;
        r.clock = .{};
        r.core = lanes.Engine.init(a, &r.cfg, .{ .ptr = &r.target, .vtable = &r.driver.vt, .chunk_invariant = true }, r.clock.clock());
        errdefer r.core.deinit();
        r.health = .{};
        r.dial = .{};
        r.box = .{};
        r.blockers = .{ .{}, .{} };
        r.next = 1;
        r.stopped = false;
        r.host = LaneHost.init(r.gate.allocator(), io, &r.core, .{ .lanes = lanes_n });
        r.host.capacity = .{ .mode = .paged };
        r.host.health = .{ .ctx = &r.health, .read = Health.read };
        if (governed) r.host.pressure = .{ .source = .{ .ctx = &r.dial, .read = Dial.read }, .calm_ns = 0, .every_ns = 0 };
        r.driver.engine = r.host.engine();
        try r.host.countPool(&r.pages.pool);
        errdefer r.host.uncountPool();
        LaneHost.Load.poolChanged(&r.host, .{ .capacity = pool_pages * layout.page_tokens, .available = pool_pages * layout.page_tokens }); // paged needs a census
        try r.host.start();
        r.ledger = .init(io);
        r.ledger.attach(r.host.engine());
    }

    /// The start values every path must come back to, checked against an idle host first.
    fn mark(r: *Rig) !void {
        r.base = try r.counts(true);
        try t.expectEqual(@as(usize, 0), r.base.host.admitted + r.base.host.queued + r.base.backend.?.live);
        try t.expectEqual(@as(u32, lanes_n), r.base.host.lanes_free);
        try t.expectEqual(@as(usize, pool_pages), r.base.backend.?.pages_free);
        try t.expectEqual(@as(u64, pool_pages * 4), r.base.backend.?.kv_free);
    }

    /// Stops the host, then the pool shows nothing promised, leased or held.
    fn close(r: *Rig) !void {
        r.ledger.detach(false);
        r.host.stop();
        r.stopped = true;
        try t.expect(r.pages.pool.idle());
        try t.expectEqual(@as(usize, 0), r.pages.leases.count());
        try t.expectEqual(@as(usize, 0), r.target.lanes.count());
    }

    fn deinit(r: *Rig) void {
        if (!r.stopped) {
            r.ledger.detach(false);
            r.host.stop();
        }
        r.core.deinit();
        r.target.deinit();
        r.pages.deinit();
        r.cfg.deinit(a);
    }

    fn engine(r: *Rig) api.Engine {
        return r.host.engine();
    }

    fn id(r: *Rig) api.Id {
        r.next += 1;
        return r.next;
    }

    /// The counts now; the backend side only when asked (read while no request runs on the engine thread).
    fn counts(r: *Rig, backend: bool) !Counts {
        const snap = r.ledger.read();
        const req = snap.requests orelse return error.NoRequestsGroup;
        const cap = snap.capacity orelse return error.NoCapacityGroup;
        r.host.lock();
        defer r.host.unlock();
        const census = r.pages.pool.census();
        return .{
            .host = .{
                .queued = r.host.queued.items.len,
                .admitted = r.host.admitted.items.len,
                .filling = r.host.filling.items.len,
                .cancels = r.host.cancels.items.len,
                .running = req.running,
                .waiting = req.waiting,
                .filling_left = req.filling_left,
                .lanes_free = cap.lanes_free,
            },
            .backend = if (!backend) null else .{
                .live = r.core.live.items.len,
                .lanes = r.target.lanes.count(),
                .leases = r.pages.leases.count(),
                .pages_free = census.free,
                .pages_promised = census.promised,
                .pages_leased = census.leased,
                .kv_free = r.host.kv.?.free(),
            },
        };
    }

    /// Waits up to 2 s for the counts to equal `want`, then checks them.
    fn settle(r: *Rig, want: Counts) !void {
        const backend = want.backend != null;
        var now = try r.counts(backend);
        for (0..2000) |_| {
            if (std.meta.eql(now, want)) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
            now = try r.counts(backend);
        }
        try t.expectEqual(want, now);
    }

    /// Waits for the engine thread to govern at `want`, waking it from its idle wait to read the dial.
    fn reach(r: *Rig, want: mp.Level) !void {
        for (0..4000) |_| {
            r.host.lock();
            const now = r.host.pressure.?.level;
            if (now != want) r.host.wake.signal(io);
            r.host.unlock();
            if (now == want) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.LevelNotReached;
    }

    fn setStale(r: *Rig, on: bool) void {
        r.host.lock();
        defer r.host.unlock();
        r.host.stale = on;
    }

    fn setFailRound(r: *Rig, on: bool) void {
        r.host.lock();
        defer r.host.unlock();
        r.target.fail_round = if (on) r.target.rounds + 1 else null;
    }

    /// One request down `path`, its end checked; the counts are the caller's to check.
    fn drive(r: *Rig, path: Path) !void {
        const e = r.engine();
        const n = r.id();
        r.box = .{};
        const sink = r.box.sink();
        switch (path) {
            .served => {
                try e.submit(n, &plain, sink);
                try t.expectEqual(@as(?Reason, .length), r.box.waitFor(5000));
                try t.expectEqual(@as(usize, 12), r.box.state()[0]);
            },
            .busy => {
                r.gate.shut.store(true, .release);
                defer r.gate.shut.store(false, .release);
                try t.expectError(error.Busy, e.submit(n, &plain, sink));
            },
            .closed => {
                r.health.up.store(false, .release);
                defer r.health.up.store(true, .release);
                try t.expectError(error.Closed, e.submit(n, &plain, sink));
            },
            .stale => {
                r.setStale(true);
                defer r.setStale(false);
                try t.expectError(error.Stale, e.submit(n, &plain, sink));
            },
            .pressure => {
                r.dial.set(.critical);
                try r.reach(.critical);
                try t.expectError(error.MemoryPressure, e.submit(n, &plain, sink));
                r.dial.set(.normal);
                try r.reach(.normal);
            },
            .invalid => {
                var out: [2]f32 = @splat(0);
                const ends = [_]u32{prompt.len};
                const embed: Request = .{ .prompt = &prompt, .max_tokens = 0, .drafts = false, .embedding = .{ .ends = &ends, .dimensions = 2, .output = &out } };
                // the host's own refusal on odd ids, the engine API's check before it on even ones
                const got = if (n % 2 == 1) e.vtable.submit(e.ctx, n, &embed, sink) else e.submit(n, &embed, sink);
                try t.expectError(error.InvalidRequest, got);
            },
            .spans => try t.expectError(error.InvalidSpans, e.submit(n, &spanned, sink)),
            .cancel_at_once => {
                try e.submit(n, &holder, sink); // it waits for the cancel: `running` can reply in full first on a loaded runner
                e.cancel(n);
                try t.expectEqual(@as(?Reason, .cancelled), r.box.waitFor(5000));
            },
            .cancel_prompt => {
                r.driver.id.store(n, .release);
                r.driver.set(.cancel_prompt);
                defer r.driver.set(.serve);
                try e.submit(n, &running, sink);
                try t.expectEqual(@as(?Reason, .cancelled), r.box.waitFor(5000));
                try t.expectEqual(@as(usize, 0), r.box.state()[0]);
            },
            .cancel_decode => {
                r.box.cancel = e;
                try e.submit(n, &running, sink);
                try t.expectEqual(@as(?Reason, .cancelled), r.box.waitFor(5000));
                try t.expect(r.box.state()[0] > 0);
            },
            .fail_prompt => {
                r.driver.set(.fail_prompt);
                defer r.driver.set(.serve);
                try e.submit(n, &running, sink);
                try t.expectEqual(@as(?Reason, .failed), r.box.waitFor(5000));
                try t.expectEqualStrings("GpuFailed", r.box.message);
            },
            .fail_round => {
                r.setFailRound(true);
                defer r.setFailRound(false);
                try e.submit(n, &running, sink);
                try t.expectEqual(@as(?Reason, .failed), r.box.waitFor(5000));
                try t.expectEqualStrings("GpuFailed", r.box.message);
            },
            .no_memory => {
                try e.submit(n, &huge, sink);
                try t.expectEqual(@as(?Reason, .failed), r.box.waitFor(5000));
                try t.expect(std.mem.indexOf(u8, r.box.message, "no memory") != null);
            },
        }
        if (refusal(path)) try t.expectEqual(@as(usize, 0), r.box.events); // a refused request never reaches its sink
    }

    /// `runs` requests down `path`, each followed by the start counts; then a normal request is served.
    fn strand(r: *Rig, path: Path) !void {
        for (0..runs) |_| {
            try r.drive(path);
            try r.settle(r.base);
        }
        try r.drive(.served);
        try r.settle(r.base);
    }
};

fn refusal(path: Path) bool {
    return switch (path) {
        .busy, .closed, .stale, .pressure, .invalid, .spans => true,
        else => false,
    };
}

/// A rig of `kind` with its start counts marked, for one test's paths.
fn rigged(r: *Rig, kind: Kind, governed: bool) !void {
    try r.init(kind, governed);
    errdefer r.deinit();
    try r.mark();
}

test "lane strand: 200 refusals of each code take no lane, page or slot, and the next request is served" {
    var r: Rig = undefined;
    try rigged(&r, .whole, true);
    defer r.deinit();
    for ([_]Path{ .busy, .closed, .stale, .pressure, .invalid, .spans }) |path| try r.strand(path);
    try t.expectEqual(@as(u64, 6), r.ledger.read().requests.?.started); // only the six served requests were queued
    try r.close();
}

test "lane strand: 200 cancels while queued behind full lanes give every count back" {
    var r: Rig = undefined;
    try rigged(&r, .whole, false);
    defer r.deinit();
    const e = r.engine();
    var ids: [lanes_n]api.Id = undefined;
    for (&r.blockers, &ids) |*b, *n| {
        n.* = r.id();
        try e.submit(n.*, &holder, b.sink());
    }
    for (&r.blockers) |*b| try t.expect(b.startsWithin(5000));
    var held: Counts = .{ .host = r.base.host, .backend = null };
    held.host.admitted = lanes_n;
    held.host.running = lanes_n;
    held.host.lanes_free = 0;
    try r.settle(held);
    var waiting = held;
    waiting.host.queued = 1;
    waiting.host.waiting = 1;
    for (0..runs) |_| {
        const n = r.id();
        r.box = .{};
        try e.submit(n, &plain, r.box.sink());
        try r.settle(waiting); // it waits: no lane is free
        e.cancel(n);
        try t.expectEqual(@as(?Reason, .cancelled), r.box.waitFor(5000));
        try t.expectEqual(@as(usize, 0), r.box.state()[0]);
        try r.settle(held);
    }
    for (ids) |n| e.cancel(n);
    for (&r.blockers) |*b| try t.expectEqual(@as(?Reason, .cancelled), b.waitFor(5000));
    try r.settle(r.base);
    try r.drive(.served);
    try r.settle(r.base);
    try r.close();
}

test "lane strand: 200 cancels at once, mid-prompt and mid-decode, whole and stepped prompts" {
    for ([_]Kind{ .whole, .stepped }) |kind| {
        var r: Rig = undefined;
        try rigged(&r, kind, false);
        defer r.deinit();
        for ([_]Path{ .cancel_at_once, .cancel_prompt, .cancel_decode }) |path| try r.strand(path);
        try r.close();
    }
}

test "lane strand: 200 backend errors in the prompt pass and in a round, whole and stepped prompts" {
    const log_level = t.log_level;
    t.log_level = .err;
    defer t.log_level = log_level;
    for ([_]Kind{ .whole, .stepped }) |kind| {
        var r: Rig = undefined;
        try rigged(&r, kind, false);
        defer r.deinit();
        for ([_]Path{ .fail_prompt, .fail_round }) |path| try r.strand(path);
        try r.close();
    }
}

test "lane strand: 200 requests with no lane memory and nothing else running, whole and stepped prompts" {
    for ([_]Kind{ .whole, .stepped }) |kind| {
        var r: Rig = undefined;
        try rigged(&r, kind, false);
        defer r.deinit();
        try r.strand(.no_memory);
        try r.close();
    }
}

test "lane strand: 200 requests mixing every path, whole and stepped prompts" {
    const log_level = t.log_level;
    t.log_level = .err;
    defer t.log_level = log_level;
    const every = [_]Path{ .served, .cancel_prompt, .busy, .fail_round, .cancel_decode, .closed, .no_memory, .fail_prompt, .stale, .cancel_at_once, .invalid, .pressure, .spans };
    for ([_]Kind{ .whole, .stepped }) |kind| {
        var r: Rig = undefined;
        try rigged(&r, kind, true);
        defer r.deinit();
        for (0..runs) |i| {
            try r.drive(every[(i * 5) % every.len]);
            try r.settle(r.base);
        }
        try r.drive(.served);
        try r.settle(r.base);
        try r.close();
    }
}
