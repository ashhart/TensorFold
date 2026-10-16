//! Keyed CUDA graphs for any family: one captured executable per key, launched again when the key comes back.

const std = @import("std");
const graph = @import("graph.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const registry = @import("graph_registry.zig");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel

/// One step's launches on `stream`; it must not synchronize, allocate device memory or read the device on the host.
pub const Body = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, stream: Stream) anyerror!void,
};

/// Every rank's verdict AND-ed on a miss, never a hit; an off cache never votes, so ranks match `Settings.digest`.
pub const Agree = struct {
    ctx: *anyopaque,
    all: *const fn (ctx: *anyopaque, ok: bool) anyerror!bool,
};

/// Capture a body into an executable graph, launch it, free it: a fake engine runs the policy in host tests.
pub const Engine = struct {
    ctx: *anyopaque,
    capture: *const fn (ctx: *anyopaque, stream: Stream, body: Body) anyerror!*anyopaque,
    launch: *const fn (ctx: *anyopaque, exec: *anyopaque, stream: Stream) anyerror!void,
    free: *const fn (ctx: *anyopaque, exec: *anyopaque) void,
};

pub const Settings = struct {
    on: bool = true, // off: every step runs eagerly
    max: u32 = 48, // graphs held at most
    min_keep: u32 = 8, // under the memory floor the cache evicts down to this many, then stops capturing
    hold: bool = false, // under the floor a miss runs eagerly with no eviction: held graphs cost next to no memory

    /// These settings under the operator's TF_GRAPH_CACHE value: `0` or `off` runs every step eagerly; else unchanged.
    pub fn operator(s: Settings, value: ?[]const u8) Settings {
        var out = s;
        const v = value orelse return out;
        if (std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off")) out.on = false;
        return out;
    }

    /// What a cache runs with: the operator's value, min_keep under max (0 with hold), and every off cache one value.
    pub fn effective(s: Settings, value: ?[]const u8) Settings {
        var out = s.operator(value);
        if (!out.on or out.max == 0) return .{ .on = false, .max = 0, .min_keep = 0, .hold = false };
        out.min_keep = if (out.hold) 0 else @min(out.min_keep, out.max);
        return out;
    }

    /// `effective` under this process's TF_GRAPH_CACHE.
    pub fn fromEnv(s: Settings) Settings {
        return s.effective(if (std.c.getenv("TF_GRAPH_CACHE")) |v| std.mem.span(v) else null);
    }

    /// The effective settings as bytes for ranks to compare at start (a family's bootstrap agreement), never per step.
    pub fn digest(s: Settings) [10]u8 {
        const e = s.effective(null);
        var out: [10]u8 = undefined;
        out[0] = @intFromBool(e.on);
        out[1] = @intFromBool(e.hold);
        std.mem.writeInt(u32, out[2..6], e.max, .little);
        std.mem.writeInt(u32, out[6..10], e.min_keep, .little);
        return out;
    }
};

pub const Outcome = enum { replayed, captured, eager };

/// Eager steps under the floor between two "still under" lines.
pub const report_every = 256;

/// One cache's name and counts as other threads read them (/health); `snapshots` copies the caches in use.
pub const Snapshot = registry.Snapshot;
pub const snapshots = registry.snapshots;

pub fn Cache(comptime Key: type) type {
    return struct {
        const Self = @This();
        const Entry = struct { exec: *anyopaque, used: u64 };

        gpa: std.mem.Allocator,
        engine: Engine,
        settings: Settings,
        agree: ?Agree = null,
        room: ?*const fn () bool = null, // false when this rank is below its memory floor; null: always room
        meter: ?*const fn () ?f64 = null, // MemAvailable now in GiB, to log a capture's cost; null: not measured
        name: []const u8 = "graphs", // the cache's name in its log lines and snapshot
        entries: std.AutoArrayHashMapUnmanaged(Key, Entry) = .empty,
        cap: u32,
        tick: u64 = 0,
        fingerprint: ?u64 = null,
        failed: bool = false, // a capture failed on some rank: eager from here on
        under: bool = false, // the last miss met no room on some rank
        under_eager: u64 = 0, // eager steps since `under` began
        slot: ?usize = null, // this cache's registry slot, claimed at its first step; null: every slot taken
        claimed: bool = false,
        stats: struct { replayed: u64 = 0, captured: u64 = 0, eager: u64 = 0, evicted: u64 = 0, dropped: u64 = 0, floor_hits: u64 = 0, capture_mib: f64 = 0 } = .{},

        /// The cache with `settings.fromEnv()`: every rank sharing an `agree` must hold an equal `Settings.digest`.
        pub fn init(gpa: std.mem.Allocator, engine: Engine, settings: Settings) Self {
            const s = settings.fromEnv();
            if (!quiet and settings.on and settings.max > 0 and !s.on) std.log.info("graph cache: off (TF_GRAPH_CACHE), every step eager", .{});
            return .{ .gpa = gpa, .engine = engine, .settings = s, .cap = s.max };
        }

        pub fn deinit(c: *Self) void {
            if (c.slot) |i| registry.release(i);
            c.dropAll();
            c.entries.deinit(c.gpa);
        }

        pub fn count(c: *const Self) usize {
            return c.entries.count();
        }

        pub fn has(c: *const Self, key: Key) bool {
            return c.entries.contains(key);
        }

        /// Every graph freed (a fingerprint change, a shutdown).
        pub fn dropAll(c: *Self) void {
            for (c.entries.values()) |e| c.engine.free(c.engine.ctx, e.exec);
            c.stats.dropped += c.entries.count();
            c.entries.clearRetainingCapacity();
        }

        fn agreed(c: *Self, ok: bool) !bool {
            const a = c.agree orelse return ok;
            return a.all(a.ctx, ok);
        }

        fn evictOldest(c: *Self) void {
            var at: ?usize = null;
            for (c.entries.values(), 0..) |e, i| {
                if (at == null or e.used < c.entries.values()[at.?].used) at = i;
            }
            const i = at orelse return;
            c.engine.free(c.engine.ctx, c.entries.values()[i].exec);
            c.entries.swapRemoveAt(i);
            c.stats.evicted += 1;
        }

        fn publish(c: *Self) void {
            if (!c.claimed) {
                c.claimed = true;
                c.slot = registry.claim(c.name);
            }
            const i = c.slot orelse return;
            registry.publish(i, .{ .held = @intCast(c.entries.count()), .captured = c.stats.captured, .replayed = c.stats.replayed, .eager = c.stats.eager, .floor_hits = c.stats.floor_hits, .under = c.under });
        }

        fn avail(c: *const Self) ?f64 {
            return if (c.meter) |m| m() else null;
        }

        /// A miss without room on some rank: the floor entered (logged), or a "still under" line every `report_every`.
        fn underFloor(c: *Self, room_here: bool) void {
            c.stats.floor_hits += 1;
            c.under_eager += 1;
            if (!c.under) {
                c.under = true;
                c.under_eager = 1;
                if (!quiet) std.log.warn("graph cache {s}: under the memory floor on {s} rank (MemAvailable {?d:.2} GiB here): new keys eager, {d} graphs held", .{ c.name, if (room_here) "another" else "this", c.avail(), c.entries.count() });
            } else if (c.under_eager % report_every == 0) {
                if (!quiet) std.log.warn("graph cache {s}: still under the memory floor: {d} eager steps since it began (MemAvailable {?d:.2} GiB here, {d} graphs held)", .{ c.name, c.under_eager, c.avail(), c.entries.count() });
            }
        }

        fn eager(c: *Self, stream: Stream, body: Body) !Outcome {
            try body.run(body.ctx, stream);
            c.stats.eager += 1;
            return .eager;
        }

        /// One step: the key's graph, captured first on a miss; a new `fingerprint` (same step on all ranks) drops all.
        pub fn run(c: *Self, key: Key, stream: Stream, fingerprint: u64, body: Body) !Outcome {
            defer c.publish();
            if (!c.settings.on or c.failed or c.settings.max == 0) return c.eager(stream, body);
            if (c.fingerprint != null and c.fingerprint.? != fingerprint) c.dropAll();
            c.fingerprint = fingerprint;
            c.tick += 1;
            if (c.entries.getPtr(key)) |e| {
                e.used = c.tick;
                try c.engine.launch(c.engine.ctx, e.exec, stream);
                c.stats.replayed += 1;
                return .replayed;
            }
            // a miss: room on every rank (a host map that cannot grow is none), else shed a quarter and lower the cap
            const reserved = if (c.entries.ensureUnusedCapacity(c.gpa, 1)) |_| true else |_| false;
            const room_here = reserved and (if (c.room) |f| f() else true);
            if (!try c.agreed(room_here)) {
                c.underFloor(room_here);
                if (c.settings.hold) return c.eager(stream, body);
                if (c.entries.count() > c.settings.min_keep) {
                    const drop = @max(1, c.entries.count() / 4);
                    for (0..drop) |_| if (c.entries.count() > c.settings.min_keep) c.evictOldest();
                }
                const was = c.cap;
                c.cap = @intCast(@max(c.entries.count(), c.settings.min_keep)); // min_keep <= max (init)
                if (!quiet and c.cap != was) std.log.warn("graph cache: under the memory floor on {s} rank: cap {d} -> {d} ({d} held, {d} evicted, {d} eager so far)", .{ if (room_here) "another" else "this", was, c.cap, c.entries.count(), c.stats.evicted, c.stats.eager + 1 });
                return c.eager(stream, body);
            }
            if (c.under) {
                if (!quiet) std.log.info("graph cache {s}: room again on every rank (MemAvailable {?d:.2} GiB here) after {d} eager steps under the floor, {d} graphs held", .{ c.name, c.avail(), c.under_eager, c.entries.count() });
                c.under = false;
            }
            if (c.cap < c.settings.max) {
                if (!quiet) std.log.info("graph cache: room again on every rank: cap {d} -> {d}", .{ c.cap, c.settings.max });
                c.cap = c.settings.max;
            }
            while (c.entries.count() >= c.cap) c.evictOldest();
            const before = c.avail();
            const exec = c.engine.capture(c.engine.ctx, stream, body) catch |err| blk: {
                // the only trace of why graphs went off (the step then runs eagerly, which may fail the same way)
                if (!quiet) std.log.warn("graph capture failed: {s} (key {any}, {d} graphs held)", .{ @errorName(err), key, c.entries.count() });
                break :blk null;
            };
            var owned = exec; // freed here until the map holds it, so a failed agreement cannot leak it
            defer if (owned) |x| c.engine.free(c.engine.ctx, x);
            if (!try c.agreed(exec != null)) {
                c.failed = true;
                c.dropAll();
                return c.eager(stream, body);
            }
            c.entries.putAssumeCapacityNoClobber(key, .{ .exec = exec.?, .used = c.tick });
            owned = null;
            try c.engine.launch(c.engine.ctx, exec.?, stream);
            c.stats.captured += 1;
            // what a graph really costs this rank (the room rule prices it)
            if (before) |b| if (c.avail()) |a| {
                c.stats.capture_mib = (b - a) * 1024;
                if (!quiet) std.log.info("graph cache {s}: captured, {d} held: MemAvailable {d:.2} -> {d:.2} GiB ({d:.0} MiB)", .{ c.name, c.entries.count(), b, a, c.stats.capture_mib });
            };
            return .captured;
        }
    };
}

/// The driver's engine: thread-local stream capture (other threads' CUDA calls stay out), instantiate, upload, launch.
pub const CudaEngine = struct {
    d: *const Driver,

    pub fn engine(e: *CudaEngine) Engine {
        return .{ .ctx = e, .capture = capture, .launch = launch, .free = free };
    }

    const Held = struct { exec: graph.Exec };

    fn capture(ctx: *anyopaque, stream: Stream, body: Body) anyerror!*anyopaque {
        const e: *CudaEngine = @ptrCast(@alignCast(ctx));
        try graph.beginCapture(stream, .thread_local);
        body.run(body.ctx, stream) catch |err| {
            // end the capture so the stream is usable again; the partial graph is dropped
            if (graph.endCapture(stream)) |g| {
                var gg = g;
                gg.deinit();
            } else |_| {}
            return err;
        };
        var g = try graph.endCapture(stream);
        defer g.deinit();
        const h = try std.heap.c_allocator.create(Held);
        errdefer std.heap.c_allocator.destroy(h);
        h.exec = try g.instantiate();
        errdefer h.exec.deinit();
        try h.exec.upload(stream);
        _ = e;
        return h;
    }

    fn launch(_: *anyopaque, exec: *anyopaque, stream: Stream) anyerror!void {
        const h: *Held = @ptrCast(@alignCast(exec));
        try h.exec.launchOn(stream);
    }

    fn free(_: *anyopaque, exec: *anyopaque) void {
        const h: *Held = @ptrCast(@alignCast(exec));
        h.exec.deinit();
        std.heap.c_allocator.destroy(h);
    }
};

/// FNV-1a over the addresses a body captures (buffer bases, sizes): the fingerprint `run` compares, on one rank only.
pub fn fingerprintOf(words: []const u64) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (words) |w| {
        var x = w;
        for (0..8) |_| {
            h = (h ^ (x & 0xff)) *% 0x100000001b3;
            x >>= 8;
        }
    }
    return h;
}

// -- host tests: the policy with a fake engine --------------------------------------------------------------------

const Fake = struct {
    captures: u32 = 0,
    launches: u32 = 0,
    live: u32 = 0,
    fail_capture: bool = false,
    slots: [64]u8 = undefined,

    fn engine(f: *Fake) Engine {
        return .{ .ctx = f, .capture = cap, .launch = lau, .free = fre };
    }
    fn cap(ctx: *anyopaque, stream: Stream, body: Body) anyerror!*anyopaque {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (f.fail_capture) return error.CaptureFailed;
        _ = stream;
        _ = body; // a capture records the body; nothing runs
        f.captures += 1;
        f.live += 1;
        return &f.slots[f.captures % f.slots.len];
    }
    fn lau(ctx: *anyopaque, _: *anyopaque, _: Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.launches += 1;
    }
    fn fre(ctx: *anyopaque, _: *anyopaque) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.live -= 1;
    }
};

const Ran = struct {
    n: u32 = 0,
    fn body(r: *Ran) Body {
        return .{ .ctx = r, .run = go };
    }
    fn go(ctx: *anyopaque, _: Stream) anyerror!void {
        const r: *Ran = @ptrCast(@alignCast(ctx));
        r.n += 1;
    }
};

const no_stream: Stream = .{ .d = undefined, .handle = null };

test "graph cache: capture on first use, replay after, LRU at the cap" {
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{ .max = 2 });
    defer c.deinit();
    try std.testing.expectEqual(Outcome.captured, try c.run(1, no_stream, 7, r.body()));
    try std.testing.expectEqual(Outcome.replayed, try c.run(1, no_stream, 7, r.body()));
    try std.testing.expectEqual(Outcome.captured, try c.run(2, no_stream, 7, r.body()));
    _ = try c.run(1, no_stream, 7, r.body()); // 1 is now the most recent
    try std.testing.expectEqual(Outcome.captured, try c.run(3, no_stream, 7, r.body())); // evicts 2
    try std.testing.expect(c.has(1) and c.has(3) and !c.has(2));
    try std.testing.expectEqual(@as(u32, 0), r.n); // the body never ran eagerly
    try std.testing.expectEqual(@as(u32, 2), f.live);
    try std.testing.expectEqual(@as(u32, 5), f.launches);
}

test "graph cache: a new fingerprint drops every graph" {
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{});
    defer c.deinit();
    _ = try c.run(1, no_stream, 7, r.body());
    _ = try c.run(2, no_stream, 7, r.body());
    try std.testing.expectEqual(Outcome.captured, try c.run(1, no_stream, 8, r.body()));
    try std.testing.expectEqual(@as(usize, 1), c.count());
    try std.testing.expectEqual(@as(u32, 1), f.live);
}

const Peer = struct {
    theirs: []const bool, // the other rank's verdicts, in call order
    at: usize = 0,
    fn agree(p: *Peer) Agree {
        return .{ .ctx = p, .all = all };
    }
    fn all(ctx: *anyopaque, ok: bool) anyerror!bool {
        const p: *Peer = @ptrCast(@alignCast(ctx));
        const t = p.theirs[p.at];
        p.at += 1;
        return ok and t;
    }
};

test "graph cache: a capture failing on another rank turns graphs off here too" {
    var f: Fake = .{};
    var r: Ran = .{};
    // per miss: room, then the capture's verdict; the peer's capture fails on the second key
    var peer: Peer = .{ .theirs = &.{ true, true, true, false } };
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{});
    c.agree = peer.agree();
    defer c.deinit();
    try std.testing.expectEqual(Outcome.captured, try c.run(1, no_stream, 7, r.body()));
    try std.testing.expectEqual(Outcome.eager, try c.run(2, no_stream, 7, r.body()));
    try std.testing.expect(c.failed);
    try std.testing.expectEqual(@as(u32, 0), f.live); // this rank's good graphs are freed as well
    try std.testing.expectEqual(Outcome.eager, try c.run(1, no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(u32, 2), r.n);
}

var room_flag = true;
fn roomFn() bool {
    return room_flag;
}

test "graph cache: below the memory floor the least recent quarter goes, the cap follows, the step runs eagerly" {
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{ .max = 48, .min_keep = 2 });
    c.room = roomFn;
    defer c.deinit();
    room_flag = true;
    for (0..8) |k| _ = try c.run(@intCast(k), no_stream, 7, r.body());
    room_flag = false;
    try std.testing.expectEqual(Outcome.eager, try c.run(100, no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(usize, 6), c.count());
    try std.testing.expectEqual(@as(u32, 6), c.cap);
    try std.testing.expect(!c.has(0) and !c.has(1) and c.has(7));
    room_flag = true;
    // room again: the cap is back to max, the miss captures without evicting
    try std.testing.expectEqual(Outcome.captured, try c.run(100, no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(usize, 7), c.count());
    try std.testing.expectEqual(@as(u32, 48), c.cap);
    try std.testing.expect(c.has(7) and c.has(2));
    try std.testing.expectEqual(@as(u64, 1), c.stats.floor_hits);
}

test "graph cache: a dip under the floor does not leave later keys churning once memory is back" {
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{ .max = 16, .min_keep = 2 });
    c.room = roomFn;
    defer c.deinit();
    room_flag = true;
    for (0..12) |k| _ = try c.run(@intCast(k), no_stream, 7, r.body());
    room_flag = false; // a long prompt's workspace held
    _ = try c.run(200, no_stream, 7, r.body());
    room_flag = true; // freed: the decode after it meets 12 new keys (a new context bucket), then replays them
    const before = c.stats.evicted;
    for (0..12) |k| _ = try c.run(@intCast(300 + k), no_stream, 7, r.body());
    for (0..12) |k| try std.testing.expectEqual(Outcome.replayed, try c.run(@intCast(300 + k), no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(u32, 16), c.cap);
    // only what the cap of 16 forces: 9 left after the dip + 12 new = 21 -> 5 evictions, none during the replays
    try std.testing.expectEqual(before + 5, c.stats.evicted);
}

test "graph cache: hold - below the floor a miss runs eagerly, nothing is evicted, the held keys replay" {
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{ .max = 48, .min_keep = 2, .hold = true });
    c.room = roomFn;
    defer c.deinit();
    room_flag = true;
    for (0..8) |k| _ = try c.run(@intCast(k), no_stream, 7, r.body());
    room_flag = false;
    try std.testing.expectEqual(Outcome.eager, try c.run(100, no_stream, 7, r.body()));
    try std.testing.expectEqual(Outcome.eager, try c.run(101, no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(usize, 8), c.count());
    try std.testing.expectEqual(@as(u32, 48), c.cap);
    for (0..8) |k| try std.testing.expectEqual(Outcome.replayed, try c.run(@intCast(k), no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(u64, 0), c.stats.evicted);
    try std.testing.expectEqual(@as(u64, 2), c.stats.floor_hits);
    room_flag = true;
    try std.testing.expectEqual(Outcome.captured, try c.run(100, no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(usize, 9), c.count());
}

test "graph cache: the floor's enter / under / leave state and its counts, as /health reads them" {
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{ .max = 48, .hold = true });
    c.room = roomFn;
    c.name = "test floor";
    defer c.deinit();
    room_flag = true;
    for (0..4) |k| _ = try c.run(@intCast(k), no_stream, 7, r.body());
    try std.testing.expect(!c.under);
    room_flag = false;
    for (0..report_every + 3) |k| _ = try c.run(@intCast(1000 + k), no_stream, 7, r.body());
    try std.testing.expect(c.under);
    try std.testing.expectEqual(@as(u64, report_every + 3), c.under_eager);
    // the held keys replay under the floor and leave the state alone
    try std.testing.expectEqual(Outcome.replayed, try c.run(2, no_stream, 7, r.body()));
    try std.testing.expect(c.under);
    var out: [8]Snapshot = undefined;
    const mine = for (out[0..snapshots(&out)]) |x| {
        if (std.mem.eql(u8, x.name(), "test floor")) break x;
    } else return error.NotPublished;
    try std.testing.expectEqual(registry.Counts{ .held = 4, .captured = 4, .replayed = 1, .eager = report_every + 3, .floor_hits = report_every + 3, .under = true }, mine.counts);
    // room again: the next miss captures and the floor is left; the counts keep their totals
    room_flag = true;
    try std.testing.expectEqual(Outcome.captured, try c.run(2000, no_stream, 7, r.body()));
    try std.testing.expect(!c.under);
    try std.testing.expectEqual(@as(u64, report_every + 3), c.stats.floor_hits);
    // a second dip starts its own count
    room_flag = false;
    _ = try c.run(3000, no_stream, 7, r.body());
    try std.testing.expectEqual(@as(u64, 1), c.under_eager);
    room_flag = true;
}

var meter_gib: f64 = 100;
var meter_reads: u32 = 0;
fn meterFn() ?f64 {
    meter_reads += 1;
    defer meter_gib -= 0.25; // each read sees a quarter GiB less: a capture's before and after differ by that
    return meter_gib;
}

test "graph cache: the meter prices a capture, replays never read it, a freed cache leaves the snapshots" {
    var f: Fake = .{};
    var r: Ran = .{};
    var out: [8]Snapshot = undefined;
    const others = snapshots(&out);
    {
        var c = Cache(u32).init(std.testing.allocator, f.engine(), .{ .max = 4 });
        defer c.deinit();
        c.meter = meterFn;
        c.name = "metered";
        meter_reads = 0;
        try std.testing.expectEqual(Outcome.captured, try c.run(1, no_stream, 7, r.body()));
        try std.testing.expectEqual(@as(f64, 256), c.stats.capture_mib);
        try std.testing.expectEqual(@as(u32, 2), meter_reads);
        try std.testing.expectEqual(Outcome.replayed, try c.run(1, no_stream, 7, r.body()));
        try std.testing.expectEqual(@as(u32, 2), meter_reads);
        try std.testing.expectEqual(others + 1, snapshots(&out));
        const mine = for (out[0 .. others + 1]) |x| {
            if (std.mem.eql(u8, x.name(), "metered")) break x;
        } else return error.NotPublished;
        try std.testing.expectEqual(@as(u64, 1), mine.counts.replayed);
    }
    try std.testing.expectEqual(others, snapshots(&out));
}

test "graph cache: off runs every step eagerly" {
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{ .on = false });
    defer c.deinit();
    try std.testing.expectEqual(Outcome.eager, try c.run(1, no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(u32, 0), f.captures);
}

test "graph cache: TF_GRAPH_CACHE=0 or off turns a family's cache off without a rebuild; anything else leaves it" {
    const on: Settings = .{};
    for ([_][]const u8{ "0", "off", "OFF" }) |v| try std.testing.expect(!on.operator(v).on);
    for ([_]?[]const u8{ null, "1", "on", "" }) |v| try std.testing.expect(on.operator(v).on);
    try std.testing.expect(!(Settings{ .on = false }).operator("1").on); // the operator only turns a cache off
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), on.operator("off"));
    defer c.deinit();
    for (0..3) |_| try std.testing.expectEqual(Outcome.eager, try c.run(1, no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(u32, 0), f.captures);
}

test "graph cache: fingerprint" {
    try std.testing.expect(fingerprintOf(&.{ 1, 2 }) != fingerprintOf(&.{ 2, 1 }));
}

test {
    _ = @import("graph_cache_regression_test.zig");
    _ = @import("graph_cache_ranks_test.zig");
}
