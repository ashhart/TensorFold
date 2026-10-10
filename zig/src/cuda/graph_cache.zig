//! Keyed CUDA graphs for any family: one captured executable per key, launched again when the key comes back.

const std = @import("std");
const graph = @import("graph.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel

/// One step's launches on `stream`; it must not synchronize, allocate device memory or read the device on the host.
pub const Body = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, stream: Stream) anyerror!void,
};

/// Every rank's verdict AND-ed, so ranks capture, evict and fall back at the same steps; never called in a capture.
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
};

pub const Outcome = enum { replayed, captured, eager };

pub fn Cache(comptime Key: type) type {
    return struct {
        const Self = @This();
        const Entry = struct { exec: *anyopaque, used: u64 };

        gpa: std.mem.Allocator,
        engine: Engine,
        settings: Settings,
        agree: ?Agree = null,
        room: ?*const fn () bool = null, // false when this rank is below its memory floor; null: always room
        entries: std.AutoArrayHashMapUnmanaged(Key, Entry) = .empty,
        cap: u32,
        tick: u64 = 0,
        fingerprint: ?u64 = null,
        failed: bool = false, // a capture failed on some rank: eager from here on
        stats: struct { replayed: u64 = 0, captured: u64 = 0, eager: u64 = 0, evicted: u64 = 0, dropped: u64 = 0, floor_hits: u64 = 0 } = .{},

        pub fn init(gpa: std.mem.Allocator, engine: Engine, settings: Settings) Self {
            return .{ .gpa = gpa, .engine = engine, .settings = settings, .cap = settings.max };
        }

        pub fn deinit(c: *Self) void {
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

        fn eager(c: *Self, stream: Stream, body: Body) !Outcome {
            try body.run(body.ctx, stream);
            c.stats.eager += 1;
            return .eager;
        }

        /// One step: the key's graph launched, captured first on a miss; a new `fingerprint` drops every graph first.
        pub fn run(c: *Self, key: Key, stream: Stream, fingerprint: u64, body: Body) !Outcome {
            if (!c.settings.on or c.failed) return c.eager(stream, body);
            if (c.fingerprint != null and c.fingerprint.? != fingerprint) c.dropAll();
            c.fingerprint = fingerprint;
            c.tick += 1;
            if (c.entries.getPtr(key)) |e| {
                e.used = c.tick;
                try c.engine.launch(c.engine.ctx, e.exec, stream);
                c.stats.replayed += 1;
                return .replayed;
            }
            // a miss: room on every rank, else shed the least recent quarter and lower the cap (once below it, eager)
            const room_here = if (c.room) |f| f() else true;
            if (!try c.agreed(room_here and c.cap > 0)) {
                if (c.settings.hold) {
                    if (!quiet and c.stats.floor_hits == 0) std.log.warn("graph cache: under the memory floor on {s} rank: this key eager, {d} graphs held (hold)", .{ if (room_here) "another" else "this", c.entries.count() });
                    c.stats.floor_hits += 1;
                    return c.eager(stream, body);
                }
                if (c.entries.count() > c.settings.min_keep) {
                    const drop = @max(1, c.entries.count() / 4);
                    for (0..drop) |_| if (c.entries.count() > c.settings.min_keep) c.evictOldest();
                }
                const was = c.cap;
                c.cap = @intCast(@max(c.entries.count(), c.settings.min_keep));
                c.stats.floor_hits += 1;
                if (!quiet and c.cap != was) std.log.warn("graph cache: under the memory floor on {s} rank: cap {d} -> {d} ({d} held, {d} evicted, {d} eager so far)", .{ if (room_here) "another" else "this", was, c.cap, c.entries.count(), c.stats.evicted, c.stats.eager + 1 });
                return c.eager(stream, body);
            }
            if (c.cap < c.settings.max) {
                if (!quiet) std.log.info("graph cache: room again on every rank: cap {d} -> {d}", .{ c.cap, c.settings.max });
                c.cap = c.settings.max;
            }
            while (c.entries.count() >= c.cap) c.evictOldest();
            const exec = c.engine.capture(c.engine.ctx, stream, body) catch |err| blk: {
                // the only trace of why graphs went off (the step then runs eagerly, which may fail the same way)
                if (!quiet) std.log.warn("graph capture failed: {s} (key {any}, {d} graphs held)", .{ @errorName(err), key, c.entries.count() });
                break :blk null;
            };
            if (!try c.agreed(exec != null)) {
                if (exec) |x| c.engine.free(c.engine.ctx, x);
                c.failed = true;
                c.dropAll();
                return c.eager(stream, body);
            }
            try c.entries.put(c.gpa, key, .{ .exec = exec.?, .used = c.tick });
            try c.engine.launch(c.engine.ctx, exec.?, stream);
            c.stats.captured += 1;
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

/// FNV-1a over the addresses a body captures (buffer bases, sizes): the fingerprint `run` compares.
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

test "graph cache: off runs every step eagerly" {
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{ .on = false });
    defer c.deinit();
    try std.testing.expectEqual(Outcome.eager, try c.run(1, no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(u32, 0), f.captures);
}

test "graph cache: fingerprint" {
    try std.testing.expect(fingerprintOf(&.{ 1, 2 }) != fingerprintOf(&.{ 2, 1 }));
}
