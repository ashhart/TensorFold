//! The router's children: started on demand, leased a request at a time, stopped when idle, crashed or out of room.
const std = @import("std");
const benchmark_server = @import("../benchmark_server.zig");
const log = @import("../log.zig");
const catalog = @import("catalog.zig");
const room = @import("room.zig");
const Managed = benchmark_server.Managed;
const Allocator = std.mem.Allocator;

pub const Options = struct {
    max_loaded: u32 = 1,
    /// Seconds a child may sit idle before the router stops it (0: never).
    idle_s: i64 = 0,
    /// Seconds a request waits for a busy model to finish and make room, before its 503.
    busy_wait_s: f64 = 300,
    /// Each stop's wait after SIGTERM, then after SIGKILL, in milliseconds: past the child's own drain.
    grace_ms: u32 = 30_000,
    /// The machine's memory in bytes, which loaded floors share; 0 checks no memory.
    capacity: u64 = 0,
    /// The folder each child's log is made in and unlinked from at once.
    log_dir: []const u8,
};

/// A request's hold on a ready child: its port and key, released once its reply has gone.
pub const Lease = struct { slot: usize, gen: u32, port: u16, key: [64]u8 };

/// Why a request got no child: the router stops, every loaded model is busy, it cannot fit, or its load failed.
pub const Refusal = union(enum) {
    stopping,
    busy,
    too_large: struct { need: u64, have: u64 },
    failed: []const u8,
    timeout: f64,
};

pub const Acquired = union(enum) { lease: Lease, refused: Refusal };

/// One loaded or loading child, as /health and the listings show it.
pub const Status = struct { slot: usize, state: room.State, in_flight: u32, port: u16, key: [64]u8 };

const Slot = struct {
    state: room.State = .unloaded,
    child: ?Managed = null,
    port: u16 = 0,
    /// The child's API key, new for each start: 32 random bytes as hex.
    key: [64]u8 = @splat('0'),
    in_flight: u32 = 0,
    waiters: u32 = 0,
    used_ns: i96 = 0,
    /// Counts starts; a request waiting on a load reads ``failure`` when ``failed_gen`` is the start it waited for.
    gen: u32 = 0,
    failed_gen: u32 = 0,
    failure: [512]u8 = undefined,
    failure_len: usize = 0,
    /// What one start allocates (its log path, environment and lines), freed when its child is gone.
    arena: std.heap.ArenaAllocator,
};

pub const Pool = struct {
    gpa: Allocator,
    io: std.Io,
    entries: []const catalog.Entry,
    env: *const std.process.Environ.Map,
    options: Options,
    slots: []Slot,
    views: []room.View,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    stopping: std.atomic.Value(bool) = .init(false),
    reaper: ?std.Thread = null,

    pub fn init(gpa: Allocator, io: std.Io, entries: []const catalog.Entry, env: *const std.process.Environ.Map, options: Options) !*Pool {
        const p = try gpa.create(Pool);
        errdefer gpa.destroy(p);
        const slots = try gpa.alloc(Slot, entries.len);
        errdefer gpa.free(slots);
        for (slots) |*s| s.* = .{ .arena = .init(gpa) };
        p.* = .{ .gpa = gpa, .io = io, .entries = entries, .env = env, .options = options, .slots = slots, .views = try gpa.alloc(room.View, entries.len) };
        return p;
    }

    /// Frees the pool after ``stopAll``.
    pub fn deinit(p: *Pool) void {
        for (p.slots) |*s| s.arena.deinit();
        p.gpa.free(p.slots);
        p.gpa.free(p.views);
        p.gpa.destroy(p);
    }

    /// A lease on entry ``i``'s child, starting it and making room first when it is not running.
    pub fn acquire(p: *Pool, a: Allocator, i: usize) Acquired {
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        const busy_until = now(p.io) + nanoseconds(p.options.busy_wait_s);
        var waited: ?u32 = null;
        var load_until: i96 = 0;
        while (true) {
            if (p.stopping.load(.acquire)) return .{ .refused = .stopping };
            const s = &p.slots[i];
            if (waited) |g| if (s.failed_gen == g) return .{ .refused = .{ .failed = a.dupe(u8, s.failure[0..s.failure_len]) catch "" } };
            switch (s.state) {
                .ready => {
                    if (p.exitedLocked(i)) continue;
                    s.in_flight += 1;
                    s.used_ns = now(p.io);
                    return .{ .lease = .{ .slot = i, .gen = s.gen, .port = s.port, .key = s.key } };
                },
                .loading, .stopping => {
                    if (s.state == .loading and waited != s.gen) {
                        waited = s.gen;
                        load_until = now(p.io) + nanoseconds(p.entries[i].startup_s + 10) + 2 * @as(i96, p.options.grace_ms) * std.time.ns_per_ms;
                    }
                    if (s.state == .loading and now(p.io) >= load_until) return .{ .refused = .{ .timeout = p.entries[i].startup_s } };
                    s.waiters += 1;
                    p.pause();
                    s.waiters -= 1;
                },
                .unloaded => switch (room.decide(p.viewsLocked(), i, p.options.max_loaded, p.options.capacity)) {
                    .fits => return p.load(a, i),
                    .stop => |v| p.stopLocked(v, "least recently used, to make room"),
                    .wait => {
                        if (now(p.io) >= busy_until) return .{ .refused = .busy };
                        p.pause();
                    },
                    .never => return .{ .refused = .{ .too_large = .{ .need = p.entries[i].floor.?, .have = p.options.capacity } } },
                },
            }
        }
    }

    /// Ends a lease: one request fewer in flight, and the child's idle time starts again.
    pub fn release(p: *Pool, lease: Lease) void {
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        const s = &p.slots[lease.slot];
        s.in_flight -|= 1;
        s.used_ns = now(p.io);
        p.cond.broadcast(p.io);
    }

    /// After a failed exchange: whether the leased child has exited, waiting briefly as its sockets close first.
    pub fn lost(p: *Pool, lease: Lease) bool {
        const until = now(p.io) + 2 * std.time.ns_per_s;
        while (true) {
            {
                p.mutex.lockUncancelable(p.io);
                defer p.mutex.unlock(p.io);
                const s = &p.slots[lease.slot];
                if (s.gen != lease.gen or s.state != .ready) return true;
                if (p.exitedLocked(lease.slot)) return true;
            }
            if (now(p.io) >= until) return false;
            p.io.sleep(.fromMilliseconds(50), .awake) catch return false;
        }
    }

    /// Starts the thread that notices exits and stops children idle past ``idle_s``.
    pub fn startReaper(p: *Pool) !void {
        p.reaper = try std.Thread.spawn(.{}, reap, .{p});
    }

    /// Stops every child: new leases are refused, a load in progress ends, each child gets SIGTERM, then SIGKILL.
    pub fn stopAll(p: *Pool) void {
        p.stopping.store(true, .release);
        p.mutex.lockUncancelable(p.io);
        p.cond.broadcast(p.io);
        // every drain starts at once; each stop below then waits on its own child
        for (p.slots) |*s| if (s.state == .ready) std.posix.kill(s.child.?.pid, .TERM) catch {};
        const until = now(p.io) + 2 * @as(i96, p.options.grace_ms + 5_000) * std.time.ns_per_ms;
        while (true) {
            var left = false;
            for (p.slots, 0..) |*s, i| {
                if (s.state == .ready) p.stopLocked(i, "the router is stopping");
                left = left or s.state != .unloaded;
            }
            if (!left or now(p.io) >= until) break;
            p.pause();
        }
        p.mutex.unlock(p.io);
        if (p.reaper) |t| t.join();
        p.reaper = null;
    }

    /// The children loading, ready or stopping, with each one's port and key.
    pub fn loaded(p: *Pool, a: Allocator) ![]const Status {
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        var out: std.ArrayList(Status) = .empty;
        for (p.slots, 0..) |*s, i| if (s.state != .unloaded)
            try out.append(a, .{ .slot = i, .state = s.state, .in_flight = s.in_flight, .port = s.port, .key = s.key });
        return out.items;
    }

    /// The requests in flight across every child.
    pub fn inFlight(p: *Pool) u32 {
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        var n: u32 = 0;
        for (p.slots) |*s| n += s.in_flight;
        return n;
    }

    fn pause(p: *Pool) void {
        p.cond.waitTimeout(p.io, &p.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
    }

    fn viewsLocked(p: *Pool) []const room.View {
        for (p.slots, p.views, p.entries) |*s, *v, e| v.* = .{ .state = s.state, .in_flight = s.in_flight, .waiters = s.waiters, .used_ns = s.used_ns, .floor = e.floor };
        return p.views;
    }

    /// Starts entry ``i``'s child with the lock released, then publishes it ready, or the failure its waiters read.
    fn load(p: *Pool, a: Allocator, i: usize) Acquired {
        const s = &p.slots[i];
        const e = p.entries[i];
        s.state = .loading;
        s.gen += 1;
        const gen = s.gen;
        var raw: [32]u8 = undefined;
        p.io.random(&raw);
        s.key = std.fmt.bytesToHex(raw, .lower);
        _ = s.arena.reset(.free_all);
        p.mutex.unlock(p.io);
        if (e.floor == null) log.line("loading {s} (no memory estimate: only --max-loaded bounds it)", .{e.id}) else log.line("loading {s}", .{e.id});
        const t0 = now(p.io);
        var failure: benchmark_server.Failure = .none;
        var tail: []const []const u8 = &.{};
        const got = p.spawn(s.arena.allocator(), e, &s.key, &failure, &tail);
        p.mutex.lockUncancelable(p.io);
        if (got) |m| {
            s.child = m;
            s.port = portOf(m.base());
            s.state = .ready;
            s.in_flight += 1;
            s.used_ns = now(p.io);
            p.cond.broadcast(p.io);
            log.line("{s} ready in {d:.1} s", .{ e.id, @as(f64, @floatFromInt(s.used_ns - t0)) / 1e9 });
            return .{ .lease = .{ .slot = i, .gen = gen, .port = s.port, .key = s.key } };
        }
        const message = describe(s.arena.allocator(), e, failure, tail) catch "it could not start";
        log.line("{s} could not start: {s}", .{ e.id, message });
        for (tail) |line| log.line("{s}: {s}", .{ e.id, line });
        s.failure_len = @min(message.len, s.failure.len);
        @memcpy(s.failure[0..s.failure_len], message[0..s.failure_len]);
        s.failed_gen = gen;
        s.state = .unloaded;
        p.cond.broadcast(p.io);
        const refusal: Refusal = switch (failure) {
            .stopped => .stopping,
            .timeout => .{ .timeout = e.startup_s },
            else => .{ .failed = a.dupe(u8, s.failure[0..s.failure_len]) catch "" },
        };
        _ = s.arena.reset(.free_all);
        return .{ .refused = refusal };
    }

    /// The benchmark's serve child, as the router runs it: its own key and process group, its log unlinked.
    fn spawn(p: *Pool, sa: Allocator, e: catalog.Entry, key: *const [64]u8, failure: *benchmark_server.Failure, tail: *[]const []const u8) ?Managed {
        failure.* = .setup;
        // the parent's variables a serve would read, less the router's own keys; spawn adds the child's key
        const env = benchmark_server.environWithout(sa, p.env, &.{ "TENSORFOLD_API_KEY", "TENSORFOLD_REQUEST_JOURNAL", "TENSORFOLD_BENCHMARK_TOKEN" }) catch return null;
        _ = std.Io.Dir.cwd().createDirPathStatus(p.io, p.options.log_dir, .fromMode(0o700)) catch {};
        var problem: []const u8 = "";
        return Managed.spawn(sa, p.io, p.env, e.argv, .{
            .startup_timeout_s = e.startup_s,
            .grace_ms = p.options.grace_ms,
            .bench_dir = p.options.log_dir,
            .tail = tail,
            .served = e.id,
            .lock = false,
            .own_group = true,
            .key = key,
            .environ = &env,
            .guard = false,
            .stop = &p.stopping,
            .failure = failure,
        }, &problem) catch {
            failure.* = .setup;
            return null;
        };
    }

    /// Stops slot ``v``'s child with the lock released; the slot counts as loaded until it has exited.
    fn stopLocked(p: *Pool, v: usize, why: []const u8) void {
        const s = &p.slots[v];
        var m = s.child.?;
        s.child = null;
        s.state = .stopping;
        p.mutex.unlock(p.io);
        log.line("stopping {s}: {s}", .{ p.entries[v].id, why });
        if (m.stop() != null) log.line("{s} did not exit after SIGKILL; its memory may still be held", .{p.entries[v].id});
        p.mutex.lockUncancelable(p.io);
        s.state = .unloaded;
        _ = s.arena.reset(.free_all);
        p.cond.broadcast(p.io);
    }

    /// Whether ready slot ``i``'s child has exited; if so it is unloaded, and its last log lines are logged.
    fn exitedLocked(p: *Pool, i: usize) bool {
        const s = &p.slots[i];
        const m = &s.child.?;
        // the tail is read only once the child has exited, so running out of memory still means it has
        const lines = (m.exitedTail(s.arena.allocator()) catch &.{}) orelse return false;
        log.line("{s} exited ({s} {d}); the next request starts it again", .{ p.entries[i].id, if (std.posix.W.IFSIGNALED(@bitCast(m.status))) "signal" else "status", exitCode(m.status) });
        for (lines) |line| log.line("{s}: {s}", .{ p.entries[i].id, line });
        _ = m.stop();
        s.child = null;
        s.state = .unloaded;
        _ = s.arena.reset(.free_all);
        p.cond.broadcast(p.io);
        return true;
    }

    fn reap(p: *Pool) void {
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        while (!p.stopping.load(.acquire)) {
            for (p.slots, 0..) |*s, i| {
                if (s.state != .ready or p.exitedLocked(i) or p.options.idle_s <= 0) continue;
                const idle = room.isIdle(.{ .state = s.state, .in_flight = s.in_flight, .waiters = s.waiters });
                if (idle and now(p.io) - s.used_ns >= @as(i96, p.options.idle_s) * std.time.ns_per_s) p.stopLocked(i, "idle past --idle-unload");
            }
            p.cond.waitTimeout(p.io, &p.mutex, .{ .duration = .{ .raw = .fromMilliseconds(500), .clock = .awake } }) catch {};
        }
    }
};

/// Why a start gave no child, for the 503 and the log: the child's own last lines when it exited.
fn describe(a: Allocator, e: catalog.Entry, failure: benchmark_server.Failure, tail: []const []const u8) ![]const u8 {
    return switch (failure) {
        .exited => if (tail.len > 0) std.fmt.allocPrint(a, "its server exited while loading: {s}", .{tail[tail.len - 1]}) else "its server exited while loading",
        .timeout => std.fmt.allocPrint(a, "its server was not ready within {d} s", .{e.startup_s}),
        .spawn => "its server could not be started; check the TensorFold installation",
        .setup => "its server's log could not be created; check the TensorFold directory permissions",
        .stopped => "the router is stopping",
        .none => "it did not answer its readiness checks",
    };
}

fn exitCode(status: c_int) u32 {
    const s: u32 = @bitCast(status);
    return if (std.posix.W.IFSIGNALED(s)) @intCast(@backingInt(std.posix.W.TERMSIG(s))) else std.posix.W.EXITSTATUS(s);
}

/// ``http://127.0.0.1:PORT``'s port.
fn portOf(base: []const u8) u16 {
    const colon = std.mem.lastIndexOfScalar(u8, base, ':') orelse return 0;
    return std.fmt.parseInt(u16, base[colon + 1 ..], 10) catch 0;
}

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn nanoseconds(seconds: f64) i96 {
    return @as(i64, @intFromFloat(@min(@max(seconds, 0), 1e7) * 1e9));
}
