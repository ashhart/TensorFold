//! The server-side idle unload: after `--idle-unload` seconds idle, close the engine; the next request reloads it.
const std = @import("std");
const api = @import("engine_api");
const cli = @import("cli.zig");
const log = @import("log.zig");
const engines = @import("engines.zig");
const server_mod = @import("server.zig");

const Allocator = std.mem.Allocator;

/// How the manager opens and closes an engine; the same calls the process made at startup.
pub const Reopen = struct {
    ctx: *anyopaque,
    /// `engines.open`'s shape: the engine for `dir`/`model_type` with `args`, or null with `problem` set.
    open: *const fn (ctx: *anyopaque, a: Allocator, gpa: Allocator, io: std.Io, dir: []const u8, model_type: []const u8, args: cli.Args, problem: *[]const u8) anyerror!?api.Opened,
    /// Closes the engine the process started with, freeing its weights; once.
    close_held: *const fn (ctx: *anyopaque) void,
    /// Adopts `opened` as the engine the process closes at shutdown.
    replace: *const fn (ctx: *anyopaque, opened: api.Opened) void,
};

/// The engine's lifetime: resident serves; unloading closes; unloaded waits; loading reopens; stopped ends the server.
pub const State = enum { resident, unloading, unloaded, loading, stopped };

/// The state `serve.run` keeps across unload/reload cycles; one mutex guards the engine's lifetime.
pub const Manager = struct {
    gpa: Allocator,
    io: std.Io,
    args: cli.Args,
    re: Reopen,
    /// The checkpoint folder and its model_type, read once at startup.
    dir: []const u8,
    model_type: []const u8,
    /// Seconds of idle past the last engine-using request before the engine closes; the timer re-arms forever.
    idle_s: i64,
    /// Guards `state`, `active`, the parked flag, the stub and every engine swap; the condition parks waiters.
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    state: State = .resident,
    /// Engine-using requests in flight (routes.post's count); the timer parks only at zero.
    active: u32 = 0,
    /// The manager's timer thread, waiting out idle windows forever until `stop`.
    thread: ?std.Thread = null,
    stopped: bool = false,
    /// The engine the process started with, still unclosed; closed by the first park, freed of the role after.
    held: ?api.Opened = null,
    /// The parked stand-in the server holds while no engine is resident; destroyed at the reload's swap.
    stub: ?*Stub = null,
    /// True while the server's engine is the stub; the reload's swap clears it.
    parked: bool = false,
    /// The startup keepalive was stopped at the first park; the reload's fresh engine arms a new one.
    keepalive_stopped: bool = false,
    /// The timer's last note of engine-using work, and the reload's re-arm, on the awake clock.
    last_activity_ns: i64 = 0,
    /// The server whose engine the manager swaps; set by `serve.run` before the timer's first window.
    srv: ?*server_mod.Server = null,

    /// A request starts engine work: raises `active` and refreshes the idle deadline; the route's `end` must follow.
    pub fn begin(m: *Manager) void {
        m.mutex.lockUncancelable(m.io);
        defer m.mutex.unlock(m.io);
        m.active += 1;
        m.last_activity_ns = now(m.io);
    }

    /// The request ended, any outcome: drops `active` and restarts the idle window.
    pub fn end(m: *Manager) void {
        m.mutex.lockUncancelable(m.io);
        defer m.mutex.unlock(m.io);
        m.active -= 1;
        m.last_activity_ns = now(m.io);
        m.cond.broadcast(m.io);
    }

    /// Whether the server's engine is the parked stub; the stub answers every read, no closed-engine touch.
    pub fn isParked(m: *Manager) bool {
        m.mutex.lockUncancelable(m.io);
        defer m.mutex.unlock(m.io);
        return m.parked;
    }
};

fn now(io: std.Io) i64 {
    return @intCast(std.Io.Clock.awake.now(io).toNanoseconds());
}

/// Starts the idle timer for `args.idle_unload` seconds; null and 0 keep the manager off.
pub fn start(gpa: Allocator, io: std.Io, args: cli.Args, re: Reopen, dir: []const u8, model_type: []const u8) ?*Manager {
    if (args.idle_unload <= 0) return null;
    const m = gpa.create(Manager) catch return null;
    m.* = .{ .gpa = gpa, .io = io, .args = args, .re = re, .dir = dir, .model_type = model_type, .idle_s = args.idle_unload, .last_activity_ns = now(io) };
    m.thread = std.Thread.spawn(.{}, timer, .{m}) catch {
        gpa.destroy(m);
        return null;
    };
    return m;
}

/// The idle timer: re-arms after every activity; parks when a whole window passed with no request.
fn timer(m: *Manager) void {
    while (true) {
        m.mutex.lockUncancelable(m.io);
        while (!m.stopped and m.state != .resident) m.cond.waitUncancelable(m.io, &m.mutex);
        if (m.stopped) return m.mutex.unlock(m.io);
        // `idle_s` is seconds: nanoseconds, clamped so a huge setting cannot overflow the window math
        const window_ns = std.math.mul(i64, m.idle_s, std.time.ns_per_s) catch std.math.maxInt(i64);
        while (!m.stopped and m.state == .resident and m.active == 0) {
            const elapsed = now(m.io) -% m.last_activity_ns;
            if (elapsed >= window_ns) break;
            m.mutex.unlock(m.io);
            std.Io.sleep(m.io, .fromMilliseconds(@intCast(@min(window_ns - elapsed, 100))), .awake) catch {};
            m.mutex.lockUncancelable(m.io);
        }
        if (m.stopped) return m.mutex.unlock(m.io);
        if (m.active > 0 or m.state != .resident or now(m.io) -% m.last_activity_ns < window_ns) {
            m.mutex.unlock(m.io);
            continue; // work arrived or a reload runs: `end` re-arms the deadline and wakes this loop
        }
        m.state = .unloading;
        m.mutex.unlock(m.io);
        park(m);
    }
}

/// Closes the startup engine and swaps the stand-in in; the timer thread is the only caller.
fn park(m: *Manager) void {
    m.mutex.lockUncancelable(m.io);
    const srv = m.srv orelse return m.mutex.unlock(m.io);
    if (m.state != .unloading) return m.mutex.unlock(m.io); // a request began: it wins, the window restarts
    if (m.stub == null) {
        const stub = m.gpa.create(Stub) catch {
            m.state = .resident; // out of memory: stay resident and try the next window
            m.mutex.unlock(m.io);
            log.line("idle unload: out of memory; the engine stays resident", .{});
            return;
        };
        stub.* = .{ .gpa = m.gpa, .info = srv.info };
        m.stub = stub;
        m.parked = true;
    }
    srv.engine = m.stub.?.engine();
    if (!m.keepalive_stopped) {
        srv.stopKeepalive(); // its target was the closed engine's queue: a tick would hit freed memory
        m.keepalive_stopped = true;
    }
    m.mutex.unlock(m.io);
    m.re.close_held(m.re.ctx); // the weights leave here, outside the lock: the reload opens with them gone
    m.mutex.lockUncancelable(m.io);
    if (m.state == .unloading) m.state = .unloaded;
    m.mutex.unlock(m.io);
    m.cond.broadcast(m.io);
    log.line("idle unload: the engine is closed; the next request reloads it", .{});
}

/// A request's reload: the parked request drives the reopen; every fallible step runs before the engine publishes.
pub fn reload(m: *Manager, srv: *server_mod.Server, problem_out: *[]const u8) error{ OutOfMemory, ReloadFailed, EngineOpenFailed }!void {
    m.mutex.lockUncancelable(m.io);
    while (m.state == .loading) {
        m.cond.waitUncancelable(m.io, &m.mutex); // one reload at a time: the others wait for publish or failure
        if (m.state == .resident) return m.mutex.unlock(m.io); // a concurrent reload won: serve with its engine
        if (m.state == .unloaded) break; // the winner failed: this request retries the reload
    }
    if (m.state == .resident) return m.mutex.unlock(m.io);
    if (m.state != .unloaded) {
        problem_out.* = "the engine is still unloading; send the request again";
        return m.mutex.unlock(m.io); // still closing: a deliberate retry, not a race
    }
    m.state = .loading;
    m.mutex.unlock(m.io); // `parked` stays true: the server's engine is the stub until the swap below publishes
    errdefer {
        m.mutex.lockUncancelable(m.io);
        if (m.state == .loading) m.state = .unloaded; // still parked: the next request retries the reload
        m.mutex.unlock(m.io);
        m.cond.broadcast(m.io);
    }
    var problem: []const u8 = "";
    const opened = (m.re.open(m.re.ctx, srv.arena.allocator(), m.gpa, m.io, m.dir, m.model_type, m.args, &problem) catch {
        problem_out.* = problem; // the open path wrote its reason into serve's arena, which outlives this request
        return error.EngineOpenFailed;
    }) orelse {
        problem_out.* = try std.fmt.allocPrint(m.gpa, "the engine could not reload ({s}); send the request again", .{problem});
        return error.ReloadFailed;
    };
    var attached = true;
    if (srv.recall) |record| if (!opened.engine.attachRecall(record)) { // the startup engine drafts from the record; so must this one
        opened.close(opened.ctx);
        attached = false;
    };
    if (!attached) {
        problem_out.* = "the reopened engine does not draft from the recall record; send the request again";
        return error.ReloadFailed;
    }
    const info = opened.engine.info();
    var keepalive: ?*api.keepalive.Keepalive = null;
    if (srv.config.keep_warm_s > 0) if (opened.engine.keepaliveTarget()) |target| {
        keepalive = api.keepalive.Keepalive.start(m.gpa, m.io, target, @as(i64, srv.config.keep_warm_s) * std.time.ns_per_s) catch |e| blk: {
            log.line("idle keepalive off: {s}", .{@errorName(e)});
            break :blk null;
        };
    };
    m.mutex.lockUncancelable(m.io);
    if (m.state == .stopped) { // the server stopped while the weights loaded: close the candidate at once
        m.mutex.unlock(m.io);
        opened.close(opened.ctx);
        if (keepalive) |k| k.stop();
        return error.ReloadFailed;
    }
    const old_stub = m.stub;
    m.stub = null;
    m.parked = false; // the swap publishes the resident engine: the stub reads end here
    m.state = .resident;
    m.last_activity_ns = now(m.io);
    // swap everything the request paths read while every reader holds this lock or waits on `cond`
    srv.engine = opened.engine;
    srv.info = info;
    srv.keepalive = keepalive;
    m.re.replace(m.re.ctx, opened); // shutdown closes this engine, not the long-gone startup one
    m.mutex.unlock(m.io);
    m.cond.broadcast(m.io);
    if (old_stub) |s| m.gpa.destroy(s);
    log.line("idle reload: the model is resident again", .{});
}

/// A request's wake: reloads the parked server; the only error is OutOfMemory, a failure returns its reason.
pub fn wakeReload(m: *Manager, srv: *server_mod.Server) error{OutOfMemory}![]const u8 {
    var problem: []const u8 = "";
    reload(m, srv, &problem) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        if (problem.len == 0) problem = "the engine could not reload; send the request again";
        log.line("idle reload failed: {s}", .{problem});
        return problem;
    };
    return "";
}

/// The drain's halt for the park-closed engine: the Closer's halt would hit a closed engine, so it routes here.
pub fn haltHeld(m: *Manager, reason: []const u8) void {
    m.mutex.lockUncancelable(m.io);
    defer m.mutex.unlock(m.io);
    if (m.held) |h| if (h.halt) |f| f(h.ctx, reason);
}

/// Stops the timer and frees the manager; the server's own shutdown path closes the resident engine.
pub fn stop(m: *Manager) void {
    m.mutex.lockUncancelable(m.io);
    m.stopped = true;
    m.cond.broadcast(m.io);
    var stub: ?*Stub = null;
    if (m.stub) |s| { // a still-parked server's stub dies here; its engine holds no weights
        m.stub = null;
        stub = s;
    }
    m.mutex.unlock(m.io);
    if (stub) |s| m.gpa.destroy(s);
    if (m.thread) |t| t.join();
    m.gpa.destroy(m);
}

/// The parked stand-in: refuses every submit, reports the server unavailable, holds no weights.
const Stub = struct {
    gpa: Allocator,
    info: api.Info,

    fn engine(s: *Stub) api.Engine {
        return .{ .ctx = s, .vtable = &.{
            .info = infoFn,
            .submit = submitFn,
            .cancel = cancelFn,
            .status = statusFn,
            .memory = memoryFn,
        } };
    }

    fn infoFn(ctx: *anyopaque) api.Info {
        return @as(*Stub, @ptrCast(@alignCast(ctx))).info;
    }

    fn submitFn(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed; // the chat path turns this into "the engine is unavailable; retry after it restarts"
    }

    fn cancelFn(_: *anyopaque, _: api.Id) void {}

    fn statusFn(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{ .healthy = false }; // /health 503 and the dashboard's unavailable state while parked
    }

    fn memoryFn(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

test "the manager stays off for 0 and negative idle seconds" {
    var args: cli.Args = .{};
    args.idle_unload = 0;
    const re: Reopen = .{ .ctx = undefined, .open = undefined, .close_held = undefined, .replace = undefined };
    const m = start(std.testing.allocator, std.testing.io, args, re, "d", "t");
    try std.testing.expect(m == null);
    args.idle_unload = -5;
    try std.testing.expect(start(std.testing.allocator, std.testing.io, args, re, "d", "t") == null);
}

test "a one-second window parks the engine without a request; the timer does not re-close" {
    const Harness = @import("idle_unload_test.zig").Harness;
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(1);
    try h.waitParked(3000); // the window passed with no request: the park landed
    try std.testing.expectEqual(@as(u32, 1), h.closes);
    h.sleepMs(1100); // a second window: the engine is already closed, the park does not re-close
    try std.testing.expectEqual(@as(u32, 1), h.closes);
}

test "active work prevents the park; its end restarts the window" {
    const Harness = @import("idle_unload_test.zig").Harness;
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(1);
    h.man.?.begin();
    h.sleepMs(1400); // a whole window with a request in flight: no park
    try std.testing.expectEqual(@as(u32, 0), h.closes);
    h.man.?.end();
    try h.waitParked(3000); // the end re-armed the deadline: the park lands after the fresh window
    try std.testing.expectEqual(@as(u32, 1), h.closes);
}

test "the wake reloads: the candidate opens, the stub is freed, and a second cycle works" {
    const Harness = @import("idle_unload_test.zig").Harness;
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(1);
    try h.waitParked(3000);
    var problem: []const u8 = "";
    try h.reload(&problem);
    try std.testing.expectEqual(@as(u32, 1), h.opens);
    try std.testing.expectEqual(@as(u32, 1), h.replaces);
    try std.testing.expect(!h.man.?.isParked());
    try h.waitParked(3000); // a second complete cycle
    try h.reload(&problem);
    try std.testing.expectEqual(@as(u32, 2), h.opens);
    try std.testing.expectEqual(@as(u32, 2), h.closes);
}

test "concurrent wake requests produce exactly one reload; both find the engine resident" {
    const Harness = @import("idle_unload_test.zig").Harness;
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(1);
    try h.waitParked(3000);
    var a_err: ?anyerror = null;
    var b_err: ?anyerror = null;
    const Wa = struct {
        fn run(x: *Harness, e: *?anyerror) void {
            var p: []const u8 = "";
            x.reload(&p) catch |err| {
                e.* = err;
            };
        }
    };
    const ta = try std.Thread.spawn(.{}, Wa.run, .{ &h, &a_err });
    const tb = try std.Thread.spawn(.{}, Wa.run, .{ &h, &b_err });
    ta.join();
    tb.join();
    try std.testing.expectEqual(@as(u32, 1), h.opens); // exactly one reload opened the engine
    try std.testing.expect(a_err == null and b_err == null);
    try std.testing.expect(!h.man.?.isParked());
}

test "a failed open cleans up and leaves the server parked and retryable" {
    const Harness = @import("idle_unload_test.zig").Harness;
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(1);
    h.fail_open = true;
    try h.waitParked(3000);
    var problem: []const u8 = "";
    try std.testing.expectError(error.EngineOpenFailed, h.reload(&problem));
    try std.testing.expect(h.man.?.isParked()); // still parked: the next request retries
    h.fail_open = false;
    try h.reload(&problem); // the retry succeeds
    try std.testing.expectEqual(@as(u32, 1), h.opens);
    try std.testing.expect(!h.man.?.isParked());
}

test "the parked stub refuses submits and reports unhealthy" {
    const Harness = @import("idle_unload_test.zig").Harness;
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(1);
    try h.waitParked(3000);
    const engine = h.srv.?.engine;
    var probe: api.Request = .{ .prompt = &.{}, .max_tokens = 0 };
    try std.testing.expectError(error.Closed, engine.submit(1, &probe, .{ .ctx = undefined, .event = undefined }));
    var status: api.Status = .{};
    engine.status(&status, &.{});
    try std.testing.expect(!status.healthy);
    try std.testing.expect(engine.memory(false) == null);
}
