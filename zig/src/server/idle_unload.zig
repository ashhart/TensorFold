//! The server-side idle unload: after `--idle-unload` seconds idle, close the engine; the next request reloads it.
const std = @import("std");
const api = @import("engine_api");
const cli = @import("cli.zig");
const log = @import("log.zig");
const hf_text = @import("hf_text.zig");
const engines = @import("engines.zig");
const server_mod = @import("server.zig");

const Allocator = std.mem.Allocator;

/// What the manager needs to open and close an engine, and to build the server around it.
pub const Reopen = struct {
    ctx: *anyopaque,
    /// `engines.open`'s shape: the engine for `dir`/`model_type` with `args`, or null with `problem` set.
    open: *const fn (ctx: *anyopaque, a: Allocator, gpa: Allocator, io: std.Io, dir: []const u8, model_type: []const u8, args: cli.Args, problem: *[]const u8) anyerror!?api.Opened,
    /// `loadText`'s shape: the checkpoint's tokenizer and templates, or an error naming `problem`.
    load_text: *const fn (ctx: *anyopaque, gpa: Allocator, io: std.Io, dir: []const u8, template: ?[]const u8, pa: Allocator, problem: *[]const u8) anyerror!*hf_text.HfText,
    /// Closes the engine the process started with, freeing its weights; once.
    close_held: *const fn (ctx: *anyopaque) void,
    /// Adopts `opened` as the engine the process closes at shutdown.
    replace: *const fn (ctx: *anyopaque, opened: api.Opened) void,
};

/// The state `serve.run` keeps across an unload/reload cycle.
pub const Manager = struct {
    gpa: Allocator,
    io: std.Io,
    args: cli.Args,
    re: Reopen,
    /// The checkpoint folder and its model_type, read once at startup.
    dir: []const u8,
    model_type: []const u8,
    /// The process environment (`TENSORFOLD_*`), handed to each fresh open.
    environ: ?*const std.process.Environ.Map,
    /// Seconds of idle past the last request before the engine closes; 0 never closes.
    idle_s: i64,
    /// Parked now; the next request reopens it. Null keeps the live engine the server holds.
    parked: std.atomic.Value(bool) = .init(false),
    /// The manager's thread, waiting out the idle window.
    thread: ?std.Thread = null,
    /// Set once `stop` must join the thread and close a parked process.
    stopped: std.atomic.Value(bool) = .init(false),
    /// The engine the process started with, still unclosed (the park closes it); null after the close.
    held: ?api.Opened = null,
    /// The parked stand-in the server holds while no engine is resident; freed at the reload's swap.
    stub: ?*Stub = null,
};

/// Starts the idle timer for `args.idle_unload` seconds; null and 0 keep the manager off.
pub fn start(gpa: Allocator, io: std.Io, args: cli.Args, re: Reopen, dir: []const u8, model_type: []const u8, environ: ?*const std.process.Environ.Map) ?*Manager {
    if (args.idle_unload <= 0) return null;
    const m = gpa.create(Manager) catch return null;
    m.* = .{ .gpa = gpa, .io = io, .args = args, .re = re, .dir = dir, .model_type = model_type, .environ = environ, .idle_s = args.idle_unload };
    m.thread = std.Thread.spawn(.{}, idle, .{m}) catch {
        gpa.destroy(m);
        return null;
    };
    return m;
}

/// The idle timer: sleep out the window, then park the engine unless a request arrived.
fn idle(m: *Manager) void {
    // `idle_s` is seconds: wait it out in 100 ms slices so a stop lands fast; the park happens at 300 s, not 300 ms
    var waited_ms: i64 = 0;
    const window_ms = m.idle_s * 1000;
    while (waited_ms < window_ms) : (waited_ms += 100) {
        if (m.stopped.load(.acquire)) return;
        std.Io.sleep(m.io, .fromMilliseconds(100), .awake) catch {};
    }
    if (m.stopped.load(.acquire)) return;
    if (!m.parked.swap(true, .acq_rel)) log.line("idle unload: the model has been idle {d} s; the next request reloads it", .{m.idle_s});
}

/// True when the request is the park's first: the caller closes the held engine and reloads before submitting.
pub fn waking(m: *Manager) bool {
    return m.parked.swap(false, .acq_rel);
}

/// The parked stand-in: refuses every submit, reports the server unavailable, holds no weights.
const Stub = struct {
    gpa: Allocator,
    info: api.Info,
    parked: std.atomic.Value(bool) = .init(false),

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

    fn statusFn(ctx: *anyopaque, out: *api.Status, _: []u32) void {
        _ = ctx;
        out.* = .{ .healthy = false }; // /health 503 and the dashboard's unavailable state while parked
    }

    fn memoryFn(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// Closes the held engine and swaps the stand-in in (the wake hook calls this once, before `reload`).
pub fn park(m: *Manager, srv: *server_mod.Server) error{OutOfMemory}!void {
    if (m.stub != null) return; // already parked (a failed reload re-parks): the stub is in place
    const stub = try m.gpa.create(Stub);
    stub.* = .{ .gpa = m.gpa, .info = srv.info };
    const held = m.held orelse api.Opened{ .engine = srv.engine, .close = noopClose, .ctx = srv.engine.ctx };
    m.held = held;
    m.stub = stub;
    srv.engine = stub.engine();
    srv.stopKeepalive(); // its target was the closed engine's queue: a tick would hit freed memory
    m.re.close_held(m.re.ctx); // the weights leave here: the reload opens with them gone
    log.line("idle unload: the engine is closed; the next request reloads it", .{});
}

fn noopClose(_: *anyopaque) void {}

/// Reopens the engine and its text after an unload; the reply names the failure and the request is refused.
pub fn reload(m: *Manager, srv: *server_mod.Server, problem_out: *[]const u8) error{ OutOfMemory, ReloadFailed, EngineOpenFailed }!void {
    var problem: []const u8 = "";
    const opened = (m.re.open(m.re.ctx, srv.arena.allocator(), m.gpa, m.io, m.dir, m.model_type, m.args, &problem) catch {
        m.parked.store(true, .release); // still parked: the next request retries the reload
        return error.EngineOpenFailed;
    }) orelse {
        m.parked.store(true, .release);
        problem_out.* = try std.fmt.allocPrint(m.gpa, "the engine could not reload ({s}); send the request again", .{problem});
        return error.ReloadFailed;
    };
    var text_problem: []const u8 = "";
    const text = m.re.load_text(m.re.ctx, m.gpa, m.io, m.dir, m.args.chat_template, srv.arena.allocator(), &text_problem) catch {
        opened.close(opened.ctx); // the fresh engine cannot serve without its text: free it at once
        m.parked.store(true, .release);
        problem_out.* = if (text_problem.len > 0) try m.gpa.dupe(u8, text_problem) else try m.gpa.dupe(u8, @errorName(error.ReloadFailed));
        return error.ReloadFailed;
    };
    // swap the fresh engine under the request paths; the stub holds no weights and is simply dropped
    srv.engine = opened.engine;
    srv.info = opened.engine.info();
    srv.text = text.text();
    srv.eos = try srv.arena.allocator().dupe(u32, srv.text.eosIds());
    if (srv.config.keep_warm_s > 0) if (srv.engine.keepaliveTarget()) |target| {
        srv.keepalive = api.keepalive.Keepalive.start(m.gpa, m.io, target, @as(i64, srv.config.keep_warm_s) * std.time.ns_per_s) catch null;
    };
    if (m.stub) |s| m.gpa.destroy(s);
    m.stub = null;
    m.held = opened;
    m.re.replace(m.re.ctx, opened); // shutdown closes this engine, not the long-gone startup one
    log.line("idle reload: the model is resident again", .{});
}

/// Stops the timer; a parked engine's server is closed by `serve.run`'s own shutdown path.
pub fn stop(m: *Manager) void {
    m.stopped.store(true, .release);
    if (m.thread) |t| t.join();
    m.thread = null;
}

test "the manager stays off for 0 and negative idle seconds" {
    var a: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer a.deinit();
    var args: cli.Args = .{};
    args.idle_unload = 0;
    const re: Reopen = .{ .ctx = undefined, .open = undefined, .load_text = undefined, .close_held = undefined, .replace = undefined };
    const m = start(std.testing.allocator, std.testing.io, args, re, "d", "t", null);
    try std.testing.expect(m == null);
    args.idle_unload = -5;
    try std.testing.expect(start(std.testing.allocator, std.testing.io, args, re, "d", "t", null) == null);
}

test "a two-second window does not park within the first second (units: seconds, not milliseconds)" {
    var a: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer a.deinit();
    var args: cli.Args = .{};
    args.idle_unload = 2;
    const re: Reopen = .{ .ctx = undefined, .open = undefined, .load_text = undefined, .close_held = undefined, .replace = undefined };
    const m = start(std.testing.allocator, std.testing.io, args, re, "d", "t", null) orelse return error.TestUnexpectedResult;
    defer {
        stop(m);
        std.testing.allocator.destroy(m);
    }
    // at 300 ms (the old units bug parked here) the window must still be open
    std.Io.sleep(std.testing.io, .fromMilliseconds(300), .awake) catch {};
    try std.testing.expect(!m.parked.load(.acquire));
    // by 2.5 s the window has passed and the park lands
    var waited: u32 = 0;
    while (!m.parked.load(.acquire) and waited < 40) : (waited += 1) {
        std.Io.sleep(std.testing.io, .fromMilliseconds(100), .awake) catch {};
    }
    try std.testing.expect(m.parked.load(.acquire));
}

test "the idle window parks the engine and waking resets the flag" {
    var a: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer a.deinit();
    var args: cli.Args = .{};
    args.idle_unload = 1;
    const re: Reopen = .{ .ctx = undefined, .open = undefined, .load_text = undefined, .close_held = undefined, .replace = undefined };
    const m = start(std.testing.allocator, std.testing.io, args, re, "d", "t", null) orelse return error.TestUnexpectedResult;
    defer {
        stop(m);
        std.testing.allocator.destroy(m);
    }
    // the window is 1 s: after it, parked flips true; waking flips it back and reports the wake
    var waited: u32 = 0;
    while (!m.parked.load(.acquire) and waited < 100) : (waited += 1) {
        std.Io.sleep(std.testing.io, .fromMilliseconds(100), .awake) catch {};
    }
    try std.testing.expect(m.parked.load(.acquire));
    try std.testing.expect(waking(m));
    try std.testing.expect(!waking(m)); // the second wake is not the park's first
}

test "the parked stub refuses submits and reports unhealthy while the manager holds the real engine" {
    var stub = Stub{ .gpa = std.testing.allocator, .info = .{ .lanes = 4 } };
    var probe: api.Request = .{ .prompt = &.{}, .max_tokens = 0 };
    var status: api.Status = .{};
    stub.engine().status(&status, &.{});
    try std.testing.expect(!status.healthy);
    try std.testing.expectError(error.Closed, stub.engine().submit(1, &probe, .{ .ctx = undefined, .event = undefined }));
}
