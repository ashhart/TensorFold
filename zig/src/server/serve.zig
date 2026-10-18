//! Server lifetime: keys, socket, startup and live lines, and a clean stop on SIGTERM or SIGINT.
const std = @import("std");
const platform = @import("sys");
const api = @import("engine_api");
const json = @import("json");
const auth = @import("auth.zig");
const cli = @import("cli.zig");
const log = @import("log.zig");
const live = @import("live.zig");
const listener_mod = @import("listener.zig");
const model_text = @import("model_text.zig");
const server_mod = @import("server.zig");
const routes = @import("routes.zig");
const recall_routes = @import("recall_routes.zig");
const Allocator = std.mem.Allocator;
const posix = platform.posix;

var stop_requested: std.atomic.Value(bool) = .init(false);
/// The signal that asked for the stop (0: none yet), for the shutdown line
var stop_signal: std.atomic.Value(i32) = .init(0);
/// Stop signals so far: a second one cuts the drain short
var stop_count: std.atomic.Value(u32) = .init(0);

fn onStop(sig: posix.SIG) callconv(.c) void {
    stop_signal.store(@intCast(@backingInt(sig)), .release);
    _ = stop_count.fetchAdd(1, .acq_rel);
    stop_requested.store(true, .release);
}

fn signalName(sig: i32) []const u8 {
    if (sig == @backingInt(posix.SIG.TERM)) return "SIGTERM";
    if (sig == @backingInt(posix.SIG.INT)) return "SIGINT";
    return "a stop signal";
}

/// SIGTERM and SIGINT stop the server cleanly; SIGHUP rereads the key file; SIGPIPE is ignored, as Python ignores it.
pub fn installSignals(keys: bool) void {
    const stop: posix.Sigaction = .{ .handler = .{ .handler = onStop }, .mask = posix.sigemptyset(), .flags = 0 };
    posix.sigaction(.TERM, &stop, null);
    posix.sigaction(.INT, &stop, null);
    const ignore: posix.Sigaction = .{ .handler = .{ .handler = posix.SIG.IGN }, .mask = posix.sigemptyset(), .flags = 0 };
    posix.sigaction(.PIPE, &ignore, null);
    if (keys) auth.installHangup();
}

pub fn stopping() bool {
    return stop_requested.load(.acquire);
}

/// The hard limit macOS reads as unlimited.
const rlim_infinity: u64 = platform.c.RLIM.INFINITY;

/// Try the finite hard limit, else 65536 then 10240; never choose an unlimited soft limit.
fn fillTargets(max: u64, out: *[2]u64) []const u64 {
    if (max != rlim_infinity) {
        out[0] = max;
        return out[0..1];
    }
    out[0] = 65536;
    out[1] = 10240;
    return out;
}

/// Try raising the soft file limit above 1024; preserve limits on failure and return null if unchanged.
fn raiseOpenFileLimit() ?[2]u64 {
    const limit = posix.getrlimit(.NOFILE) catch return null;
    const cur: u64 = @intCast(limit.cur);
    if (cur == rlim_infinity) return null; // already unlimited
    var buf: [2]u64 = undefined;
    for (fillTargets(@intCast(limit.max), &buf)) |target| {
        if (target <= cur) continue;
        posix.setrlimit(.NOFILE, .{ .cur = @intCast(target), .max = limit.max }) catch continue;
        return .{ cur, target };
    }
    return null;
}

/// What a server needs besides its flags.
pub const Setup = struct {
    engine: api.Engine,
    text: model_text.Text,
    served: []const u8,
    /// The checkpoint's folder (structured output reads its tokenizer.json and config.json, /slide shows its shards).
    model_dir: []const u8 = "",
    /// Sampling defaults from generation_config.json and the flags, as a JSON object; null: greedy.
    sampling: ?json.Value = null,
    environ: ?*const std.process.Environ.Map = null,
    started: i96 = 0,
    /// The release, backend and GPU the open /health names.
    identity: server_mod.Identity = .{},
    /// A test build's TF_SLIDE_FORGET_FAIL: the fact whose check a delete fails (0: none, as in every release).
    forget_fail: u32 = 0,
    /// Called once the socket listens, with its port (tests read it).
    on_listen: ?*const fn (port: u16) void = null,
    /// Recall's record, attached to the engine already (--recall); null: off.
    recall: ?*api.recall_store.Store = null,
    /// Run by `run` so the engine stops before the server is freed; null: the caller closes it after `run`.
    stop: ?EngineStop = null,
};

/// `halt` (optional) ends the rounds at a boundary at the drain's deadline, so `close` never meets a running round.
pub const EngineStop = struct {
    ctx: *anyopaque,
    halt: ?*const fn (ctx: *anyopaque, reason: []const u8) void = null,
    close: *const fn (ctx: *anyopaque) void,
};

fn env(s: Setup, name: []const u8) ?[]const u8 {
    const m = s.environ orelse return null;
    return m.get(name);
}

/// Serves until a stop signal; the process exit status.
pub fn run(gpa: Allocator, io: std.Io, args: cli.Args, s: Setup) u8 {
    stop_requested.store(false, .release);
    stop_count.store(0, .release);
    stop_signal.store(0, .release);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var store = auth.Store.init(gpa, args.api_key, env(s, "TENSORFOLD_API_KEY") orelse "", args.api_key_file, args.metrics_open, &problem) catch {
        std.debug.print("tensorfold: {s}\n", .{if (problem.len > 0) problem else "API key file cannot be opened safely"});
        return 1;
    };
    defer store.deinit();
    if (!store.enabled() and !auth.loopback(args.host)) log.line("warning: non-loopback server has no API key; requests are open", .{});
    var ids: std.ArrayList([]const u8) = .empty;
    for ([_][]const []const u8{ &.{s.served}, args.alias }) |group| for (group) |raw| {
        const id = std.mem.trim(u8, raw, " \t\r\n");
        if (id.len == 0) continue;
        for (ids.items) |seen| {
            if (std.mem.eql(u8, seen, id)) break;
        } else ids.append(a, id) catch return 1;
    };
    const salt = if (env(s, "TENSORFOLD_SEED_SALT")) |t| std.fmt.parseInt(i64, std.mem.trim(u8, t, " "), 10) catch {
        std.debug.print("tensorfold: TENSORFOLD_SEED_SALT={s}: an integer\n", .{t});
        return 1;
    } else 0;
    const drain_s = seconds(env(s, "TF_DRAIN_S"), 20) orelse {
        std.debug.print("tensorfold: TF_DRAIN_S: seconds, a number >= 0\n", .{});
        return 1;
    };
    const keepalive_s = seconds(env(s, "TF_SSE_KEEPALIVE_S"), 15) orelse {
        std.debug.print("tensorfold: TF_SSE_KEEPALIVE_S: seconds, a number >= 0\n", .{});
        return 1;
    };
    const config: server_mod.Config = .{
        .served_name = s.served,
        .model_dir = s.model_dir,
        .model_ids = ids.items,
        .default_max_tokens = args.max_tokens,
        .enable_thinking = args.thinking,
        .reasoning_effort = args.reasoning_effort,
        .thinking_budget = args.thinking_budget,
        .loop_guard = args.loop_guard,
        .keep_warm_s = args.keep_warm,
        .default_sampling = s.sampling,
        .use_drafts = !args.no_drafts,
        .vision_urls = args.vision_urls,
        .seed_salt = salt,
        .sse_keepalive_s = keepalive_s,
        .request_log = env(s, "TENSORFOLD_REQUEST_LOG"),
        .dashboard = args.dashboard,
        .compact_at = if (args.compact_auto) .{ .auto = {} } else if (args.compact_fraction) |f| .{ .fraction = f } else null,
        .compact_keep = args.compact_keep,
        .compact_memory = args.compact_memory,
        .slide_graph = args.slide_graph,
        .slide_forget_fail = s.forget_fail,
        .cortheon = args.cortheon,
        .cortheon_search_key = env(s, "BRAVE_SEARCH_API_KEY"),
        .saved_conversations = args.learn_conversations,
        .identity = s.identity,
        .environ = s.environ,
    };
    const srv = server_mod.Server.init(gpa, io, s.engine, s.text, config, if (store.enabled()) &store else null) catch {
        std.debug.print("tensorfold: the server could not start\n", .{});
        return 1;
    };
    defer srv.deinit();
    srv.recall = s.recall;
    if (args.loop_guard and !srv.info.loop_guard) {
        std.debug.print("tensorfold: --loop-guard is not supported by this native engine\n", .{});
        return 2;
    }
    if (raiseOpenFileLimit()) |r| log.line("open-file limit raised: soft {d} -> {d}", .{ r[0], r[1] });
    const address = resolve(io, args.host, args.port) orelse {
        std.debug.print("tensorfold: cannot resolve --host {s}\n", .{args.host});
        return 1;
    };
    const lis = listener_mod.Listener.open(address) catch |e| {
        std.debug.print("tensorfold: cannot listen on {s}:{d} ({s})\n", .{ args.host, args.port, @errorName(e) });
        return 1;
    };
    installSignals(store.enabled());
    const port = lis.port();
    if (s.on_listen) |f| f(port);
    const window = srv.info.context_window;
    if (srv.info.startup.len > 0) log.line("{s}", .{srv.info.startup});
    const now = std.Io.Clock.awake.now(io).toNanoseconds();
    const loaded = if (s.started > 0) @as(f64, @floatFromInt(now - s.started)) / 1e9 else 0;
    var window_text: [24]u8 = undefined;
    log.line("serving {s} at http://{s}:{d}/v1 (sampling: {s}; drafts: {s}; context: {s}; loaded in {d:.1}s)", .{
        s.served,                                                   args.host,                                                                                   port,   shownSampling(a, s.sampling),
        if (args.no_drafts or srv.info.plain_only) "off" else "on", if (window > 0) std.fmt.bufPrint(&window_text, "{d}", .{window}) catch "?" else "unlimited", loaded,
    });
    if (args.thinking and std.mem.indexOf(u8, s.text.templateSource(), "enable_thinking") != null)
        log.line("thinking on (the chat template's default): replies reason in reasoning_content before the answer in content, and max_tokens counts both. --no-thinking turns it off; a request can send chat_template_kwargs {{\"enable_thinking\": false}}", .{});
    if (s.recall) |r| recall_routes.line(r);
    if (srv.cortheon) |c| log.line("cortheon on: chat requests get Cortheon's reasoning program, web research before building, and cortheon_search/cortheon_fetch run here ({s}); a request can send \"cortheon\": false", .{c.engine});
    live.columns_env = env(s, "COLUMNS");
    var ticker: live.Ticker = .{ .engine = s.engine };
    const drawing = live.wanted(io, if (env(s, "TENSORFOLD_NO_LIVE")) |v| std.mem.eql(u8, v, "1") else false);
    if (drawing) ticker.start();
    // the listener stays open through the drain: new requests get 503 + Retry-After, /health "draining"
    var accept_stop: std.atomic.Value(bool) = .init(false);
    const accept = std.Thread.spawn(.{}, server_mod.Server.serve, .{ srv, lis, &accept_stop }) catch return 1;
    while (!stop_requested.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(50), .awake) catch {};
    drain(srv, io, s.stop, drain_s);
    accept_stop.store(true, .release);
    accept.join();
    lis.close();
    if (drawing) ticker.finish();
    s.engine.drain(); // kept states waiting for disk are written (under the engine's deadline) before the engine closes
    var waited: u32 = 0;
    while (srv.open_connections.load(.acquire) > 0 and waited < 40) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(50), .awake) catch {};
    if (!closeIdle(srv, s.stop)) {
        log.line("shutting down: exiting with {d} idle connection(s) still open", .{srv.open_connections.load(.acquire)});
        std.process.exit(0); // connections still open: end without freeing what they read
    }
    log.line("shutting down: stopped", .{});
    return 0;
}

/// Work in progress gets `drain_s` to finish (new work 503s) before the rest is halted; a second signal cuts it short.
fn drain(srv: *server_mod.Server, io: std.Io, stop: ?EngineStop, drain_s: f64) void {
    srv.draining.store(true, .seq_cst);
    const signals = stop_count.load(.acquire);
    log.line("shutting down ({s}): draining: new requests get 503; {d} request(s) in progress may finish within {d:.1} s", .{ signalName(stop_signal.load(.acquire)), srv.generating.load(.seq_cst), drain_s });
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    const deadline = t0 + @as(i96, @intFromFloat(drain_s * 1e9));
    while (srv.generating.load(.seq_cst) > 0 and std.Io.Clock.awake.now(io).toNanoseconds() < deadline and stop_count.load(.acquire) == signals)
        std.Io.sleep(io, .fromMilliseconds(20), .awake) catch {};
    const left = srv.generating.load(.seq_cst);
    if (left == 0) {
        log.line("shutting down: drained", .{});
        return;
    }
    const cut = if (stop_count.load(.acquire) != signals) "a second stop signal" else "the drain deadline (TF_DRAIN_S)";
    log.line("shutting down: {s}: {d} request(s) still in progress end \"{s}\" at the next round boundary", .{ cut, left, routes.restarting });
    if (stop) |st| if (st.halt) |h| h(st.ctx, routes.restarting);
    var waited: u32 = 0;
    while (srv.generating.load(.seq_cst) > 0 and waited < 100) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(20), .awake) catch {};
}

/// A seconds setting (TF_DRAIN_S, TF_SSE_KEEPALIVE_S): unset or blank keeps the default; null: not a number >= 0.
fn seconds(raw: ?[]const u8, default: f64) ?f64 {
    const t = std.mem.trim(u8, raw orelse "", " \t");
    if (t.len == 0) return default;
    const x = std.fmt.parseFloat(f64, t) catch return null;
    const limit = @as(f64, @floatFromInt(std.math.maxInt(i64))) / std.time.ns_per_s;
    return if (std.math.isFinite(x) and x >= 0 and x < limit) x else null;
}

/// ``--host`` as an address: an IP literal, else a name the resolver knows (localhost first).
fn resolve(io: std.Io, host: []const u8, port: u16) ?std.Io.net.IpAddress {
    if (std.Io.net.IpAddress.parse(host, port)) |ip| return ip else |_| {}
    if (std.ascii.eqlIgnoreCase(host, "localhost")) return .{ .ip4 = .loopback(port) };
    return std.Io.net.IpAddress.resolve(io, host, port) catch null;
}

/// The serving line's sampling: greedy, or each default as ``name value``.
fn shownSampling(a: Allocator, sampling: ?json.Value) []const u8 {
    const v = sampling orelse return "greedy";
    if (v != .object) return "greedy";
    const t = v.get("temperature") orelse return "greedy";
    const temp: f64 = switch (t) {
        .float => |f| f,
        .int => |i| std.fmt.parseFloat(f64, i) catch 0,
        else => 0,
    };
    if (temp <= 0) return "greedy";
    var parts: std.ArrayList([]const u8) = .empty;
    for (v.object.keys(), v.object.values()) |k, x| {
        var buf: [40]u8 = undefined;
        const text = switch (x) {
            .float => |f| json.floatRepr(&buf, f),
            .int => |i| i,
            else => "?",
        };
        parts.append(a, std.fmt.allocPrint(a, "{s} {s}", .{ k, text }) catch return "greedy") catch return "greedy";
    }
    return std.mem.join(a, ", ", parts.items) catch "greedy";
}

test "the targets are the hard limit, or the fallbacks under an unlimited one" {
    var buf: [2]u64 = undefined;
    try std.testing.expectEqualSlices(u64, &.{524288}, fillTargets(524288, &buf));
    try std.testing.expectEqualSlices(u64, &.{ 65536, 10240 }, fillTargets(rlim_infinity, &buf));
}

test "the raise lifts a low soft limit, and a second raise does nothing" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // no descriptor limits to raise
    const before = try posix.getrlimit(.NOFILE);
    defer posix.setrlimit(.NOFILE, before) catch {};
    if (@as(u64, @intCast(before.max)) <= 256) return error.SkipZigTest;
    try posix.setrlimit(.NOFILE, .{ .cur = 256, .max = before.max });
    const raised = raiseOpenFileLimit() orelse return error.TestExpectedRaise;
    try std.testing.expectEqual(@as(u64, 256), raised[0]);
    try std.testing.expectEqual(raised[1], @as(u64, @intCast((try posix.getrlimit(.NOFILE)).cur)));
    try std.testing.expect(raiseOpenFileLimit() == null);
}

test "Failure boundaries: drain and SSE duration settings refuse non-finite or unrepresentable values" {
    try std.testing.expectEqual(@as(?f64, null), seconds("inf", 20));
    try std.testing.expectEqual(@as(?f64, null), seconds("1e300", 20));
    try std.testing.expectEqual(@as(?f64, null), seconds("nan", 20));
}

fn closeIdle(srv: *server_mod.Server, stop: ?EngineStop) bool {
    if (srv.open_connections.load(.acquire) != 0 or srv.generating.load(.seq_cst) != 0) return false;
    srv.stopKeepalive();
    if (stop) |st| st.close(st.ctx);
    return true;
}

test "Failure boundaries: engine close waits for handlers and connections to leave" {
    const Count = struct {
        fn close(p: *anyopaque) void {
            const count: *usize = @ptrCast(@alignCast(p));
            count.* += 1;
        }
    };
    var count: usize = 0;
    var server: server_mod.Server = undefined;
    server.open_connections = .init(1);
    server.generating = .init(0);
    server.keepalive = null;
    const stop: EngineStop = .{ .ctx = &count, .close = Count.close };
    try std.testing.expect(!closeIdle(&server, stop));
    server.open_connections.store(0, .seq_cst);
    server.generating.store(1, .seq_cst);
    try std.testing.expect(!closeIdle(&server, stop));
    try std.testing.expectEqual(@as(usize, 0), count);
    server.generating.store(0, .seq_cst);
    try std.testing.expect(closeIdle(&server, stop));
    try std.testing.expectEqual(@as(usize, 1), count);
}
