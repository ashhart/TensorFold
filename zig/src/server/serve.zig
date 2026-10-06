//! A server's life: keys, the listening socket, the startup lines, the live line, and a clean stop on SIGTERM or SIGINT.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const auth = @import("auth.zig");
const cli = @import("cli.zig");
const log = @import("log.zig");
const live = @import("live.zig");
const listener_mod = @import("listener.zig");
const model_text = @import("model_text.zig");
const server_mod = @import("server.zig");
const Allocator = std.mem.Allocator;
const posix = std.posix;

var stop_requested: std.atomic.Value(bool) = .init(false);

fn onStop(_: posix.SIG) callconv(.c) void {
    stop_requested.store(true, .release);
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
const rlim_infinity: u64 = std.c.RLIM.INFINITY;

/// The targets to try, in order: the hard limit when it is finite; when it reads as unlimited, the PR's fallbacks,
/// 65536 then 10240, first that works (PR #294). The soft limit is never set to unlimited: anything that loops over
/// every possible descriptor would do absurd work. `out` holds the list; the caller gives room for two.
fn fillTargets(max: u64, out: *[2]u64) []const u64 {
    if (max != rlim_infinity) {
        out[0] = max;
        return out[0..1];
    }
    out[0] = 65536;
    out[1] = 10240;
    return out;
}

/// Raises the soft open-file limit toward the hard limit, so idle keep-alive connections cannot exhaust a default
/// 1024 (PR #294). Best effort: a failure keeps the current limits, and an unlimited soft limit, or one already at
/// every target, does nothing. Returns the old and new soft limits, or null when nothing changed; the caller logs.
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
    /// Sampling defaults from generation_config.json and the flags, as a JSON object; null: greedy.
    sampling: ?json.Value = null,
    environ: ?*const std.process.Environ.Map = null,
    started: i96 = 0,
    /// Called once the socket listens, with its port (tests read it).
    on_listen: ?*const fn (port: u16) void = null,
};

fn env(s: Setup, name: []const u8) ?[]const u8 {
    const m = s.environ orelse return null;
    return m.get(name);
}

/// Serves until a stop signal; the process exit status.
pub fn run(gpa: Allocator, io: std.Io, args: cli.Args, s: Setup) u8 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var store = auth.Store.init(gpa, args.api_key, env(s, "TENSORFOLD_API_KEY") orelse "", args.api_key_file, args.metrics_open, &problem) catch {
        std.debug.print("tensorfold: {s}\n", .{if (problem.len > 0) problem else "API key file cannot be opened safely"});
        return 1;
    };
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
    const config: server_mod.Config = .{
        .served_name = s.served,
        .model_ids = ids.items,
        .default_max_tokens = args.max_tokens,
        .enable_thinking = args.thinking,
        .reasoning_effort = args.reasoning_effort,
        .thinking_budget = args.thinking_budget,
        .loop_guard = args.loop_guard,
        .default_sampling = s.sampling,
        .use_drafts = !args.no_drafts,
        .seed_salt = salt,
        .request_log = env(s, "TENSORFOLD_REQUEST_LOG"),
        .dashboard = args.dashboard,
    };
    const srv = server_mod.Server.init(gpa, io, s.engine, s.text, config, if (store.enabled()) &store else null) catch {
        std.debug.print("tensorfold: the server could not start\n", .{});
        return 1;
    };
    defer srv.deinit();
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
    const now = std.Io.Clock.awake.now(io).toNanoseconds();
    const loaded = if (s.started > 0) @as(f64, @floatFromInt(now - s.started)) / 1e9 else 0;
    var window_text: [24]u8 = undefined;
    log.line("serving {s} at http://{s}:{d}/v1 (sampling: {s}; drafts: {s}; context: {s}; loaded in {d:.1}s)", .{
        s.served,                            args.host,                                                                                   port,   shownSampling(a, s.sampling),
        if (args.no_drafts) "off" else "on", if (window > 0) std.fmt.bufPrint(&window_text, "{d}", .{window}) catch "?" else "unlimited", loaded,
    });
    if (args.thinking and std.mem.indexOf(u8, s.text.templateSource(), "enable_thinking") != null)
        log.line("thinking on (the chat template's default): replies reason in reasoning_content before the answer in content, and max_tokens counts both. --no-thinking turns it off; a request can send chat_template_kwargs {{\"enable_thinking\": false}}", .{});
    live.columns_env = env(s, "COLUMNS");
    var ticker: live.Ticker = .{ .engine = s.engine };
    const drawing = live.wanted(io, if (env(s, "TENSORFOLD_NO_LIVE")) |v| std.mem.eql(u8, v, "1") else false);
    if (drawing) ticker.start();
    const accept = std.Thread.spawn(.{}, server_mod.Server.serve, .{ srv, lis, &stop_requested }) catch return 1;
    accept.join();
    lis.close();
    if (drawing) ticker.finish();
    var waited: u32 = 0;
    while (srv.open_connections.load(.acquire) > 0 and waited < 40) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(50), .awake) catch {};
    if (srv.open_connections.load(.acquire) > 0) std.process.exit(0); // replies still open: end without freeing what they read
    return 0;
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
    const before = try posix.getrlimit(.NOFILE);
    defer posix.setrlimit(.NOFILE, before) catch {};
    if (@as(u64, @intCast(before.max)) <= 256) return error.SkipZigTest;
    try posix.setrlimit(.NOFILE, .{ .cur = 256, .max = before.max });
    const raised = raiseOpenFileLimit() orelse return error.TestExpectedRaise;
    try std.testing.expectEqual(@as(u64, 256), raised[0]);
    try std.testing.expectEqual(raised[1], @as(u64, @intCast((try posix.getrlimit(.NOFILE)).cur)));
    try std.testing.expect(raiseOpenFileLimit() == null);
}
