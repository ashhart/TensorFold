//! The serve, stop and status tools: `tensorfold serve` children on loopback, each with its own key.
const std = @import("std");
const builtin = @import("builtin");
const json = @import("json");
const checkpoint = @import("checkpoint_cli");
const mcp = @import("mcp.zig");
const rpc = @import("mcp_rpc.zig");
const tools = @import("mcp_tools.zig");
const cache = @import("mcp_cache.zig");
const cli = @import("cli.zig");
const hub = @import("hub.zig");
const posix_only = builtin.os.tag != .windows;
const server = if (posix_only) @import("benchmark_server.zig") else struct {};
const weights = if (posix_only) @import("benchmark_model.zig") else struct {};
const child = if (posix_only) @import("chat_cli_child.zig") else struct {};
const Allocator = std.mem.Allocator;
const Value = json.Value;

/// The serve child as `tensorfold benchmark` runs it; Windows has none, as it stops children by POSIX signals.
const Child = if (posix_only) server.Managed else void;

/// How often a loading server's progress is reported, in seconds.
const report_every_s = 5;

/// What a server this session starts says when it cannot start or stop.
const words = if (posix_only) server.Words{
    .log = "tensorfold serve could not start: its log under ~/.tensorfold/mcp could not be created",
    .spawn = "tensorfold serve could not start; check the TensorFold installation",
    .exited = "tensorfold serve exited while loading",
    .timed_out = "tensorfold serve did not answer within",
    .stuck = "tensorfold serve did not exit after SIGTERM and SIGKILL; cleanup is incomplete",
} else {};

pub const State = enum { starting, running, exited, stopped };

pub const Entry = struct {
    id: []const u8,
    model: []const u8,
    name: []const u8,
    dir: []const u8,
    context: ?u32,
    drafter: ?[]const u8,
    key: []const u8,
    url: []const u8 = "",
    state: State = .starting,
    child: ?Child = null,
    /// An exited child's last log lines.
    last: []const []const u8 = &.{},
    /// The loading child's stop: set by stop, by the session's end and by the client's cancel.
    cancel: std.atomic.Value(bool) = .init(false),
};

/// The servers this session started, under one lock that every tool call and the session's end take.
pub const Registry = struct {
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    arena: std.heap.ArenaAllocator,
    entries: std.ArrayList(*Entry) = .empty,
    next: u32 = 1,
    ending: bool = false,

    pub fn init(gpa: Allocator, io: std.Io) Registry {
        return .{ .io = io, .arena = .init(gpa) };
    }

    pub fn deinit(r: *Registry) void {
        r.arena.deinit();
    }

    /// The session's end: every server stops, and one a call registers later is stopped on arrival.
    pub fn stopAll(r: *Registry) void {
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        r.ending = true;
        for (r.entries.items) |e| {
            e.cancel.store(true, .release);
            _ = halt(e);
        }
    }

    const Claim = union(enum) { new: *Entry, same: *Entry, loading: *Entry, ending };

    /// A new entry for these settings, or the server already running or loading them.
    fn claim(r: *Registry, model: []const u8, name: []const u8, dir: []const u8, context: ?u32, drafter: ?[]const u8) !Claim {
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        if (r.ending) return .ending;
        const a = r.arena.allocator();
        for (r.entries.items) |e| {
            const same = std.mem.eql(u8, e.dir, dir) and e.context == context and std.mem.eql(u8, e.drafter orelse "", drafter orelse "");
            if (!same) continue;
            if (e.state == .starting) return .{ .loading = e };
            if (e.state == .running and alive(e, a)) return .{ .same = e };
        }
        const e = try a.create(Entry);
        e.* = .{
            .id = try std.fmt.allocPrint(a, "{d}", .{r.next}),
            .model = try a.dupe(u8, model),
            .name = try a.dupe(u8, name),
            .dir = try a.dupe(u8, dir),
            .context = context,
            .drafter = if (drafter) |d| try a.dupe(u8, d) else null,
            .key = try child.freshKey(a, r.io),
        };
        try r.entries.append(a, e);
        r.next += 1;
        return .{ .new = e };
    }

    /// A started child into its entry; false when stop or the session's end came first, and the child is stopped.
    fn register(r: *Registry, e: *Entry, started: Child) !bool {
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        e.child = started;
        errdefer _ = halt(e);
        if (r.ending or e.cancel.load(.acquire)) {
            _ = halt(e);
            return false;
        }
        const a = r.arena.allocator();
        e.child.?.base_url = try a.dupe(u8, started.base_url); // the call's arena ends with the call
        e.url = try std.fmt.allocPrint(a, "{s}/v1", .{e.child.?.base_url});
        e.state = .running;
        return true;
    }

    fn drop(r: *Registry, e: *Entry) void {
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        for (r.entries.items, 0..) |x, i| if (x == e) {
            _ = r.entries.orderedRemove(i);
            return;
        };
    }
};

/// Whether the entry's child still runs; an exit seen here marks it exited with its last log lines, kept in ``a``.
fn alive(e: *Entry, a: Allocator) bool {
    if (comptime !posix_only) return false;
    const m = if (e.child) |*m| m else return false;
    if (m.closed) return false;
    const lines = (m.exitedTail(a) catch return true) orelse return true;
    if (e.state == .running) {
        e.state = .exited;
        e.last = lines;
    }
    return false;
}

/// Stops the entry's child, if it runs; null when it is gone, else why it is not.
fn halt(e: *Entry) ?[]const u8 {
    if (comptime !posix_only) return null;
    const why = if (e.child) |*m| m.stop() else null;
    if (e.state != .exited) e.state = .stopped;
    return why;
}

/// serve's arguments, checked against its schema.
pub const Wanted = struct { model: []const u8, context: ?i64, drafter: ?[]const u8 };

/// While a server loads: the client's cancel passed on as its stop, and its progress every few seconds.
const Loading = struct {
    call: *mcp.Call,
    entry: *Entry,
    done: std.atomic.Value(bool) = .init(false),

    fn watch(l: *Loading) void {
        const io = l.call.ctx().io;
        var polls: u32 = 0;
        while (!l.done.load(.acquire)) : (polls += 1) {
            if (l.call.stopping()) l.entry.cancel.store(true, .release);
            if (polls > 0 and polls % (5 * report_every_s) == 0) {
                var buf: [512]u8 = undefined;
                l.call.report(std.fmt.bufPrint(&buf, "loading {s}: {d} s", .{ l.entry.model, polls / 5 }) catch "loading");
            }
            io.sleep(.fromMilliseconds(200), .awake) catch {};
        }
    }
};

/// The serve tool: a cached model served on 127.0.0.1 with a fresh key, answered once it answers requests.
pub fn start(c: *mcp.Call, w: Wanted) !*json.Object {
    const a = c.a;
    if (comptime !posix_only)
        return tools.failed(a, "serve is not supported on Windows yet, as it stops its servers with POSIX signals; run tensorfold serve {s} in a terminal instead", .{w.model});
    const ctx = c.ctx();
    var problem: []const u8 = "";
    const dir = try cache.resolve(a, ctx.io, ctx.environ, w.model, &problem) orelse return tools.failed(a, "{s}", .{problem});
    const model_type = checkpoint.hub.modelType(a, ctx.io, dir);
    if (checkpoint.hub.family(model_type) == null) return tools.failed(a, "{s}: no TensorFold family in this build serves model_type {s}", .{ w.model, model_type });
    const drafter = if (w.drafter) |d| try cache.resolve(a, ctx.io, ctx.environ, d, &problem) orelse return tools.failed(a, "drafter: {s}", .{problem}) else null;
    const context: ?u32 = if (w.context) |n| @intCast(n) else null;
    const r = &c.session.servers;
    const e = switch (try r.claim(w.model, hub.servedName("", w.model, dir), dir, context, drafter)) {
        .new => |e| e,
        .same => |e| return served(a, e, true),
        .loading => |e| return tools.failed(a, "{s} is already loading as server {s}; status shows when it runs", .{ w.model, e.id }),
        .ending => return tools.failed(a, "the session is ending, so no server starts", .{}),
    };
    errdefer r.drop(e);
    var extra: std.ArrayList([]const u8) = .empty;
    try extra.append(a, "--no-update-check");
    if (context) |n| try extra.appendSlice(a, &.{ "--context", try std.fmt.allocPrint(a, "{d}", .{n}) });
    if (drafter) |d| try extra.appendSlice(a, &.{ "--drafter", d });
    const argv = try child.argv(a, ctx.exe orelse try std.process.executablePathAlloc(ctx.io, a), dir, e.name, extra.items);
    var u: cli.Usage = .{};
    _ = cli.parse(a, argv[2..], &u) catch |err| switch (err) {
        error.Usage => {
            r.drop(e);
            return tools.failed(a, "tensorfold serve refuses these settings: {s}", .{u.message});
        },
        else => |x| return x,
    };
    // the parent's variables a serve reads, less its own key and the benchmark's upload token, as the router has them
    const env = try server.environWithout(a, ctx.environ, &.{ "TENSORFOLD_API_KEY", "TENSORFOLD_REQUEST_JOURNAL", "TENSORFOLD_BENCHMARK_TOKEN" });
    const logs = try std.fs.path.join(a, &.{ ctx.home, ".tensorfold", "mcp" });
    _ = std.Io.Dir.cwd().createDirPathStatus(ctx.io, logs, .fromMode(0o700)) catch {};
    var failure: server.Failure = .none;
    var tail: []const []const u8 = &.{};
    const timeout_s = server.startupTimeout(weights.weightBytes(ctx.io, dir));
    c.report(try std.fmt.allocPrint(a, "starting tensorfold serve for {s}", .{w.model}));
    var loading: Loading = .{ .call = c, .entry = e };
    const watcher = std.Thread.spawn(.{}, Loading.watch, .{&loading}) catch null;
    const started = server.Managed.spawn(a, ctx.io, ctx.environ, argv, .{
        .startup_timeout_s = timeout_s,
        .grace_ms = ctx.grace_ms,
        .bench_dir = logs,
        .tail = &tail,
        .served = e.name,
        .lock = false,
        .own_group = true,
        .key = e.key,
        .words = words,
        .environ = &env,
        .guard = false,
        .stop = &e.cancel,
        .failure = &failure,
    }, &problem);
    loading.done.store(true, .release); // no progress may follow the answer
    if (watcher) |t| t.join();
    if (try started) |m| {
        if (c.cancelled.load(.acquire)) e.cancel.store(true, .release); // its client will never learn the key
        if (try r.register(e, m)) return served(a, e, false);
        return tools.failed(a, "server {s} was stopped while {s} loaded", .{ e.id, w.model });
    }
    r.drop(e);
    return switch (failure) {
        .exited => tools.failed(a, "tensorfold serve exited while loading {s}{s}", .{ w.model, try lastLog(a, tail) }),
        .timeout => tools.failed(a, "{s} did not answer within {d} s of starting, so it was stopped; check the checkpoint, memory and backend", .{ w.model, timeout_s }),
        .stopped => tools.failed(a, "{s} was stopped while it loaded", .{w.model}),
        .setup, .spawn, .none => tools.failed(a, "{s}", .{problem}),
    };
}

fn lastLog(a: Allocator, tail: []const []const u8) ![]const u8 {
    if (tail.len == 0) return "";
    return std.fmt.allocPrint(a, "; its last log lines: {s}", .{try std.mem.join(a, " | ", tail)});
}

fn served(a: Allocator, e: *const Entry, again: bool) !*json.Object {
    const text = try std.fmt.allocPrint(a, "{s} {s} as {s} at {s} (server {s}).\nAPI key: {s}; send it as Authorization: Bearer KEY, or as x-api-key to /v1/messages.\n" ++
        "Chat completions, Responses and Anthropic Messages are under that base URL with model {s}; stop it with stop and id {s}.\n", .{ if (again) "Already serving" else "Serving", e.model, e.name, e.url, e.id, e.key, e.name, e.id });
    const o = try json.newObject(a);
    try o.put(a, "id", .{ .string = e.id });
    try o.put(a, "model", .{ .string = e.model });
    try o.put(a, "name", .{ .string = e.name });
    try o.put(a, "base_url", .{ .string = e.url });
    try o.put(a, "api_key", .{ .string = e.key });
    try o.put(a, "context", try tools.optInt(a, if (e.context) |n| @as(u64, n) else null));
    try o.put(a, "already_running", .{ .bool = again });
    return tools.answered(a, text, o);
}

/// The stop tool: the server ``id`` stopped, or its loading cancelled; never a process this session did not start.
pub fn stop(c: *mcp.Call, id: []const u8) !*json.Object {
    const a = c.a;
    const r = &c.session.servers;
    r.mutex.lockUncancelable(r.io);
    defer r.mutex.unlock(r.io);
    const e = for (r.entries.items) |e| {
        if (std.mem.eql(u8, e.id, id)) break e;
    } else return tools.failed(a, "this session started no server {s}; status lists the ones it did", .{id});
    const o = try json.newObject(a);
    try o.put(a, "id", .{ .string = e.id });
    try o.put(a, "model", .{ .string = e.model });
    const text = switch (e.state) {
        .starting => blk: {
            e.cancel.store(true, .release);
            break :blk try std.fmt.allocPrint(a, "Server {s} ({s}) was still loading; it stops instead of serving.\n", .{ e.id, e.model });
        },
        .running => blk: {
            if (!alive(e, r.arena.allocator())) break :blk try std.fmt.allocPrint(a, "Server {s} ({s}) had already exited.\n", .{ e.id, e.model });
            if (halt(e)) |why| return tools.failed(a, "server {s}: {s}", .{ e.id, why });
            break :blk try std.fmt.allocPrint(a, "Stopped server {s} ({s} at {s}).\n", .{ e.id, e.model, e.url });
        },
        .exited => try std.fmt.allocPrint(a, "Server {s} ({s}) had already exited.\n", .{ e.id, e.model }),
        .stopped => try std.fmt.allocPrint(a, "Server {s} ({s}) was already stopped.\n", .{ e.id, e.model }),
    };
    try o.put(a, "state", .{ .string = @tagName(if (e.state == .starting) State.stopped else e.state) });
    return tools.answered(a, text, o);
}

/// The status tool: every server this session started, whether each still runs, and an exited one's last log lines.
pub fn status(c: *mcp.Call) !*json.Object {
    const a = c.a;
    const r = &c.session.servers;
    r.mutex.lockUncancelable(r.io);
    defer r.mutex.unlock(r.io);
    var text: std.Io.Writer.Allocating = .init(a);
    const items = try a.alloc(Value, r.entries.items.len);
    for (r.entries.items, items) |e, *slot| {
        const running = e.state == .running and alive(e, r.arena.allocator());
        try text.writer.print("server {s}: {s} as {s}, {t}", .{ e.id, e.model, e.name, e.state });
        if (e.url.len > 0) try text.writer.print(", {s}", .{e.url});
        if (e.state == .exited and e.last.len > 0) try text.writer.print("; last log: {s}", .{try std.mem.join(a, " | ", e.last)});
        try text.writer.writeAll("\n");
        const o = try json.newObject(a);
        try o.put(a, "id", .{ .string = e.id });
        try o.put(a, "model", .{ .string = e.model });
        try o.put(a, "name", .{ .string = e.name });
        try o.put(a, "base_url", if (e.url.len > 0) .{ .string = e.url } else .null);
        try o.put(a, "state", .{ .string = @tagName(e.state) });
        try o.put(a, "running", .{ .bool = running });
        if (e.state == .exited) try o.put(a, "last_log", try rpc.strings(a, e.last));
        slot.* = .{ .object = o };
    }
    if (items.len == 0) try text.writer.writeAll("this session has started no servers\n");
    const o = try json.newObject(a);
    try o.put(a, "servers", .{ .array = items });
    return tools.answered(a, text.written(), o);
}
