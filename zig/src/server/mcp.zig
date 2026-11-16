//! `tensorfold mcp`: a Model Context Protocol server on stdio for this machine's checkpoints and local servers.
const std = @import("std");
const builtin = @import("builtin");
const json = @import("json");
const cli = @import("cli.zig");
const rpc = @import("mcp_rpc.zig");
const tools = @import("mcp_tools.zig");
const serve = @import("mcp_serve.zig");
const cache = @import("mcp_cache.zig");
const posix_only = builtin.os.tag != .windows;
const stream = if (posix_only) @import("benchmark_stream.zig") else struct {};
const stop = if (posix_only) @import("benchmark_stop.zig") else struct {};
const Allocator = std.mem.Allocator;
const Value = json.Value;
const none: Value = .null;

pub const usage_line = "usage: tensorfold mcp   (a Model Context Protocol server on stdin and stdout; see docs/mcp.md)\n";

/// The pulls and serves that may run at once, each on its own thread.
const max_running = 8;

const instructions = "TensorFold runs models on this machine. machine describes the machine; models lists the cached " ++
    "checkpoints and info describes one; pull downloads a Hugging Face repo; serve starts a local OpenAI- and " ++
    "Anthropic-compatible server and returns its base URL, served name and API key; status and stop manage the " ++
    "servers this session started, which all stop when the session ends.";

/// What the tools reach, so a test can run a session with fakes, a temporary home and a stand-in serve child.
pub const Context = struct {
    io: std.Io,
    /// Thread-safe: calls on their own threads and the server registry allocate from it.
    gpa: Allocator,
    environ: *const std.process.Environ.Map,
    home: []const u8,
    version: []const u8,
    engines: *const fn (Allocator) cli.Engines,
    /// The serve child's executable; null is this process's own.
    exe: ?[]const u8 = null,
    grace_ms: u32 = 10_000,
    /// How long the end of a session, its servers stopped, waits for a pull still running.
    drain_s: u32 = 10,
};

/// One tools/call: its request, revision, arguments, progress token and whether the client cancelled it.
pub const Call = struct {
    session: *Session,
    a: Allocator,
    id: Value,
    version: rpc.Version,
    tool: tools.Tool,
    args: Value,
    token: ?Value,
    cancelled: std.atomic.Value(bool) = .init(false),
    done: std.atomic.Value(bool) = .init(false),
    arena: std.heap.ArenaAllocator,
    thread: ?std.Thread = null,
    sent: u64 = 0,

    pub fn ctx(c: *const Call) *const Context {
        return c.session.ctx;
    }

    /// True once the client cancelled this call or the session is ending.
    pub fn stopping(c: *const Call) bool {
        return c.cancelled.load(.acquire) or c.session.ending.load(.acquire);
    }

    /// A progress notification saying ``message``, when the client sent a progress token and still waits.
    pub fn report(c: *Call, message: []const u8) void {
        const token = c.token orelse return;
        if (c.cancelled.load(.acquire)) return;
        var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
        defer arena.deinit();
        const a = arena.allocator();
        c.sent += 1;
        const p = json.newObject(a) catch return;
        p.put(a, "progressToken", token) catch return;
        p.put(a, "progress", json.intValue(a, c.sent) catch return) catch return;
        p.put(a, "message", .{ .string = message }) catch return;
        c.session.wire.send(rpc.notification(a, "notifications/progress", p) catch return);
    }

    /// The call's thread: its tool, then its answer unless the client cancelled it.
    fn run(c: *Call) void {
        defer c.done.store(true, .release);
        c.session.answer(c, tools.call(c));
    }
};

pub const Session = struct {
    ctx: *const Context,
    wire: *rpc.Wire,
    handshake: ?rpc.Version = null,
    ending: std.atomic.Value(bool) = .init(false),
    calls: std.ArrayList(*Call) = .empty,
    servers: serve.Registry,
    pulls: cache.Pulls = .{},

    pub fn init(ctx: *const Context, wire: *rpc.Wire) Session {
        return .{ .ctx = ctx, .wire = wire, .servers = .init(ctx.gpa, ctx.io) };
    }

    pub fn deinit(s: *Session) void {
        s.calls.deinit(s.ctx.gpa);
        s.pulls.deinit(s.ctx.gpa);
        s.servers.deinit();
    }

    /// One line from the client, answered on the wire; pull and serve answer later from their own threads.
    pub fn handle(s: *Session, a: Allocator, line: []const u8) !void {
        s.reap();
        switch (try rpc.read(a, line)) {
            .dropped => |why| if (why.len > 0) std.debug.print("tensorfold mcp: dropped {s}\n", .{why}),
            .refused => |r| s.wire.send(try rpc.failure(a, r)),
            .notification => |n| s.notified(n),
            .request => |r| try s.request(a, r),
        }
    }

    fn request(s: *Session, a: Allocator, r: rpc.Request) !void {
        const modern = (r.params.get("_meta") orelse none).get(rpc.meta_version) != null;
        if (!modern and eql(r.method, "initialize")) return s.initialize(a, r);
        if (!modern and eql(r.method, "ping")) return s.wire.send(try rpc.result(a, r.id, try json.newObject(a)));
        const v = switch (try rpc.version(a, r, s.handshake)) {
            .ok => |v| v,
            .refused => |why| return s.wire.send(try rpc.failure(a, why)),
        };
        if (eql(r.method, "tools/list")) return s.reply(a, r.id, v, try tools.list(a, v));
        if (eql(r.method, "tools/call")) return s.call(a, r, v);
        if (eql(r.method, "server/discover") and v.stateless()) return s.reply(a, r.id, v, try discover(a));
        s.wire.send(try rpc.failure(a, .{ .id = r.id, .code = rpc.method_not_found, .message = try std.fmt.allocPrint(a, "Method not found: {s}", .{r.method}) }));
    }

    /// The handshake revisions' initialize: the negotiated version, the tools capability and who this server is.
    fn initialize(s: *Session, a: Allocator, r: rpc.Request) !void {
        const v = rpc.negotiate(r.params.strField("protocolVersion"));
        s.handshake = v;
        const body = try json.newObject(a);
        try body.put(a, "protocolVersion", .{ .string = @tagName(v) });
        try body.put(a, "capabilities", try capabilities(a));
        try body.put(a, "serverInfo", try s.identity(a));
        try body.put(a, "instructions", .{ .string = instructions });
        s.wire.send(try rpc.result(a, r.id, body));
    }

    pub fn identity(s: *const Session, a: Allocator) !Value {
        const o = try json.newObject(a);
        try o.put(a, "name", .{ .string = "tensorfold" });
        try o.put(a, "title", .{ .string = "TensorFold" });
        try o.put(a, "version", .{ .string = s.ctx.version });
        return .{ .object = o };
    }

    fn reply(s: *Session, a: Allocator, id: Value, v: rpc.Version, body: *json.Object) !void {
        try rpc.complete(a, body, v, try s.identity(a));
        s.wire.send(try rpc.result(a, id, body));
    }

    /// tools/call: the envelope checked here, the tool run inline, or on its own thread when it can take minutes.
    fn call(s: *Session, a: Allocator, r: rpc.Request, v: rpc.Version) !void {
        const name = r.params.strField("name") orelse
            return s.wire.send(try rpc.failure(a, .{ .id = r.id, .code = rpc.invalid_params, .message = "Invalid params: tools/call needs a tool name" }));
        const tool = std.meta.stringToEnum(tools.Tool, name) orelse
            return s.wire.send(try rpc.failure(a, .{ .id = r.id, .code = rpc.invalid_params, .message = try std.fmt.allocPrint(a, "Unknown tool: {s}", .{name}) }));
        const args = r.params.get("arguments") orelse none;
        if (args != .null and args != .object)
            return s.wire.send(try rpc.failure(a, .{ .id = r.id, .code = rpc.invalid_params, .message = "Invalid params: arguments must be an object" }));
        if (tool.long() and s.calls.items.len >= max_running)
            return s.reply(a, r.id, v, try tools.failed(a, "{d} pulls and serves are running already; call again once one has answered", .{max_running}));
        const token = (r.params.get("_meta") orelse none).get("progressToken");
        const c = try s.ctx.gpa.create(Call);
        c.* = .{ .session = s, .a = undefined, .id = .null, .version = v, .tool = tool, .args = .null, .token = null, .arena = .init(s.ctx.gpa) };
        c.a = c.arena.allocator();
        errdefer {
            c.arena.deinit();
            s.ctx.gpa.destroy(c);
        }
        c.id = try json.deepCopy(c.a, r.id);
        c.args = try json.deepCopy(c.a, args);
        if (token) |t| if (t == .string or t == .int) {
            c.token = try json.deepCopy(c.a, t);
        };
        try s.calls.ensureUnusedCapacity(s.ctx.gpa, 1);
        if (tool.long()) if (std.Thread.spawn(.{}, Call.run, .{c})) |t| {
            c.thread = t;
            s.calls.appendAssumeCapacity(c);
            return;
        } else |_| {};
        defer {
            c.arena.deinit();
            s.ctx.gpa.destroy(c);
        }
        s.answer(c, tools.call(c));
    }

    /// A call's result, unless the client cancelled it; a tool's own error becomes a tool error, never a crash.
    fn answer(s: *Session, c: *Call, got: anyerror!*json.Object) void {
        if (c.cancelled.load(.acquire)) return;
        const body = got catch |e| tools.failed(c.a, "{s} failed: {t}", .{ @tagName(c.tool), e }) catch return;
        s.reply(c.a, c.id, c.version, body) catch return;
    }

    fn notified(s: *Session, n: rpc.Notification) void {
        if (!eql(n.method, "notifications/cancelled")) return; // initialized and the rest need nothing
        const id = n.params.get("requestId") orelse return;
        for (s.calls.items) |c| if (json.equal(c.id, id)) c.cancelled.store(true, .release);
    }

    /// Joins the calls that finished and frees them.
    fn reap(s: *Session) void {
        var i: usize = 0;
        while (i < s.calls.items.len) {
            const c = s.calls.items[i];
            if (!c.done.load(.acquire)) {
                i += 1;
                continue;
            }
            if (c.thread) |t| t.join();
            c.arena.deinit();
            s.ctx.gpa.destroy(c);
            _ = s.calls.swapRemove(i);
        }
    }

    /// The end of a session: every server stops first, a loading one stopping itself, then running calls finish.
    pub fn finish(s: *Session) void {
        s.ending.store(true, .release);
        s.servers.stopAll();
        const io = s.ctx.io;
        const started = std.Io.Clock.awake.now(io).toNanoseconds();
        // a loading server's own stop takes two grace periods at most; a pull gets drain_s and resumes next time
        const serve_ms = 2 * @as(i96, s.ctx.grace_ms) + 1000 * @as(i96, s.ctx.drain_s);
        while (true) {
            s.reap();
            if (s.calls.items.len == 0) break;
            const waited_ms = @divTrunc(std.Io.Clock.awake.now(io).toNanoseconds() - started, std.time.ns_per_ms);
            const serving = for (s.calls.items) |c| {
                if (c.tool == .serve) break true;
            } else false;
            if (waited_ms >= (if (serving) serve_ms else 1000 * @as(i96, s.ctx.drain_s))) break;
            io.sleep(.fromMilliseconds(20), .awake) catch {};
        }
        for (s.calls.items) |c| if (c.thread) |t| t.detach(); // left to the process's exit
        s.calls.clearRetainingCapacity();
    }
};

fn capabilities(a: Allocator) !Value {
    const o = try json.newObject(a);
    try o.put(a, "tools", .{ .object = try json.newObject(a) });
    return .{ .object = o };
}

/// 2026-07-28's server/discover: the revisions served, the capabilities and the instructions, cacheable an hour.
fn discover(a: Allocator) !*json.Object {
    const body = try json.newObject(a);
    try body.put(a, "supportedVersions", try rpc.strings(a, &rpc.supported));
    try body.put(a, "capabilities", try capabilities(a));
    try body.put(a, "instructions", .{ .string = instructions });
    try body.put(a, "ttlMs", try json.intValue(a, tools.ttl_ms));
    try body.put(a, "cacheScope", .{ .string = "public" });
    return body;
}

fn eql(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

const Next = union(enum) { line: []const u8, too_long, end, stopped };

/// One line from the client, read as its own task so a stop signal ends the wait, as benchmark's prompts are read.
fn next(io: std.Io, r: *std.Io.Reader) Next {
    const Read = struct {
        r: *std.Io.Reader,
        got: Next = .end,
        done: std.atomic.Value(bool) = .init(false),
        fn go(t: *@This()) void {
            t.got = if (t.r.takeDelimiter('\n')) |line| (if (line) |l| .{ .line = l } else .end) else |e| switch (e) {
                error.StreamTooLong => blk: {
                    _ = t.r.discardDelimiterInclusive('\n') catch {};
                    break :blk .too_long;
                },
                error.ReadFailed => .end,
            };
        }
    };
    var task: Read = .{ .r = r };
    if (comptime posix_only) {
        switch (stream.bounded(io, null, Read.go, &task)) {
            .finished => return task.got,
            .stopped, .deadline => return .stopped,
            .unavailable => {},
        }
    }
    task.go();
    return task.got;
}

/// Serves messages from ``in`` until it ends or a stop signal arrives, then stops what the session started.
pub fn run(ctx: *const Context, in: *std.Io.Reader, out: *std.Io.Writer) !u8 {
    var wire: rpc.Wire = .{ .out = out, .io = ctx.io };
    var s: Session = .init(ctx, &wire);
    defer s.deinit();
    var arena: std.heap.ArenaAllocator = .init(ctx.gpa);
    defer arena.deinit();
    loop: while (true) {
        _ = arena.reset(.retain_capacity);
        const a = arena.allocator();
        switch (next(ctx.io, in)) {
            .line => |line| s.handle(a, line) catch |e| std.debug.print("tensorfold mcp: a message could not be answered ({t})\n", .{e}),
            .too_long => wire.send(rpc.failure(a, .{ .code = rpc.invalid_request, .message = "Invalid Request: the message is longer than 4 MiB" }) catch continue),
            .end, .stopped => break :loop,
        }
    }
    s.finish();
    if (comptime posix_only) if (stop.requested()) |sig| return stop.exitCode(sig);
    return 0;
}

pub fn main(init: std.process.Init, argv: []const []const u8, version: []const u8, engines: *const fn (Allocator) cli.Engines) !u8 {
    const io = init.io;
    if (argv.len == 1 and (eql(argv[0], "--help") or eql(argv[0], "-h"))) {
        try std.Io.File.stdout().writeStreamingAll(io, usage_line);
        return 0;
    }
    if (argv.len > 0) {
        std.debug.print("{s}tensorfold mcp: error: unrecognized arguments: {s}\n", .{ usage_line, argv[0] });
        return 2;
    }
    if (comptime posix_only) {
        stop.install(); // TERM, INT and HUP end the session as stdin's end does, so the servers it started stop too
        const ignore: std.posix.Sigaction = .{ .handler = .{ .handler = std.posix.SIG.IGN }, .mask = std.posix.sigemptyset(), .flags = 0 };
        std.posix.sigaction(.PIPE, &ignore, null); // a client gone mid-reply is a write error, not this process's death
    }
    defer if (comptime posix_only) stop.restore();
    const buf = try init.gpa.alloc(u8, rpc.max_line);
    defer init.gpa.free(buf);
    var in = std.Io.File.stdin().readerStreaming(io, buf);
    var out_buf: [64 << 10]u8 = undefined;
    var out = std.Io.File.stdout().writerStreaming(io, &out_buf);
    const env = init.environ_map;
    const ctx: Context = .{ .io = io, .gpa = init.gpa, .environ = env, .home = env.get("HOME") orelse env.get("USERPROFILE") orelse ".", .version = version, .engines = engines };
    return run(&ctx, &in.interface, &out.interface);
}
