//! `tensorfold launch <agent> [args...]`: a coding agent against a local server, set up in its environment only.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Environ = std.process.Environ.Map;

pub const Agent = struct {
    name: []const u8,
    command: []const u8,
    /// The variables each tool documents for its API base URL.
    base_vars: []const []const u8,
    /// The variables each tool uses for the API key.
    key_vars: []const []const u8,
    /// The tool appends /v1/... itself, so it takes the server's root (Anthropic clients: docs/api.md).
    root: bool = false,
};

/// Agents whose base URL and key come from their environment. Names are short so they are easy to type.
pub const agents = [_]Agent{
    .{ .name = "claude", .command = "claude", .base_vars = &.{"ANTHROPIC_BASE_URL"}, .key_vars = &.{"ANTHROPIC_API_KEY"}, .root = true },
    .{ .name = "codex", .command = "codex", .base_vars = &.{"OPENAI_BASE_URL"}, .key_vars = &.{"OPENAI_API_KEY"} },
    .{ .name = "opencode", .command = "opencode", .base_vars = &.{"OPENAI_BASE_URL"}, .key_vars = &.{"OPENAI_API_KEY"} },
    .{ .name = "hermes", .command = "hermes", .base_vars = &.{"TENSORFOLD_BASE_URL"}, .key_vars = &.{"TENSORFOLD_API_KEY"} },
};

pub fn find(name: []const u8) ?Agent {
    for (agents) |agent| if (std.mem.eql(u8, agent.name, name)) return agent;
    return null;
}

pub fn names(a: Allocator) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (agents, 0..) |agent, i| {
        if (i > 0) try out.appendSlice(a, ", ");
        try out.appendSlice(a, agent.name);
    }
    return out.items;
}

/// The terminal, probe and process boundaries are injectable for host-only command tests.
pub const Context = struct {
    a: Allocator,
    io: std.Io,
    environ: *const Environ,
    out: *std.Io.Writer,
    test_ctx: *anyopaque = undefined,
    hooks: Hooks = .{},
};

pub const Hooks = struct {
    /// The model id a reachable server answers with, or null when it does not answer.
    probe: *const fn (*const Context, []const u8) anyerror!?[]const u8 = probeServer,
    /// Runs the child and returns its exit code.
    spawn: *const fn (*const Context, []const []const u8, *const Environ) anyerror!u8 = spawnAndWait,
};

/// The URL a child is pointed at; tests set this to a loopback port.
pub fn baseUrl(environ: *const Environ) []const u8 {
    return environ.get("TENSORFOLD_LAUNCH_URL") orelse "http://127.0.0.1:8080/v1";
}

/// BASE without a trailing /v1 (and without a trailing slash).
pub fn rootUrl(base: []const u8) []const u8 {
    const b = std.mem.trimEnd(u8, base, "/");
    return if (std.mem.endsWith(u8, b, "/v1")) b[0 .. b.len - 3] else b;
}

/// The key a child sends: the first of TENSORFOLD_API_KEY's comma-separated keys, or "local" when it is unset.
pub fn keyFor(environ: *const Environ) []const u8 {
    const keys = environ.get("TENSORFOLD_API_KEY") orelse return "local";
    var it = std.mem.splitScalar(u8, keys, ',');
    const first = std.mem.trim(u8, it.first(), " ");
    return if (first.len == 0) "local" else first;
}

pub fn command(ctx: *const Context, args: []const []const u8) !u8 {
    if (args.len == 0) {
        try ctx.out.print("usage: tensorfold launch <agent> [args...]\nagents: {s}\n", .{try names(ctx.a)});
        return 2;
    }
    const agent = find(args[0]) orelse {
        try ctx.out.print("unknown agent '{s}'; known agents: {s}\n", .{ args[0], try names(ctx.a) });
        return 2;
    };
    const base = baseUrl(ctx.environ);
    const model = try ctx.hooks.probe(ctx, base) orelse {
        try ctx.out.print("no TensorFold server answered at {s}; start one before launching {s}\n", .{ base, agent.name });
        return 1;
    };
    try ctx.out.print("launching {s} against {s} with model {s}\n", .{ agent.name, base, model });
    try ctx.out.flush(); // before the agent takes the terminal

    var env = try ctx.environ.clone(ctx.a);
    defer env.deinit();
    const url = if (agent.root) rootUrl(base) else base;
    // a server started with keys wants one of them: TENSORFOLD_API_KEY when set, else the keyless placeholder
    const key = keyFor(ctx.environ);
    for (agent.base_vars) |name| try env.put(name, url);
    for (agent.key_vars) |name| try env.put(name, key);

    const argv = try ctx.a.alloc([]const u8, args.len);
    argv[0] = agent.command;
    @memcpy(argv[1..], args[1..]);
    return try ctx.hooks.spawn(ctx, argv, &env);
}

/// Asks a local server which model it serves; null when it cannot be reached.
fn probeServer(ctx: *const Context, base: []const u8) !?[]const u8 {
    var client: std.http.Client = .{ .allocator = ctx.a, .io = ctx.io };
    defer client.deinit();
    const url = try std.fmt.allocPrint(ctx.a, "{s}/models", .{base});
    var body: std.Io.Writer.Allocating = .init(ctx.a);
    defer body.deinit();
    const result = client.fetch(.{ .location = .{ .url = url }, .response_writer = &body.writer }) catch return null;
    if (result.status != .ok) return null;
    var parsed = std.json.parseFromSlice(std.json.Value, ctx.a, body.written(), .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const data = parsed.value.object.get("data") orelse return null;
    if (data != .array or data.array.items.len == 0) return null;
    if (data.array.items[0] != .object) return null;
    const id = data.array.items[0].object.get("id") orelse return null;
    if (id != .string) return null;
    return try ctx.a.dupe(u8, id.string); // the parsed tree is freed on return
}

pub fn spawnAndWait(ctx: *const Context, argv: []const []const u8, env: *const Environ) !u8 {
    var child = try std.process.spawn(ctx.io, .{ .argv = argv, .environ_map = env });
    defer child.kill(ctx.io);
    const term = try child.wait(ctx.io);
    return switch (term) {
        .exited => |code| @intCast(@min(code, 255)),
        .signal => |sig| 128 + @as(u8, @intCast(@min(@backingInt(sig), 127))),
        else => 1,
    };
}

const Capture = struct {
    argv: []const []const u8 = &.{},
    env: Environ = undefined,
    calls: usize = 0,

    /// Copies what the child would get: command() frees its environment once the child returns.
    fn spawn(ctx: *const Context, argv: []const []const u8, env: *const Environ) anyerror!u8 {
        const c: *Capture = @ptrCast(@alignCast(ctx.test_ctx));
        c.argv = try ctx.a.dupe([]const u8, argv);
        c.env = try env.clone(ctx.a);
        c.calls += 1;
        return 0;
    }
    fn noServer(_: *const Context, _: []const u8) anyerror!?[]const u8 {
        return null;
    }
    fn answers(_: *const Context, _: []const u8) anyerror!?[]const u8 {
        return "local-model";
    }
};

fn harness(a: Allocator, environ: *const Environ, capture: *Capture, out: *std.Io.Writer) Context {
    return .{
        .a = a,
        .io = std.testing.io,
        .environ = environ,
        .out = out,
        .test_ctx = capture,
        .hooks = .{ .probe = Capture.answers, .spawn = Capture.spawn },
    };
}

test "launch sets only each known agent's documented variables and passes its arguments through" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var environ: Environ = .{ .array_hash_map = .empty, .allocator = a };
    defer environ.deinit();
    try environ.put("HOME", "test-home");
    try environ.put("PATH", "/usr/bin");
    for (agents) |agent| {
        var capture: Capture = .{};
        var out: std.Io.Writer.Allocating = .init(a);
        const ctx = harness(a, &environ, &capture, &out.writer);
        try std.testing.expectEqual(@as(u8, 0), try command(&ctx, &.{ agent.name, "--flag", "value" }));
        try std.testing.expectEqual(@as(usize, 1), capture.calls);
        try std.testing.expectEqual(@as(usize, 3), capture.argv.len);
        try std.testing.expectEqualStrings(agent.command, capture.argv[0]);
        try std.testing.expectEqualStrings("--flag", capture.argv[1]);
        try std.testing.expectEqualStrings("value", capture.argv[2]);
        const want = if (agent.root) "http://127.0.0.1:8080" else "http://127.0.0.1:8080/v1";
        for (agent.base_vars) |key| try std.testing.expectEqualStrings(want, capture.env.get(key).?);
        for (agent.key_vars) |key| try std.testing.expectEqualStrings("local", capture.env.get(key).?);
        try std.testing.expectEqualStrings("test-home", capture.env.get("HOME").?);
        try std.testing.expect(std.mem.indexOf(u8, out.written(), "local-model") != null);
    }
}

test "launch refuses an unreachable server before the agent starts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var environ: Environ = .{ .array_hash_map = .empty, .allocator = a };
    defer environ.deinit();
    var capture: Capture = .{};
    var out: std.Io.Writer.Allocating = .init(a);
    var ctx = harness(a, &environ, &capture, &out.writer);
    ctx.hooks.probe = Capture.noServer;
    try std.testing.expectEqual(@as(u8, 1), try command(&ctx, &.{"claude"}));
    try std.testing.expectEqual(@as(usize, 0), capture.calls);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "no TensorFold server answered") != null);
}

test "launch refuses an unknown agent and names the known ones" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var environ: Environ = .{ .array_hash_map = .empty, .allocator = a };
    defer environ.deinit();
    var capture: Capture = .{};
    var out: std.Io.Writer.Allocating = .init(a);
    const ctx = harness(a, &environ, &capture, &out.writer);
    try std.testing.expectEqual(@as(u8, 2), try command(&ctx, &.{"missing"}));
    try std.testing.expectEqual(@as(usize, 0), capture.calls);
    const text = out.written();
    try std.testing.expect(std.mem.indexOf(u8, text, "unknown agent 'missing'") != null);
    for (agents) |agent| try std.testing.expect(std.mem.indexOf(u8, text, agent.name) != null);
}

test "launch hands the agent the server's key when TENSORFOLD_API_KEY is set" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var environ: Environ = .{ .array_hash_map = .empty, .allocator = a };
    defer environ.deinit();
    try environ.put("TENSORFOLD_API_KEY", "k-one, k-two");
    try environ.put("TENSORFOLD_LAUNCH_URL", "http://127.0.0.1:9000/v1/");
    var capture: Capture = .{};
    var out: std.Io.Writer.Allocating = .init(a);
    const ctx = harness(a, &environ, &capture, &out.writer);
    try std.testing.expectEqual(@as(u8, 0), try command(&ctx, &.{"claude"}));
    try std.testing.expectEqualStrings("k-one", capture.env.get("ANTHROPIC_API_KEY").?);
    try std.testing.expectEqualStrings("http://127.0.0.1:9000", capture.env.get("ANTHROPIC_BASE_URL").?);
    try std.testing.expectEqualStrings("http://127.0.0.1:9000", rootUrl("http://127.0.0.1:9000"));
}
