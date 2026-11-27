//! `tensorfold start`: pick a fitting checkpoint, consent to its pull, serve it and prove one reply.
const std = @import("std");
const builtin = @import("builtin");
const registry = @import("checkpoint_cli").start;
const machine = @import("sys").machine;
const child = @import("chat_cli_child.zig");
const credential = @import("sys").credential;
const turn = @import("chat_cli_turn.zig");
const history = @import("chat_cli_history.zig");
const stream = @import("benchmark_stream.zig");
const stop = @import("benchmark_stop.zig");
const chat = @import("chat_cli.zig");
const Allocator = std.mem.Allocator;

const windows = builtin.os.tag == .windows;
const Child = if (windows) void else child.Child;
pub const usage = "usage: tensorfold start [--yes] [--check] [--startup-timeout SECONDS]\n";

/// Everything the command touches; tests supply a machine, registry, cache, child executables and inert effects.
pub const Context = struct {
    a: Allocator,
    io: std.Io,
    environ: *const std.process.Environ.Map,
    out: *std.Io.Writer,
    err: *std.Io.Writer,
    input: *std.Io.Reader,
    interactive: bool,
    /// The invoking low-level executable, or a test's deliberately different current executable.
    self_exe: ?[]const u8 = null,
    /// The native serving executable, explicitly injected by CUDA's production route.
    serve_exe: ?[]const u8 = null,
    signals: bool = false,
    given: ?machine.Machine = null,
    registry: ?[]const registry.Entry = null,
    override: ?[]const u8 = null,
    pull: ?Pull = null,
};

/// A pull effect for tests; its output goes through the same privacy sink as production pulls.
pub const Pull = struct {
    ctx: *anyopaque,
    run: *const fn (*anyopaque, []const u8) anyerror!u8,
};

const Args = struct { yes: bool = false, check: bool = false, startup_timeout: ?f64 = null, help: bool = false };

fn parse(args: []const []const u8) ?Args {
    var out: Args = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--yes")) {
            out.yes = true;
        } else if (std.mem.eql(u8, arg, "--check")) {
            out.check = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            out.help = true;
        } else if (std.mem.eql(u8, arg, "--startup-timeout")) {
            i += 1;
            if (i == args.len) return null;
            const seconds = std.fmt.parseFloat(f64, args[i]) catch return null;
            if (!(seconds > 0 and seconds <= 3600)) return null;
            out.startup_timeout = seconds;
        } else return null;
    }
    return out;
}

pub fn main(init: std.process.Init, argv: []const []const u8) !u8 {
    return mainWithServeExe(init, argv, null);
}

/// CUDA's low-level `tensorfold` routes start to its native serving sibling, never back to itself.
pub fn mainCuda(init: std.process.Init, argv: []const []const u8) !u8 {
    const current = try std.process.executablePathAlloc(init.io, init.arena.allocator());
    return mainWithServeExe(init, argv, try nativeSibling(init.arena.allocator(), current));
}

pub fn nativeSibling(a: Allocator, current: []const u8) ![]const u8 {
    const name = std.fs.path.basename(current);
    const sibling = if (std.ascii.eqlIgnoreCase(name, "tensorfold"))
        "tensorfold-native"
    else if (std.ascii.eqlIgnoreCase(name, "tensorfold.exe"))
        "tensorfold-native.exe"
    else
        return current;
    const directory = std.fs.path.dirname(current) orelse ".";
    if (std.ascii.eqlIgnoreCase(std.fs.path.basename(directory), "bin")) {
        const prefix = std.fs.path.dirname(directory) orelse ".";
        return std.fs.path.join(a, &.{ prefix, "native", "bin", sibling });
    }
    return std.fs.path.join(a, &.{ directory, sibling });
}

fn mainWithServeExe(init: std.process.Init, argv: []const []const u8, serve_exe: ?[]const u8) !u8 {
    const io = init.io;
    var out_buf: [16 << 10]u8 = undefined;
    var err_buf: [4 << 10]u8 = undefined;
    var in_buf: [4 << 10]u8 = undefined;
    var out = std.Io.File.stdout().writerStreaming(io, &out_buf);
    var err = std.Io.File.stderr().writerStreaming(io, &err_buf);
    var in = std.Io.File.stdin().readerStreaming(io, &in_buf);
    defer out.interface.flush() catch {};
    defer err.interface.flush() catch {};
    const ctx: Context = .{
        .a = init.arena.allocator(),
        .io = io,
        .environ = init.environ_map,
        .out = &out.interface,
        .err = &err.interface,
        .input = &in.interface,
        .interactive = std.Io.File.stdin().isTty(io) catch false,
        .serve_exe = serve_exe,
        .signals = true,
    };
    return command(&ctx, argv);
}

pub fn command(ctx: *const Context, argv: []const []const u8) !u8 {
    const args = parse(argv) orelse {
        try ctx.err.writeAll(usage);
        try ctx.err.flush();
        return 2;
    };
    if (args.help) {
        try ctx.out.writeAll(usage ++
            "Pick the largest catalog model that fits this machine, ask before any download, serve on loopback, and send one reply.\n" ++
            "--yes consents to the registered download; without it a non-interactive run downloads nothing.\n" ++
            "--check stops the server after a successful first reply and uses a private temporary key; persistent mode requires TENSORFOLD_API_KEY.\n");
        try ctx.out.flush();
        return 0;
    }
    if (comptime windows) return fail(ctx, 2, "starting a server is not supported on Windows yet: use tensorfold serve MODEL in one console and tensorfold chat --url URL MODEL in another", .{}) else return run(ctx, args);
}

fn run(ctx: *const Context, args: Args) !u8 {
    const caller_key = if (args.check) null else chat.managedKey(ctx.environ) catch return fail(ctx, 2, chat.managed_bad_key, .{});
    if (!args.check and caller_key == null)
        return fail(ctx, 2, "persistent start requires a nonempty TENSORFOLD_API_KEY entry in the environment, or use --check for a private temporary key", .{});
    const mc = ctx.given orelse machine.detect(ctx.a, ctx.io);
    const choice = (try registry.choose(ctx.a, ctx.io, ctx.environ, ctx.override, ctx.registry orelse &registry.entries, &mc)) orelse
        return fail(ctx, 1, "no catalog model that this machine can start automatically; tensorfold models --all shows the reasons and manual recipes", .{});
    const e = choice.entry;
    try say(ctx.out, "Selected {s}, {s}: {s}", .{ e.title, e.format, e.repo });
    if (choice.cached) {
        try say(ctx.out, "Checkpoint and any drafter are already cached; no download is needed", .{});
    } else {
        try say(ctx.out, "Download {s}: {d:.1} GiB ({d} bytes) of model weights, including any drafter", .{ choice.model, gibs(choice.download_bytes), choice.download_bytes });
        if (choice.drafter) |d| try say(ctx.out, "Drafter download: {s}", .{d});
        if (!args.yes) {
            if (!ctx.interactive) return fail(ctx, 2, "stdin is not a terminal to ask on; pass --yes to consent in a non-interactive run; nothing was downloaded", .{});
            try ctx.err.writeAll("Download these model weights? [y/N] ");
            try ctx.err.flush();
            const answer = std.mem.trim(u8, (ctx.input.takeDelimiter('\n') catch null) orelse "", " \t\r");
            if (!std.ascii.eqlIgnoreCase(answer, "y") and !std.ascii.eqlIgnoreCase(answer, "yes")) {
                try say(ctx.out, "download declined; nothing was downloaded", .{});
                return 1;
            }
        }
        if (try pull(ctx, choice.model) != 0) return fail(ctx, 1, "pull {s} failed; fix the reported error and run start again", .{choice.model});
        if (choice.drafter) |d| if (try pull(ctx, d) != 0) return fail(ctx, 1, "pull {s} failed; fix the reported error and run start again", .{d});
    }
    if (comptime !windows) if (ctx.signals) stop.install();
    defer if (comptime !windows) if (ctx.signals) stop.restore();
    return serve(ctx, args, choice, caller_key);
}

fn pull(ctx: *const Context, spec: []const u8) !u8 {
    try say(ctx.out, "Pulling {s}", .{spec});
    var buffer: [4 << 10]u8 = undefined;
    const discard: std.Io.Writer.Discarding = .init(&buffer);
    const saved = ctx.out.*;
    ctx.out.* = discard.writer;
    defer ctx.out.* = saved;
    if (ctx.pull) |p| return p.run(p.ctx, spec);
    return registry.pull.run(ctx.a, ctx.io, ctx.out, ctx.err, ctx.environ, ctx.override, spec);
}

fn serve(ctx: *const Context, args: Args, choice: registry.Choice, caller_key: ?[]const u8) !u8 {
    var problem: []const u8 = "";
    var tail: []const []const u8 = &.{};
    const serving_exe = ctx.serve_exe orelse if (ctx.self_exe) |current| try nativeSibling(ctx.a, current) else null;
    var setup: child.Setup = .{ .a = ctx.a, .io = ctx.io, .environ = ctx.environ, .exe = serving_exe, .key = caller_key, .environment_key = if (args.check) null else ctx.environ.get("TENSORFOLD_API_KEY"), .log_dir = ctx.environ.get("TMPDIR") orelse "/tmp" };
    var extra: std.ArrayList([]const u8) = .empty;
    try extra.appendSlice(ctx.a, &.{ "--name", std.fs.path.basename(choice.entry.repo) });
    if (choice.drafter_dir) |d| try extra.appendSlice(ctx.a, &.{ "--drafter", d });
    const plan = try child.plan(setup, choice.model_dir, extra.items, args.startup_timeout, &problem) orelse return fail(ctx, 1, "{s}", .{problem});
    const credentials = try turn.Credentials.init(ctx.a, try chat.callerCredentials(ctx.a, ctx.environ, plan.key));
    setup.secrets = credentials.forms;
    try say(ctx.err, "Loading {s}; Ctrl-C cancels", .{choice.entry.repo});
    var own = try child.start(setup, plan, &tail, &problem) orelse {
        if (problem.len == 0) return stop.exitCode(stop.requested() orelse .INT);
        const code = try fail(ctx, 1, "{s}", .{(try credentials.safe(ctx.a, problem)) orelse credential.withheld});
        try lastLines(ctx, tail);
        return code;
    };
    errdefer _ = own.managed.stop();
    const code = try firstReply(ctx, &own, credentials);
    if (code != 0) {
        if (own.managed.stop()) |why| try say(ctx.err, "tensorfold start: {s}", .{why});
        return code;
    }
    if (args.check) {
        if (own.managed.stop()) |why| return fail(ctx, 1, "{s}", .{why});
        try say(ctx.out, "first-reply check passed; server stopped", .{});
        return 0;
    }
    try say(ctx.out, "Serving {s} at {s}/v1; Ctrl-C stops the server", .{ own.served, own.managed.base() });
    return wait(ctx, &own);
}

/// One deterministic short request proves answer content, not only a listening socket.
fn firstReply(ctx: *const Context, own: *Child, credentials: turn.Credentials) !u8 {
    var problem: []const u8 = "";
    const url = try stream.endpoint(ctx.a, own.managed.base(), true, &problem) orelse return fail(ctx, 1, "{s}", .{problem});
    var h: history.History = .{ .a = ctx.a };
    const request = try turn.body(ctx.a, .{ .model = own.served, .temperature = 0, .max_tokens = 32 }, &h, "Reply with a short confirmation that the server is ready.");
    try say(ctx.out, "First reply:", .{});
    switch (try turn.runManagedStart(ctx.a, ctx.io, url, own.key, credentials, request, ctx.out, .{ .terminal = true }, false)) {
        .finished => |f| {
            if (f.content.len == 0) return fail(ctx, 1, "first reply failed: the model returned no answer content", .{});
            return 0;
        },
        .stopped => return stop.exitCode(stop.requested() orelse .INT),
        .failed => |f| {
            const reason = try std.fmt.allocPrint(ctx.a, "first reply failed: {s}", .{f.reason});
            return fail(ctx, 1, "{s}", .{try credentials.check(ctx.a, reason)});
        },
    }
}

/// Keep the checked server in the foreground until a signal or its own exit.
fn wait(ctx: *const Context, own: *Child) !u8 {
    while (true) {
        if (stop.requested()) |sig| {
            if (own.managed.stop()) |why| try say(ctx.err, "tensorfold start: {s}", .{why});
            return stop.exitCode(sig);
        }
        if (try own.managed.exitedTail(ctx.a)) |tail| {
            const code = try fail(ctx, 1, "the model server exited after its successful first reply", .{});
            try lastLines(ctx, tail);
            return code;
        }
        ctx.io.sleep(.fromMilliseconds(200), .awake) catch {};
    }
}

fn lastLines(ctx: *const Context, tail: []const []const u8) !void {
    if (tail.len == 1) try ctx.err.print("Last server log line: {s}\n", .{tail[0]});
    if (tail.len > 1) try ctx.err.writeAll("Last server log lines:\n");
    if (tail.len > 1) for (tail) |line| try ctx.err.print("  {s}\n", .{line});
    try ctx.err.flush();
}

fn gibs(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(machine.gib));
}

fn say(w: *std.Io.Writer, comptime fmt: []const u8, args: anytype) !void {
    try w.print(fmt ++ "\n", args);
    try w.flush();
}

fn fail(ctx: *const Context, code: u8, comptime fmt: []const u8, args: anytype) !u8 {
    try ctx.err.print("tensorfold start: " ++ fmt ++ "\n", args);
    try ctx.err.flush();
    return code;
}

test "start CLI: Windows refusal compiles through the complete command context" {
    if (comptime !windows) return;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var env: std.process.Environ.Map = .init(a);
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    var input: std.Io.Reader = .fixed("");
    const ctx: Context = .{ .a = a, .io = std.testing.io, .environ = &env, .out = &out.writer, .err = &err.writer, .input = &input, .interactive = false };
    try std.testing.expectEqual(@as(u8, 2), try command(&ctx, &.{}));
    try std.testing.expect(std.mem.indexOf(u8, err.written(), "not supported on Windows") != null);
}
