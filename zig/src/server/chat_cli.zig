//! ``tensorfold chat``: a conversation in the terminal with a model it serves itself, or with a server already running.
const std = @import("std");
const builtin = @import("builtin");
const json = @import("json");
const args_mod = @import("chat_cli_args.zig");
const history = @import("chat_cli_history.zig");
const turn = @import("chat_cli_turn.zig");
const child = @import("chat_cli_child.zig");
const stream = @import("benchmark_stream.zig");
const stop = @import("benchmark_stop.zig");
const Allocator = std.mem.Allocator;
const Sig = std.posix.SIG;

const windows = builtin.os.tag == .windows;
/// The serve child waits on POSIX signals, as the benchmark's does, so Windows chats only with a running server.
const windows_child = "starting a server is not supported on Windows yet: run tensorfold serve MODEL, then tensorfold chat --url http://127.0.0.1:8080";
const Child = if (windows) void else child.Child;
/// The longest line a conversation reads, and the most a piped message holds.
const max_line = 1 << 20;
const max_message = 16 << 20;
const banner = "Chatting with {s}. /help lists the commands; /bye or Ctrl-D ends the chat.\n";
const prompt_hint = "(Type /bye or press Ctrl-D to end the chat.)\n";
const cut_note = "The reply stopped at its token limit; --max-tokens raises it.";
const thought_note = "The reply held only reasoning, which is hidden; --show-thinking shows it.";

/// Everything the command touches, so tests run it with captured writers, fixed input and a stand-in server.
pub const Context = struct {
    a: Allocator,
    io: std.Io,
    environ: *const std.process.Environ.Map,
    out: *std.Io.Writer,
    err: *std.Io.Writer,
    input: *std.Io.Reader,
    /// stdin is a terminal: a conversation; otherwise all of stdin is one message.
    interactive: bool,
    /// stdout is a terminal: reasoning dimmed, a hint while it is hidden, and control characters dropped.
    terminal: bool,
    self_exe: ?[]const u8 = null,
    signals: bool = false,
};

pub fn main(init: std.process.Init, argv: []const []const u8) !u8 {
    const io = init.io;
    var out_buf: [16 << 10]u8 = undefined;
    var err_buf: [4 << 10]u8 = undefined;
    var in_buf: [16 << 10]u8 = undefined;
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
        .terminal = std.Io.File.stdout().isTty(io) catch false,
        .signals = true,
    };
    return command(&ctx, argv);
}

pub fn command(ctx: *const Context, argv: []const []const u8) !u8 {
    var problem: []const u8 = "";
    const args = args_mod.parse(ctx.a, argv, &problem) catch |e| switch (e) {
        error.Usage => {
            try ctx.err.print("{s}\ntensorfold chat: error: {s}\n", .{ args_mod.usage_line, problem });
            try ctx.err.flush();
            return 2;
        },
        error.OutOfMemory => |x| return x,
    };
    if (args.help) {
        try args_mod.help(ctx.out);
        try ctx.out.flush();
        return 0;
    }
    if (windows and args.url == null) return fail(ctx, windows_child);
    if (comptime !windows) if (ctx.signals) signals.install();
    defer if (comptime !windows) if (ctx.signals) signals.restore();
    // a piped message is read before any server starts, so an empty one costs no model load
    var message: ?[]const u8 = null;
    if (!ctx.interactive) switch (try piped(ctx)) {
        .text => |text| message = text,
        .code => |code| return code,
    };
    if (args.url) |url| return session(ctx, args, url, args.model, null, message);
    if (comptime windows) return fail(ctx, windows_child) else return served(ctx, args, message);
}

/// The chat with the server it starts: loaded, then the session, then stopped whatever the session's end.
fn served(ctx: *const Context, args: args_mod.Args, message: ?[]const u8) !u8 {
    var problem: []const u8 = "";
    var tail: []const []const u8 = &.{};
    const setup: child.Setup = .{ .a = ctx.a, .io = ctx.io, .environ = ctx.environ, .exe = ctx.self_exe, .log_dir = ctx.environ.get("TMPDIR") orelse "/tmp" };
    const plan = try child.plan(setup, args.model.?, args.serve, args.startup_timeout, &problem) orelse return fail(ctx, problem);
    try say(ctx.err, "Loading {s}; Ctrl-C cancels.", .{args.model.?});
    var own = try child.start(setup, plan, &tail, &problem) orelse {
        if (problem.len == 0) return stop.exitCode(stop.requested() orelse .INT);
        const code = try fail(ctx, problem);
        try lastLines(ctx, tail);
        return code;
    };
    errdefer _ = own.managed.stop();
    const code = try session(ctx, args, own.managed.base(), own.served, &own, message);
    if (own.managed.stop()) |why| {
        const failed = try fail(ctx, why);
        return if (code == 0) failed else code;
    }
    return code;
}

/// An exited child's last log lines after its failure, as the benchmark prints them.
fn lastLines(ctx: *const Context, tail: []const []const u8) !void {
    if (tail.len == 1) try ctx.err.print("Last server log line: {s}\n", .{tail[0]});
    if (tail.len > 1) try ctx.err.writeAll("Last server log lines:\n");
    if (tail.len > 1) for (tail) |line| try ctx.err.print("  {s}\n", .{line});
    try ctx.err.flush();
}

const Session = struct {
    ctx: *const Context,
    /// The chat route.
    url: []const u8,
    settings: turn.Settings,
    show: turn.Show,
    history: history.History,
    own: ?*Child,

    /// One message and its streamed reply, in an arena of its own that lives until the turn is kept.
    fn ask(s: *Session, a: Allocator, message: []const u8) !turn.Ended {
        const request = try turn.body(a, s.settings, &s.history, message);
        return turn.run(a, s.ctx.io, s.url, request, s.ctx.out, s.show);
    }

    /// A finished reply's notes: cut at its token limit, or nothing but reasoning that was hidden.
    fn notes(s: *const Session, f: turn.Finished) !void {
        if (f.cut) try say(s.ctx.err, cut_note, .{});
        if (f.content.len == 0 and f.reasoned and !s.show.thinking) try say(s.ctx.err, thought_note, .{});
    }

    /// A failed turn said; a conversation goes on (null) unless the server it started has exited.
    fn failed(s: *Session, reason: []const u8, connect: bool) !?u8 {
        _ = try fail(s.ctx, reason);
        if (comptime !windows) if (connect) if (s.own) |own| if (try own.managed.exitedTail(s.ctx.a)) |tail| {
            try s.ctx.err.writeAll("The model server has exited.\n");
            try lastLines(s.ctx, tail);
            return 1;
        };
        return if (s.ctx.interactive) null else 1;
    }

    /// A message of the conversation: null to read the next line, else the chat's exit code.
    fn talk(s: *Session, text: []const u8) !?u8 {
        var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
        defer arena.deinit();
        switch (try s.ask(arena.allocator(), text)) {
            .finished => |f| {
                if (f.content.len > 0) try s.history.keep(text, f.content);
                try s.notes(f);
            },
            // a stopped reply keeps what it printed, so the next message can redirect it
            .stopped => |content| {
                const sig = stop.requested() orelse .INT;
                if (sig != .INT) return stop.exitCode(sig);
                stop.reset();
                if (content.len > 0) try s.history.keep(text, content);
            },
            .failed => |f| return s.failed(f.reason, f.connect),
        }
        return null;
    }
};

/// Finds the model and its thinking switch, then holds one exchange (a piped message) or a conversation.
fn session(ctx: *const Context, args: args_mod.Args, base: []const u8, wanted: ?[]const u8, own: ?*Child, message: ?[]const u8) !u8 {
    var problem: []const u8 = "";
    const url = try stream.endpoint(ctx.a, base, true, &problem) orelse return fail(ctx, problem);
    const found = try discover(ctx, url, wanted);
    if (found.stopped) |sig| return stop.exitCode(sig);
    const model = found.model orelse return fail(ctx, found.problem);
    if (args.no_thinking and found.thinking_switch == false)
        return fail(ctx, "--no-thinking: this model's chat template cannot turn thinking off (its /v1/models thinking_switch is false)");
    var s: Session = .{
        .ctx = ctx,
        .url = url,
        .settings = .{ .model = model, .temperature = args.temperature, .max_tokens = args.max_tokens, .no_thinking = args.no_thinking },
        .show = .{ .thinking = args.show_thinking, .terminal = ctx.terminal },
        .history = .{ .a = ctx.a },
        .own = own,
    };
    if (args.system) |text| try s.history.setSystem(text);
    if (message) |text| {
        var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
        defer arena.deinit();
        return switch (try s.ask(arena.allocator(), text)) {
            .finished => |f| blk: {
                try s.notes(f);
                break :blk 0;
            },
            .stopped => stop.exitCode(stop.requested() orelse .INT),
            .failed => |f| (try s.failed(f.reason, f.connect)) orelse 1,
        };
    }
    return converse(&s);
}

/// The conversation: a prompt, a line, then a command or a turn, until /bye, end of input or a stop signal.
fn converse(s: *Session) !u8 {
    const ctx = s.ctx;
    try ctx.out.print(banner, .{s.settings.model});
    while (true) {
        try ctx.out.writeAll("> ");
        try ctx.out.flush();
        var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
        defer arena.deinit();
        const text = switch (readLine(ctx, arena.allocator())) {
            .line => |line| line,
            .eof => {
                try ctx.out.writeAll("\n");
                try ctx.out.flush();
                return 0;
            },
            .stopped => |sig| {
                if (sig != .INT) return stop.exitCode(sig);
                stop.reset();
                try ctx.out.writeAll("\n" ++ prompt_hint);
                continue;
            },
            .too_long => {
                try say(ctx.err, "That line is over 1 MiB and was not sent; pipe long text through stdin instead.", .{});
                continue;
            },
            .failed => return fail(ctx, "could not read the terminal"),
        };
        switch (history.command(text)) {
            .empty => {},
            .bye => return 0,
            .help => try ctx.out.writeAll(history.help_text),
            .clear => {
                s.history.clear();
                try ctx.out.writeAll("Conversation cleared.\n");
            },
            .system => |prompt| {
                try s.history.setSystem(prompt);
                try ctx.out.writeAll(if (prompt.len == 0) "System prompt removed.\n" else "System prompt set.\n");
            },
            .unknown => |word| try ctx.out.print("Unknown command {s}; /help lists the commands.\n", .{word}),
            .say => |said| if (try s.talk(said)) |code| return code,
        }
    }
}

const Found = struct { model: ?[]const u8 = null, thinking_switch: ?bool = null, problem: []const u8 = "", stopped: ?Sig = null };

/// One bounded look at ``/v1/models``: its first model when none was named, and whether thinking can be turned off.
fn discover(ctx: *const Context, chat_url: []const u8, wanted: ?[]const u8) !Found {
    var client: std.http.Client = .{ .allocator = std.heap.smp_allocator, .io = ctx.io };
    defer client.deinit();
    const url = try std.mem.concat(ctx.a, u8, &.{ chat_url[0 .. chat_url.len - "/chat/completions".len], "/models" });
    const got = stream.fetch(ctx.a, ctx.io, &client, .{ .url = url, .limit = 4 << 20, .timeout_s = 10, .agent = turn.user_agent });
    if (got.failed == .stopped) return .{ .stopped = stop.requested() orelse .INT };
    var doc: ?json.Value = null;
    if (got.failed == .none and got.status == 200) switch (try json.parse(ctx.a, got.body)) {
        .ok => |v| doc = v,
        .err => {},
    };
    const name = wanted orelse first(doc) orelse {
        const why = switch (got.failed) {
            .none => if (got.status != 200) try std.fmt.allocPrint(ctx.a, "answered HTTP {d}", .{got.status}) else "listed no models",
            .network => "could not connect",
            .deadline => "gave no answer in 10 s",
            .too_large => "answered with more than 4 MiB",
            .stopped => unreachable,
        };
        return .{ .problem = try std.fmt.allocPrint(ctx.a, "no model to chat with: {s} {s}; name one after the URL", .{ url, why }) };
    };
    return .{ .model = name, .thinking_switch = if (doc) |d| stream.thinkingSwitch(d, name) else null };
}

/// The first ``data[].id`` of a ``/v1/models`` document.
fn first(doc: ?json.Value) ?[]const u8 {
    const data = (doc orelse return null).get("data") orelse return null;
    if (data != .array or data.array.len == 0) return null;
    return data.array[0].strField("id");
}

const Read = union(enum) { line: []const u8, eof, too_long, failed, stopped: Sig };

/// One typed line, read as its own task so a stop signal ends the wait.
const LineTask = struct {
    a: Allocator,
    r: *std.Io.Reader,
    got: Read = .failed,
    done: std.atomic.Value(bool) = .init(false),

    fn go(t: *LineTask) void {
        var line: std.Io.Writer.Allocating = .init(t.a);
        _ = t.r.streamDelimiterLimit(&line.writer, '\n', .limited(max_line)) catch |e| {
            if (e == error.StreamTooLong) _ = t.r.discardDelimiterInclusive('\n') catch 0;
            t.got = if (e == error.StreamTooLong) .too_long else .failed;
            return;
        };
        // the newline is left in the buffer; an empty buffer means the input ended without one
        if (t.r.bufferedLen() > 0) {
            t.r.toss(1);
            t.got = .{ .line = line.written() };
        } else t.got = if (line.written().len == 0) .eof else .{ .line = line.written() };
    }
};

fn readLine(ctx: *const Context, a: Allocator) Read {
    var task: LineTask = .{ .a = a, .r = ctx.input };
    return switch (stream.bounded(ctx.io, null, LineTask.go, &task)) {
        .finished => task.got,
        .stopped, .deadline => .{ .stopped = stop.requested() orelse .INT },
        .unavailable => .failed,
    };
}

const Piped = union(enum) { text: []const u8, code: u8 };

/// All of stdin, read as its own task so a stop signal ends the wait.
const AllTask = struct {
    a: Allocator,
    r: *std.Io.Reader,
    text: ?[]u8 = null,
    too_long: bool = false,
    done: std.atomic.Value(bool) = .init(false),

    fn go(t: *AllTask) void {
        t.text = t.r.allocRemaining(t.a, .limited(max_message)) catch |e| {
            t.too_long = e == error.StreamTooLong;
            return;
        };
    }
};

/// The piped message, trimmed, or the exit code once its problem is said.
fn piped(ctx: *const Context) !Piped {
    var task: AllTask = .{ .a = ctx.a, .r = ctx.input };
    switch (stream.bounded(ctx.io, null, AllTask.go, &task)) {
        .finished => {},
        .stopped, .deadline => return .{ .code = stop.exitCode(stop.requested() orelse .INT) },
        .unavailable => return .{ .code = try fail(ctx, "could not read stdin") },
    }
    if (task.too_long) return .{ .code = try fail(ctx, "stdin holds more than 16 MiB; a message must be smaller") };
    const text = std.mem.trim(u8, task.text orelse return .{ .code = try fail(ctx, "could not read stdin") }, " \t\r\n");
    if (text.len == 0) return .{ .code = try fail(ctx, "stdin held no message to send") };
    return .{ .text = text };
}

/// Ctrl-C, SIGTERM and SIGHUP to the stop flag the waits read; a closed stdout fails a write instead of the process.
const signals = struct {
    var pipe_saved: std.posix.Sigaction = undefined;

    fn install() void {
        stop.install();
        const ignore: std.posix.Sigaction = .{ .handler = .{ .handler = std.posix.SIG.IGN }, .mask = std.posix.sigemptyset(), .flags = 0 };
        std.posix.sigaction(.PIPE, &ignore, &pipe_saved);
    }

    fn restore() void {
        std.posix.sigaction(.PIPE, &pipe_saved, null);
        stop.restore();
    }
};

fn say(w: *std.Io.Writer, comptime fmt: []const u8, args: anytype) !void {
    try w.print(fmt ++ "\n", args);
    try w.flush();
}

fn fail(ctx: *const Context, message: []const u8) !u8 {
    try ctx.err.print("tensorfold chat: {s}\n", .{message});
    try ctx.err.flush();
    return 1;
}
