//! ``tensorfold chat`` flags in one pass, the serve flags after ``--``, the usage line and the help.
const std = @import("std");
const splitUrl = @import("benchmark_stream.zig").splitUrl;
const Allocator = std.mem.Allocator;

pub const Args = struct {
    help: bool = false,
    /// The checkpoint to serve, or the model to ask a ``--url`` server for.
    model: ?[]const u8 = null,
    url: ?[]const u8 = null,
    system: ?[]const u8 = null,
    temperature: ?f64 = null,
    max_tokens: ?i64 = null,
    show_thinking: bool = false,
    no_thinking: bool = false,
    startup_timeout: ?f64 = null,
    /// What follows ``--``: flags for the started ``tensorfold serve``.
    serve: []const []const u8 = &.{},
};

pub const usage_line = "usage: tensorfold chat [-h] [--url URL] [--system TEXT] [--temperature T] [--max-tokens N]" ++
    " [--show-thinking | --no-thinking] [--startup-timeout SECONDS] [MODEL] [-- SERVE_FLAGS...]";

const Flag = struct { name: []const u8, value: bool = true, help: []const u8 };

pub const flags = [_]Flag{
    .{ .name = "--url", .help = "chat with a server already running at this http(s) address instead of starting one" },
    .{ .name = "--system", .help = "the system prompt (/system changes it in the chat)" },
    .{ .name = "--temperature", .help = "sampling temperature, 0 for greedy (default: the server's)" },
    .{ .name = "--max-tokens", .help = "the longest reply in tokens (default: the server's)" },
    .{ .name = "--show-thinking", .value = false, .help = "show the model's reasoning, dimmed, before its answer" },
    .{ .name = "--no-thinking", .value = false, .help = "ask the server to skip thinking (chat_template_kwargs.enable_thinking false)" },
    .{ .name = "--startup-timeout", .help = "seconds the started server gets to load (default: 600, plus 6 per GB of weights past 20 GB, at most 3600)" },
    .{ .name = "--help", .value = false, .help = "show this help message and exit" },
};

fn usage(a: Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) error{ Usage, OutOfMemory } {
    problem.* = try std.fmt.allocPrint(a, fmt, args);
    return error.Usage;
}

fn find(name: []const u8) ?Flag {
    for (flags) |f| if (std.mem.eql(u8, f.name, name)) return f;
    return null;
}

/// The arguments after ``chat``; ``Usage`` names the problem, and the caller prints the usage line and exits 2.
pub fn parse(a: Allocator, argv: []const []const u8, problem: *[]const u8) error{ Usage, OutOfMemory }!Args {
    var out: Args = .{};
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const token = argv[i];
        if (std.mem.eql(u8, token, "--")) {
            out.serve = argv[i + 1 ..];
            break;
        }
        if (std.mem.eql(u8, token, "-h")) return .{ .help = true };
        if (!std.mem.startsWith(u8, token, "--")) {
            if (out.model != null) return usage(a, problem, "unrecognized arguments: {s}", .{token});
            out.model = token;
            continue;
        }
        const eq = std.mem.indexOfScalar(u8, token, '=');
        const name = token[0 .. eq orelse token.len];
        const flag = find(name) orelse return usage(a, problem, "unrecognized arguments: {s}", .{token});
        var value: []const u8 = "";
        if (flag.value) {
            if (eq) |e| value = token[e + 1 ..] else {
                i += 1;
                if (i >= argv.len or std.mem.startsWith(u8, argv[i], "--")) return usage(a, problem, "argument {s}: expected one argument", .{name});
                value = argv[i];
            }
        } else if (eq) |e| return usage(a, problem, "argument {s}: ignored explicit argument '{s}'", .{ name, token[e + 1 ..] });
        if (is(name, "--help")) return .{ .help = true };
        if (is(name, "--url")) out.url = value;
        if (is(name, "--system")) out.system = value;
        if (is(name, "--show-thinking")) out.show_thinking = true;
        if (is(name, "--no-thinking")) out.no_thinking = true;
        if (is(name, "--temperature")) {
            const t = std.fmt.parseFloat(f64, value) catch null;
            if (t == null or !std.math.isFinite(t.?) or t.? < 0) return usage(a, problem, "argument --temperature: expected a number, 0 (greedy) or more: '{s}'", .{value});
            out.temperature = t;
        }
        if (is(name, "--max-tokens")) {
            const n = std.fmt.parseInt(i64, value, 10) catch 0;
            if (n < 1) return usage(a, problem, "argument --max-tokens: expected a positive token count: '{s}'", .{value});
            out.max_tokens = n;
        }
        if (is(name, "--startup-timeout")) {
            const s = std.fmt.parseFloat(f64, value) catch null;
            if (s == null or !std.math.isFinite(s.?) or s.? <= 0 or s.? > 3600) return usage(a, problem, "argument --startup-timeout: expected seconds above 0 and at most 3600: '{s}'", .{value});
            out.startup_timeout = s;
        }
    }
    try checks(a, out, problem);
    return out;
}

fn is(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

/// The combinations a chat refuses, the first that fails named.
fn checks(a: Allocator, out: Args, problem: *[]const u8) error{ Usage, OutOfMemory }!void {
    if (out.show_thinking and out.no_thinking) return usage(a, problem, "argument --show-thinking: not allowed with argument --no-thinking", .{});
    const url = out.url orelse {
        if (out.model == null) return usage(a, problem, "the following arguments are required: MODEL (a model to serve), or --url URL", .{});
        return;
    };
    if (splitUrl(url) == null) return usage(a, problem, "argument --url: expected an http or https URL without credentials, query or fragment: '{s}'", .{url});
    if (out.serve.len > 0) return usage(a, problem, "flags after -- go to the server tensorfold chat starts; --url names one already running", .{});
    if (out.startup_timeout != null) return usage(a, problem, "argument --startup-timeout: applies only to the server tensorfold chat starts, not to --url", .{});
}

/// ``-h``: the usage line, what the command does and one line per flag.
pub fn help(w: *std.Io.Writer) !void {
    try w.print("{s}\n\n", .{usage_line});
    try w.writeAll(
        \\Chat with a model in the terminal. tensorfold chat MODEL starts tensorfold serve MODEL on a free loopback port,
        \\waits until it is ready, and stops it when the chat ends; --url chats with a server that is already running.
        \\When stdin is not a terminal, all of stdin is one message and only the reply is written to stdout.
        \\
        \\positional arguments:
        \\  MODEL               a cached repo ID or checkpoint directory to serve, or with --url the model to ask for
        \\                      (default with --url: the first in its /v1/models)
        \\  SERVE_FLAGS         after --, flags for the started tensorfold serve, such as --parallel 1
        \\
        \\options:
        \\
    );
    for (flags) |f| try w.print("  {s:<19} {s}\n", .{ f.name, f.help });
    try w.writeAll("\nIn the chat, /help lists the commands: /bye (or Ctrl-D) ends it, /clear, /system TEXT. Ctrl-C stops a reply.\n");
}

test "chat CLI: flags, the serve flags after --, and the model or --url a chat needs" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const full = try parse(a, &.{ "example-org/model", "--system", "Be brief.", "--temperature=0.5", "--max-tokens", "64", "--show-thinking", "--", "--parallel", "1", "--url" }, &problem);
    try std.testing.expectEqualStrings("example-org/model", full.model.?);
    try std.testing.expectEqualStrings("Be brief.", full.system.?);
    try std.testing.expect(full.temperature.? == 0.5 and full.max_tokens.? == 64 and full.show_thinking and !full.no_thinking);
    try std.testing.expectEqual(@as(usize, 3), full.serve.len);
    try std.testing.expectEqualStrings("--url", full.serve[2]);
    const attached = try parse(a, &.{ "--url", "http://127.0.0.1:8080/v1", "--no-thinking" }, &problem);
    try std.testing.expect(attached.model == null and attached.no_thinking and attached.serve.len == 0);
    try std.testing.expect((try parse(a, &.{ "m", "-h" }, &problem)).help);
    try std.testing.expect((try parse(a, &.{"--help"}, &problem)).help);
    try std.testing.expectEqual(@as(?f64, 30), (try parse(a, &.{ "m", "--startup-timeout", "30" }, &problem)).startup_timeout);
}

test "chat CLI: every refused argument is a usage error that names it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const cases = [_]struct { []const []const u8, []const u8 }{
        .{ &.{}, "the following arguments are required: MODEL (a model to serve), or --url URL" },
        .{ &.{ "a", "b" }, "unrecognized arguments: b" },
        .{ &.{ "m", "--bogus" }, "unrecognized arguments: --bogus" },
        .{ &.{ "m", "--system" }, "argument --system: expected one argument" },
        .{ &.{ "m", "--system", "--no-thinking" }, "argument --system: expected one argument" },
        .{ &.{ "m", "--no-thinking=1" }, "argument --no-thinking: ignored explicit argument '1'" },
        .{ &.{ "m", "--temperature", "-1" }, "argument --temperature: expected a number, 0 (greedy) or more: '-1'" },
        .{ &.{ "m", "--temperature", "inf" }, "argument --temperature: expected a number, 0 (greedy) or more: 'inf'" },
        .{ &.{ "m", "--temperature", "warm" }, "argument --temperature: expected a number, 0 (greedy) or more: 'warm'" },
        .{ &.{ "m", "--max-tokens", "0" }, "argument --max-tokens: expected a positive token count: '0'" },
        .{ &.{ "m", "--max-tokens", "1.5" }, "argument --max-tokens: expected a positive token count: '1.5'" },
        .{ &.{ "m", "--startup-timeout", "3601" }, "argument --startup-timeout: expected seconds above 0 and at most 3600: '3601'" },
        .{ &.{ "m", "--show-thinking", "--no-thinking" }, "argument --show-thinking: not allowed with argument --no-thinking" },
        .{ &.{ "--url", "ftp://127.0.0.1:1" }, "argument --url: expected an http or https URL without credentials, query or fragment: 'ftp://127.0.0.1:1'" },
        .{ &.{ "--url", "http://user:secret@127.0.0.1:1" }, "argument --url: expected an http or https URL without credentials, query or fragment: 'http://user:secret@127.0.0.1:1'" },
        .{ &.{ "--url", "http://127.0.0.1:1", "--", "--parallel", "1" }, "flags after -- go to the server tensorfold chat starts; --url names one already running" },
        .{ &.{ "--url", "http://127.0.0.1:1", "--startup-timeout", "5" }, "argument --startup-timeout: applies only to the server tensorfold chat starts, not to --url" },
    };
    for (cases) |c| {
        try std.testing.expectError(error.Usage, parse(a, c[0], &problem));
        try std.testing.expectEqualStrings(c[1], problem);
    }
}

test "chat CLI: the help names every flag and the in-chat commands" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try help(&out.writer);
    for (flags) |f| try std.testing.expect(std.mem.indexOf(u8, out.written(), f.name) != null);
    for ([_][]const u8{ usage_line, "/bye", "/clear", "/system TEXT", "Ctrl-C stops a reply", "-- SERVE_FLAGS" }) |want|
        try std.testing.expect(std.mem.indexOf(u8, out.written(), want) != null);
}
